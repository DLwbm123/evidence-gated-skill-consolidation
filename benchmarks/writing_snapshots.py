"""Load synchronized TP1 Writing policy weights and keep a bounded final delta."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time
import zlib

import numpy as np
from safetensors import safe_open
import torch


def load_engine(engine, tensors):
    worker = engine.llm_engine.model_executor.driver_worker
    model = worker.get_model()
    with torch.no_grad():
        loaded = model.load_weights(tensors)
    missing = set(dict(model.named_parameters())) - loaded
    if missing:
        raise RuntimeError(f"Incomplete rollout policy load: {sorted(missing)[:5]}")
    torch.cuda.synchronize(0)


def capture(engine, directory, update):
    scratch = directory/"scratch"
    index = json.loads((scratch/"model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(index.values()))
    timing = directory/"timings.jsonl"
    while True:
        count = len(timing.read_text().splitlines()) if timing.exists() else 0
        if count >= update and update < 300:
            return False
        if (count == update-1 or (update==300 and count==300)) and all(
            (scratch/s).is_file() and not (scratch/s).is_symlink() for s in shards
        ):
            # Atomic shard publication, followed by synchronous vLLM reload and cleanup.
            # Open all files before taking a snapshot. No disk copies.
            with ExitStack() as stack:
                try:
                    files = [stack.enter_context(safe_open(str(scratch/s),framework="pt")) for s in shards]
                except FileNotFoundError:
                    continue
                load_engine(engine, ((k, f.get_tensor(k)) for f in files for k in f.keys()))
            return True
        if (directory/"exit.json").exists():
            return False
        time.sleep(.4)


def store_delta(actor, base, destination, limit=3_000_000_000):
    """Persist exact BF16 XOR against the frozen base. Cap output before every write."""
    directory = destination.with_suffix(".part")
    directory.mkdir()
    mapping = json.loads((base/"model.safetensors.index.json").read_text())["weight_map"]
    state = actor.state_dict()
    total = 0
    manifest = []
    for shard in sorted(set(mapping.values())):
        with safe_open(str(base/shard), framework="pt") as f:
            for key in f.keys():
                original = f.get_tensor(key)
                trained = state[key].detach().cpu().contiguous()
                assert original.dtype == trained.dtype == torch.bfloat16 and original.shape == trained.shape
                xor = (trained.view(torch.int16) ^ original.view(torch.int16)).numpy().view(np.uint8).reshape(-1)
                filename = str(len(manifest))+".z"
                with (directory/filename).open("xb") as out:
                    obj = zlib.compressobj(1)
                    for offset in range(0,xor.size,4*1024**2):
                        block = obj.compress(xor[offset:offset+4*1024**2])
                        total += len(block)
                        if total >= limit:
                            raise RuntimeError("final delta exceeded storage allowance")
                        out.write(block)
                    tail = obj.flush()
                    total += len(tail)
                    if total >= limit:
                        raise RuntimeError("final delta exceeded storage allowance")
                    out.write(tail)
                manifest.append(dict(key=key,shape=list(trained.shape),file=filename))
    (directory/"index.json").write_text(json.dumps(manifest))
    os.replace(directory,destination)


def restore_tensor(original, directory, item):
    raw = zlib.decompress((directory/item["file"]).read_bytes())
    delta = torch.from_numpy(np.frombuffer(raw,dtype=np.int16).copy()).reshape(item["shape"])
    return (original.view(torch.int16) ^ delta).view(torch.bfloat16)
