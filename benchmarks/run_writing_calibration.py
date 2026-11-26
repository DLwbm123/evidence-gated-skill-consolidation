"""Run the serial prefix of the FSOL Writing stream in a prepared private run directory."""
import json
import os
from pathlib import Path
from random import Random
import re
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from verl.trainer.ppo.core_algos import compute_policy_loss, kl_penalty
from vllm import LLM, SamplingParams


def log(name, record):
    with Path(name).open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def batches_update(actor, ref, optimizer, masters, rows, kl_coefficient):
    rewards = torch.tensor([r["reward"] for r in rows], dtype=torch.float)
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
    count = sum(len(r["tokens"]) for r in rows)
    actor.zero_grad(set_to_none=True)
    observed_loss = 0.
    for row, advantage in zip(rows, advantages):
        obs = row["prefix"]
        actions = row["tokens"]
        ids = torch.tensor([obs + actions], device=actor.device)
        def score(model):
            logits = model(ids, use_cache=False, logits_to_keep=len(actions)+1).logits[:, :-1].float()
            return -F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[0,-len(actions):], reduction="none")[None]
        with torch.no_grad():
            reference = score(ref)
        current = score(actor)
        old = current.new_tensor([row["old"]])
        mask = torch.ones_like(current)
        pg = compute_policy_loss(old, current, advantage.to(current.device).expand_as(current),
                                 mask, cliprange=.2, clip_ratio_c=float(1e8))[0]
        loss = (pg + kl_coefficient * kl_penalty(current, reference, "low_var_kl").mean()) * (len(actions)/count)
        loss.backward()
        observed_loss += loss.detach().item()
    norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True).item()
    with torch.no_grad():
        for param, master in zip(actor.parameters(), masters, strict=True):
            master.grad = param.grad.to(device="cpu", dtype=torch.float32)
            param.grad = None
        optimizer.step()
        for param, master in zip(actor.parameters(), masters, strict=True):
            param.copy_(master)
        optimizer.zero_grad(set_to_none=True)
    return dict(loss=observed_loss, grad_norm=norm, tokens=count, rewards=int(rewards.sum()))


def publish_weights(actor, live):
    mapping = json.loads((live/"model.safetensors.index.json").read_text())["weight_map"]
    state = actor.state_dict()
    for filename in sorted(set(mapping.values())):
        tensors = {k: state[k].detach().cpu().contiguous() for k, shard in mapping.items() if shard == filename}
        temp = live/(filename+".part")
        save_file(tensors, str(temp), metadata={"format":"pt"})
        temp.replace(live/filename)


def main():
    cfg = json.loads(Path("config.json").read_text())
    templates = json.loads(Path("templates.json").read_text())
    method = cfg.get("method", "fsol")
    if method not in {"fsol", "memory", "memory_matched", "zero", "rl"}:
        raise ValueError("unknown Writing method")
    train = method in {"fsol", "rl"}
    use_memory = method not in {"zero", "rl"}
    kl_coefficient = .01 if method == "rl" else .05
    seed = cfg["seed"]
    torch.manual_seed(seed)
    prompts = list(dict.fromkeys(x.strip() for x in Path("input.txt").read_text().splitlines() if x.strip()))
    Random(seed).shuffle(prompts)
    assert len(prompts)>=4864
    Path("split.json").write_text(json.dumps(dict(stream=prompts[:4800], probe=prompts[4800:4864])))
    live = Path("live")
    live.mkdir(exist_ok=False)
    for original in Path("m0").iterdir():
        (live/original.name).symlink_to(Path("..")/original)
    tok, judge_tok, embed_tok = [AutoTokenizer.from_pretrained(k, local_files_only=True) for k in ("m0","m1","m2")]
    engine = LLM(model=str(Path(os.environ["RUN_ALIAS"])/live), dtype="bfloat16", tensor_parallel_size=1, enforce_eager=True,
                 max_model_len=5120, max_num_seqs=1, max_num_batched_tokens=5120, seed=seed,
                 gpu_memory_utilization=.20, enable_prefix_caching=False, generation_config="vllm")
    opts = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    judge = AutoModelForCausalLM.from_pretrained("m1", **opts).to("cuda:0").eval()
    if use_memory:
        embed = AutoModel.from_pretrained("m2", **opts).to("cuda:0").eval()
    if train:
        actor = AutoModelForCausalLM.from_pretrained("m0", **opts).to("cuda:1").train()
        actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        ref = AutoModelForCausalLM.from_pretrained("m0", **opts).to("cuda:1").eval().requires_grad_(False)
        masters = [torch.nn.Parameter(p.detach().to(device="cpu", dtype=torch.float32)) for p in actor.parameters()]
        optimizer = torch.optim.AdamW(masters, lr=1e-6, betas=(.9,.999), weight_decay=.01, foreach=False)
    memory, vectors, batch = [], [], []
    print("models_ready", flush=True)
    updates = successes = 0
    temperature, top_p, top_k = (.7, .8, 20) if method in {"memory", "zero"} else (1., 1., -1)
    for i, prompt in enumerate(prompts[:cfg["queries"]]):
        started = time.time()
        indices = []
        if use_memory:
            with torch.inference_mode():
                inputs = embed_tok(prompt,return_tensors="pt",truncation=True,max_length=4096).to(embed.device)
                vector = F.normalize(embed(**inputs).last_hidden_state[0,-1].float(),dim=0).cpu()
            indices = (torch.stack(vectors) @ vector).argsort(descending=True,stable=True)[:3].tolist() if vectors else []
        task = templates["policy"].replace("{task_description}",prompt)
        def prefix():
            items = [f"Example Query {k}: {memory[j][1]}\n\nExample Response {k}: {memory[j][2]}" for k,j in enumerate(indices,1)]
            user = "\n\n".join([templates["memory"], *items, task]) if indices else task
            return tok.apply_chat_template([dict(role="user",content=user)],tokenize=True,add_generation_prompt=True)
        ids = prefix()
        while len(ids)>4096 and indices:
            indices.pop()
            ids = prefix()
        if len(ids)>4096: raise ValueError("bare task exceeds context budget")
        sampler = SamplingParams(temperature=temperature,top_p=top_p,top_k=top_k,max_tokens=1024,logprobs=0,seed=seed+i)
        out = engine.generate([dict(prompt_token_ids=ids)],sampler,use_tqdm=False)[0].outputs[0]
        tokens = list(out.token_ids)
        old = [choices[t].logprob for choices,t in zip(out.logprobs,tokens,strict=True)]
        assert tokens and len(tokens)==len(old)
        generated = time.time()
        passage = " ".join(out.text.split()[:500]) if len(out.text.split())>500 else out.text
        dialog = [dict(role="system",content=templates["judge_system"]),
                  dict(role="user",content=templates["judge_user"].replace("{response}",passage))]
        jids = judge_tok.apply_chat_template(dialog,tokenize=True,add_generation_prompt=True,return_tensors="pt").to(judge.device)
        with torch.inference_mode():
            judged = judge.generate(jids,attention_mask=torch.ones_like(jids),do_sample=False,max_new_tokens=32,
                                    pad_token_id=judge_tok.eos_token_id)
        verdict = judge_tok.decode(judged[0,jids.size(1):],skip_special_tokens=True).strip()
        match = re.fullmatch(r"(Classical|Cute|Chuunibyou)\s*,\s*(good|bad)\.?",verdict,re.I)
        reward = int(bool(match and match[1].casefold()==cfg["target"].casefold() and match[2].lower()=="good"))
        row = dict(i=i+1,prefix=ids,tokens=tokens,old=old,reward=reward)
        if train:
            batch.append(row)
        refs = [memory[j][0] for j in indices]
        successes += reward
        if use_memory and reward:
            memory.append((i+1,prompt,passage)); vectors.append(vector)
        log("rows.jsonl",dict(**row,retrieved=refs,verdict=verdict,parse_ok=bool(match),policy_version=updates,
                             retrieval_generation_s=generated-started,judge_s=time.time()-generated))
        if len(batch)==16:
            begin = time.time()
            metrics = batches_update(actor, ref, optimizer, masters, batch, kl_coefficient)
            publish_weights(actor,live)
            engine.collective_rpc("reload_weights")
            log("updates.jsonl",dict(after_query=i+1,seconds=time.time()-begin,**metrics))
            print("update",i+1,metrics,flush=True)
            updates += 1
            batch.clear()
    Path("done.json").write_text(json.dumps(dict(queries=cfg["queries"],updates=updates,
                                                   pending=len(batch),successes=successes,end=time.time())))


if __name__=="__main__":
    main()
