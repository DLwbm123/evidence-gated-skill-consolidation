"""Audit saved Writing policy inputs and replay the original single-text judge."""
from collections import Counter
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parent = Path("..")
    policy = AutoTokenizer.from_pretrained("m0")
    tokenizer = AutoTokenizer.from_pretrained("m1")
    templates = json.loads(Path("templates.json").read_text())
    stream = json.loads(Path("split.json").read_text())["stream"]
    rows = records(parent / "c1/rows.jsonl")
    assert len(rows) == 200
    counts = {}
    for group, folder in [("zero", "c3"), ("memory", "c1"), ("matched_memory", "c2"), ("joint", ".")]:
        counts[group] = dict(Counter(r["verdict"].lower().strip().rstrip(".") for r in records(parent/folder/"rows.jsonl")))
    sample, inputs, positions = [], [], Counter()
    matches = valid_refs = full_context = successes = 0
    for r in rows:
        n, refs, ids = r["i"], r["retrieved"], r["prefix"]
        positions[len(refs)] += 1
        items = []
        for k, ref in enumerate(refs, 1):
            source = rows[ref-1]
            assert ref < n and source["reward"] == 1
            ans = passage(policy, source)
            items.append(f"Example Query {k}: {stream[ref-1]}\n\nExample Response {k}: {ans}")
            valid_refs += int(source["verdict"].lower().strip().rstrip(".").replace(" ", "") == "cute,good")
        task = templates["policy"].replace("{task_description}", stream[n-1])
        content = "\n\n".join([templates["memory"], *items, task]) if refs else task
        reconstructed = policy.apply_chat_template([dict(role="user", content=content)], add_generation_prompt=True)
        matches += int(reconstructed == ids)
        full_context += int(len(refs) == min(successes, 3))
        successes += r["reward"]
        if n in (10, 50, 100, 150, 200):
            sample.append(f"## Query {n} · refs {refs} · {len(ids)} tokens\n\n" + policy.decode(ids, clean_up_tokenization_spaces=False))
        dialog = [dict(role="system",content=templates["judge_system"]),
                  dict(role="user",content=templates["judge_user"].replace("{response}",passage(policy, r)))]
        inputs.append(tokenizer.apply_chat_template(dialog,add_generation_prompt=True,return_tensors="pt"))
    Path("samples.md").write_text("\n\n".join(sample))
    metrics = dict(styles=counts, retrieved_counts=dict(positions), examples=sum(k*v for k,v in positions.items()),
                   qualified_examples=valid_refs, same_input_tokens=matches, full_available_context=full_context,
                   max_prefix=max(len(r["prefix"]) for r in rows), n=len(rows))
    Path("audit.json").write_text(json.dumps(metrics))
    judge = AutoModelForCausalLM.from_pretrained("m1",torch_dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval()
    passes = []
    with torch.inference_mode():
        for _ in range(2):
            labels = []
            for ids in inputs:
                ids = ids.to(judge.device)
                out = judge.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=32,
                                     pad_token_id=tokenizer.eos_token_id)
                labels.append(tokenizer.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip())
            passes.append(labels)
    metrics.update(equal_verdict=sum(a==b for a,b in zip(*passes,strict=True)),
                   equal_original=[sum(s==r["verdict"] for s,r in zip(group, rows, strict=True)) for group in passes])
    Path("audit.json").write_text(json.dumps(metrics))
    Path("done.json").write_text(json.dumps({"passes":2,"n":len(rows)}))


def passage(policy, row):
    text = policy.decode(row["tokens"],skip_special_tokens=True,clean_up_tokenization_spaces=False)
    return " ".join(text.split()[:500]) if len(text.split()) > 500 else text


if __name__ == "__main__":
    main()
