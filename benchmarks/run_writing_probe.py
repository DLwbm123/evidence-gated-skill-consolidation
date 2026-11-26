"""Paired fixed-bank Writing probe for base, FSOL and pure-RL checkpoints."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def prepare():
    split = json.loads(Path("split.json").read_text())
    probes = split["probe"]
    assert len(set(probes)) == 64 and not set(probes).intersection(split["stream"])
    rows = [json.loads(x) for x in Path("history.jsonl").read_text().splitlines()]
    bank = [x for x in rows[:192] if x["reward"]]
    assert len(bank) == 62 and [x["i"] for x in rows[:192]] == list(range(1,193))
    template = json.loads(Path("templates.json").read_text())
    tok = AutoTokenizer.from_pretrained("m0")
    embed_tok = AutoTokenizer.from_pretrained("m2")
    embed = AutoModel.from_pretrained("m2",torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval()
    def vector(text):
        inp = embed_tok(text,return_tensors="pt",truncation=True,max_length=4096).to(embed.device)
        with torch.inference_mode():
            return F.normalize(embed(**inp).last_hidden_state[0,-1].float(),dim=0).cpu()
    texts = [split["stream"][x["i"]-1] for x in bank]
    vectors = torch.stack([vector(x) for x in texts])
    passages = [tok.decode(x["tokens"],skip_special_tokens=True,clean_up_tokenization_spaces=False) for x in bank]
    passages = [" ".join(x.split()[:500]) if len(x.split())>500 else x for x in passages]
    prepared = []
    for i,prompt in enumerate(probes):
        neighbors = (vectors @ vector(prompt)).argsort(descending=True,stable=True)[:3].tolist()
        for with_memory in (False,True):
            chosen = neighbors.copy() if with_memory else []
            while True:
                task = template["policy"].replace("{task_description}",prompt)
                items = [f"Example Query {k}: {texts[j]}\n\nExample Response {k}: {passages[j]}" for k,j in enumerate(chosen,1)]
                user = "\n\n".join([template["memory"],*items,task]) if chosen else task
                ids = tok.apply_chat_template([dict(role="user",content=user)],tokenize=True,add_generation_prompt=True)
                if len(ids)<=4096: break
                if not chosen: raise ValueError("bare probe exceeds budget")
                chosen.pop()
            prepared.append(dict(query=i+1,memory=with_memory,refs=[bank[j]["i"] for j in chosen],ids=ids))
    Path("inputs.json").write_text(json.dumps(prepared))


def evaluate(index):
    from vllm import LLM, SamplingParams
    root = Path(os.environ["RUN_ALIAS"])
    inputs = json.loads(Path("inputs.json").read_text())
    template = json.loads(Path("templates.json").read_text())
    judge_tok = AutoTokenizer.from_pretrained("m1")
    engine = LLM(model=str(root/f"a{index}"),dtype="bfloat16",tensor_parallel_size=1,
                 enforce_eager=True,max_model_len=5120,max_num_seqs=8,max_num_batched_tokens=5120,
                 enable_chunked_prefill=True,gpu_memory_utilization=.25,enable_prefix_caching=False,generation_config="vllm",seed=0)
    judge = AutoModelForCausalLM.from_pretrained("m1",torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval()
    with Path(f"scores{index}.jsonl").open("x") as stream:
        for offset in range(0,len(inputs),8):
            batch = inputs[offset:offset+8]
            params = [SamplingParams(temperature=.7,top_p=.8,top_k=20,seed=10000+x["query"],max_tokens=1024) for x in batch]
            generated = engine.generate([dict(prompt_token_ids=x["ids"]) for x in batch],params,use_tqdm=False)
            for x,response in zip(batch,generated,strict=True):
                passage = response.outputs[0].text
                if len(passage.split())>500: passage = " ".join(passage.split()[:500])
                dialog = [dict(role="system",content=template["judge_system"]),dict(role="user",content=template["judge_user"].replace("{response}",passage))]
                ids = judge_tok.apply_chat_template(dialog,tokenize=True,add_generation_prompt=True,return_tensors="pt").to(judge.device)
                with torch.inference_mode():
                    result = judge.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=32,pad_token_id=judge_tok.eos_token_id)
                verdict = judge_tok.decode(result[0,ids.size(1):],skip_special_tokens=True).strip()
                match = re.fullmatch(r"(Classical|Cute|Chuunibyou)\s*,\s*(good|bad)\.?",verdict,re.I)
                scored = dict(query=x["query"],memory=x["memory"],reward=int(bool(match and match[1].lower()=="cute" and match[2].lower()=="good")),verdict=verdict,parse_ok=bool(match))
                stream.write(json.dumps(scored)+"\n")
            stream.flush()


if __name__=="__main__":
    if len(sys.argv)==2:
        if sys.argv[1]=="0": prepare()
        else: evaluate(int(sys.argv[1])-1)
    else:
        for stage in range(4):
            subprocess.run([sys.executable,sys.argv[0],str(stage)],check=True)
        Path("done.json").write_text(json.dumps(dict(checkpoints=3,queries=64,arms=6,end=time.time())))
