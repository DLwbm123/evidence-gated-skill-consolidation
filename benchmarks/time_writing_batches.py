"""Time two 16-query Writing batches, starting from an early FSOL checkpoint."""
import json
import os
from pathlib import Path
import re
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from run_writing_calibration import batches_update, log, publish_weights


def main():
    root = Path(os.environ["RUN_ALIAS"])
    split = json.loads(Path("split.json").read_text())
    cfg = json.loads(Path("config.json").read_text())
    mode = cfg.get("mode")
    factorial = mode in {"joint_100", "rl_100"}
    rl_only = mode in {"rl", "rl_100"}
    batches = 100 if factorial else 300
    kl = float(cfg["kl"]) if factorial else (.01 if rl_only else .05)
    if factorial:
        assert kl in (.01, .05)
        from writing_reference_probe import REFERENCES, measure
    controls = factorial or mode in {"rl", "memory", "snapshot"}
    full = mode == "full" or controls
    training = mode not in {"memory", "snapshot"}
    fixed = json.loads(Path("../c5/inputs.json").read_text()) if controls else []
    if controls:
        assert len(fixed) == 128 and all(x["query"] == i//2+1 and x["memory"] == bool(i%2) for i,x in enumerate(fixed))
        from writing_snapshots import capture, load_engine, store_delta
    history = [] if full else [json.loads(x) for x in Path("history.jsonl").read_text().splitlines()]
    stream = split["stream"]
    assert len(set(stream)) == 4800 and set(split["probe"]).isdisjoint(stream)
    template = json.loads(Path("templates.json").read_text())
    live = Path("scratch")
    live.mkdir()
    for f in Path("initial").iterdir():
        (live/f.name).symlink_to(Path("..")/f)
    tok, jt, et = [AutoTokenizer.from_pretrained(s) for s in ("m0", "m1", "m2")]
    jt.padding_side = "left"
    opts = dict(torch_dtype=torch.bfloat16,attn_implementation="sdpa")
    llm = LLM(model=str(root/live), dtype="bfloat16", tensor_parallel_size=1, enforce_eager=True,
              max_model_len=5120, max_num_seqs=16, max_num_batched_tokens=8192, enable_chunked_prefill=True,
              gpu_memory_utilization=.25, enable_prefix_caching=False, generation_config="vllm", seed=0)
    judge = AutoModelForCausalLM.from_pretrained("m1",**opts).to("cuda:0").eval()
    embed = AutoModel.from_pretrained("m2",**opts).to("cuda:0").eval()
    if training:
        actor = AutoModelForCausalLM.from_pretrained("initial",**opts).to("cuda:1").train()
        actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        ref = AutoModelForCausalLM.from_pretrained("m0",**opts).to("cuda:1").eval().requires_grad_(False)
        masters = [torch.nn.Parameter(p.detach().to(device="cpu",dtype=torch.float32)) for p in actor.parameters()]
        optim = torch.optim.AdamW(masters,lr=1e-6,betas=(.9,.999),weight_decay=.01,foreach=False)
    def vector(text):
        ids = et(text,return_tensors="pt",truncation=True,max_length=4096).to(embed.device)
        with torch.inference_mode():
            return F.normalize(embed(**ids).last_hidden_state[0,-1].float(),dim=0).cpu()
    def cap(text):
        return " ".join(text.split()[:500]) if len(text.split())>500 else text
    memory = [(r["i"],stream[r["i"]-1],cap(tok.decode(r["tokens"],skip_special_tokens=True,clean_up_tokenization_spaces=False)))
              for r in history[:192] if r["reward"]]
    assert len(memory)==(0 if full else 62)
    vectors = [vector(q) for _,q,_ in memory]
    print("models_ready",flush=True)
    def generate(prompts, numbers, *, probe=False, use_memory=True, draw=0, frozen=False):
        begin = time.monotonic()
        queries = [vector(x) for x in prompts] if use_memory and not frozen else []
        rankings = ([v.argsort(descending=True,stable=True)[:3].tolist() for v in
                     (torch.stack(queries) @ torch.stack(vectors).T)] if vectors and use_memory and not frozen else [[] for _ in prompts])
        prefixes, refs = [], []
        for idx, (prompt, neighbors) in enumerate(zip(prompts,rankings,strict=True)):
            if frozen:
                item = fixed[(numbers[idx]-1)*2 + int(use_memory)]
                prefixes.append(dict(prompt_token_ids=item["ids"]))
                refs.append(item["refs"])
                continue
            while True:
                items = [f"Example Query {k}: {memory[j][1]}\n\nExample Response {k}: {memory[j][2]}" for k,j in enumerate(neighbors,1)]
                task = template["policy"].replace("{task_description}",prompt)
                text = "\n\n".join([template["memory"],*items,task]) if items else task
                ids = tok.apply_chat_template([dict(role="user",content=text)],tokenize=True,add_generation_prompt=True)
                if len(ids)<=4096: break
                if not neighbors: raise ValueError("bare prompt too long")
                neighbors.pop()
            prefixes.append(dict(prompt_token_ids=ids))
            refs.append([memory[j][0] for j in neighbors])
        if not probe:
            assert all(i < numbers[0] for group in refs for i in group)
        prepared=time.monotonic()
        params=[SamplingParams(seed=(10000+n+100000*draw if probe else n-1),temperature=.7 if probe or mode=="memory" else 1.,
                               top_p=.8 if probe or mode=="memory" else 1.,top_k=20 if probe or mode=="memory" else -1,
                               max_tokens=1024,logprobs=None if probe or not training else 0) for n in numbers]
        outputs=llm.generate(prefixes,params,use_tqdm=False)
        generated=time.monotonic()
        passages=[cap(o.outputs[0].text) for o in outputs]
        dialogs=[[dict(role="system",content=template["judge_system"]),dict(role="user",content=template["judge_user"].replace("{response}",s))] for s in passages]
        encoded=jt.apply_chat_template(dialogs,add_generation_prompt=True,tokenize=True,padding=True,return_tensors="pt",return_dict=True).to(judge.device)
        with torch.inference_mode():
            judged=judge.generate(**encoded,do_sample=False,max_new_tokens=32,pad_token_id=jt.eos_token_id)
        verdicts=jt.batch_decode(judged[:,encoded.input_ids.size(1):],skip_special_tokens=True)
        rows=[]; parsed=0
        for i,(o,verdict) in enumerate(zip(outputs,verdicts,strict=True)):
            match=re.fullmatch(r"(Classical|Cute|Chuunibyou)\s*,\s*(good|bad)\.?",verdict.strip(),re.I)
            parsed+=int(bool(match))
            reward=int(bool(match and match[1].lower()=="cute" and match[2].lower()=="good"))
            out=o.outputs[0];tokens=list(out.token_ids)
            if probe:
                rows.append(dict(query=numbers[i],memory=use_memory,reward=reward,parsed=bool(match),refs=refs[i],verdict=verdict.strip(),draw=draw,bank="fixed" if frozen and use_memory else ("online" if use_memory else "none")))
            else:
                old=[d[t].logprob for d,t in zip(out.logprobs,tokens,strict=True)] if training else []
                rows.append(dict(prefix=prefixes[i]["prompt_token_ids"],tokens=tokens,old=old,reward=reward))
                if full:
                    log("queries.jsonl",dict(query=numbers[i],version=(numbers[i]-1)//16,reward=reward,
                                             verdict=verdict.strip(),refs=refs[i],tokens=len(tokens)))
                if reward and use_memory:
                    memory.append((numbers[i],prompts[i],passages[i]));vectors.append(queries[i])
        return rows, refs, parsed, [prepared-begin,generated-prepared,time.monotonic()-generated]
    def probes(update, *, frozen, draws):
        bank_size=len(memory)
        for use in (False,True):
            for draw in range(draws):
                for pos in range(0,64,16):
                    results,_,_,_=generate(split["probe"][pos:pos+16],list(range(pos+1,pos+17)),
                                           probe=True,use_memory=use,draw=draw,frozen=frozen)
                    for item in results: log("probes.jsonl",dict(update=update,**item))
        assert len(memory)==bank_size

    if mode == "snapshot":
        for update in (200,250,300):
            captured = capture(llm,Path("../c7"),update)
            log("captures.jsonl",dict(update=update,captured=captured))
            if captured:
                probes(update,frozen=True,draws=4)
        Path("done.json").write_text(json.dumps(dict(end=time.time())))
        return
    if factorial:
        assert set(stream + split["probe"]).isdisjoint(item[0] for item in REFERENCES)
        log("reference.jsonl",dict(update=0,items=measure(actor,tok,template["policy"])))
    for start in (range(0,batches*16,16) if full else (320,336)):
        begin=time.monotonic()
        width=1 if full and start<320 else 16
        batch=[];refs=[];parsed=0;times=[0.,0.,0.]
        for pos in range(start,start+16,width):
            rows,selected,ok,elapsed=generate(stream[pos:pos+width],list(range(pos+1,pos+width+1)),use_memory=not rl_only)
            batch.extend(rows);refs.extend(selected);parsed+=ok
            times=[a+b for a,b in zip(times,elapsed,strict=True)]
        scored=time.monotonic()
        metrics = batches_update(actor,ref,optim,masters,batch,kl) if training else {}
        updated=time.monotonic()
        if rl_only:
            load_engine(llm,((name,p.detach().to("cuda:0")) for name,p in actor.named_parameters()))
        elif training:
            publish_weights(actor,live)
            llm.collective_rpc("reload_weights")
            for f in ([] if mode=="full" and start+16==4800 else live.glob("*.safetensors")):
                assert not f.is_symlink()
                f.unlink()
        synced=time.monotonic()
        log("timings.jsonl",dict(first=start+1,n=16,parsed=parsed,refs=refs,retrieval_s=times[0],
                                 generation_s=times[1],judge_s=times[2],update_s=updated-scored,
                                 sync_s=synced-updated,total_s=time.monotonic()-begin,**metrics))
        update=start//16+1
        if factorial and update%10==0:
            log("reference.jsonl",dict(update=update,items=measure(actor,tok,template["policy"])))
        if full and update%50==0:
            probes(update,frozen=rl_only or factorial,draws=4 if controls else 1)
    if mode=="rl":
        store_delta(actor,Path("m0"),Path("final_delta"))
    Path("done.json").write_text(json.dumps(dict(batches=batches if full else 2,end=time.time())))


if __name__=="__main__":
    main()
