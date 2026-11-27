"""Blind cross-judge agreement on 200 preselected, cached pilot responses."""
from collections import Counter
import json
import os
from pathlib import Path
import random
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from run_writing_calibration import log


def main():
    root = Path(os.environ["RUN_ALIAS"])
    tok = AutoTokenizer.from_pretrained("m0")
    judge = AutoTokenizer.from_pretrained("j")
    templates = json.loads(Path("templates.json").read_text())
    rng = random.Random(2711)
    # One answer per query; each arm contributes 50 distinct queries.
    order = list(range(200))
    rng.shuffle(order)
    examples = []
    for arm, name in enumerate(("c3","c1",".","c4")):
        rows = [json.loads(x) for x in (Path("..")/name/"rows.jsonl").read_text().splitlines()]
        for pos in order[50*arm:50*(arm+1)]:
            r = rows[pos]
            assert r["i"] == pos+1
            passage = tok.decode(r["tokens"],skip_special_tokens=True,clean_up_tokenization_spaces=False)
            if len(passage.split())>500:
                passage = " ".join(passage.split()[:500])
            dialog = [dict(role="system",content=templates["judge_system"]),
                      dict(role="user",content=templates["judge_user"].replace("{response}",passage))]
            ids = judge.apply_chat_template(dialog,add_generation_prompt=True,enable_thinking=False)
            examples.append((arm,pos+1,r["verdict"],ids))
    rng.shuffle(examples)
    assert len({x[1] for x in examples}) == 200
    engine = LLM(model=str(root/"j"),tensor_parallel_size=1,dtype="half",enforce_eager=True,
                 max_model_len=4096,max_num_seqs=8,gpu_memory_utilization=.58,enable_prefix_caching=False,
                 generation_config="vllm",seed=0)
    pattern = re.compile(r"(Classical|Cute|Chuunibyou)\s*,\s*(good|bad)\.?",re.I)
    matrix = Counter()
    for i in range(0,200,8):
        batch=examples[i:i+8]
        outputs=engine.generate([dict(prompt_token_ids=item[3]) for item in batch],
                                SamplingParams(temperature=0,max_tokens=32),use_tqdm=False)
        for (arm,q,original,_),o in zip(batch,outputs,strict=True):
            a=pattern.fullmatch(original.strip())
            assert a
            b=pattern.fullmatch(o.outputs[0].text.strip())
            style=b[1].lower() if b else "unparsed"
            matrix[a[1].lower(),style]+=1
            log("pairs.jsonl", dict(group=arm,query=q,first_style=a[1].lower(),second_style=style,
                                    first_quality=a[2].lower(),second_quality=b[2].lower() if b else "unparsed"))
    assert sum(matrix.values())==200
    Path("done.json").write_text(json.dumps({"pairs":200,"agreement":sum(v for (a,b),v in matrix.items() if a==b)}))


if __name__=="__main__":
    main()
