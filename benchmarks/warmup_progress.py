"""Record throughput at completed optimizer steps; pin the shared checkpoint on commit."""
import argparse
import json
from pathlib import Path
import re
import shutil
import signal
import subprocess
from statistics import fmean
import time

from warmup_state import generations, pin_shared, retained_dir, rotate_fallback

METRIC = re.compile(r"(?:^| - )([\w/]+):([0-9.eE+\-]+)")


def timing_rows(lines):
    steps = {}
    for line in lines:
        if "timing_s/step:" not in line:
            continue
        fields = dict(METRIC.findall(line[line.index("step:"):]))
        if "training/global_step" in fields:
            steps[int(float(fields["training/global_step"]))] = {
                k: float(v) for k, v in fields.items() if k.startswith("timing_s/")}
    return steps


def forecast(steps, val_seconds, budget_left):
    if len(steps) < 2:
        return {}
    recent = [steps[k] for k in sorted(steps)[-3:]]
    train = fmean(r["timing_s/step"]-r.get("timing_s/testing", 0)-r.get("timing_s/save_checkpoint", 0)
                  for r in recent)
    testing = [r["timing_s/testing"] for r in steps.values() if "timing_s/testing" in r]
    saving = [r["timing_s/save_checkpoint"] for r in steps.values() if "timing_s/save_checkpoint" in r]
    val = fmean(testing) if testing else val_seconds
    last = max(steps)
    scenarios = {}
    for freq in (5, 25):
        for target in (100, 150):
            lower = (target-last)*train
            v = target//freq - last//freq
            saves = target//10 - last//10
            estimate = lower + v*val + saves*(fmean(saving) if saving else 0) if val is not None else None
            scenarios[f"step{target}_val{freq}"] = dict(seconds=estimate, train_only_seconds=lower,
                exceeds_remaining=estimate > budget_left if estimate is not None else None)
    return dict(training_seconds=train, validation_seconds=val,
                save_seconds=fmean(saving) if saving else None, scenarios=scenarios)


def disk_usage(work):
    paths = [work]
    if retained_dir(work).exists():
        paths.append(retained_dir(work))
    # One du invocation counts hard links across both directories only once.
    last = subprocess.check_output(["du", "-skc", *paths], text=True).splitlines()[-1]
    return int(last.split()[0])*1024


def watch(work, log, initial_start, pid, offset=0):
    from os import kill
    work = work.absolute()
    start = float((work/"start").read_text())
    promoted = 0
    val = None
    if (work/"progress.json").exists():
        val = json.loads((work/"progress.json").read_text()).get("initial_validation_seconds")
    steps = {}
    report = work / "progress.json"
    with log.open(errors="replace") as stream:
        stream.seek(offset)
        while time.time() < start+86400:
            used = disk_usage(work)
            if used >= 90_000_000_000 or shutil.disk_usage(work).free < 100_000_000_000:
                print(f"disk guard stop: {used} allocated bytes", flush=True)
                kill(pid, signal.SIGTERM)
                return
            lines = stream.readlines()
            if not initial_start and any("test_gen_batch meta info" in line for line in lines):
                initial_start = time.time()
            first_validation = initial_start and val is None and any("Initial validation metrics" in line for line in lines)
            if first_validation:
                val = time.time()-initial_start
            changed = timing_rows(lines)
            steps.update(changed)
            now = time.time()
            if changed or not report.exists() or first_validation:
                result = dict(completed_step=max(steps, default=0), initial_validation_seconds=val,
                              observed_at=now, disk_allocated_bytes=used, time_left_seconds=start+86000-now,
                              forecast=forecast(steps, val, start+86000-now))
                tmp = report.with_suffix(".tmp")
                tmp.write_text(json.dumps(result, indent=2)+"\n")
                tmp.replace(report)
            # The tracker is written only after rank saves and dataloader save.
            marker = work/"ckpt/latest_checkpointed_iteration.txt"
            value = marker.read_text().strip() if marker.is_file() else ""
            if value.isdecimal():
                latest = int(value)
                if latest == 100:
                    pin_shared(work/"ckpt", 2)
                    return
                if latest != promoted and (not generations(retained_dir(work)) or max(steps, default=0) >= latest+9):
                    checkpoint = work/"ckpt"/f"global_step_{latest}"
                    allocated = int(subprocess.check_output(["du", "-sk", checkpoint], text=True).split()[0])*1024
                    overhead = int(subprocess.check_output(
                        ["du", "-sk", "--exclude=ckpt", "--exclude=protected", work], text=True).split()[0])*1024
                    if 2*allocated+overhead >= 90_000_000_000:
                        print("two generations exceed the working disk allowance", allocated, overhead, flush=True)
                        kill(pid, signal.SIGTERM)
                        return
                    rotate_fallback(work, latest, 2)
                    promoted = latest
            try:
                kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(3)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("work", type=Path)
    cli.add_argument("log", type=Path)
    cli.add_argument("start_val", type=float)
    cli.add_argument("supervisor", type=int)
    cli.add_argument("--offset", type=int, default=0)
    a = cli.parse_args()
    watch(a.work, a.log, a.start_val, a.supervisor, a.offset)
