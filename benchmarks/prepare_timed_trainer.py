"""Install two narrowly scoped compatibility changes in a private trainer copy."""
from pathlib import Path
import argparse


def change(path, old, new):
    source = path.read_text()
    if new in source:
        return
    if source.count(old) != 1:
        raise ValueError(f"unexpected upstream revision: {path.name}")
    temp = path.with_suffix(".new")
    temp.write_text(source.replace(old, new))
    temp.replace(path)


def prepare(source):
    worker = source / "verl/workers/fsdp_workers.py"
    change(worker, 'cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)',
           'cpu_offload = CPUOffload(offload_params=True) if role == "ref" and fsdp_config.get("param_offload", True) else None')
    trainer = source / "verl/trainer/ppo/ray_trainer.py"
    save = "if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):"
    change(trainer, save, 'timing_stop = self.global_steps == self.config.trainer.get("timing_stop_step", -1)\n                    ' +
           save.replace("is_last_step or", "timing_stop or is_last_step or"))
    change(trainer, "progress_bar.update(1)\n                self.global_steps += 1",
           'progress_bar.update(1)\n                if timing_stop:\n                    progress_bar.close()\n                    return\n                self.global_steps += 1')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    prepare(parser.parse_args().source)
