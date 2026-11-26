"""Launch one original trainer update with a small ALFWorld batch."""

from pathlib import Path
import os
import runpy
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    selected = os.environ.get("CUDA_VISIBLE_DEVICES")
    if selected is not None and selected.replace(" ", "") != "2,3":
        raise RuntimeError("the two-GPU smoke requires physical CUDA devices 2,3")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    reloading = os.environ.get("BENCH_RELOAD") == "1"
    root = Path(os.environ["BENCH_ROOT"])
    launch_file = Path(__file__).resolve().relative_to(root.resolve())
    model = root / "models/m0"
    source = root / "source/f"
    work = root / "runs/r0"
    checkpoint = work / "global_step_1"
    work.mkdir(parents=True, exist_ok=True)
    free = subprocess.check_output(
        ["nvidia-smi", "-i", "2,3", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    if len(free) != 2 or min(map(int, free)) < 54000:
        print("Headroom changed before model initialization:", free, flush=True)
        raise SystemExit(75)
    print("free MiB at launch:", free, flush=True)

    if not reloading:
        for split in ("train", "test"):
            rows = [{"data_source": "text", "prompt": [{"role": "user", "content": ""}],
                     "ability": "agent", "extra_info": {"split": split, "index": i}}
                    for i in range(4)]
            pq.write_table(pa.Table.from_pylist(rows), work / f"{split}.parquet")

    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.argv = [
        "runner",
        "algorithm.adv_estimator=grpo", "algorithm.use_kl_in_reward=False",
        f"data.train_files={work}/train.parquet", f"data.val_files={work}/test.parquet",
        "data.train_batch_size=4", "data.val_batch_size=4", "data.max_prompt_length=2048",
        "data.max_response_length=128", "data.return_raw_chat=True",
        f"actor_rollout_ref.model.path={model}",
        "+actor_rollout_ref.model.override_config._attn_implementation=sdpa",
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_torch_compile=False",
        "actor_rollout_ref.actor.use_kl_loss=True", "actor_rollout_ref.actor.kl_loss_coef=0.01",
        "actor_rollout_ref.actor.use_invalid_action_penalty=True",
        "actor_rollout_ref.actor.invalid_action_penalty_coef=0.1",
        "actor_rollout_ref.actor.checkpoint.contents=[model,extra]",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.28",
        "actor_rollout_ref.rollout.max_num_seqs=16",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.4",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "env.seed=0", "env.max_steps=6", "env.rollout.n=2",
        "trainer.logger=[console]", "trainer.project_name=r0", "trainer.experiment_name=r0",
        f"trainer.default_local_dir={work}",
        "trainer.n_gpus_per_node=2", "trainer.nnodes=1", "trainer.total_training_steps=1",
        "trainer.total_epochs=2", f"trainer.val_before_train={reloading}",
        "trainer.save_freq=1", "trainer.test_freq=1", "trainer.max_actor_ckpt_to_keep=1",
        f"trainer.resume_mode={'resume_path' if reloading else 'disable'}",
        "ray_init.num_cpus=16",
        "+ray_init.include_dashboard=False", "+ray_init._node_ip_address=127.0.0.1",
        f"+ray_init._temp_dir={work}/{'reload' if reloading else 'ray'}",
        "+algorithm.pr_opd.oprd_coef=0.01", "+algorithm.pr_opd.oprd_layers=all",
        "+algorithm.pr_opd.oprd_normalize=True", "+algorithm.pr_opd.oprd_proj_mode=mlp",
        "+algorithm.pr_opd.oprd_proj_hidden=512", "+algorithm.pr_opd.log_skill_examples=0",
    ]
    if reloading:
        sys.argv.extend(("trainer.val_only=True", f"trainer.resume_from_path={checkpoint}"))
    import ray

    try:
        runpy.run_module("verl.trainer.main_pr_opd", run_name="__main__")
    finally:
        ray.shutdown()
    if reloading:
        print("Saved checkpoint reloaded through original trainer validation", flush=True)
    else:
        actor = checkpoint / "actor"
        assert len(list(actor.glob("model_world_size_2_rank_*.pt"))) == 2, "model shards incomplete"
        assert (actor / "oprd_projectors.pt").is_file(), "projection checkpoint absent"
        bytes_written = sum(item.stat().st_size for item in checkpoint.rglob("*") if item.is_file())
        print("Saved checkpoint bytes:", bytes_written, flush=True)
        subprocess.run([str(root / "env/b/bin/python"), str(launch_file)], cwd=root,
                       env={**os.environ, "BENCH_RELOAD": "1"}, check=True)


if __name__ == "__main__":
    main()
