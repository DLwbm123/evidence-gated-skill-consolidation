"""Run the shared 100-step ALFWorld GRPO warm start with the original trainer."""
import os
from pathlib import Path
import runpy
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from warmup_state import resume_step, pin_shared


def main():
    root = Path(os.environ["BENCH_ROOT"])
    work = Path(os.environ["RUN_DIR"])
    source = root / "source/f-speed"
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    ranks = len(devices)
    if ranks not in (2, 4) or len(set(devices)) != ranks:
        raise ValueError("expected two or four distinct visible devices")
    gate = -1 if os.environ.get("BENCH_RELEASE") == "1" else 2
    work.mkdir(parents=True, exist_ok=True)
    step = resume_step(work, ranks)
    if step == 100:
        pin_shared(work/"ckpt", ranks)
        return
    if gate == 2 and step >= 2:
        raise RuntimeError("timing gate is already complete; review timings before release")
    if step > 100:
        raise ValueError("warmup directory contains a post-warmup checkpoint")
    # Upstream assigns games at environment reset. Parquet rows supply slots.
    for split, size in (("train", 1600), ("test", 128)):
        if (work / f"{split}.parquet").is_file():
            continue
        rows = [dict(data_source="text", prompt=[dict(role="user", content="")], ability="agent",
                     extra_info=dict(split=split, index=i)) for i in range(size)]
        pq.write_table(pa.Table.from_pylist(rows), work / f"{split}.parquet")
    sys.path.insert(0, str(source))
    os.chdir(source)
    sys.argv = ["runner",
        "algorithm.adv_estimator=grpo", "algorithm.use_kl_in_reward=False",
        f"data.train_files={work}/train.parquet", f"data.val_files={work}/test.parquet",
        "data.train_batch_size=16", "data.val_batch_size=128", "data.max_prompt_length=2048",
        "data.max_response_length=512", "data.filter_overlong_prompts=True", "data.truncation=error",
        "data.return_raw_chat=True", f"actor_rollout_ref.model.path={root}/models/m0",
        "+actor_rollout_ref.model.override_config._attn_implementation=sdpa",
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=1e-6", "actor_rollout_ref.actor.ppo_mini_batch_size=256",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.use_torch_compile=False", "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.01", "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0.001", "actor_rollout_ref.actor.use_invalid_action_penalty=True",
        "actor_rollout_ref.actor.invalid_action_penalty_coef=0.1",
        "actor_rollout_ref.rollout.name=vllm", "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.40", "actor_rollout_ref.rollout.max_num_seqs=64",
        "actor_rollout_ref.rollout.max_num_batched_tokens=8192",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False", "actor_rollout_ref.rollout.free_cache_engine=False",
        "actor_rollout_ref.rollout.enforce_eager=False", "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.4", "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.ref.fsdp_config.param_offload=False", "env.env_name=alfworld/AlfredTWEnv",
        "env.seed=0", "env.max_steps=50", "env.rollout.n=8", "env.resources_per_worker.num_cpus=0.1",
        "trainer.logger=[console]", "trainer.project_name=w", "trainer.experiment_name=w",
        f"trainer.default_local_dir={work}/ckpt", f"trainer.n_gpus_per_node={ranks}", "trainer.nnodes=1",
        "trainer.total_training_steps=100", "trainer.total_epochs=1", "trainer.val_before_train=False",
        "trainer.save_freq=10", "trainer.test_freq=25", f"+trainer.timing_stop_step={gate}", "trainer.max_actor_ckpt_to_keep=1",
        "trainer.resume_mode=auto", "ray_init.num_cpus=36", "+ray_init.include_dashboard=False",
        "+ray_init._node_ip_address=127.0.0.1", "+ray_init.object_store_memory=2147483648",
        f"+ray_init._temp_dir={work}/r", f"hydra.run.dir={work}/cfg"]
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__")
    saved = resume_step(work, ranks)
    if saved != (2 if gate == 2 else 100):
        raise RuntimeError(f"unexpected committed terminal step {saved}")
    if saved == 100:
        pin_shared(work/"ckpt", ranks)


if __name__ == "__main__":
    main()
