"""Run the shared 100-step ALFWorld GRPO warm start with the original trainer."""
import os
from pathlib import Path
import runpy
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    root = Path(os.environ["BENCH_ROOT"])
    work = Path(os.environ["RUN_DIR"])
    source = root / "source/f"
    work.mkdir(parents=True, exist_ok=True)
    # Upstream assigns games at environment reset. Parquet rows supply slots.
    for split, size in (("train", 1600), ("test", 128)):
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
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_torch_compile=False", "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.01", "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0.001", "actor_rollout_ref.actor.use_invalid_action_penalty=True",
        "actor_rollout_ref.actor.invalid_action_penalty_coef=0.1",
        "actor_rollout_ref.rollout.name=vllm", "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.28", "actor_rollout_ref.rollout.max_num_seqs=16",
        "actor_rollout_ref.rollout.max_num_batched_tokens=2560",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False", "actor_rollout_ref.rollout.free_cache_engine=False",
        "actor_rollout_ref.rollout.enforce_eager=False", "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.4", "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True", "env.env_name=alfworld/AlfredTWEnv",
        "env.seed=0", "env.max_steps=50", "env.rollout.n=8", "env.resources_per_worker.num_cpus=0.1",
        "trainer.logger=[console]", "trainer.project_name=w", "trainer.experiment_name=w",
        f"trainer.default_local_dir={work}/ckpt", "trainer.n_gpus_per_node=2", "trainer.nnodes=1",
        "trainer.total_training_steps=100", "trainer.total_epochs=1", "trainer.val_before_train=True",
        "trainer.save_freq=10", "trainer.test_freq=5", "trainer.max_actor_ckpt_to_keep=1",
        "trainer.resume_mode=disable", "ray_init.num_cpus=36", "+ray_init.include_dashboard=False",
        "+ray_init._node_ip_address=127.0.0.1", "+ray_init.object_store_memory=2147483648",
        f"+ray_init._temp_dir={work}/r", f"hydra.run.dir={work}/cfg"]
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__")


if __name__ == "__main__":
    main()
