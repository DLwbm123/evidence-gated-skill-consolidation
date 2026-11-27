# Evidence-gated skill consolidation

研究交互式强化学习中应在何时将技能库中的经验巩固进策略模型。计划用技能掌握度选择训练样本，以事后技能为教师上下文对齐策略的隐藏状态，并在关闭技能库时评估内化效果。当前代码覆盖 ALFWorld 的任务分组、表示对齐诊断及双卡训练验收。[实验进展](reports/PROGRESS.md)

Writing 的纯 RL、纯记忆和 FSOL 对照结果见[阶段报告及逐题数据](reports/WRITING_INTERIM.md)。

## 入口和依赖

- `benchmarks/prepare_alfworld_families.py` 使用 Python 标准库，输入包含 `official_split`、`game_file` 和 `task_type` 的逐行 JSON 清单。游戏文件须能从清单上两级目录按 `game_file` 解析。
- `benchmarks/check_representation.py` 在 CUDA 上执行一个 ALFWorld 游戏，并检查教师停止梯度、策略梯度和投影头更新。需要 TextWorld、ALFWorld、PyTorch、Transformers 和独立取得的兼容 trainer 源码。脚本从当前目录的 `source/f` 读取该实现，从 `models/m0` 读取 Qwen2.5-3B-Instruct，从 `a-protocol` 读取任务索引及游戏文件。
- `benchmarks/run_training_smoke.py` 是原 trainer 的短程启动器。它使用环境变量 `BENCH_ROOT` 下的 `source/f`、`models/m0` 和 `env/b/bin/python`，需要 PyArrow、Ray、vLLM 及两张可用 GPU。启动器须位于 `BENCH_ROOT` 内。输入环境与 FSDP 兼容性要求见进展报告。

```sh
python benchmarks/prepare_alfworld_families.py path/to/processed/games.jsonl path/to/families.json
```
