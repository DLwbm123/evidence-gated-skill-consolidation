# 技能巩固实验进展

FSOL Writing 已完成种子 0 的 200 条串行校准，获得 65 次奖励，完成 12 次全参数更新。同一前缀的纯记忆对照为 66/200；统一解码后，更新前 16 条逐 token 一致，后 184 条 FSOL 命中 59 条、冻结策略命中 63 条。已启动零样本对照。[结果与协议偏离](FSOL_WRITING_CALIBRATION.md)

2026-11-26。已用 Qwen2.5-3B-Instruct 在一个 ALFWorld 训练任务上检验事后技能的生成，以及用该技能监督策略隐藏表示的计算路径。

| 单卡诊断 | 观测 |
|---|---:|
| 交互轮数 | 8 |
| 自生成事后技能 token | 63 |
| 教师和学生对齐的响应 token | 96 |
| 有监督的隐藏层 | 36 |
| 表示对齐损失 | 2.0071 |
| PyTorch 峰值已分配显存 | 13.45 GiB |

教师输出已停止梯度，策略梯度非零，投影头完成一次 AdamW 更新。该游戏的终局任务失败。技能选择和内化是否改善任务成功率，仍需带技能库及关闭技能库的对照实验。[诊断指标](direct-smoke.json)

## Benchmark 和集成验证

ALFWorld 共 3,553 个训练、200 个 valid_train、140 个 valid_seen 和 134 个 valid_unseen 游戏。六类任务各完成了一次参考动作回放。固定上游 verl-agent 提交 `20bd331bdbc9026a5668e11362178e10ab7400c8` 的技能库适配已通过语法编译。正式阶段按 Pick、Look、Clean、Heat、Cool、Pick2 的顺序评估迁移与遗忘。[ALFWorld](https://github.com/alfworld/alfworld)

双卡短程配置使用 PyTorch 2.8.0+cu128、vLLM 0.11.0、Transformers 4.55.2 和 Ray 2.59.0。以 4 个训练任务、每题 2 条 rollout、6 轮上限检验 trainer 的一次联合更新和 checkpoint 重载。使用 SDPA 和有填充批次，去填充和 GiGPO 辅助依赖按所选路径加载。已排查 Triton 的 Python 3.12 头文件路径和 Ray worker 的解释器路径。双卡首步已写出两份 actor 模型分片、extra state 和投影头 checkpoint，共 14,512,062,120 字节。重载进程在显存预检时退出，张量一致性和连续训练仍待验证。[执行与 checkpoint 核查记录](INITIAL_EXECUTION.md)

已从基础权重启动 ALFWorld 的 100 步共享 GRPO 预热；目标为在第 150 步按论文口径独立评估两组。有关采样和硬件差异见[复现协议与启动记录](ALFWORLD_REPRODUCTION.md)。
