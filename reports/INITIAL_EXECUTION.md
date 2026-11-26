# 算法方向首轮执行记录

2026-11-26。

- 实验服务器 的 GPU 2 上，Qwen2.5-3B-Instruct 在 ALFWorld 的一个训练游戏里完成了八轮短轨迹。策略自产 63 个事后技能 token，教师和学生对齐同一段 96 个响应 token。作者的 36 层损失有限，策略梯度非零，投影头完成一次 AdamW 更新。PyTorch 峰值已分配显存为 13.45 GiB。原始记录为 `.benchmarks/logs/pr-opd-direct-smoke.log`。这条轨迹未成功完成任务。[模型发布](https://www.modelscope.cn/models/Qwen/Qwen2.5-3B-Instruct)
- TrackRL 的 22 个作者补丁应用到了固定的 `verl-agent` 版本 `20bd331bdbc9026a5668e11362178e10ab7400c8`。改动和新增的 35 个 Python 文件通过语法编译。工作副本在本机 `.benchmarks/source/h/`。
- 实验服务器 已装好隔离环境，核对了 PyTorch 2.8.0+cu128、vLLM 0.11.0、Transformers 4.55.2、Ray 2.59.0 和 trainer、模型 worker、轨迹收集器、环境模块的导入。首次启动到达 Ray，但进程导入了匿名包缺失的 `gigpo`。这次使用 GRPO 优势，已把 GiGPO 导入移到实际使用该优势的分支。相对于远程项目根目录，`logs/n1.log` 保存安装，`logs/n3.log` 保存首次尝试，`logs/n4.log` 保存原时限内的重试。每张卡空闲达到 54,000 MiB 后才运行两个模型 worker。一次 actor 更新、模型和投影头 checkpoint 保存与重载仍需运行验收。
- 后续 `logs/n5.log` 已加载双卡模型并调用 vLLM 采样，Triton 编译因缺少 `Python.h` 退出。匹配的 Python 3.12 头文件经 C 预处理验证后通过 `CPATH` 接入。`logs/n8.log` 记录了 Ray worker 的含空格解释器路径错误。恢复中性的绝对解释器路径后在 `logs/n10.log` 重跑，沿用本轮原始时限。
- 正式第 100 至 150 步训练以及 TrackRL 的门控对照属于后继训练。

## 双卡 checkpoint 核查

原 trainer 已在首个训练步写出两份 actor 模型分片、extra state 和投影头 checkpoint，共 14,512,062,120 字节。重载子进程因显存余量降至 48,558/42,448 MiB，在加载任何状态前以 75 退出。源码中的 `_load_checkpoint()` 位于 `val_only` 返回之前，FSDP worker 的恢复函数包含投影头恢复调用。当前产物省略 actor 优化器状态，仅支持轻量验收；完整训练续跑需在保存时包含 actor 优化器，核对恢复后投影头权重及其优化器状态，并在相同输入和 RNG 下与未中断训练比较下一步。单纯观察相邻随机批次的 loss 无法检验恢复一致性。启动脚本现显式限定 CUDA 2、3，遇到冲突的可见设备掩码会退出。原实现的 all 层范围为 1 至 num_layers，不含索引 0。
