# 首轮工程验收的协议偏离

- **PR-OPD 单卡诊断**。使用已验证的 PyTorch 2.8.0/CUDA 12.8 和 Transformer SDPA。基础策略最多执行八轮，最后一轮对齐响应。使用作者损失、投影头和事后技能提示，检查真实环境采样、教师停止梯度、策略回传与投影头更新。正式比较采用共同 GRPO 预热 checkpoint 和作者评测配置。
- **原版 trainer 冒烟配置**。使用两张 RTX PRO 5000 Blackwell、单卡张量并行、4 个训练任务、每任务 2 条 rollout、6 轮上限和单步更新。FSDP 模型暂用 SDPA 和有填充批次，vLLM 安装目标为 0.11.0。仅保存模型和运行状态的短任务 checkpoint。
- **原版 trainer 的 checkpoint 验收**。短任务保存后，用启动的第二个 trainer 进程在验证模式重载模型与投影头。保留作者的有填充 actor 路径，只在实际去填充时才导入 FlashAttention 的辅助函数。
- **GRPO 导入兼容**。匿名 trainer 包缺少 GiGPO 模块。将该模块延迟至选择 GiGPO 优势时导入，保留 GRPO 路径原有计算和评测。

- **FSOL Writing 的隐藏目标与绝对成绩**。本地在运行前选定 Cute 且质量为 good 作为平稳流的成功条件。《Fast and Slow Online Learning for Agents》表 1 报告零样本 17.9%，正文与附录给出了三个候选风格和评分量表，未能核实作者平稳流究竟奖励哪种风格。本地无记忆评测明显偏向 Chuunibyou，仅凭两边的零样本比例不能反推出作者的目标风格。作者题序、查询子集与本地划分的对应关系也未确认。[诊断与配对对照](reports/WRITING_KL_FOLLOWUP.md)
