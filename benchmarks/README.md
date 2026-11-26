# 算法 Benchmark 准备记录

## 算法创新

### ALFWorld

官方文本游戏和元数据位于 `.benchmarks/a/`，包含 3,553 个 train、200 个 valid_train、140 个 valid_seen、134 个 valid_unseen，共 4,027 个。DuetHarness 作者的独立划分校验通过 55/55 项。其训练池划为 1,776 个冷启动任务和 1,777 个在线 RL 任务。

实验服务器 上的 ALFWorld 0.4.2 和 TextWorld 1.7.0 已通过六类任务各一条训练 episode 的参考动作回放。六条轨迹都完成了 reset、逐步交互和成功评分。远程日志位于项目目录 `logs/a-smoke.log`，依赖版本位于 `logs/a-requirements.txt`，本机日志副本位于 `.benchmarks/logs/a-remote-smoke.log`。后续仿真默认使用 实验服务器 的游戏目录 `a/json_2.1.1/` 和划分索引 `a-protocol/families.json`。

TrackRL 使用 seen 和 unseen 两个评测划分，并在同一模型权重下分别评测带技能和关闭整个技能库的表现。PR-OPD 原版 `scripts/eval.sh` 使用 seen 划分的 128 个评测 episode，采样温度为 0.4。两者共享游戏文件，各自按论文或原仓库的采样协议评测。

六阶段实验的索引按照 Pick → Look → Clean → Heat → Cool → Pick2 生成，各任务类别保留原始划分。训练任务数量依次为 790、308、650、459、533、813。各阶段评测时使用固定的 seen 和 unseen 任务，保留单独的结果。索引位于本机 `.benchmarks/source/d/data/alfworld/splits/families.json` 和 实验服务器 的项目目录 `a-protocol/families.json`。其中的游戏路径相对于 ALFWorld 数据根目录。

### 其他环境

| Benchmark | 本轮状态 | 路径 |
|---|---|---|
| AppWorld | 本机 `0.1.3.post1`，train/dev/test_normal/test_challenge 分别为 90/57/168/417。参考轨迹通过了一个 train 任务的 reset、API 调用和评分 | `.benchmarks/b/`，日志 `.benchmarks/logs/b-smoke.log` |
| DBBench | DuetHarness 的 SQLite 后端。test 300、valid 200、train 4,632。作者的数据校验 54/54 通过，实际 SELECT/INSERT 评分及错误答案拒绝检查通过 | `.benchmarks/source/d/data/dbbench/` |
| WebShop | DuetHarness 固定的官方代码版本 `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd` 和作者补丁已落盘。两篇论文的代码均采用完整商品库。本机全量下载未完成，Lucene 索引和交互验收待完成 | 后续下载写入 实验服务器 项目目录的 `data/webshop/raw/`。本机初次下载日志在 `.benchmarks/logs/w-transfer.log` |

WebShop 续传入口在 实验服务器 项目目录的 `migration/20261126-local/artifacts/downloads/fetch.sh`。它继续使用 `data/webshop/raw/` 中的断点文件。

私有作者工件保存在 `.benchmarks/source/d/`（DuetHarness）、`.benchmarks/source/e/`（TrackRL）和 `.benchmarks/source/f/`（PR-OPD）。TrackRL README 指定的上游 `verl-agent` 提交 `20bd331bdbc9026a5668e11362178e10ab7400c8` 已检出至 `.benchmarks/source/g/`。
