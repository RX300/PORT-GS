# LiSA 补充实验

本目录是本轮实验的唯一阅读入口。模型固定为 LiSA-staged；排除 SSS-GS **方法**，保留 Synthetic_SSS-GS **数据**。

本轮已全部验收：65项训练/精修、9项测量、18场景复评。

## 先看这些文件

- [results.md](results.md)：结果概览、各组结论及耗时/速度对照。
- [verification.json](verification.json)：最终核验记录。
- [progress.json](progress.json)：65 项训练/精修任务的完成、运行、等待与失败数量，每小时及队列结束时更新。
- [results.csv](results.csv)：逐场景、逐配置、逐 seed 指标及原始结果路径；`reused` 标明复用结果。
- [summary.csv](summary.csv)：同组、同配置、同 seed 的场景等权平均；组未完成时均值留空。
- `render_efficiency.csv`：9项测量已完成，记录固定/变化灯位的FPS、显存和模型大小。
- `repeatability.csv`、`paired_deltas.csv`：三seed统计已齐全，包含逐场景均值/标准差，以及完整模型相对无传输、局部残差的配对差值。
- [protocol.md](protocol.md)：固定实验矩阵、对照含义、训练与评价口径。

详细执行记录只在 `runs/lisa_supplementary_experiments/`：
`validation.json` 为启动配置，`manifest.json` 为精确命令，`source.tar` 为源码，
`status.json` 为逐阶段状态，`hourly_checks.jsonl` 为真实进程/GPU/训练进展的小时检查，`queue.log` 为队列事件。
训练与测量分开计数：65项训练/精修、9项性能/视频/图集测量、18场景共同GT复评。
其中10项局部残差重复性检查在seed0结构消融完成后追加，依据和边界见协议；
追加命令保存在`local_repeatability_manifest.json`。
后两组的精确命令在`measurement_manifest.json`、`comparison_manifest.json`，补充分析与调度源码在`measurement_source.tar`。
各任务按 `实验组/配置/数据家族/场景/` 存放，完整流程包含 `geometry/` 和 `appearance/`。
GS³/SSD-GS 的新模型和日志在各自 `runs/lisa_supplementary_experiments/`，此处只引用指标。
`render_efficiency/<family>/<scene>/`保存性能JSON、三条连续轨迹视频及实际图集/分支预览；
`common_ground_truth/<family>/<scene>/`保存五种方法在同一GT上的复评和GT像素差异审计，原基线数据不改。

## 执行

沿用 `configs/validation.json` → `make_validation_manifest.py` → `run_benchmark.py`，
启动入口仍为 `bash launch_validation.sh`。已有运行不重新生成或覆盖。
自动衔接训练、精修、完整评价和下一任务。最新授权使用GPU0/1/2，东京时间01:00–08:00加入GPU3；
每卡最多两个worker，启动第二项前显存占用须低于65%。GPU0已有约4GiB空闲占用获得使用许可。
训练耗时对比/推理测速仍单卡独占；其余任务的耗时列明确标记是否允许同卡共享。
原始数据、已有模型和历史实验记录保留。新评价仅保存四组图像预览，但逐帧指标覆盖完整划分。

目录内的均值是本地协议结果；未完成、失败或帧数/数值审计不通过的结果不计为完成。
