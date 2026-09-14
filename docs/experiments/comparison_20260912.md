# 跨数据集评价协议 — 2026-09-12–13

**状态：第二轮已接受并冻结，三个场景完整 test 已完成，整体 3/3，完成日期 2026-09-13。** 共完成两轮结构迭代。
固定 30k Cat validation 为 22.316405/.778143/.228932（PSNR/SSIM/标准 LPIPS）；
相对首轮，细节能量提高 26.4%，PSNR 下降 .224227 dB，轮廓 IoU 下降。
相对历史同划分验证，PSNR 仍高 .114553 dB。接受依据为感知和细节收益。
冻结源码为 `runs/research_20260912/cat_r2_source.tar`，具体证据见 [结果](results.md)。
Cat 的 522 train / 66 test 全流程得到 **21.501174 dB / .766281 / .227281**，
达到原始标定下超过 20 dB 的要求；相对历史 full Cat，PSNR 下降 .498897 dB，
SSIM 下降 .001245，LPIPS 改善 .021971（约 8.81%）。纹理与感知收益伴随广域误差。
Translucent 完整 400 帧 test 为 **28.304924/.960368/.051791**，30k、399830 点，
训练 1539.79 秒、评价 26.35 秒。Bunny 完整 500 帧 test 为
**37.600658/.986484/.019684**，30k、354592 点，训练 2328.69 秒、评价 19.11 秒。
Bunny 中位 PSNR 为 38.047665 dB，但 7/500 帧低于 20 dB，最差两帧仅
12.14355/12.43680 dB；图 332 存在严重视角、轮廓与形状错误，平均值掩盖了失败视角。GPU 0 已释放。
`run_full_sequence.py` 沿用冻结方案执行，测试分数不用于调整模型或超参数。

## 冻结后的执行顺序

所有阶段在同一块 GPU 0 串行运行，复用 `ssd-gs` 环境。每个场景从头初始化，
`--fit-all` 消费其完整官方 train；训练结束后评价固定 30k checkpoint 的完整 test。
全部样本加载至 GPU，按 48GB 显存容量安排。固定 seed 0、20k 初始点、400k 点上限、
feature 32、width 128、rank 32，其余训练设置继承冻结的第二轮 Cat 配置。

| 顺序 | 场景 | train → test | 分辨率 / 背景 | 观测设置 | 冻结的运行目录 |
| --- | --- | --- | --- | --- | --- |
| 1，已完成 | Real_NRHints/Cat | 522 → 66 | 512 / 黑 | PNG，gamma 2.2 预测前景编码 | `runs/research_20260912/cat_full_s0` |
| 2，已完成 | Synthetic_GS3/Translucent | 2000 → 400 | 512 / 白 | HDR 合成后 GT/预测均 gamma 2.2 | `runs/research_20260912/translucent_full_s0` |
| 3，已完成 | Synthetic_SSS-GS/bunny_small | 500 → 500 | 256 / 黑 | PNG，gamma 2.2；`unit_light_intensity=1` | `runs/research_20260912/bunny_full_s0` |

场景均位于 `/workspace/datasets/SSD-GS/data/`，现有图像引用完整。
Bunny 的完整实际路径为 `/workspace/datasets/SSD-GS/data/Synthetic_SSS-GS/bunny_small`。
外部场景使用冻结配置，不再按外部验证或测试结果调参；本次执行只训练 PORT。
关闭分支的对照实验及其他方法重训练均不在这份协议中。

## 评价与比较边界

测试保留原始相机和灯光标定，不用测试 GT 优化参数。主要指标为显示域 RGB
裁剪、uint8 量化后的 unit-peak PSNR、零 padding 的 11x11/sigma 1.5 SSIM、
标准 VGG LPIPS（[-1,1]）；同时保存逐帧指标与 GT/预测图。
训练内浮点 validation 与最终量化 test 分开报告。

- **Cat：**历史 PORT 22.000071 dB 为旧方法的完整 test 结果，作为历史参照。
  已接受候选的 validation 值与本轮最终 test 分开报告。
- **GS³：**Translucent 论文参考 32.34 dB / 0.9740 / 0.0318；官方脚本训练
  100k 步，本轮冻结预算为 30k。官方 EXR 白背景与 gamma 导出路径已核对匹配，最终论文
  metric 实现仍有缺口；公开训练日志 PSNR 使用线性域逐通道计算，LPIPS helper
  默认 AlexNet。结果表应标记预算和度量边界。
- **SSS-GS：**本地 small 为 256px、500 输入，等功率强度 1 是 PORT 的明确假设。
  官方 35.01±1.01 dB 是合成集合聚合值，已核对资料未提供可确认的 Bunny small
  单场景 PSNR。官方离线 TorchMetrics 的动态范围、SSIM 边界口径与主要指标不同；
  聚合值仅作论文背景参考；本次 Bunny 37.60 dB 不据此声称超过 SSS-GS。

详细官方链接及纯评价适配建议见
[协议审计](../research/related_work_20260912.md)。这些结果用于检查跨数据集可用性，
不构成严格同协议的 SOTA 排名。

## 记录位置

每个运行目录保存启动命令、配置、划分、日志、源码来源与 checkpoint。
最终报告已在各目录的 `test/metrics.json` 保存：
`cat_full_s0/test/metrics.json`、`translucent_full_s0/test/metrics.json`、
`bunny_full_s0/test/metrics.json`。三个运行均保存完整结果及 `result.json`。
[full_results.json](../../runs/research_20260912/full_results.json) 最终记录 3/3 完成。数据全部复用共享目录中完整已有文件，本轮没有重复下载。
