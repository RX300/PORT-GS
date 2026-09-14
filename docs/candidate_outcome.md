# Candidate 三场景结果

候选配置为 `localized-ports + gaussian-frame-local + encoded-alpha`，固定
seed 0、最后 checkpoint、无测试期拟合。Real/Cat 与 GS3/AnisoMetal 使用
100K 步和 512px，SSS/bunny_small 使用 60K 步和 256px；背景分别为黑、白、黑。
SSS 使用显式 `unit-light-intensity=1` 的等功率假设。该批次只覆盖三个场景的
开发验证面板，不代表完整 18 场景结果，也不能据此宣称 PORT-GS 达到 SOTA。

## 完成性核对

| 场景 | checkpoint step | fit / test 帧数 | 最终点数 | 训练秒数 |
|---|---:|---:|---:|---:|
| Real_NRHints/Cat | 100000 | 522 / 66 | 308806 | 7742.901 |
| Synthetic_GS3/AnisoMetal | 100000 | 2000 / 400 | 276889 | 5808.004 |
| Synthetic_SSS-GS/bunny_small | 60000 | 500 / 500 | 33600 | 1140.857 |

三个配置均核对为 `fit_all=true`、`encoded_alpha=true`、validation 为空，且
metadata 为对应的 `transforms_train.json`；test 评价均使用 `last.pt` 的全部
记录。候选 test 结果保留在
[`runs/candidate_comparison.json`](../runs/candidate_comparison.json) 和各场景
的 `test/metrics.json` 中。

## 指标

下表为 **PSNR / SSIM / LPIPS[0,1]**，其中候选 PORT 的 LPIPS 列使用
`LPIPS_baseline_01`，以匹配 SSD-GS 和论文记录；候选标准 `[-1,1]` LPIPS
依次为 Cat **0.295151**、AnisoMetal **0.052325**、bunny_small **0.023148**。
旧 PORT 来自 [`runs/benchmark_comparison.json`](../runs/benchmark_comparison.json)，
SSD 与论文数值来自 [`docs/baseline_metrics.json`](baseline_metrics.json)。

| 场景 | 候选 PORT | 旧 PORT | SSD local | 论文 |
|---|---|---|---|---|
| Cat | 18.805021 / 0.742850 / 0.273379 | 18.900711 / 0.744192 / 0.261327 | 27.243343 / 0.900045 / 0.138131 | 27.6844 / 0.9027 / 0.1357 |
| AnisoMetal | 28.212358 / 0.954148 / 0.042896 | 28.094611 / 0.956731 / 0.039761 | 28.000637 / 0.956572 / 0.039422 | 30.0448 / 0.9698 / 0.0295 |
| bunny_small | 38.798491 / 0.987675 / 0.018013 | 36.281493 / 0.983080 / 0.024851 | 38.480034 / 0.989095 / 0.014756 | 37.296 / 0.9861 / 0.0173 |

括号内的结果按 `ΔPSNR / ΔSSIM / ΔLPIPS` 表示候选减参考；PSNR、SSIM 越高越好，
LPIPS 越低越好。

| 场景 | 对旧 PORT | 对 SSD local | 对论文 |
|---|---|---|---|
| Cat | −0.095690 / −0.001342 / +0.012052；三项均输 | −8.438323 / −0.157195 / +0.135248；三项均输 | −8.879379 / −0.159850 / +0.137679；三项均输 |
| AnisoMetal | +0.117748 / −0.002583 / +0.003135；仅 PSNR 赢 | +0.211721 / −0.002424 / +0.003474；仅 PSNR 赢 | −1.832442 / −0.015652 / +0.013396；三项均输 |
| bunny_small | +2.516997 / +0.004595 / −0.006838；三项均赢 | +0.318457 / −0.001420 / +0.003257；仅 PSNR 赢 | +1.502491 / +0.001575 / +0.000713；PSNR、SSIM 赢，LPIPS 输 |

结果显示候选在 bunny_small 上相对旧 PORT 三项均改善，在 AnisoMetal 上只有
PSNR 改善，在 Cat 上三项均退化；相对 SSD local 仍没有三项同时胜出的场景。
这些是固定三场景面板的 test 结果，不应外推为完整数据集结论。
