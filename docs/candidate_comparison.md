# PORT-GS / SSD-GS 三场景开发验证对照

按用户最新范围，仅选择 Real/Cat、GS3/AnisoMetal、SSS/bunny_small；不再运行其余场景。
固定 seed 0。Real/GS3：100K、512px；SSS：60K、256px。全部官方 train，固定最后 checkpoint，完整官方 test。
Real/SSS 黑背景，GS3 白背景。无测试期相机、曝光或颜色拟合。SSS 使用正确世界灯光坐标、共同功率 gauge=1。
表中成对数值为 **PORT / SSD-GS**；ΔPSNR=PORT−SSD-GS。LPIPS 采用基线 [0,1] 输入约定，越低越好。
这是场景级开发验证面板，使用每个场景原有train训练、原有test评分；不计算或声称完整数据集宏平均。

候选：空间局部传输 + Gaussian 坐标系局部响应 + PNG 编码域透明度合成（HDR 不变）；固定原始相机。旧版对照保留于 benchmark_comparison.md。

## Real_NRHints

| 场景 | 状态 | 测试帧 | PSNR | ΔPSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|---:|
| Cat | 完成 | 66 | 18.8050 / 27.2433 | -8.4383 | 0.7428 / 0.9000 | 0.2734 / 0.1381 |

## Synthetic_GS3

| 场景 | 状态 | 测试帧 | PSNR | ΔPSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|---:|
| AnisoMetal | 完成 | 400 | 28.2124 / 28.0006 | +0.2117 | 0.9541 / 0.9566 | 0.0429 / 0.0394 |

## Synthetic_SSS-GS

| 场景 | 状态 | 测试帧 | PSNR | ΔPSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|---:|
| bunny_small | 完成 | 500 | 38.7985 / 38.4800 | +0.3185 | 0.9877 / 0.9891 | 0.0180 / 0.0148 |

