# PORT-GS / SSD-GS 三场景开发验证对照

按用户最新范围，仅选择 Real/Cat、GS3/AnisoMetal、SSS/bunny_small；不再运行其余场景。
固定 seed 0。Real/GS3：100K、512px；SSS：60K、256px。全部官方 train，固定最后 checkpoint，完整官方 test。
Real/SSS 黑背景，GS3 白背景。无测试期相机、曝光或颜色拟合。SSS 使用正确世界灯光坐标、共同功率 gauge=1。
表中成对数值为 **PORT / SSD-GS**；ΔPSNR=PORT−SSD-GS。LPIPS 采用基线 [0,1] 输入约定，越低越好。
这是场景级开发验证面板，使用每个场景原有train训练、原有test评分；不计算或声称完整数据集宏平均。

## Real_NRHints

| 场景 | 状态 | 测试帧 | PSNR | ΔPSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|---:|
| Cat | 完成 | 66 | 18.9007 / 27.2433 | -8.3426 | 0.7442 / 0.9000 | 0.2613 / 0.1381 |

## Synthetic_GS3

| 场景 | 状态 | 测试帧 | PSNR | ΔPSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|---:|
| AnisoMetal | 完成 | 400 | 28.0946 / 28.0006 | +0.0940 | 0.9567 / 0.9566 | 0.0398 / 0.0394 |

## Synthetic_SSS-GS

| 场景 | 状态 | 测试帧 | PSNR | ΔPSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|---:|
| bunny_small | 完成 | 500 | 36.2815 / 38.4800 | -2.1985 | 0.9831 / 0.9891 | 0.0249 / 0.0148 |

