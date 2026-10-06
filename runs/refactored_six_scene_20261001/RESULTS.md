# 重构后六场景最终结果

12/12组完成，0失败；每组30,000步，合计3,874个完整官方test评估帧。队列结束UTC：2026-10-01T07:38:11.708660+00:00。

原生fresh初始化；默认方法3DGS，neural_material为2DGS并保留法线/深度先验和冻结材质decoder。真实训练相机校正；test固定原始标定。不能直接作为旧GGGS初始化R2b的重构前后对比。

| 方法 | 类别 | 场景 | 帧数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---:|---:|---:|---:|
| directional_port_v1 | Real_NRHints | Cat | 66 | 25.2582 | 0.820729 | 0.167277 |
| directional_port_v1 | Real_NRHints | Pixiu | 71 | 26.3195 | 0.881218 | 0.105643 |
| directional_port_v1 | Synthetic_GS3 | AnisoMetal | 400 | 28.2639 | 0.958128 | 0.045295 |
| directional_port_v1 | Synthetic_GS3 | Translucent | 400 | 29.3204 | 0.965120 | 0.051269 |
| directional_port_v1 | Synthetic_SSS-GS | bunny_small | 500 | 38.1696 | 0.987923 | 0.018612 |
| directional_port_v1 | Synthetic_SSS-GS | dragon_small | 500 | 36.1379 | 0.976614 | 0.034180 |
| neural_material | Real_NRHints | Cat | 66 | 24.2169 | 0.808192 | 0.194377 |
| neural_material | Real_NRHints | Pixiu | 71 | 26.1378 | 0.880558 | 0.109523 |
| neural_material | Synthetic_GS3 | AnisoMetal | 400 | 23.3651 | 0.930194 | 0.057281 |
| neural_material | Synthetic_GS3 | Translucent | 400 | 22.8779 | 0.931340 | 0.071972 |
| neural_material | Synthetic_SSS-GS | bunny_small | 500 | 26.6389 | 0.949349 | 0.068948 |
| neural_material | Synthetic_SSS-GS | dragon_small | 500 | 32.4012 | 0.957245 | 0.053153 |

## 六场景等权平均

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| directional_port_v1 | 30.5783 | 0.931622 | 0.070379 |
| neural_material | 25.9396 | 0.909480 | 0.092543 |

默认方法在六场景的PSNR、SSIM和LPIPS上均更好。本次比较包含原生几何及先验差异，不能将差距单独归因于材质分支。高光代理指标详见各场景metrics.json；全图指标优势不代表每项局部高光指标都占优。

已核验12组最终checkpoint、30k训练记录、完整测试帧/逐帧记录/配对PNG数量和有限主指标。CSV保留原始精度；预检未计入。
