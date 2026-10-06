# Surface Attention 六场景最终结果

6/6完成，0失败，30,000步；完整官方test共1,937帧。结束UTC：2026-10-01T10:15:48.098183+00:00。

与refactored_six_scene_20261001保持训练/评价协议一致，原生2DGS，复用相同训练先验；真实训练相机校正，测试保持原始标定。源码、参数与命令见source.tar、validation.json、manifest.json。

| 场景 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| Cat | 25.1440 | 0.820790 | 0.164655 |
| Pixiu | 26.3876 | 0.883174 | 0.106395 |
| AnisoMetal | 28.4298 | 0.960195 | 0.040142 |
| Translucent | 28.6373 | 0.966593 | 0.045753 |
| bunny_small | 36.5686 | 0.985537 | 0.022535 |
| dragon_small | 34.6736 | 0.970634 | 0.041084 |

## 三种方法六场景等权平均

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| directional_port_v1 | 30.5783 | 0.931622 | 0.070379 |
| surface_attention | 29.9735 | 0.931154 | 0.070094 |
| neural_material | 25.9396 | 0.909480 | 0.092543 |

默认方法平均PSNR更高约0.605 dB，平均SSIM略高；Attention平均LPIPS略低。Attention在Pixiu、AnisoMetal的PSNR更高，在Cat、Pixiu、AnisoMetal、Translucent的SSIM更高，在Cat、AnisoMetal、Translucent的LPIPS更低。单seed结果，不作显著性结论。

三方法原生几何/先验不同，不能把全部差异仅归因于传输表示。前置尺寸不匹配预检已保留，修正预检输入后通过；正式六组均成功，无重试。
