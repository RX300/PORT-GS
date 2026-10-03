# LiSA：三类数据集完整实验记录

覆盖 Real_NRHints 7场景、Synthetic_GS3 6场景、Synthetic_SSS-GS 5场景，**18/18完成**。2026-10-03全量比较完成并经中央collector核验。本页统一替代此前六场景、困难场景及剩余场景的分批结果记录。

[实验环境与协议](setup.md) · [完整机器记录（配置、命令、阶段日志、GPU及来源）](records.json) · [四方法总表与单场景对比](../../../benchmarks/full_dataset_comparison/RESULTS.md)

## 固定协议与结果口径

最终配置为 light_atlas + svbrdf 材质头；每场景30k步、seed0、原生3DGS从头初始化，无外部预训练或几何先验。真实场景仅训练相机旋转优化及平移规范，测试使用原始标定。此前六场景参与过研发观察，不能把整个研发流程描述为独立盲测。

共享 ssd-gs：Python 3.10 / PyTorch 2.4.1 / CUDA 12.1 / gsplat 1.5.3。

全部official train、完整official test，固定最后checkpoint，测试原始相机与灯位，无测试时拟合。Real/GS3最长边512px，SSS256px；Synthetic_GS3白底、线性HDR与gamma2.2展示，其余黑底。PSNR单位RGB、11×11 sigma1.5零padding SSIM、标准VGG LPIPS[-1,1]。先对每场景完整测试帧取均值，再对场景等权平均；单种子，各方法训练预算不同。

## 平均指标

| 范围 | 场景数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| all | 18 | 31.3760 | 0.9431 | 0.0646 |
| Real_NRHints | 7 | 28.4729 | 0.9072 | 0.0988 |
| Synthetic_GS3 | 6 | 28.3498 | 0.9480 | 0.0622 |
| Synthetic_SSS-GS | 5 | 39.0717 | 0.9874 | 0.0196 |

## Real_NRHints

| 场景 | train/test | 步数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 原始指标 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Cat | 522/66 | 30000 | 25.1991 | 0.8197 | 0.1671 | [metrics.json](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Real_NRHints/Cat/test/metrics.json) |
| CatSmall | 1258/158 | 30000 | 33.5302 | 0.9642 | 0.0721 | [metrics.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/CatSmall/test/metrics.json) |
| CupFabric | 1153/145 | 30000 | 30.6019 | 0.9670 | 0.0650 | [metrics.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/CupFabric/test/metrics.json) |
| Fish | 526/66 | 30000 | 27.6822 | 0.8893 | 0.1091 | [metrics.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/Fish/test/metrics.json) |
| FurScene | 676/85 | 30000 | 23.8440 | 0.8599 | 0.1246 | [metrics.json](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Real_NRHints/FurScene/test/metrics.json) |
| Pikachu | 1500/200 | 30000 | 32.1703 | 0.9691 | 0.0524 | [metrics.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/Pikachu/test/metrics.json) |
| Pixiu | 562/71 | 30000 | 26.2828 | 0.8810 | 0.1015 | [metrics.json](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Real_NRHints/Pixiu/test/metrics.json) |

## Synthetic_GS3

| 场景 | train/test | 步数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 原始指标 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AnisoMetal | 2000/400 | 30000 | 27.5993 | 0.9606 | 0.0366 | [metrics.json](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_GS3/AnisoMetal/test/metrics.json) |
| Drums | 2000/400 | 30000 | 23.1751 | 0.9450 | 0.0626 | [metrics.json](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Synthetic_GS3/Drums/test/metrics.json) |
| FurBall | 2000/400 | 30000 | 36.8497 | 0.9736 | 0.0435 | [metrics.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/FurBall/test/metrics.json) |
| Hotdog | 2000/400 | 30000 | 30.2002 | 0.9686 | 0.0425 | [metrics.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/Hotdog/test/metrics.json) |
| Lego | 2000/400 | 30000 | 21.2351 | 0.8639 | 0.1561 | [metrics.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/Lego/test/metrics.json) |
| Translucent | 2000/400 | 30000 | 31.0396 | 0.9762 | 0.0323 | [metrics.json](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_GS3/Translucent/test/metrics.json) |

## Synthetic_SSS-GS

| 场景 | train/test | 步数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 原始指标 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| bunny_small | 500/500 | 30000 | 40.1564 | 0.9913 | 0.0165 | [metrics.json](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_SSS-GS/bunny_small/test/metrics.json) |
| candle_small | 500/500 | 30000 | 45.2610 | 0.9966 | 0.0067 | [metrics.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_SSS-GS/candle_small/test/metrics.json) |
| dragon_small | 500/500 | 30000 | 37.9232 | 0.9831 | 0.0239 | [metrics.json](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_SSS-GS/dragon_small/test/metrics.json) |
| soap_small | 500/500 | 30000 | 35.8331 | 0.9812 | 0.0224 | [metrics.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_SSS-GS/soap_small/test/metrics.json) |
| statue_small | 500/500 | 30000 | 36.1847 | 0.9847 | 0.0285 | [metrics.json](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Synthetic_SSS-GS/statue_small/test/metrics.json) |

## 逐场景完整证据

权重、数据、原始日志及不可变源码快照留在原位置。下列链接和records.json将分散批次合并为一个可追溯入口；原始配置中的旧pending/failed字段仅是历史快照，正式完成状态以本页和中央审计为准。存量记录未保存的信息不反推或补造。

### Real_NRHints/Cat

[输出目录](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Real_NRHints/Cat) · [最终权重](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Real_NRHints/Cat/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Real_NRHints/Cat/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_six_scene_20261002/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Real_NRHints/Cat/config.json)
- 命令与阶段记录：[benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_six_scene_20261002/manifest.json](../../runs/light_atlas_svbrdf_six_scene_20261002/manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_six_scene_20261002/status.json](../../runs/light_atlas_svbrdf_six_scene_20261002/status.json)
- 原始日志：[Cat.test.log](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Real_NRHints/Cat.test.log) · [Cat.train.log](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Real_NRHints/Cat.train.log)
- 环境及代码版本：见源码快照、命令清单及setup.md；未记录的历史版本不推定。

### Real_NRHints/CatSmall

[输出目录](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/CatSmall) · [最终权重](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/CatSmall/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/CatSmall/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_remaining_scenes/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/CatSmall/config.json)
- 命令与阶段记录：[benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [benchmarks/full_dataset_comparison/jobs.json](../../../benchmarks/full_dataset_comparison/jobs.json) · [benchmarks/full_dataset_comparison/queue_state.json](../../../benchmarks/full_dataset_comparison/queue_state.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/jobs.json](../../runs/light_atlas_svbrdf_remaining_scenes/jobs.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/manifest.json](../../runs/light_atlas_svbrdf_remaining_scenes/manifest.json)
- 原始日志：[CatSmall.test.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/CatSmall.test.log) · [CatSmall.train.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/CatSmall.train.log)
- 环境及代码版本：[environment.txt](../../runs/light_atlas_svbrdf_remaining_scenes/environment.txt) · [provenance.json](../../runs/light_atlas_svbrdf_remaining_scenes/provenance.json)

### Real_NRHints/CupFabric

[输出目录](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/CupFabric) · [最终权重](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/CupFabric/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/CupFabric/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_remaining_scenes/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/CupFabric/config.json)
- 命令与阶段记录：[benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [benchmarks/full_dataset_comparison/jobs.json](../../../benchmarks/full_dataset_comparison/jobs.json) · [benchmarks/full_dataset_comparison/queue_state.json](../../../benchmarks/full_dataset_comparison/queue_state.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/jobs.json](../../runs/light_atlas_svbrdf_remaining_scenes/jobs.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/manifest.json](../../runs/light_atlas_svbrdf_remaining_scenes/manifest.json)
- 原始日志：[CupFabric.test.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/CupFabric.test.log) · [CupFabric.train.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/CupFabric.train.log)
- 环境及代码版本：[environment.txt](../../runs/light_atlas_svbrdf_remaining_scenes/environment.txt) · [provenance.json](../../runs/light_atlas_svbrdf_remaining_scenes/provenance.json)

### Real_NRHints/Fish

[输出目录](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/Fish) · [最终权重](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/Fish/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/Fish/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_remaining_scenes/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/Fish/config.json)
- 命令与阶段记录：[benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [benchmarks/full_dataset_comparison/jobs.json](../../../benchmarks/full_dataset_comparison/jobs.json) · [benchmarks/full_dataset_comparison/queue_state.json](../../../benchmarks/full_dataset_comparison/queue_state.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/jobs.json](../../runs/light_atlas_svbrdf_remaining_scenes/jobs.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/manifest.json](../../runs/light_atlas_svbrdf_remaining_scenes/manifest.json)
- 原始日志：[Fish.test.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/Fish.test.log) · [Fish.train.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/Fish.train.log)
- 环境及代码版本：[environment.txt](../../runs/light_atlas_svbrdf_remaining_scenes/environment.txt) · [provenance.json](../../runs/light_atlas_svbrdf_remaining_scenes/provenance.json)

### Real_NRHints/FurScene

[输出目录](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Real_NRHints/FurScene) · [最终权重](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Real_NRHints/FurScene/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Real_NRHints/FurScene/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_hard_scenes/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Real_NRHints/FurScene/config.json)
- 命令与阶段记录：[benchmarks/supplement_hard_scenes/comparison_manifest.json](../../../benchmarks/supplement_hard_scenes/comparison_manifest.json) · [benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_hard_scenes/manifest.json](../../runs/light_atlas_svbrdf_hard_scenes/manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_hard_scenes/status.json](../../runs/light_atlas_svbrdf_hard_scenes/status.json)
- 原始日志：[FurScene.test.log](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Real_NRHints/FurScene.test.log) · [FurScene.train.log](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Real_NRHints/FurScene.train.log)
- 环境及代码版本：见源码快照、命令清单及setup.md；未记录的历史版本不推定。

### Real_NRHints/Pikachu

[输出目录](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/Pikachu) · [最终权重](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/Pikachu/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/Pikachu/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_remaining_scenes/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/Pikachu/config.json)
- 命令与阶段记录：[benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [benchmarks/full_dataset_comparison/jobs.json](../../../benchmarks/full_dataset_comparison/jobs.json) · [benchmarks/full_dataset_comparison/queue_state.json](../../../benchmarks/full_dataset_comparison/queue_state.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/jobs.json](../../runs/light_atlas_svbrdf_remaining_scenes/jobs.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/manifest.json](../../runs/light_atlas_svbrdf_remaining_scenes/manifest.json)
- 原始日志：[Pikachu.test.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/Pikachu.test.log) · [Pikachu.train.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Real_NRHints/Pikachu.train.log)
- 环境及代码版本：[environment.txt](../../runs/light_atlas_svbrdf_remaining_scenes/environment.txt) · [provenance.json](../../runs/light_atlas_svbrdf_remaining_scenes/provenance.json)

### Real_NRHints/Pixiu

[输出目录](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Real_NRHints/Pixiu) · [最终权重](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Real_NRHints/Pixiu/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Real_NRHints/Pixiu/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_six_scene_20261002/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Real_NRHints/Pixiu/config.json)
- 命令与阶段记录：[benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_six_scene_20261002/manifest.json](../../runs/light_atlas_svbrdf_six_scene_20261002/manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_six_scene_20261002/status.json](../../runs/light_atlas_svbrdf_six_scene_20261002/status.json)
- 原始日志：[Pixiu.test.log](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Real_NRHints/Pixiu.test.log) · [Pixiu.train.log](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Real_NRHints/Pixiu.train.log)
- 环境及代码版本：见源码快照、命令清单及setup.md；未记录的历史版本不推定。

### Synthetic_GS3/AnisoMetal

[输出目录](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_GS3/AnisoMetal) · [最终权重](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_GS3/AnisoMetal/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_GS3/AnisoMetal/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_six_scene_20261002/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_GS3/AnisoMetal/config.json)
- 命令与阶段记录：[benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_six_scene_20261002/manifest.json](../../runs/light_atlas_svbrdf_six_scene_20261002/manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_six_scene_20261002/status.json](../../runs/light_atlas_svbrdf_six_scene_20261002/status.json)
- 原始日志：[AnisoMetal.test.log](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_GS3/AnisoMetal.test.log) · [AnisoMetal.train.log](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_GS3/AnisoMetal.train.log)
- 环境及代码版本：见源码快照、命令清单及setup.md；未记录的历史版本不推定。

### Synthetic_GS3/Drums

[输出目录](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Synthetic_GS3/Drums) · [最终权重](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Synthetic_GS3/Drums/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Synthetic_GS3/Drums/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_hard_scenes/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Synthetic_GS3/Drums/config.json)
- 命令与阶段记录：[benchmarks/supplement_hard_scenes/comparison_manifest.json](../../../benchmarks/supplement_hard_scenes/comparison_manifest.json) · [benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_hard_scenes/manifest.json](../../runs/light_atlas_svbrdf_hard_scenes/manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_hard_scenes/status.json](../../runs/light_atlas_svbrdf_hard_scenes/status.json)
- 原始日志：[Drums.test.log](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Synthetic_GS3/Drums.test.log) · [Drums.train.log](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Synthetic_GS3/Drums.train.log)
- 环境及代码版本：见源码快照、命令清单及setup.md；未记录的历史版本不推定。

### Synthetic_GS3/FurBall

[输出目录](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/FurBall) · [最终权重](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/FurBall/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/FurBall/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_remaining_scenes/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/FurBall/config.json)
- 命令与阶段记录：[benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [benchmarks/full_dataset_comparison/jobs.json](../../../benchmarks/full_dataset_comparison/jobs.json) · [benchmarks/full_dataset_comparison/queue_state.json](../../../benchmarks/full_dataset_comparison/queue_state.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/jobs.json](../../runs/light_atlas_svbrdf_remaining_scenes/jobs.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/manifest.json](../../runs/light_atlas_svbrdf_remaining_scenes/manifest.json)
- 原始日志：[FurBall.test.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/FurBall.test.log) · [FurBall.train.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/FurBall.train.log)
- 环境及代码版本：[environment.txt](../../runs/light_atlas_svbrdf_remaining_scenes/environment.txt) · [provenance.json](../../runs/light_atlas_svbrdf_remaining_scenes/provenance.json)

### Synthetic_GS3/Hotdog

[输出目录](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/Hotdog) · [最终权重](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/Hotdog/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/Hotdog/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_remaining_scenes/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/Hotdog/config.json)
- 命令与阶段记录：[benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [benchmarks/full_dataset_comparison/jobs.json](../../../benchmarks/full_dataset_comparison/jobs.json) · [benchmarks/full_dataset_comparison/queue_state.json](../../../benchmarks/full_dataset_comparison/queue_state.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/jobs.json](../../runs/light_atlas_svbrdf_remaining_scenes/jobs.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/manifest.json](../../runs/light_atlas_svbrdf_remaining_scenes/manifest.json)
- 原始日志：[Hotdog.test.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/Hotdog.test.log) · [Hotdog.train.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/Hotdog.train.log)
- 环境及代码版本：[environment.txt](../../runs/light_atlas_svbrdf_remaining_scenes/environment.txt) · [provenance.json](../../runs/light_atlas_svbrdf_remaining_scenes/provenance.json)

### Synthetic_GS3/Lego

[输出目录](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/Lego) · [最终权重](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/Lego/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/Lego/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_remaining_scenes/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/Lego/config.json)
- 命令与阶段记录：[benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [benchmarks/full_dataset_comparison/jobs.json](../../../benchmarks/full_dataset_comparison/jobs.json) · [benchmarks/full_dataset_comparison/queue_state.json](../../../benchmarks/full_dataset_comparison/queue_state.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/jobs.json](../../runs/light_atlas_svbrdf_remaining_scenes/jobs.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/manifest.json](../../runs/light_atlas_svbrdf_remaining_scenes/manifest.json)
- 原始日志：[Lego.test.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/Lego.test.log) · [Lego.train.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_GS3/Lego.train.log)
- 环境及代码版本：[environment.txt](../../runs/light_atlas_svbrdf_remaining_scenes/environment.txt) · [provenance.json](../../runs/light_atlas_svbrdf_remaining_scenes/provenance.json)

### Synthetic_GS3/Translucent

[输出目录](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_GS3/Translucent) · [最终权重](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_GS3/Translucent/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_GS3/Translucent/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_six_scene_20261002/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_GS3/Translucent/config.json)
- 命令与阶段记录：[benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_six_scene_20261002/manifest.json](../../runs/light_atlas_svbrdf_six_scene_20261002/manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_six_scene_20261002/status.json](../../runs/light_atlas_svbrdf_six_scene_20261002/status.json)
- 原始日志：[Translucent.test.log](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_GS3/Translucent.test.log) · [Translucent.train.log](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_GS3/Translucent.train.log)
- 环境及代码版本：见源码快照、命令清单及setup.md；未记录的历史版本不推定。

### Synthetic_SSS-GS/bunny_small

[输出目录](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_SSS-GS/bunny_small) · [最终权重](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_SSS-GS/bunny_small/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_SSS-GS/bunny_small/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_six_scene_20261002/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_SSS-GS/bunny_small/config.json)
- 命令与阶段记录：[benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_six_scene_20261002/manifest.json](../../runs/light_atlas_svbrdf_six_scene_20261002/manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_six_scene_20261002/status.json](../../runs/light_atlas_svbrdf_six_scene_20261002/status.json)
- 原始日志：[bunny_small.test.log](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_SSS-GS/bunny_small.test.log) · [bunny_small.train.log](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_SSS-GS/bunny_small.train.log)
- 环境及代码版本：见源码快照、命令清单及setup.md；未记录的历史版本不推定。

### Synthetic_SSS-GS/candle_small

[输出目录](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_SSS-GS/candle_small) · [最终权重](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_SSS-GS/candle_small/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_SSS-GS/candle_small/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_remaining_scenes/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_SSS-GS/candle_small/config.json)
- 命令与阶段记录：[benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [benchmarks/full_dataset_comparison/jobs.json](../../../benchmarks/full_dataset_comparison/jobs.json) · [benchmarks/full_dataset_comparison/queue_state.json](../../../benchmarks/full_dataset_comparison/queue_state.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/jobs.json](../../runs/light_atlas_svbrdf_remaining_scenes/jobs.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/manifest.json](../../runs/light_atlas_svbrdf_remaining_scenes/manifest.json)
- 原始日志：[candle_small.test.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_SSS-GS/candle_small.test.log) · [candle_small.train.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_SSS-GS/candle_small.train.log)
- 环境及代码版本：[environment.txt](../../runs/light_atlas_svbrdf_remaining_scenes/environment.txt) · [provenance.json](../../runs/light_atlas_svbrdf_remaining_scenes/provenance.json)

### Synthetic_SSS-GS/dragon_small

[输出目录](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_SSS-GS/dragon_small) · [最终权重](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_SSS-GS/dragon_small/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_SSS-GS/dragon_small/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_six_scene_20261002/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_SSS-GS/dragon_small/config.json)
- 命令与阶段记录：[benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_six_scene_20261002/manifest.json](../../runs/light_atlas_svbrdf_six_scene_20261002/manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_six_scene_20261002/status.json](../../runs/light_atlas_svbrdf_six_scene_20261002/status.json)
- 原始日志：[dragon_small.test.log](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_SSS-GS/dragon_small.test.log) · [dragon_small.train.log](../../runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_SSS-GS/dragon_small.train.log)
- 环境及代码版本：见源码快照、命令清单及setup.md；未记录的历史版本不推定。

### Synthetic_SSS-GS/soap_small

[输出目录](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_SSS-GS/soap_small) · [最终权重](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_SSS-GS/soap_small/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_SSS-GS/soap_small/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_remaining_scenes/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_SSS-GS/soap_small/config.json)
- 命令与阶段记录：[benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [benchmarks/full_dataset_comparison/jobs.json](../../../benchmarks/full_dataset_comparison/jobs.json) · [benchmarks/full_dataset_comparison/queue_state.json](../../../benchmarks/full_dataset_comparison/queue_state.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/jobs.json](../../runs/light_atlas_svbrdf_remaining_scenes/jobs.json) · [PORT-GS/runs/light_atlas_svbrdf_remaining_scenes/manifest.json](../../runs/light_atlas_svbrdf_remaining_scenes/manifest.json)
- 原始日志：[soap_small.test.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_SSS-GS/soap_small.test.log) · [soap_small.train.log](../../runs/light_atlas_svbrdf_remaining_scenes/lisa_svbrdf/Synthetic_SSS-GS/soap_small.train.log)
- 环境及代码版本：[environment.txt](../../runs/light_atlas_svbrdf_remaining_scenes/environment.txt) · [provenance.json](../../runs/light_atlas_svbrdf_remaining_scenes/provenance.json)

### Synthetic_SSS-GS/statue_small

[输出目录](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Synthetic_SSS-GS/statue_small) · [最终权重](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Synthetic_SSS-GS/statue_small/last.pt) · [评估指标](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Synthetic_SSS-GS/statue_small/test/metrics.json) · [训练源码快照](../../runs/light_atlas_svbrdf_hard_scenes/source.tar)

- 配置：[config.json](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Synthetic_SSS-GS/statue_small/config.json)
- 命令与阶段记录：[benchmarks/supplement_hard_scenes/comparison_manifest.json](../../../benchmarks/supplement_hard_scenes/comparison_manifest.json) · [benchmarks/full_dataset_comparison/comparison_manifest.json](../../../benchmarks/full_dataset_comparison/comparison_manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_hard_scenes/manifest.json](../../runs/light_atlas_svbrdf_hard_scenes/manifest.json) · [PORT-GS/runs/light_atlas_svbrdf_hard_scenes/status.json](../../runs/light_atlas_svbrdf_hard_scenes/status.json)
- 原始日志：[statue_small.test.log](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Synthetic_SSS-GS/statue_small.test.log) · [statue_small.train.log](../../runs/light_atlas_svbrdf_hard_scenes/lisa_svbrdf/Synthetic_SSS-GS/statue_small.train.log)
- 环境及代码版本：见源码快照、命令清单及setup.md；未记录的历史版本不推定。

## 研发实验的区别

[材质头选择、消融及留出光照实验](development.md)是独立研发证据，不混入上述18场景均值。原六场景泄漏审计已并入[协议](setup.md)，审计范围不自动扩展为18场景的全量训练历史审计。SOTA与成因分析仍按用户要求暂停。
