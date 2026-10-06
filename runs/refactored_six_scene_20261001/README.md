# 重构后的PORT-GS六场景实验

方法：directional_port_v1与neural_material；各6场景，共12组。GPU0、1、2。

场景：Real_NRHints/Cat、Pixiu；Synthetic_GS3/AnisoMetal、Translucent；Synthetic_SSS-GS/bunny_small、dragon_small。

从头训练30000步，原生初始化（原runs已空，不使用缺失GGGS）；完整官方train与test，测试帧66/71/400/400/500/500。真实训练相机使用rotation+translation gauge；官方test固定标定。

- `<方法>/<数据类别>/<场景>/last.pt`：最终模型。
- `<方法>/<数据类别>/<场景>/test/`：完整测试渲染、GT和指标。
- `manifest.json`、`validation.json`、`source.tar`：参数、命令与源码来源。
- `status.json`、`queue.log`：队列状态与日志。
- `dependencies/surface_priors/`：训练图法线与深度先验。
- 预检独立保存，不能作为正式结果；详见preflight_validation.json。

协议说明：../../docs/experiments/refactored_six_scene_20261001.md。已完成12/12组，0失败。最终结果见[RESULTS.md](RESULTS.md)，原始精度表格见[results.csv](results.csv)，完整性核验见completion_audit.json。
