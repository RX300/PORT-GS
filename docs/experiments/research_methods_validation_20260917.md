# 两项新方法六场景30k实验 — 2026-09-17

用户要求方案一/二分别跑现有验证集合，最多两张GPU并行，每场景训练30000步。
这里验证集合沿用项目现有六场景全train/官方test协议，不切换成训练内holdout。
本轮直接对预先确定的两个候选做对照，不按test挑选或修改超参数。

## 冻结协议

- 方案A：paired_port，GPU1串行处理六场景。
- 方案B：local_frame，GPU3串行处理六场景，frame_width=32。
- 两组同时运行，总计12次训练；每个GPU同时最多一个训练或评估子进程。
- 场景：Cat、Pixiu、AnisoMetal、Translucent、bunny_small、dragon_small。
- 各自从头训练30k、seed0，rank512、dir_dim4、dir_width32。
- shadow_start=port_start=5000，refine_stop=25000，validate_every=0；只保存最终last.pt。
- 使用完整官方train，每场景完成后自动评估完整官方test，计算PSNR/SSIM/标准LPIPS。
- 分辨率、背景、光强及其余训练设置与directional_port512_validation_20260915保持原协议。
- 原ssd-gs环境与CUDA12.1，模型代码来自4a9039e；精确配置和源码版本由manifest记录。

## 启动与产物

```bash
bash launch_validation.sh configs/paired_port512_validation_20260917.json
bash launch_validation.sh configs/local_frame512_validation_20260917.json
```

分别输出到runs/paired_port512_validation_20260917/和runs/local_frame512_validation_20260917/。
各目录保存validation.json、manifest.json、source.tar、status.json、queue.log，
每场景有config、split、history、loss曲线、最终checkpoint和test/metrics.json。
源码快照包含methods包；原30k基线输出保留，不被覆盖。

启动状态与实际进度证据在完成启动核查后补充。
