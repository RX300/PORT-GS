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

## 启动核查

两组后台tmux队列已启动，真实训练进程、日志递增与GPU活动均已核实。
- paired_port：GPU1，训练PID 222580，启动核查已到step 5000，loss有限。
- local_frame：GPU3，训练PID 222649，启动核查已到step 4200，loss有限。

GPU1/3从空闲状态启动，其余GPU未使用。Cat实际保存的训练配置与已完成30k基线对比，
差异仅representation/output，以及方案B的frame_width。完整六场景图像存在性检查通过，
每组完整test共1937帧。方法构造和源码快照内容检查通过，复用上一轮已通过的真实CUDA集成验证。

各实验目录的preflight.json和startup_evidence.json记录配置、进度与GPU证据。
NVML返回的PID与容器内PID不同，启动验证通过/proc核对训练进程及CUDA_VISIBLE_DEVICES，
通过GPU UUID对应NVML计算进程；不以两种PID数值直接相等为前提。
manifest记录源码版本767575f；模型实现未在启动期间修改。

训练及评估结果尚未完成；各场景结束后自动保存test/metrics.json，队列状态写入status.json。
