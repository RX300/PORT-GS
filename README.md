# PORT-GS

独立的 3DGS 重光照研究项目，沿用 SSD-GS 的环境与数据协议。
当前默认 **directional_port_v1：512 个可学习空间端口 × 4 个方向通道**。
Gaussian 保留32维材质特征；共享方向网络生成入射/出射方向基；每个端口
通过非负RGB方向矩阵传递非局部光。直接光仍由165→128×4→3网络计算。

流程：读取训练帧 → Gaussian属性与深度光栅化 → 重建像素三维接收点 →
源Gaussian方向汇总与端口矩阵变换 → 像素方向读取 → 直接光与非局部RGB相加 →
原alpha、背景及观察变换 → 损失优化与几何细化。
默认第5000步同时启用阴影和端口，第25000步停止细化。不声称新方向算子保持旧算子的守恒性。

## 环境与数据

复用 `/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs`：
PyTorch2.4.1、CUDA12.1、gsplat，无新增依赖。
数据根目录 `/workspace/datasets/SSD-GS/data/`，每个场景包含 transforms JSON
及图像目录。本轮六场景：

- Real_NRHints：Cat、Pixiu，512px，黑背景。
- Synthetic_GS3：AnisoMetal、Translucent，512px，白背景。
- Synthetic_SSS-GS：bunny_small、dragon_small，256px，黑背景。

## 训练、渲染与评价

在 PORT-GS 目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python train.py \
  --scene /workspace/datasets/SSD-GS/data/Real_NRHints/Cat \
  --output runs/cat_directional_reproduction_s0 --fit-all

CUDA_VISIBLE_DEVICES=0 /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python evaluate.py \
  runs/cat_directional_reproduction_s0/last.pt --split test \
  --output runs/cat_directional_reproduction_s0/test --lpips

bash launch_validation.sh
```

默认30k步、seed0、rank512、dir_dim4、dir_width32。
`evaluate.py` 渲染全部所选帧，保存PSNR/SSIM/LPIPS、逐帧指标与前四组对照图。
六场景参数见 [validation.json](configs/validation.json)，训练使用完整官方train，
最后固定last.pt评估完整官方test；这里“验证集”指六场景实验集合。
每场景保存config.json、split.json、history.jsonl、last.pt和loss.png。
已完成的30k实验位于 `runs/directional_port512_validation_20260915/`，包含源码快照及精确命令。

旧32/512空间端口checkpoint仍按保存的representation加载原Transport。
显式 `--representation learned_anchor_exchange --rank 512` 可训练旧模型。
`--init-checkpoint`要求同架构同尺寸；新方向网络需要重新训练。
旧HashGrid实验使用其自己的源码快照，当前运行不依赖tinycudann。

## 文档与历史结果

[方向化架构](docs/architecture/modules/directional_transport.md) ·
[原始修改说明](docs/architecture/PORT_GS_Architecture_Change.md) ·
[六场景实验](docs/experiments/directional_port_validation_20260915.md) ·
[状态](docs/project/status.md) · [指标](docs/experiments/results.md) ·
[设置](docs/experiments/setup.md) · [决策](docs/project/decisions.md)

旧512端口结果：`runs/rank512_validation_20260914/`。
[512端口记录](docs/experiments/rank512_validation_20260914.md) ·
[此前恢复记录](docs/project/restore_anchor512_20260915.md) ·
[历史全量比较](docs/experiments/comparison_20260913.md) ·
[输出清理记录](docs/experiments/output_cleanup_20260914.md)

## 已完成的64端口六场景结果

六场景30k/seed0训练与1937帧完整测试已完成。场景均值：PSNR **28.2392**、
SSIM **0.91407**、LPIPS **0.09069**。相对旧512端口版，PSNR下降0.1853dB。
AnisoMetal提升0.5337dB，bunny_small下降1.6259dB；本轮整体没有改善。
逐场景指标与对照见[实验报告](docs/experiments/directional_port_validation_20260915.md)。

## 512方向端口复验

按用户要求将端口数从64增加到512，方向维数仍为4；同六场景、30k步、seed0，
阴影与端口均从5000步启用，细化停止改为25000步。
`validate-every=0` 关闭中间验证和模型保存，只在30000步保存最终last.pt并评估完整test。
原512端口实验已按用户要求停止并删除输出，在同一路径从头重跑。
[512端口实验记录](docs/experiments/directional_port512_validation_20260915.md)。

## 当前状态：实验已停止

按用户要求取消60k实验并删除该轮输出目录，默认训练步数恢复为30000。
保留512端口、阴影/端口从5000步启用、25000步停止细化、仅保存最终模型的设置。
已完成的30k结果保留；没有启动新实验。下一次默认实验名为
`directional_port512_validation_20260916`，避免覆盖已有结果。
[取消记录](docs/experiments/directional_port512_60k_validation_20260916.md)。

后续实验最多使用两张空闲GPU；当前配置为GPU1/3，启动前需重新确认空闲状态。
