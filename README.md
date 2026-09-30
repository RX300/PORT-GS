# PORT-GS

基于 Gaussian Splatting 的点光源重光照研究项目，代码、输出与 SSD-GS 基线隔离。

## 当前方法

2026-09-30 按用户要求只保留三个重光照方法：

- `directional_port_v1`：默认方向端口光传输。
- `surface_attention`：空间聚合与单次 attention 光传输。
- `neural_material`：冻结共享神经 BRDF 与方向端口。

其它方法及三个独立几何训练入口已删除。历史结果、最终权重和复现归档保留；旧方法须使用对应源码归档复现，当前入口不再加载它们。
已有 GGGS 几何仍可作为默认/神经材质的初始化；只保留其读取和深度诊断依赖，不再提供 GGGS 重建训练。
[清理范围与验证](docs/experiments/method_retirement_20260930.md) · [方法接口](docs/architecture/modules/methods.md)。

## 环境与数据

复用 `ssd-gs`（CUDA 12.1、PyTorch 2.4.1、gsplat 1.5.3），不升级共享依赖：

```bash
cd /workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS
conda activate /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs
```

共享数据：`/workspace/datasets/SSD-GS/data/<family>/<scene>/`。
Real_NRHints：Cat/Pixiu；Synthetic_GS3：AnisoMetal/Translucent；Synthetic_SSS-GS：bunny_small/dragon_small。
数据只读。Attention/Neural 的 2DGS、法线/深度先验和 Neural 的材质预训练依赖保留。
[环境与评价协议](docs/experiments/setup.md)。

## 训练、渲染与评价

流程：初始化高斯（可导入已有 GGGS 几何）→ 联合优化几何与重光照表示 → 可选训练相机校正 → 保存模型 → 渲染与评价官方 test。
三种方法共用训练/评价入口。相机校正须显式开启；官方 test 主指标使用原始标定。
`train.py` 只编排训练循环，选项、初始化、相机/灯位与辅助分支位于 `training/`（[训练模块](docs/architecture/modules/training.md)）。

```bash
# 查看三种方法参数
python train.py --representation directional_port_v1 --help
python train.py --representation surface_attention --help
python train.py --representation neural_material --help

# 修改规范配置的 name 和目标方法后启动；不得覆盖已有结果
# configs/validation.json 仍保留上次 no-ports 消融配置，并非 R2b 配方
bash launch_validation.sh

# 渲染与评价已有模型，output 必须是新目录
python evaluate.py runs/gggs_neural_material_joint/Real_NRHints/Cat/last.pt \
  --split test --lpips --highlights --output runs/cat_new_evaluation
```

Neural Material 从头训练需要 `--material-decoder`。GGGS 初始化当前仅支持方向端口与 Neural Material；Attention 使用原有 2DGS 路线。
模型、指标、配置与复现源码位于 `runs/`，日志位于 `logs/`。[实验保留索引](runs/README.md)。

## 当前结果

推荐配置仍是 **R2b：directional_port_v1 + GGGS 初始化 + 旋转相机校正 + 平移规范**。
Cat/Pixiu 固定标定 test 的 PSNR/SSIM/LPIPS 为 **25.203/.8201/.1674** 和 **26.264/.8811/.1041**。
默认与 Neural Material 已完成该相机协议下的比较；Attention 尚未完成同协议实验。
[完整标定结果](docs/experiments/calib_camrot.md) · [标定审计](docs/experiments/calibration_audit_20260930.md)。

[当前状态](docs/project/status.md) · [方法流程](docs/method/pipeline.md) ·
[架构](docs/architecture/system.md) · [实验结果](docs/experiments/results.md) · [决策记录](docs/project/decisions.md)
