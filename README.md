# PORT-GS

基于 Gaussian Splatting 的点光源重光照研究项目，代码、输出与 SSD-GS 基线隔离。

## 当前方法

2026-09-30 按用户要求只保留前三个重光照方法；2026-10-01 新增第四个方法 `light_atlas`：

- `directional_port_v1`：默认方向端口光传输。
- `surface_attention`：空间聚合与单次 attention 光传输。
- `neural_material`：冻结共享神经 BRDF 与方向端口。
- `light_atlas`（2026-10-01 新增，LiSA）：从光源 splat 学习的通量特征与深度矩，多尺度金字塔供每个接收像素读取；
  矩阴影检验+学习残差给出逐像素可见度，对通量线性的学习核给出半透明/散射/近邻反弹。3DGS，无需先验或预训练。
  [模块](docs/architecture/modules/light_atlas.md) · [提案与新颖性审查](docs/research/lisa_light_space_atlas_20261001.md) ·
  [完整实验记录](docs/experiments/results.md)。

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
python train.py --representation light_atlas --help

# LiSA 单场景训练（最终配置 svbrdf 局部材质头；光源通道自2000步启用；真实场景另加与 configs/validation.json 相同的相机校正选项）
python train.py --scene /workspace/datasets/SSD-GS/data/Synthetic_GS3/AnisoMetal --output runs/<new_dir> \
  --representation light_atlas --material-head svbrdf --fit-all --background 1.0 --shadow-start 2000 --port-start 2000

# 修改规范配置的 name 和目标方法后启动；不得覆盖已有结果
# configs/validation.json 仍保留上次 no-ports 消融配置，并非 R2b 配方
bash launch_validation.sh

# 渲染与评价已有模型，output 必须是新目录
python evaluate.py runs/gggs_neural_material_joint/Real_NRHints/Cat/last.pt \
  --split test --lpips --highlights --output runs/cat_new_evaluation
```

Neural Material 从头训练需要 `--material-decoder`。GGGS 初始化当前仅支持方向端口与 Neural Material；Attention 使用原有 2DGS 路线。
模型、指标、配置与复现源码位于 `runs/`，日志位于 `logs/`。[完整实验索引](docs/experiments/records.json)。

## 当前实验结果

LiSA（light_atlas + svbrdf）三类数据集全部18场景完成。
[完整实验记录](docs/experiments/results.md) · [配置/命令/日志/权重索引](docs/experiments/records.json) · [环境与协议](docs/experiments/setup.md) · [四方法总表](../benchmarks/full_dataset_comparison/RESULTS.md)

2026-10-03核查：本地总体SSIM/LPIPS最佳，总PSNR尚低于SSD-GS/GS³；优先问题为几何覆盖与局部分支退化。
[结果分析](docs/experiments/full_dataset_analysis.md) · [SOTA判断](docs/research/sota_assessment.md) · [改进提案（未实施）](docs/research/improvement_proposal.md)
[2025年3月起逐场景OLAT论文与三类数据出处](docs/research/per_scene_olat_literature_202503_202610.md)

[材质头、消融与留出光照实验](docs/experiments/development.md)单独记录研发证据。其他PORT方法的历史结果见其各自实验文档；不混入LiSA主结果。
[当前状态](docs/project/status.md) · [方法流程](docs/method/pipeline.md) · [架构](docs/architecture/system.md) · [决策记录](docs/project/decisions.md)
