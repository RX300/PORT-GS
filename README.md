# PORT-GS

基于 Gaussian Splatting 的点光源重光照研究项目，代码、输出与 SSD-GS 基线隔离。

**补充实验已完成：** [LiSA补充实验总入口](docs/experiments/supplementary/README.md)：
65项结构消融、多种子、精修及耗时实验；SSS-GS基线不纳入。
配置为`configs/validation.json`，模型按实验组保存在`runs/lisa_supplementary_experiments/`。

## 当前方法

当前保留 `directional_port_v1`、`surface_attention`、`neural_material`、`light_atlas` 四种方法。
LiSA（`light_atlas`）从光源splat学习的通量特征和深度矩构建多尺度图集，接收点读取图集学习阴影、散射与近邻反弹。
[模块](docs/architecture/modules/light_atlas.md) · [提案与新颖性审查](docs/research/lisa_light_space_atlas_20261001.md)。

**LiSA-staged（2026-10-04，当前完整实验）**：训练轮廓初始化自身3DGS，8k几何预热与辐射目标过渡，
22k线性域逐高斯着色联合训练，再冻结几何/标定、以像素着色精修外观8k。所有场景使用同一38k流程。
[方法与受控证据](docs/experiments/shading_refinement.md)。纯LiSA-v2代码保存在Git标签 `lisa-v2`（`14f28d5`），后续修改未加入该标签。

其它方法及独立几何训练入口已删除。历史结果、最终权重和源码归档保留；旧方法按其源码归档复现。
已有GGGS几何仍可供方向端口/神经材质初始化；当前LiSA不使用外部几何或预训练。
[清理范围](docs/experiments/method_retirement_20260930.md) · [方法接口](docs/architecture/modules/methods.md)。

## 环境与数据

复用 `ssd-gs`（CUDA 12.1、PyTorch 2.4.1、gsplat 1.5.3），不升级共享依赖：

```bash
cd /workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS
conda activate /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs
```

共享数据：`/workspace/datasets/SSD-GS/data/<family>/<scene>/`，只读。
完整实验覆盖Real_NRHints 7场景、Synthetic_GS3 6场景、Synthetic_SSS-GS 5场景；场景列表见规范配置。
Attention/Neural既有2DGS、法线/深度先验及材质预训练依赖保留。[环境与评价协议](docs/experiments/setup.md)。

## 训练、渲染与评价

LiSA流程：训练轮廓种子 → 几何预热 → 线性域联合训练 → 固定几何的观测域外观精修 → 完整官方test。
四种方法共用训练/评价入口。真实场景训练相机校正须显式开启，官方test主指标使用原始标定。
`train.py`编排训练循环，选项、初始化及相机/灯位位于`training/`（[训练模块](docs/architecture/modules/training.md)）。

```bash
# 查看LiSA参数；其他方法由representation选择
python train.py --representation light_atlas --help

# configs/validation.json为当前补充实验矩阵；启动前确认worker_gpus为空闲物理GPU
# 既有18场景配置保存在其run的validation.json；禁止覆盖旧结果
bash launch_validation.sh

# 渲染与评价已有模型，output必须是新目录
python evaluate.py runs/lisa_staged_full_dataset_20261004/Real_NRHints/Cat/appearance/last.pt --split test --lpips --highlights --output runs/cat_new_evaluation
```

Neural Material从头训练需要`--material-decoder`；Attention使用其原有2DGS路线。
当前run的`geometry/`与`appearance/`分别保存30k及8k阶段，最终评价在`appearance/test/`。
模型、配置、命令、日志与复现源码位于对应`runs/<name>/`，启动日志位于`logs/`。

## 当前实验结果

LiSA-staged全部18场景、5691张官方test完成，场景等权PSNR/SSIM/LPIPS为 **33.4722 / .95199 / .05636**；
LiSA-v2为32.4122 / .94666 / .06154，本地GS³为31.4885 / .93925 / .06870。
Lego/Drums比v2提高7.6091/5.4169dB，距GS³分别.1053/1.0652dB。
总体三项均值提高，但PSNR仅8场景提高、10场景下降；Real/SSS均值分别下降.2157/.3976dB，不能称为全场景改进。
预算为38k，v2为30k、GS³为100k；单种子，非等预算或文献级SOTA结论。原结果、模型和Git标签均保留。

[完整逐场景结果](docs/experiments/results.md) · [六方法总表](../benchmarks/full_dataset_comparison/RESULTS.md) ·
[受控实验与残余问题](docs/experiments/shading_refinement.md) · [机器记录](docs/experiments/radiometric_curriculum_results.json) ·
[环境与协议](docs/experiments/setup.md) · [文献可比性](docs/research/sota_assessment.md)
[当前状态](docs/project/status.md) · [方法流程](docs/method/pipeline.md) · [架构](docs/architecture/system.md) · [决策记录](docs/project/decisions.md)
