# PORT-GS

独立的3DGS重光照研究项目，复用SSD-GS环境和数据协议。
Gaussian属性与期望深度光栅化 → 像素接收点 → 所选光传输方法 → alpha合成与观察变换。
训练、渲染、评价共用同一套流程；方法实现位于 `methods/`。

## 方法选择

| `--representation` | 实现 | 作用 |
| --- | --- | --- |
| `directional_port_v1`（默认） | DirectionalTransport | 当前512空间端口×4方向通道基线 |
| `paired_port` | PairedPortTransport | 方案A：入光/出光端独立空间中心和宽度 |
| `local_frame` | LocalFrameTransport | 方案B：直接光使用学习的局部坐标系，非局部方向端口保持原定义 |
| `learned_anchor_exchange` | AnchorTransport | 原空间端口irradiance交换对照 |

A、B是两个独立候选，未默认组合。两者默认R=512、B=4；B的frame_width=32。
新增方法继承基类或现有方法，并在注册表增加一项即可接入训练和checkpoint评估。
接口、扩展步骤和参数约定见[方法架构](docs/architecture/modules/methods.md)。

## 环境与数据

复用 `/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs`，CUDA12.1、PyTorch2.4.1、gsplat；无新增依赖。
数据在 `/workspace/datasets/SSD-GS/data/<family>/<scene>/`，保留原transforms JSON及图像目录。
Real_NRHints为Cat/Pixiu（512px黑底）；Synthetic_GS3为AnisoMetal/Translucent（512px白底）；
Synthetic_SSS-GS为bunny_small/dragon_small（256px黑底，显式 `--unit-light-intensity 1`）。

## 训练与评价

在已确认空闲的GPU上运行。以下示例使用GPU1：

```bash
cd /workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS
conda activate /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs

# 方案A；默认使用train内灯光留出集，最终全train训练时加 --fit-all
CUDA_VISIBLE_DEVICES=1 python train.py \
  --scene /workspace/datasets/SSD-GS/data/Real_NRHints/Cat \
  --output runs/cat_paired_s0 --representation paired_port

# 方案B
CUDA_VISIBLE_DEVICES=1 python train.py \
  --scene /workspace/datasets/SSD-GS/data/Real_NRHints/Cat \
  --output runs/cat_frame_s0 --representation local_frame --frame-width 32

# 评估自动从checkpoint恢复方法；同时保存渲染对照与逐帧指标
CUDA_VISIBLE_DEVICES=1 python evaluate.py runs/cat_paired_s0/last.pt \
  --split validation --output runs/cat_paired_s0/validation --lpips
```

默认30k步、seed0，阴影/端口从5000步启用，25000步停止细化，只保存最终last.pt。
输出包含config.json、split.json、history.jsonl、loss.png、last.pt。
正式test使用 `--split test`；方法选择与超参数应先在train内验证完成。
`python train.py --representation local_frame --help`列出该方法参数。
`--init-checkpoint`只初始化相同方法/尺寸的权重，optimizer重新创建；
跨方法只迁移几何时用 `--init-geometry`，要求相同训练划分。
现有anchor/directional checkpoint继续按原参数键加载。

六场景批量实验使用[validation.json](configs/validation.json)：将 `train.representation`
改为所需方法，并设置新的name以分离输出，再执行 `bash launch_validation.sh <配置路径>`。
选择anchor时移除dir-dim/dir-width，选择local_frame可增加frame-width；未指定的参数由方法默认值补齐。
每个job和checkpoint记录方法，源码快照包含methods包。后续实验最多使用两张空闲GPU。

## 验证与文档

```bash
python test_methods.py -v
CUDA_VISIBLE_DEVICES=1 python test_method_integration.py --output runs/method_smoke
```

第一条检查公式、初始化退化、梯度、光强线性与序列化；第二条使用真实Cat数据，
对四种方法分别训练3步、重载checkpoint并评估。短测不代表重光照质量提升。

[系统架构](docs/architecture/system.md) · [新增方法说明](docs/architecture/modules/research_methods.md) ·
[研究方案及自审](docs/research/cache_material_proposals_20260916.md) ·
[本次验证](docs/experiments/method_refactor_20260917.md) ·
[历史指标](docs/experiments/results.md) · [项目状态](docs/project/status.md)

已完成的30k方向端口结果保留在 `runs/directional_port512_validation_20260915/`。
60k实验已按此前要求取消。历史HashGrid实验使用各自源码归档。
重构前代码保存在Git提交 `2204cdc`；本次工作分支为 `feature/selectable-transport-methods`。

当前两项新方法各自的六场景30k实验已启动，使用GPU1/3，每场景训练后自动测试。
见[实验协议与输出位置](docs/experiments/research_methods_validation_20260917.md)。
