# PORT-GS

Paired-Origin Radiance Transport Gaussians：独立的 3DGS 重光照研究项目。
采用 **material-conditioned conservative irradiance exchange**：完整 Gaussian
源积分与像素接收点着色。守恒与可逆性质适用于源节点离散算子，像素查询是其连续延拓。

当前使用 NVIDIA tiny-cuda-nn 多分辨率 HashGrid 替换可学习空间锚点；
512 个交换通道与网格分辨率独立。当前哈希网格从 16 到约 2048，共 16 层。
以下表格是此前 32 节点方案的历史结果，HashGrid 六场景实验已启动。

| 完整官方 test | 帧数 | PSNR | SSIM | 标准 LPIPS |
| --- | ---: | ---: | ---: | ---: |
| Cat | 66 | 21.501174 | .766281 | .227281 |
| GS³ Translucent | 400 | 28.304924 | .960368 | .051791 |
| SSS-GS Bunny small | 500 | 37.600658 | .986484 | .019684 |

Cat 达到原始标定下 >20 dB；相对历史 22.000071 dB，PSNR 下降 .4989 dB，
LPIPS 在 65/66 帧改善，alpha 误差在全部 66 帧增加。Bunny 仍有约 12 dB 的失败视角。
论文数字存在训练预算、指标或聚合范围差异，以上结果不代表严格 SOTA 排名。
生产核心由 2009 减至 1531 行（23.79%）；含测试的顶层 Python 减少 45.75%。

## 环境与复现

复用 `/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs`：PyTorch 2.4.1、
CUDA 12.1、gsplat。数据位于 `/workspace/datasets/SSD-GS/data/`，包含
Real_NRHints/Cat、Synthetic_GS3/Translucent、Synthetic_SSS-GS/bunny_small；
场景由 transforms JSON 与图像目录组成。本轮复用完整已有数据。

旧可学习锚点源码已保存为 Git commit `9e9596a`，历史权重使用对应源码加载。
HashGrid 依赖已在 `third_party/python` 本地编译，复用原环境；重装命令见
[HashGrid 实验说明](docs/experiments/hashgrid_validation_20260914.md)。
以下在 PORT-GS 目录运行：

```bash
export PYTHONPATH="$PWD/third_party/python"
CUDA_VISIBLE_DEVICES=0 /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python train.py \
  --scene /workspace/datasets/SSD-GS/data/Real_NRHints/Cat \
  --output runs/cat_reproduction_s0 --fit-all

CUDA_VISIBLE_DEVICES=0 /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python evaluate.py \
  runs/cat_reproduction_s0/last.pt --split test \
  --output runs/cat_reproduction_eval --lpips
```

默认 30k、512px、seed 0。流程为读取训练划分 → GPU 初始化 → 属性与深度栅格化 →
像素接收点光传输/着色 → 图像优化与几何细化 → checkpoint → 显式 test 渲染评价。
指标、对照图和命令位于 `runs/research_20260912/*_full_s0/`。
首轮使用 `cat_r1_source.tar`，更早源码使用 `source_before.tar`；各归档复现自身权重，
旧实验全部保留。`--init-checkpoint` 初始化当前表示，重新开始优化。

HashGrid 配置在 [`configs/hashgrid.json`](configs/hashgrid.json)；六场景、训练参数、
GPU 与实验名在 [`configs/validation.json`](configs/validation.json)。每个 Gaussian
的 `feature_dim=32`，每场景 30k、seed 0、官方 train/test 划分。

```bash
bash launch_validation.sh
```

该入口冻结配置到 `runs/hashgrid_validation_20260914/`，复用原队列调度器，
自动完成训练和完整 test 评价。详情见
[HashGrid 验证](docs/experiments/hashgrid_validation_20260914.md)。

## 历史全量队列恢复与查看

下列命令属于锚点实现，须在对应 Git/源码归档下使用。

首次启动或 tmux 会话不存在时运行：

```bash
./launch_full_benchmark.sh --resume
```

如果 `port-full-benchmark` 会话已经存在，只在对应 pane 已退出且确认没有活动
scheduler/worker 时 respawn 原 pane；不要重复启动 launcher：

```bash
tmux -L port-full-benchmark list-panes -a -F '#{session_name}:#{window_name} pid=#{pane_pid} exited=#{pane_dead} exit_status=#{pane_dead_status}'
tmux -L port-full-benchmark respawn-pane -t full:queue
tmux -L port-full-benchmark respawn-pane -t full:monitor
```

一次性查看状态：

```bash
/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python \
  ./run_benchmark.py --manifest runs/full_benchmark_20260913/manifest.json --status
```

恢复边界和中断证据见
[recovery review](docs/experiments/recovery_20260913.md)；SSD 与 PORT 的逐场景结果见
[full comparison](docs/experiments/comparison_20260913.md)。

## 文档

[方法](docs/method/principles.md) · [架构](docs/architecture/system.md) ·
[最终结果与失败图](docs/experiments/results.md) · [复现设置](docs/experiments/setup.md) ·
[跨数据协议](docs/experiments/comparison_20260912.md) · [独立审查](docs/project/review_20260912.md) ·
[外部研究](docs/research/related_work_20260912.md) · [状态](docs/project/status.md) ·
[决策](docs/project/decisions.md) · [数据合同](docs/data_contract.md)
