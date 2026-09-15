# PORT-GS

Paired-Origin Radiance Transport Gaussians：独立的 3DGS 重光照研究项目。
当前已恢复 **512 个可学习空间节点 + 光照交换 + 材质响应 MLP**，
核心实现来自 Git `9e9596a`。节点位置和宽度参与训练；源光照汇总后在像素接收点
查询，再由 165→128×4→3 材质响应网络着色。源节点离散交换算子具有守恒/可逆性质，
不将这些性质扩展声称为完整图像的物理保证。阴影在第1500步启用，交换在第5000步启用。
以下表格是此前 32 节点方案的历史结果。

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

当前代码可直接加载保留的32/512节点同架构权重。HashGrid及残差实验需使用各自
的Git/源码快照加载；当前运行不依赖tinycudann。
恢复范围见 [512节点恢复记录](docs/project/restore_anchor512_20260915.md)。
以下在 PORT-GS 目录运行：

```bash
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
已清理的过时实验见文末清理记录。`--init-checkpoint` 初始化同架构且rank相符的权重，重新开始优化。

六场景、训练参数、GPU与实验名在 [`configs/validation.json`](configs/validation.json)。每个 Gaussian
的 `feature_dim=32`，每场景 30k、seed 0、官方 train/test 划分。

```bash
bash launch_validation.sh
```

空间节点数由 `configs/validation.json` 的 `train.rank` 配置，默认512。
上述命令会启动新实验并冻结配置到 `runs/anchor512_validation_20260915/`；本次仅恢复代码，
尚未启动该实验。已完成的512节点结果在 `runs/rank512_validation_20260914/`。

每场景保存 `history.jsonl`（总 loss 和加权分项，每 100 步）和训练完成后的
`loss.png`。可运行 `python plot_loss.py <场景目录>/history.jsonl` 重画当前训练曲线。

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

过时试验输出已清理；保留当前 HashGrid、32/512 对照和测试依赖。
删除清单与保留范围见 [输出清理记录](docs/experiments/output_cleanup_20260914.md)。

[方法](docs/method/principles.md) · [架构](docs/architecture/system.md) ·
[最终结果与失败图](docs/experiments/results.md) · [复现设置](docs/experiments/setup.md) ·
[跨数据协议](docs/experiments/comparison_20260912.md) · [独立审查](docs/project/review_20260912.md) ·
[外部研究](docs/research/related_work_20260912.md) · [状态](docs/project/status.md) ·
[决策](docs/project/decisions.md) · [数据合同](docs/data_contract.md)
