> **2026-09-30 方法清理更新：** 当前只保留 `directional_port_v1`、`surface_attention`、`neural_material`。
> 其余方法和独立几何训练入口已删除；已有 GGGS 几何读取/诊断和三种方法的公共依赖保留。
> 以下涉及已删除方法的内容仅作历史记录，不是可运行入口。清理详情见 `docs/experiments/method_retirement_20260930.md`（项目根目录相对路径）。

## 当前高光研究 — 2026-09-23

既有7种可选方法与默认方法保留。SDF辅助分支虽改善轮廓/自一致性，尚未恢复准确细小高光，
不据此声称真实几何或未见灯光改善。当前研究在较强的neural_material主GS上添加
零初始化光照条件响应残差，固定全部源状态，只优化新头，按完整来源协议比较拟合与未训练帧。
[实验与结果](experiments/gs_radiance_residual.md) · [研究接续](project/research_handoff.md)。

以下为各阶段的历史架构。

## 当前可选择方法架构 — 2026-09-17

当前支持directional_port_v1（默认）、paired_port（方案A）、local_frame（方案B）
及learned_anchor_exchange。公共训练和渲染流程不区分具体方法；按representation构造。
详细接口见 [方法架构](architecture/modules/methods.md)。
以下日期更早的内容保留作为历史方案记录，新架构不继承超出各方法适用范围的物理声明。

## Current architecture — 2026-09-15

The active model is directional_port_v1: 512 spatial ports, four direction
channels and a shared material direction MLP. Direct and nonlocal radiance are
combined at pixel receivers. See [directional architecture](architecture/modules/directional_transport.md).
The earlier architecture below is historical.

# PORT-GS overview

## Active selection — anchor512 restored, 2026-09-15

The user reverted to the original 512 learned spatial nodes, source light
pooling and material-response MLP. HashGrid/direct/residual descriptions below
are historical experiments. Loss curves and JSON configuration remain active.
See [restore record](project/restore_anchor512_20260915.md).

## Historical representation — direct HashGrid queries, 2026-09-14

2026-09-15 update: queried features now feed a residual RGB MLP with two
width128 blocks. See [residual experiment](experiments/residual_hashgrid_validation_20260915.md).

The user corrected the spatial-encoding experiment: remove the 512-channel
source pooling and query NVIDIA HashGrid directly into a per-pixel RGB decoder
conditioned on material, light/view geometry and visibility. Hash features are
active from step 1. This representation no longer claims conservative exchange.
The six-scene 30k/seed0 protocol and automatic loss plots are described in
[the current experiment](experiments/direct_hashgrid_validation_20260914.md).

## Historical second-round exchange representation

The user reopened PORT-GS research on 2026-09-12. This instruction supersedes the
September 11 retirement decision; the [historical record](project/retirement.md)
and all existing experiments remain available. The second-round candidate is accepted and structural iteration is complete. The old Cat official-test score of 22.000071 dB
belongs to the previous representation and is not a result of this candidate.

PORT-GS remains an independent 3D Gaussian relighting project in the shared
`ssd-gs` environment. First-round fixed-last Cat validation reaches 22.540632 dB,
up 0.338780 dB over the earlier 22.201852 dB validation result. Detail energy
remains 26.83% of GT versus 27.10% previously, with 399,358 Gaussians. The first
round improves average RGB error while leaving blur unresolved, motivating a
second round rather than acceptance.

The current second-round candidate retains material-conditioned irradiance
exchange over all Gaussian sources and shades reconstructed pixel receivers.
Rasterized features, base appearance, visibility and expected camera depth define
each covered pixel's material and world-space location. A spatially encoded
angular response operates there to avoid one constant shaded color per broad
Gaussian. GPU audits and its 30k validation run are complete. Conservation
and reversibility hold for source-node queries of the discrete operator;
arbitrary pixel receivers use a continuous extension with expected-depth mixing.

The second-round fixed-last validation is 22.316405 dB / .778143 SSIM /
.228932 standard LPIPS. Relative to round one, detail energy increases 26.4% and
LPIPS improves, while PSNR decreases .224227 dB and silhouette IoU decreases.
Its PSNR remains .114553 dB above the historical matched validation. The decision
prioritizes measured perceptual/detail gains, with outline errors retained as a
limitation. Two structural rounds are complete; early first-round startup
corrections were engineering restarts within that round.

Code and training settings were frozen in `cat_r2_source.tar` before official
testing. Fresh full Cat training used 522 frames and its fixed 30k checkpoint
completed all 66 original-calibration test frames: 21.501174 dB / .766281 SSIM /
.227281 standard LPIPS. This meets the >20 dB target. Relative to historical
full Cat, PSNR decreases .498897 dB and LPIPS improves .021971 (8.81%). Texture
improves while broad lighting/position errors remain, so the result preserves
the accepted perceptual tradeoff.

The cross-data sequence finished 3/3 on September 13 JST. Translucent completed
at 28.304924 dB / .960368 SSIM / .051791 LPIPS; Bunny small completed at
37.600658 / .986484 / .019684. Bunny's lowest two views are only 12.14/12.44 dB,
so the high average coexists with substantial view-specific failures. These
runs retained frozen settings and original test calibration, with no external
tuning, ablations or other-method training. Shared datasets were complete and
reused without duplicate downloads.

All work used one GPU 0, now released. Two structural rounds and three fresh
30k full fits are complete. Core production code decreased 2009 → 1531 physical
lines (23.79%); all top-level Python including tests decreased 4133 → 2242
(45.75%). All archives and earlier experiments remain intact. SSD-GS was used
only as a metric reference; external paper comparisons retain their budget,
metric and aggregation boundaries and do not establish an unqualified SOTA rank.

Read [principles](method/principles.md), [pipeline](method/pipeline.md),
[system](architecture/system.md), [setup](experiments/setup.md),
[status](project/status.md), and [independent review](project/review_20260912.md).
[Earlier results](experiments/results.md) and
[Cat image diagnosis](experiments/cat_image_diagnosis.md) provide prior evidence.
External datasets and protocol boundaries are in the
[related-work report](research/related_work_20260912.md).


## 2026-09-24: PORT-DNA-2DGS

Implemented 8DNA-inspired photo-supervised distribution material on 2DGS.
No SDF or pretrained geometry supervision. See [module](architecture/modules/distribution_material.md) and the distribution-material experiment report.
