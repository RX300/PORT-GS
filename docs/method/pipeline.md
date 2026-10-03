> **2026-09-30 方法清理更新：** 当前只保留 `directional_port_v1`、`surface_attention`、`neural_material`。
> 其余方法和独立几何训练入口已删除；已有 GGGS 几何读取/诊断和三种方法的公共依赖保留。
> 以下涉及已删除方法的内容仅作历史记录，不是可运行入口。清理详情见 `docs/experiments/method_retirement_20260930.md`（项目根目录相对路径）。

## LiSA（`light_atlas`）流程 — 2026-10-01

随机20k点原生3DGS → 每步：相机光栅化（基色、32维材质code、协方差法线、期望深度）得到像素接收点 →
局部神经材质 ρ（code、法线、光/视/半程方向编码、余弦、SG高光提示）→ 若光源通道启用（默认2000步起）：
从光源光栅化通量特征与深度矩（512²）→ 7级金字塔 → 接收点投影到光源空间逐级双线性读取 →
可见度 V（矩检验+学习残差）与传输项（对通量线性的学习核）→ L = E·(V·ρ + Σ W·Φ)，E = I/r² →
alpha合成、观察变换、0.8 L1 + 0.2 D-SSIM + 0.05 mask。增密、真实场景训练相机校正与评价流程与其他方法共用。
[模块](../architecture/modules/light_atlas.md)。

## 可选主GS残差流程 — 2026-09-23

载入同方法源checkpoint并冻结GS/传输/相机 → 原像素接收点与前景 →
零初始化光照条件有符号修正 → 非负截断 → 原alpha及观察变换。
训练仅更新新头，抽样峰/邻环/前景/定义域像素并保留重复权重；评价自动恢复头、查询所有覆盖点。
当前Pixiu源已拟合全部562train，fit16用于拟合诊断；完整71test原相机只用于开发期未训练帧评价。
[固定协议](../experiments/gs_radiance_residual.md) · [实现](../architecture/modules/radiance_residual.md)。

以下为既有通用流程及历史记录。

## 当前流程 — 2026-09-22

当前支持directional_port_v1（默认）
及learned_anchor_exchange；2DGS方法为surface_attention、neural_material和distribution_material。
按representation构造光传输方法，按该方法的geometry选择几何光栅化与增密。
2DGS训练可读取预先逐图预测的StableNormal法线与DA3相对深度；预测器不参与推理。

默认30k步：直接光从开始参与拟合，表面损失从1000步启用，阴影及非局部光从5000步启用，
25000步停止细化。可选geometry_warmup_steps=5000则先做纯圆盘RGB重建与表面约束，
第5001步重置RGB并开始重光照；这项实验已完成，结果见[实验记录](../experiments/results.md)。
每帧先光栅化属性与期望深度，反投影出像素接收点，再计算直接光及所选非局部项。
surface_attention将全部源圆盘聚合到固定非空网格，
接收像素经单次cross-attention读取受光信号。最终统一做alpha合成和观察变换。
neural_material在场景训练前独立预训练共享BRDF decoder；两个Real场景共用冻结权重。
接收点查询6维材质code、漫反射色与表面法线，BRDF和余弦形成直接光响应，
再与原方向端口组合。预训练与场景训练成本分别记录。

方法选择使用train内灯光留出validation；选定参数后从头全train训练并评价完整official test。
只保存最终last.pt，使用原始测试标定。实时实验状态见[项目状态](../project/status.md)。
详细接口见 [方法架构](../architecture/modules/methods.md)。
以下日期更早的内容保留作为历史方案记录；其中的旧默认参数及执行状态不适用于当前实验。

## Current30k default — 2026-09-16

The60k experiment was cancelled and its outputs deleted at user request.
The default budget is30000 steps, with shadows/ports starting5000 and
refinement stopping25000. Save only final last.pt, then evaluate full official
test when an experiment is explicitly launched. No new run is active.

## Current training schedule — 2026-09-15

The user-requested restart enables both shadows and directional ports at step5000,
stops Gaussian refinement at step25000, and trains through step30000.
validate_every=0 disables periodic validation and intermediate checkpoint saves;
only final last.pt is saved, followed by full official test evaluation.

## Current architecture — 2026-09-15

The active model is directional_port_v1: 512 spatial ports, four direction
channels and a shared material direction MLP. Direct and nonlocal radiance are
combined at pixel receivers. See [directional architecture](../architecture/modules/directional_transport.md).
The earlier architecture below is historical.

> Active representation restored on 2026-09-15: 512 learned spatial anchors,
> source irradiance pooling and the original material-response MLP (Git `9e9596a`).
> HashGrid and residual decoders are archived experiments. Current training
> also retains weighted loss logs/plots and the six-scene JSON launcher.

# Training and evaluation pipeline

1. Read official train metadata and create the deterministic training-light
   holdout. Cat uses 470 fit / 52 validation frames. Current-representation
   checkpoint initialization inherits its saved membership; `--fit-all` uses
   all 522 official train frames and an empty validation set.
2. Upload decoded samples to the selected GPU, estimate bounds from camera
   geometry on GPU, and initialize Gaussian geometry and the exchange model.
3. Sample a fit frame and compute source visibility. Rasterize base appearance,
   features, visibility and expected camera-Z using `RGB+ED`. Normalize attributes
   by alpha and back-project pixel centers through `K` and the view transform to
   form covered-pixel receivers. Integrate irradiance over all Gaussian sources,
   query exchange at receivers, and apply the spatial/angular material response
   per pixel. Alpha-composite the resulting foreground radiance.
4. Apply the shared observation model. PNG foreground encoding precedes alpha
   composition; HDR targets and predictions receive the same display transform.
   Optimize image, alpha and representation losses. Optional training-camera
   corrections belong only to the fit frames.
5. Refine geometry with the established opacity-reset and point-budget schedule.
   Defaults enable shadows at step 1,500, exchange at step 5,000, and stop
   refinement at step 15,000. During refinement, Gaussians whose screen radius
   exceeds 3% of the image's long edge enter split candidates, including broad
   supports with weak position gradients. Split and duplication are mutually
   exclusive. Screen-radius pruning is disabled because inherited parent radii
   are stale for newly split children; opacity/world-size pruning remains active.
   Save validation checkpoints during the 30k run.
6. Two structural rounds have completed their fixed-last validation and image
   diagnosis. The second round is accepted for perceptual/detail improvements,
   with its PSNR/outline tradeoffs recorded. Code and hyperparameters are frozen;
   first-round engineering startup corrections remain within round one.
7. Train a fresh model on all 522 Cat train frames and evaluate its final
   checkpoint on 66 official test frames using original camera/light calibration.
   Then train/evaluate Translucent and Bunny in sequence on the same GPU 0,
   applying only their predefined data/observation settings. This full-fit phase completed for all three scenes on September 13 JST;
   results retain the predefined data and metric boundaries.

`evaluate.py --split fit` restores saved training-camera corrections for fit
frames. Validation and test use original calibration. Standard LPIPS and the
quantized final metric protocol are described in
[setup](../experiments/setup.md). Actual completion and metrics belong in the
experiment record; this page describes the pipeline, not its execution status.

## 连续场着色对照

`--sdf-shading`模式要求预先拟合SDF。2DGS仍提供像素接收点深度、材质码与可见性，
基础法线改为归一化SDF梯度；PBR的图像误差反向更新field，原Gaussian法线仍参与双向几何约束。
验证/推理从checkpoint恢复同一field，详见[SDF模块](../architecture/modules/sdf.md)。


## 可选材质与光强尺度对照

neural_material默认继续使用冻结共享decoder；`--material-model ggx`将同一6维码解释为
RGB F0、两个log-alpha和混合系数，不需要材质预训练。跨表示续训需显式`--reset-material`，
只重置材质码，保留已拟合的着色法线、底色与PORT；当前GGX对照未胜出，不改变默认。

`--optimize-light-scale`在训练中优化一个全场景log尺度，正值写入原transport.light_scale buffer。
所有视图共用同一个参数，点光与非局部源照度同倍变化；checkpoint直接保存拟合值，推理不再优化。
它不修改测试相机、光位置、每帧相对强度或display_gamma。物理尺度与材质仍不可唯一辨识，
因此效果须以独立图像评价判断，不能仅用albedo饱和降低或scale变化作成功指标。


## 2026-09-24: PORT-DNA-2DGS

Implemented 8DNA-inspired photo-supervised distribution material on 2DGS.
No SDF or pretrained geometry supervision. See [module](../architecture/modules/distribution_material.md) and the distribution-material experiment report.


## 2026-09-26 Geometry-first stage

The user now prioritizes author-native 2DGS reconstruction before relighting.
A separate native_reconstruction module uses SH and author CUDA/model with calibrated
NRHints adaptation; new material training waits for geometric review. See
[protocol](../experiments/native_2dgs_geometry.md).
