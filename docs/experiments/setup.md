> **2026-09-30 方法清理更新：** 当前只保留 `directional_port_v1`、`surface_attention`、`neural_material`。
> 其余方法和独立几何训练入口已删除；已有 GGGS 几何读取/诊断和三种方法的公共依赖保留。
> 以下涉及已删除方法的内容仅作历史记录，不是可运行入口。清理详情见 `docs/experiments/method_retirement_20260930.md`（项目根目录相对路径）。

## 2026-09-30：训练重构后的入口与来源记录

命令与规范配置不变，仍用 `launch_validation.sh` → `make_validation_manifest.py` → `run_benchmark.py`。
manifest 新增 `source_provenance`（HEAD、`dirty`、`git status` 行、`git diff HEAD` 的 SHA-256、third_party 各仓库修订/脏文件数/diff 哈希）
及每个作业的 `source_dirty`；精确项目源码仍以运行目录的 `source.tar` 为准，third_party 不打包。
新选项 `--opacity-reset-every`（默认 3000，0 关闭 opacity 重置，剪枝调度不变）；`--val-limit` 须为正。
[训练模块](../architecture/modules/training.md)。

## 2026-09-30：三种真实场景评价口径（主指标不变）

1. **Fixed（主指标）**：原始official test相机与灯光，无任何拟合；与此前所有表格一致。
2. **Shift-aligned（诊断）**：`evaluate.py --shift-align 24`，或对其他方法已保存PNG用
   `diagnose_image_errors.py --external-renders DIR --scene SCENE --output NEW`。每个test视角对渲染图做一个
   全局二维平移（±24px整数搜索后1/8px双线性细化、重新量化）使其与GT的MSE最小，再算PSNR/SSIM/LPIPS。
   平移在GT上拟合，只用于把标定误差与外观质量分开，不能替代主指标；对所有方法完全同一流程。
3. **Test-time calibrated（文献口径对照）**：`evaluate.py --calibrate-test-views 100`，场景/材质/网络全部冻结，
   仅对每个test视角拟合绕原相机中心的旋转（3维）及灯位偏移（3维），与SSD-GS发布代码在训练中拟合test相机/灯光的做法对应。
   用于和论文报告数值比较，不作为主指标。配置键`eval_shift_align`、`eval_calibrate_test`自动加入manifest评价阶段。

训练相机/灯光自标定：`--optimize-cameras --camera-mode rotation --camera-start 2000 --camera-lr 1e-3 --camera-lr-final 1e-5`，
可选`--camera-gauge translation`与`--optimize-lights --light-start 10000`；GGGS初始化在几何可训练时允许相机优化。
依据见[标定审计](calibration_audit_20260930.md)，模块见[cameras](../architecture/modules/cameras.md)。

## GGGS初始化的默认/DNA联合重光照

仍使用ssd-gs/CUDA12.1环境。规范入口为configs/validation.json与launch_validation.sh。
新选项`--init-geometry-format gggs --init-geometry SOURCE`导入过滤后的世界坐标3D协方差/透明度。
默认继续优化几何；显式`--freeze-geometry`才冻结。支持directional_port_v1和distribution_material。
DNA模式保留三维尺度并输入协方差短轴法线，必须显式将normal/depth/surface-consistency/distortion四个2D先验权重设0。
使用普通evaluate.py评价，无local-transport/native-2dgs标志。
`diagnose_image_errors.py CHECKPOINT --relight-preview --output NEW_DIR`生成固定相机换光预览；
`--gggs-default-review`生成含同GGGS连续深度回放的几何对照。输出目录须为新目录。
[两轮完整协议与结果](gggs_dna_joint.md)。

## 2026-09-27：GGGS几何后的逐贡献局部神经材质

当前canonical为gggs_local_transport_final，GPU0/1，材质各30000步、512px、全train522/562。
复用冻结的GGGS+先验几何；逐贡献三点积分、两视图patch梯度累积，官方test66/71和全fit终端评价。
主环境ssd-gs/CUDA12.1不变。GGGS扩展新增只读投影缓存导出，补丁为
[local_transport_backend_patch.diff](local_transport_backend_patch.diff)，在其原submodule目录运行
`python setup.py build_ext --inplace`；local_fragments.py首次使用时在项目内编译csrc/local_fragments.cu。
使用已激活ssd-gs的PATH/ninja并设置CUDA_HOME=/usr/local/cuda-12.1、TORCH_CUDA_ARCH_LIST=8.9、MAX_JOBS=8。
最终网络为PyTorch SiLU，不依赖tinycudann；无预训练材质。缓存仅保存固定几何投影与遮挡提示。
[完整协议](local_transport.md) · [预检记录](local_transport_preflight.json)。

## 2026-09-27：GGGS加入法线和深度先验

规范配置gggs_normal_depth_geometry，GPU0/1，Cat/Pixiu各fresh30000/512px。
StableNormal复用上一轮fit-only张量，DA3-Large重新对fit图逐帧预测；没有validation/test上下文。
法线.05、深度.1，从1000步起2000步渐进启用；原GGGS其余设置保持。
训练仍使用ssd-gs；新增项目内da3_env隔离推理依赖，继承共享PyTorch2.4.1/CUDA12.1，
用--no-deps安装；精确版本、创建方式、作者/权重revision见
[实验协议](gggs_normal_depth.md)和[依赖记录](gggs_prior_environment.txt)。
不使用SDF，不以教师输出作为几何GT。

## 2026-09-27：Gaussian Wrapping / StableNormal 对照

训练和评价复用 ssd-gs（PyTorch2.4.1、CUDA12.1），作者 CUDA 扩展在项目内
`setup.py build_ext --inplace`，未升级共享环境。源码与协议见
[Wrapping 实验](gaussian_wrapping.md)和[实现](../architecture/modules/wrapping_reconstruction.md)。

StableNormal 依赖旧 diffusers/transformers，因此推理单独使用项目内 venv，复用共享 torch：

```bash
/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python -m venv --system-site-packages third_party/stablenormal_env
third_party/stablenormal_env/bin/python -m pip install --no-deps \
  diffusers==0.28.0 transformers==4.36.1 huggingface-hub==0.23.0 \
  accelerate==0.30.1 tokenizers==0.15.2 safetensors==0.4.3 einops==0.7.0 \
  importlib_metadata==7.1.0 zipp==3.19.1 regex==2024.5.15
```

另用 `python3 -m venv third_party/hf_cli_env` 和该环境的
`pip install 'huggingface-hub[cli]==0.35.3'` 提供下载 CLI，避免新 Hub 与旧推理栈冲突。
官方 fp16 权重位于 `third_party/weights/{yoso-normal-v0-3,stable-normal-v0-1}`，
模型要求的 DINOv2-L 位于 `third_party/torch_cache/hub/`。
权重精确 revision、法线坐标预检、训练划分及变动项记于实验文档。
唯一启动入口仍为 `configs/validation.json` / `bash launch_validation.sh`；
manifest 先用独立 Python 生成 fit-only normal.pt，再用 ssd-gs 训练。
未使用 SDF、深度教师或官方 test；先验不是几何真值。

> 2026-09-24清理更新：runs已按用户要求仅保留5套最终结果。中间实验、训练先验/种子和独立审计导出目录已删除；最终模型/指标/图片/源码仍在。历史路径仅作来源记录，重训缓存须重新生成。见[清理说明](runs_cleanup_final_only_20260924.md)。

## 2026-09-24：PORT-DNA-2DGS

当前canonical为distribution_material_final，fresh30000 Cat/Pixiu，512px、40000 surfels，
GPU0/1，ssd-gs原环境。训练轮廓占据表面初始化，无SDF、预训练几何或材质，
全官方训练集拟合后一次完整test66/71。[完整协议](distribution_material.md)。

## 2026-09-23：完整数据残差预算链

canonical为gs_residual_budget_pilot：worker_gpus[0]、steps30000、lr-decay-steps3000、
save-steps[3000,12000]，取消少帧variants与residual-frames，保持全部562fit来源。
已有manifest在训练后依序fit3k/fit12k/fit30k；eval_test:true仅追加终端71test，
新增eval_full_fit:true追加终端562fit，二者不继承fit16 limit。
--highlights现在也生成完整组件桶raw sums及派生指标，空桶None，旧全峰指标/渲染流程不变。
[协议与检查](gs_residual_budget.md)。

## 2026-09-23：GS残差少帧拟合诊断已完成

canonical目前为已完成`gs_residual_subset_pilot`：single[0]3k、four[0,35,70,105]12k，
各512px/2048像素/seed0，仅当前stage采样池与曝光改变，完整源562fit与相机映射保留。
--residual-frames显式保序/唯一/源fit成员检查；省略仍全部fit（resume不隐式继承）。
checkpoint/history分开保存stage pool/count与累积residual_steps，不能以stage池替代来源链。
只评价同16fit，四目标帧与其它12source-fit外溢单列；eval_test=false。
[结果与边界](gs_residual_subset.md)。

## 2026-09-23：固定主GS的光照残差对照

当前canonical为`gs_radiance_residual_pilot`，GPU0/1，geometry/material两法线输入，
每组512px、3k、2048抽样像素、seed0；原GS/传输/相机完全固定，只训练新残差。
新增`eval_test:true`在终端fit16后自动完整test71，不继承fit的limit、不评价中间checkpoint。
原source已训练全部562帧；71test没有参与拟合，但曾被开发观察，不能称新的盲测。
[固定协议与结果](gs_radiance_residual.md)。以下为历史环境和实验记录。

> 文件清理说明（2026-09-22）：旧试验目录已清理，历史命令描述过去运行。
> 当前入口仍为configs/validation.json与launch_validation.sh；surface_priors原路径保持有效。
> 当前实验已停止；可用输出见[清理清单](output_cleanup_20260914.md)。

## 2026-09-22：退化诊断与控制变量实验

唯一配置入口configs/validation.json新增可选variants映射，null值删除该条件不适用的
方法参数。仍用make_validation_manifest.py/run_benchmark.py/launch_validation.sh。
本次9个job，GPU0/1/2，各30k/seed0，3场景AnisoMetal/bunny_small/dragon_small；
fit-all=false，eval_split=validation，holdout按原train灯光分组生成。
原教师预测文件仅按fit索引读取。control与weak-normal仅normal-weight .05→.005；
weak-normal与attention仅替换非局部表示（grid-size8、attention-dim32）。
其余shadow/port5000、refine25000、surface1000及深度权重.05不变。
输出runs/surface_attention_ablation_20260922/{variant}/{family}/{scene}/。
选型以场景等权validation PSNR为主并报告SSIM/LPIPS，之后运行六场景official test。
硬件、源码归档、配置快照和所有命令均记录在运行目录。

## 2026-09-21：纯2DGS预热六场景重跑

用户确认前5000步使用2DGS RGB重建+表面约束，冻结所有重光照网络；
5001步开始阴影/端口联合训练。复用原六场景全train→官方test协议，
总30k/seed0，GPU0/1/2，细化至25000。先验路径仍指向上一轮的完整预测，
本轮没有surface_preprocessing阶段，不重新下载或预测教师。

现有configs/validation.json原地修改：name=directional_surfel512_warmup5000_20260921，
geometry-warmup-steps=5000，port-start=shadow-start=5001；其余数据/表面监督参数不变。
网络学习率以重光照阶段为时间起点；切换时重置预热RGB及其Adam状态，
几何与不透明度保持连续。命令仍为 `bash launch_validation.sh`，不新增启动脚本。
输出 `runs/directional_surfel512_warmup5000_20260921/`，上一轮结果不覆盖。

启动前检查：8项CPU测试通过；真实Cat短测验证warmup不依赖灯光、网络/features无梯度
且不更新、几何更新、阶段边界一次切换及颜色重置、重光照网络恢复更新；
预热阶段和切换后的checkpoint均完成CLI加载/评估，圆盘阴影和增密检查通过。
证据位于 `runs/surfel_warmup_check/`，这些短测不构成质量结论。

## 方法选择 — 2026-09-17

训练用--representation选择directional_port_v1、paired_port、local_frame或learned_anchor_exchange。
方法参数由methods注册表生成；local_frame额外允许--frame-width（默认32）。
评估从checkpoint配置自动恢复，不接受覆盖为另一方法。
六场景JSON通过train.representation选择，实验name需独立；改为anchor时移除dir-dim/dir-width。
默认30k/seed0/512端口，阴影与端口5000启用，细化25000停止；新增方法沿用这些设置。
现有环境不变；方法初筛使用train内灯光留出，再冻结配置进行full train与official test。

## Current status: stopped; 30k default restored — 2026-09-16

User cancelled the60k experiment. Its training/scheduler/collector processes
were stopped and `runs/directional_port512_60k_validation_20260916/` deleted.
Existing completed30k results remain intact. No experiment is currently launched
by this task. Defaults:30000 steps, rank512, shadow/port start5000,
refine_stop25000, validate_every0; only the final model is saved.
Next configured run name:directional_port512_validation_20260916.
[Cancellation record](../experiments/directional_port512_60k_validation_20260916.md).

## Directional rank512 restart — 2026-09-15

Previous rank512 training/collector stopped and outputs deleted at user request.
Fresh six-scene30k/seed0 runs started onGPU0/1; actual Cat/Pixiu step100 and GPU activity verified: shadows and ports start5000,
refinement stops25000, validate_every=0 saves only final last.pt.
The comparison with rank64 includes schedule changes as well as port count.
See [protocol](../experiments/directional_port512_validation_20260915.md).

## Directional port results — 2026-09-15

Completed6/6 fresh30k seed0 fits and all1937 official test frames.
PSNR28.239156 / SSIM0.914073 / LPIPS0.090693 (equal scene mean).
Versus legacy rank512: −0.185288dB / −0.000540 / +0.001729.
AnisoMetal improves+0.533654dB; bunny_small drops−1.625855dB; overall quality
is not improved in this run. New architecture and preflight/checkpoint checks
are complete. Full protocol, per-scene metrics and evidence:
[directional experiment](../experiments/directional_port_validation_20260915.md).

# September 12 exchange experiment setup

Status: all three fresh 30k full fits and complete official tests finished on
September 13 JST. The accepted second-round code `cat_r2_source.tar` was selected
and frozen before test. Cat/Translucent/Bunny PSNR is 21.501174/28.304924/37.600658,
with complete 66/400/500-frame tests. GPU 0 is released. Final metrics, tradeoffs
and failure views are in [results](results.md); the original-calibration test
protocol and parameter freeze were retained throughout.
The previous September 11 experiments remain documented below as history.

## Current rank-512 validation subset

The active follow-up keeps the frozen training/evaluation protocol and changes
only the spatial exchange-node count to `rank=512`; `feature_dim=32` remains
unchanged. It trains exactly two fixed scenes from each dataset family:
Real_NRHints/Cat and Pixiu, Synthetic_GS3/AnisoMetal and Translucent, and
Synthetic_SSS-GS/bunny_small and dragon_small. Exact commands, output paths,
the run-local source snapshot, and startup evidence are in
[`rank512_validation_20260914.md`](rank512_validation_20260914.md).

Environment: `/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs`, Python 3.10,
PyTorch 2.4.1+cu121, CUDA 12.1 and installed gsplat. This task uses only GPU 0.
Data: `/workspace/datasets/SSD-GS/data/Real_NRHints/Cat`, 512px black background,
original camera/light calibration. Seed 0, 30,000 steps, 20k initial Gaussians,
400k maximum, feature dimension 32, network width 128 and 32 exchange anchors.

## Completed second-round candidate validation

The second-round canonical path rasterizes material attributes and expected
camera-Z, reconstructs covered-pixel world receivers, and shades them using an
eight-band spatial encoding. Source transport integrates all Gaussians. A
fresh model used the same 30k budget and fixed-last reporting after engineering
audits. This second round is accepted; structural iteration ends at two rounds.

First-round reproduction uses `runs/research_20260912/cat_r1_source.tar` and
`cat_r1_s0/last.pt`. Its quantized validation PSNR is 22.540632 at 30k; the earlier
20k unquantized monitor reported 22.682633. Current source belongs to round two.

### Shared training settings

The deterministic training-light split has 470 fit and 52 validation frames.
Defaults enable deep shadows at 1,500 steps, exchange at 5,000, and stop geometry
refinement at 15,000; absolute projected gradients and object-scale densification
are enabled. Gamma is 2.2, PNG uses automatic encoded-foreground alpha composition.
Camera optimization is optional via `--optimize-cameras` and is disabled by default.
During refinement, screen radius above 3% of the long edge triggers split
candidacy (`grow_scale2d=0.03`, `refine_scale2d_stop_iter=refine_stop`). Split and
duplication are mutually exclusive. `prune_scale2d=inf` keeps child pruning based
on opacity/world size; the parent's stored screen radius is unsuitable for its
new children. These geometry changes and exchange are one combined candidate.

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python train.py \
  --scene /workspace/datasets/SSD-GS/data/Real_NRHints/Cat \
  --output runs/research_20260912/cat_r2_s0

CUDA_VISIBLE_DEVICES=0 /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python evaluate.py \
  runs/research_20260912/cat_r2_s0/last.pt --split validation \
  --output runs/research_20260912/cat_r2_s0/validation --lpips
```

These commands document the completed second-round interface. Existing outputs
are preserved; use a new output directory for any future reproduction. Actual launch
commands and resolved configs accompany their logs. Selection ended after two
structural rounds. The completed full fits used the frozen second-round settings.

## First-round engineering validation and restart provenance

The first-round main task reports operator checks on 2,048 actual Cat Gaussians: analytical
identity errors around 4e-16 in high precision, FP32/TF32 errors around 1.7e-7,
and two directional finite-difference checks around 1e-9. These validate the
implemented exchange numerics; quality remains an experiment question.
The initial startup was interrupted before producing validation because the
screen-splitting switch also enabled pruning with stale parent radii. Its logs
are retained under the `interrupted_screen_pruning` label. The corrected startup
remained within the first structural round, whose final quality is recorded in
[results](results.md).

## Accepted full fit and official test

The accepted configuration has completed a fresh model with `--fit-all` at
`runs/research_20260912/cat_full_s0`, training all 522 official train frames with
empty validation. The following command reproduces the completed 66-frame
fixed-last test evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python evaluate.py \
  runs/research_20260912/cat_full_s0/last.pt --split test \
  --output runs/research_20260912/cat_full_s0/test --lpips
```

Held-out evaluation retains original calibration. Fit/train evaluation restores
saved training-camera offsets for fitted frames and reports that ownership.
Training validation is unquantized. Explicit evaluation uses uint8-quantized RGB,
unit-peak PSNR, zero-padded 11x11 SSIM with sigma 1.5 and standard LPIPS input
range [-1, 1]; it also records unclipped observation MSE. Compare matching metric
protocols. Rendered `pair_*.png` images show GT on the left and prediction on the right.

## Provenance and external dataset

The pre-replacement source is `runs/research_20260912/source_before.tar`.
Historical checkpoints use that source. Current checkpoints save weights, config,
step, radius, split indices and optional camera offsets; initialization restarts
optimizers. Historical Cat 22.000071 dB is a prior-method result.

GS³ Translucent was the first completed external scene, already present at
`/workspace/datasets/SSD-GS/data/Synthetic_GS3/Translucent` with 2,000 train and
400 test EXR images. Its 512px white-background/gamma export domain matches the
official path; budget and final metric differences remain recorded as comparison
boundaries. After Cat, fresh Translucent then Bunny full fits completed serially
on GPU 0. The actual Bunny scene path is
`/workspace/datasets/SSD-GS/data/Synthetic_SSS-GS/bunny_small` (500 train,
500 test), with output `runs/research_20260912/bunny_full_s0`. Their predetermined data/observation settings and
metrics paths are in [frozen cross-data protocol](comparison_20260912.md).
External runs received no new tuning or ablations. See
[data research](../research/related_work_20260912.md).

## Active anchor512 setup — 2026-09-15

The active code is restored to 512 learned spatial anchors. Use the existing
ssd-gs environment directly; tinycudann is not a runtime dependency. The current
`configs/validation.json` emits rank512/port-start5000 arguments. Training keeps
loss histories and plots. No new experiment was launched by the restore.
See [restore record](../project/restore_anchor512_20260915.md). The HashGrid
setup sections below describe archived experiments only.

## Historical direct-query setup — 2026-09-14

2026-09-15: the canonical decoder is now a residual MLP. Configure its two-block
default with `train.residual-blocks` in `configs/validation.json`. Current output
root and launch/verification details are in
[residual validation](residual_hashgrid_validation_20260915.md).

The active model removes pooled exchange and directly decodes NVIDIA HashGrid
features with light/view/material conditioning. Use `bash launch_validation.sh`
with the tracked `configs/validation.json` and `configs/hashgrid.json`.
Per-scene loss.png and history.jsonl are generated automatically. See
[direct-query protocol](direct_hashgrid_validation_20260914.md) for the six-scene
rerun and loss-plot command. The dependency setup below remains applicable.

## Preceding pooled HashGrid setup — 2026-09-14

The active implementation uses NVIDIA tiny-cuda-nn HashGrid. Experiment settings
are `configs/validation.json`, encoding settings are `configs/hashgrid.json`.
The local `third_party/python` build avoids the incompatible shared tinycudann
binary. See [HashGrid protocol/setup](hashgrid_validation_20260914.md) for build,
training, evaluation, provenance and the exact six-scene comparison.

# Historical September 11 repair experiment setup

Environment: `/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs`, Python 3.10,
PyTorch 2.4.1+cu121, installed gsplat; RTX 6000 Ada GPUs 0 and 1. Source datasets
are read from `/workspace/datasets/SSD-GS/data/Real_NRHints/`.

Round 1 uses recorded historical arguments, fixed seed 0 and 30,000 steps.
Cat records the relevant modern settings explicitly; Pixiu resolves later-added
options using current defaults, as qualified below:

| Scene | GPU | Configuration source | Fit / validation |
|---|---:|---|---:|
| Cat | 0 | `runs/cat_localized_s0/config.json` | 470 / 52 |
| Pixiu | 1 | `runs/pixiu_radiometric/config.json` | 506 / 56 |

Both use 512px, 20k initial points, 400k cap and refinement through step 15k.
Cat retains localized ports, four angular layers and deep shadows. Pixiu retains
its earlier global ports, two layers and depth shadows. The repair is shared;
the scene configurations preserve their respective historical settings.

Exact argument arrays are saved in `runs/cat_refinement_r1_s0_launch.json` and
`runs/pixiu_refinement_r1_s0_launch.json`. Logs use the corresponding `.log` names;
checkpoints and metrics live in those named directories. Existing runs remain
intact. New baseline, control and ablation runs are outside this cycle.

Evaluate the fixed last checkpoint with:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python evaluate.py \
  runs/cat_refinement_r1_s0/last.pt --split validation \
  --output runs/cat_refinement_r1_s0/validation --lpips
```

Final evaluation records full validation and a deterministic 32-frame fit subset.
Training validation covers all 52 Cat frames and the historical 32-frame Pixiu
subset; full Pixiu validation covers 56 frames. Quantized evaluation and
unquantized training metrics are labeled distinctly. Configuration selection uses
training-derived holdouts. A successful repair can then be fitted to all official
train frames at the frozen budget and evaluated once on official test.


The older Pixiu source config lacks later-added options; round 1 records the
current parser defaults explicitly. Its full-fit stage uses that resolved
configuration for 30k steps and all 562 training frames. GPU 1, seed 0 and
`runs/pixiu_refinement_full_s0` are frozen before official test evaluation.


## Cat round 2

After the image diagnosis, `runs/cat_shadow_gradient_r2_s0` repeats Cat's round-1
resolved configuration and original 470/52 split on GPU 0. 30k steps, fixed seed
0 and last checkpoint. The changed backward includes geometry/opacity shadow
derivatives with per-forward fixed sampling settings. Exact commands are in
`runs/cat_shadow_gradient_r2_s0_launch.json`; data selection, shading network and
forward observation model stay identical. No paired control is launched.


## Accepted Cat full fit

The second-round 30k configuration is frozen. A fresh model fits all 522 original
train frames with empty validation; last-step evaluation uses all official test
frames once. GPU 1, seed 0, 512px, black background. Exact commands:
`runs/cat_refinement_full_s0_launch.json`, with source configuration explicitly
pointing to `cat_shadow_gradient_r2_s0/config.json`. The new full fit uses 30k
steps; the old candidate's 100k test result is a historical reference with a
different budget and optional angular/compositing settings.


Both full-fit stages have completed. Cat evaluated all 66 official test frames;
Pixiu evaluated all 71. Their full-fit validation sets are empty, completion logs
report `best_validation_psnr=null`, and both evaluation reports use `limit=0`.
Final metrics and image inspection are in [results](results.md).
# 2DGS表面监督运行入口（2026-09-21）

新方法使用directional_surfel；两个冻结教师先分别通过同一个
prepare_surface_priors.py生成normal.pt与depth.pt，再使用既有train.py/evaluate.py。
环境隔离、已下载模型、六场景路径模板和完整命令见
[表面监督说明](../architecture/modules/surface.md)。
默认训练仍使用ssd-gs环境，未升级PyTorch/CUDA。
# 2026-09-21：directional_surfel六场景实验

用户确认沿用历史六场景协议：全official train，30k步，seed0，固定last.pt，完整
official test（不是train内validation），原始相机/灯光标定，PSNR/SSIM/LPIPS。
用户本次明确授权三张GPU，worker_gpus=[0,1,2]，GPU3未使用。

复用configs/validation.json，representation=directional_surfel；默认512端口/4方向，
5000步启用阴影与端口，25000停止细化；normal/depth权重均0.05，surface consistency/
distortion均0.01，surface-start=1000。Real与GS3为512px，SSS为256px/单位光强1；
GS3白底，其余黑底。预处理分辨率512，只读取train，输出与训练分辨率一致。

manifest新增可选预处理阶段，仍用原make_validation_manifest.py、run_benchmark.py及
launch_validation.sh，不创建专用启动脚本。每个GPU串行执行一个场景的
prepare_normal→prepare_depth→train→eval，完成后从队列取下一场景。
环境沿用ssd-gs和既有两个教师venv，设置项目内HF_HOME/TORCH_HOME。

运行命令：在PORT-GS下 `bash launch_validation.sh`。
tmux socket：`port-validation-directional_surfel512_validation_20260921`；session：validation。
源码基于47028ea工作树（含本次改动），精确代码保存在该运行的source.tar。
配置、命令和进度分别保存在以下目录的validation.json、manifest.json、status.json：
`runs/directional_surfel512_validation_20260921/`。
教师产物保存在同目录surface_priors/{family}/{scene}，训练结果按{family}/{scene}保存。
首批Cat/Pixiu/AnisoMetal分别分配GPU0/1/2，其余3场景排队。

启动核查：调度器PID13102；Cat训练PID23124到1900步（loss0.090533），
Pixiu训练PID29132到1000步（loss0.084882），训练进程/GPU活动/log进度一致。
GPU2法线预测694/2000，GPU3空闲；队列无failed任务。PID仅为当次启动记录。
## 2026-09-22：数值修复与补充对照

surface_attention_followup_20260922包含4个fresh job：重跑原队列失败的dragon
attention弱法线条件，以及AnisoMetal/bunny/dragon的attention原法线权重0.05条件。
所有训练/数据设置保持首轮约定，renderer仅修复gsplat未初始化padding的数值问题。
启动安排GPU0/2；GPU1继续原队列的bunny attention，合计不超过3张卡。
configs/validation.json原地更新，variants支持scenes列表以选择该条件需要运行的场景。
没有新增启动脚本；旧配置、源码快照、已完成结果及失败日志全部保留。
## 2026-09-22：归一化attention补充消融

dragon的未归一化attention出现单源格饱和，增加attention_score=cosine（Q/K L2归一化，
固定sqrt(32)缩放）的对照，无新增网络参数。先复用validation.json启动
runs/surface_attention_cosine_dragon_20260922，GPU1，dragon_small，30k/seed0，
normal_weight=0.05，其余与attention_normal_control相同。
已有GPU0/2任务继续dot模式；旧checkpoint缺省评分为dot，评估行为保持原定义。
之后补齐AnisoMetal/bunny的同条件validation，再选择六场景正式test设置。
归一化参数、源码和精确命令记录于该run的validation.json、source.tar、manifest.json。
为在各GPU释放后接续计算，bunny与AnisoMetal分别使用
surface_attention_cosine_bunny_20260922（GPU2）和
surface_attention_cosine_anisometal_20260922（GPU0）。三者是同一逻辑条件：
score=cosine、normal_weight=0.05；各次均原地复用同一configs/validation.json及启动器。
新增诊断仍复用diagnose_image_errors.py，`--attention-frames 2`记录每个split前两帧的实际权重分布。
## 2026-09-22：surface_attention六场景正式评估

统一复用configs/validation.json，输出surface_attention_validation_20260922。
GPU0/1/2，从头全train训练30k/seed0，再评估完整official test，预计66/71/400/400/500/500帧。
representation=surface_attention，attention_score=cosine，grid_size=8，attention_dim=32，
normal_weight=depth_weight=0.05，surface_consistency_weight=distortion_weight=0.01，surface_start=1000；
geometry_warmup_steps=0，shadow_start=port_start=5000，refine_stop=25000，validate_every=0。
六场景的分辨率、背景、光强、数据/相机/指标协议与原directional_surfel完整实验一致，
仅替换非局部方法；复用该实验全部train的StableNormal/DA3先验，未重新预测。
按三场景validation选择attention候选后冻结设置，质量尚未整体优于端口，详见results.md。


## 本地SSD-GS统一指标参考

复用SSD-GS保存的六场景PNG预测，在PORT-GS中按相同uint8 GT、固定范围PSNR、
零padding SSIM与VGG LPIPS[-1,1]重新计算全部1937帧。未重训练、改权重、改标定或重渲染基线。
逐帧核对保存GT与PORT-GS目标，所有像素差不超过1个uint8等级；复算用PORT-GS共同目标。
结果与逐帧证据存于docs/experiments/research_summary.json的ssdgs_reference。
合成四场景的cfg_args禁用cam_opt/pl_opt；训练代码在scene.optimizing=false时只抽train。
Real使用已保存的original-test-calibration渲染，但旧权重训练过程曾使用test相机/灯光优化；
即便渲染丢弃test校准状态，也不能证明严格零test来源。Real结果只作带说明的参考，不参与干净SOTA排名。
## 2026-09-22：共享神经材质，限定Cat/Pixiu

复用ssd-gs环境，PyTorch2.4.1/CUDA12.1/RTX6000 Ada，不升级训练依赖。
先在GPU0运行`CUDA_VISIBLE_DEVICES=0 python -u pretrain_material.py`，
读取configs/material.json，50k×4096程序化材质样本，seed0；独立验证seed1。
输出runs/neural_material_prior，含config/source/history/metrics/last.pt，学习率由0.001余弦衰减至0.00001。
此阶段不读取场景图像。

随后复用`bash launch_validation.sh`，configs/validation.json仅选择
Real_NRHints的Cat/Pixiu、GPU0/1，name=neural_material_real_validation_20260922。
两场景共享上述冻结decoder。30k/seed0、全train、512px黑底、4e5点上限，
surface_start1000、shadow/port5000、refine_stop25000、warmup0。
法线/深度权重0.05、一致性/distortion0.01，复用原完整train的surface_priors。
训练完成后分别对66/71帧official test计算PSNR/SSIM/VGG LPIPS并输出对比图。
初始化为随机几何，不复用其他方法的训练权重，不优化test标定。
预训练成本与场景训练成本分开报告。
# 高光恢复实验

当前canonical validation.json为Cat/Pixiu原30k checkpoint上的8000步固定几何修复，
不是原30k从头训练。使用同一份runs/neural_material_highlight_prior/last.pt，
--reset-material替换decoder并初始化材质码/着色法线偏移，保留几何和原传输权重。
--freeze-geometry固定means/scales/quats/opacities；--highlight-weight 0.05只使用训练GT亮点代理。
阴影/端口从第1步启用，不增密，不做相机优化；仅末端模型做完整official test。
GPU0/1，seed0，512px黑底，ssd-gs环境。命令仍为bash launch_validation.sh。
复现旧神经材质结果须用其run/source.tar，避免套用修复后的着色公式。
[原因、诊断与最终结果](highlight_recovery.md)。

## SDF双向几何

现有validation.json包含control/sdf两个variant；从38k神经材质模型追加3000步。
使用同一launch_validation.sh与GPU0/1，具体固定协议见[sdf_geometry.md](sdf_geometry.md)。
真实梯度和重载验证：`python test_method_integration.py --output runs/sdf_geometry_smoke --sdf-checkpoint runs/neural_material_highlight_recovery/Real_NRHints/Pixiu/last.pt`。


## 2026-09-26 native 2DGS reconstruction setup

Author source is local under `third_party/2d-gaussian-splatting`, root commit `f3e3b9fa67bbd1c75e05167ff37391d8dab2a678`; recursive submodule revisions are recorded in [the module document](../architecture/modules/native_reconstruction.md). Build `diff-surfel-rasterization` and `simple-knn` extensions in place with the existing ssd-gs Python, `CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=8 python setup.py build_ext --inplace`. No shared pip/Conda package replacement. Local simple-knn avoids the preexisting installed binary's GLIBC2.34 incompatibility.

The normal canonical config/launcher schedules Cat/Pixiu on GPUs0/1. The three recorded geometry profiles share512px,40k train-only occupancy initial positions,30k author SH reconstruction, camera calibration, internal470/52 and506/56 splits and seed0. The first profile uses original default distortion0; the second adds explicit alphaL1.2 and distortion1000; the third changes only training coordinates to unit camera extent. Model/CUDA remain author code, while initialization/full-K/data/loss/coordinate adaptations are explicit. Reconstruction coordinates are exported back before relighting geometry transfer.

Reconstruction evaluation uses `evaluate.py CHECKPOINT --native-2dgs --split validation --output NEW_PATH --lpips`; no light-conditioned output is claimed. The stage uses all internal validation and fit frames; official test is not evaluated. For exact controls use their saved `validation.json`/source archive, rather than the evolving canonical config. [Protocol and outcomes](native_2dgs_geometry.md).

GGGS + neural material uses `--representation neural_material --material-model neural --material-decoder runs/neural_material_highlight_prior/last.pt --init-geometry-format gggs` with the same explicit zero2D surface/prior losses as DNA. Pretrain the procedural decoder via `python pretrain_material.py --config configs/material.json`, then `bash launch_validation.sh`; [full protocol](gggs_neural_material_joint.md).
