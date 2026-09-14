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

## Current HashGrid setup — 2026-09-14

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
