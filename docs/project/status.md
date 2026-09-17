## Current status: stopped; 30k default restored — 2026-09-16

User cancelled the60k experiment. Its training/scheduler/collector processes
were stopped and `runs/directional_port512_60k_validation_20260916/` deleted.
Existing completed30k results remain intact. No experiment is currently launched
by this task. Defaults:30000 steps, rank512, shadow/port start5000,
refine_stop25000, validate_every0; only the final model is saved.
Next configured run name:directional_port512_validation_20260916.
[Cancellation record](../experiments/directional_port512_60k_validation_20260916.md).

## Directional rank512 outcome — 2026-09-15

Completed6/6 fresh30k/seed0 experiments and1937 test frames. Mean PSNR29.089976, SSIM0.916603, LPIPS0.089701. Versus directional rank64: +0.850820dB, +0.002530, -0.000992.

[Full comparison](../experiments/directional_port512_validation_20260915.md).

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

# Current — restored anchor512, 2026-09-15

The user selected the original 512 learned spatial-node implementation again.
Transport/render/evaluation code is restored from `9e9596a`; loss logging/plots
and the JSON launcher remain. No new experiment has been launched. Existing
anchor512, HashGrid and residual results are retained. See
[restore record](restore_anchor512_20260915.md).

# Completed residual experiment — 2026-09-15

The user requested a residual network after HashGrid. The direct decoder now
uses a width128 stem, two residual blocks and RGB head. A fresh same-six-scene
30k/seed0 experiment completed with mean PSNR27.501420, SSIM0.905337 and
LPIPS0.096917; Pixiu remained at13.157520dB. Source revision `cdba5d2`; preflight query,
residual-branch gradients, full rendering and checkpoint reload passed.
See [residual protocol](../experiments/residual_hashgrid_validation_20260915.md).

# Completed direct-query run — 2026-09-15

The six-scene rank512 anchor experiment completed; its scene-mean PSNR gain over
rank32 was only +0.043174 dB. Baseline code is saved in Git commit `9e9596a`.
The pooled HashGrid run (`20cd679`) also completed, with scene-mean PSNR
28.402343 dB versus anchor512 28.424444 dB. The user's correction now removes
all 512-channel mixing: direct HashGrid features feed a light/view/material
conditioned RGB decoder. That six-scene 30k/seed0 rerun completed with mean
PSNR27.264570, SSIM0.904588 and LPIPS0.097787. Pixiu regressed to13.157520dB;
its loss plateaued near0.1. All loss plots and weighted histories were saved.
Source revision `4eb3f01`; direct-query gradients, full-renderer derivatives,
150-step training, loss-plot generation and checkpoint reload passed.
See [direct query protocol](../experiments/direct_hashgrid_validation_20260914.md).

# Completed — 2026-09-13

Two structural rounds and three fresh 30k full fits are complete. Round two was
selected on training-derived validation before official testing and frozen in
`cat_r2_source.tar`. The first-round screen-pruning and layout corrections were
engineering restarts within that round. All computation used GPU 0, now released. Final independent review verified
3/3 reports, source ownership and process exit; the final GPU snapshot shows
0% utilization, 14MiB and no compute process.

| Complete official test | Train / test | PSNR | SSIM | Standard LPIPS |
| --- | ---: | ---: | ---: | ---: |
| Cat | 522 / 66 | 21.501174 | .766281 | .227281 |
| Translucent | 2000 / 400 | 28.304924 | .960368 | .051791 |
| Bunny small | 500 / 500 | 37.600658 | .986484 | .019684 |

Cat meets the original-calibration >20 dB target. Relative to historical full
Cat, PSNR decreases .498897 dB, SSIM decreases .001245, and LPIPS improves
.021971 (8.81%). LPIPS improves on 65/66 frames; alpha L1 worsens on all 66.
The result reflects perceptual/detail gains with pixel/outline costs. Broad
lighting and position errors remain. Bunny's weakest views score 12.143550 and
12.436795 dB; 7/500 views fall below 20 dB despite median 38.047665 dB.
Frame 332 has severe view/outline and shape errors, with no unique cause
established by that image. The high mean does not imply robust performance
across all views.

The accepted validation change improved detail energy .268290 → .339026 of GT
(+26.4%) and LPIPS .247135 → .228932, while PSNR decreased .224227 dB from round
one. Source-node conservation/reversibility has a precise discrete scope;
pixel receiver queries and expected-depth mixing remain approximations.

The full sequence used frozen settings, original test metadata and zero test
fitting. Existing shared datasets were reused; no duplicate downloads, other
baseline training or ablations were added. Published GS³ values use a different
budget and incompletely specified final metrics; SSS-GS's small-data score is
an aggregate rather than a Bunny-specific result. No strict SOTA claim follows.

Source archives `cat_r1_source.tar`, `cat_r2_source.tar` and `source_before.tar`
preserve their own checkpoint implementations and earlier outputs. Final code
counts are 2009 → 1531 lines for eight production modules (23.79% reduction),
and 4133 → 2242 for all top-level Python including tests (45.75%). Folder names
retain the September 12 launch date. The final aggregate records **3/3 complete**.

[Final results](../experiments/results.md) · [Decision](decisions.md) ·
[Setup](../experiments/setup.md) · [Cross-data protocol](../experiments/comparison_20260912.md) ·
[Independent review](review_20260912.md) ·
[Full results](../../runs/research_20260912/full_results.json) ·
[Code counts](../../runs/research_20260912/cleanup_files.json)
