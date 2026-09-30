# Foreground-conditioned surface initialization

## Why this is the next intervention

The completed cutoff-only recovery study restored rendering gradients but
recovered no tiny GT peaks on either all506fit or all56validation frames.
Only2.683% of the original random100000 centers satisfy at least95% of fit
silhouettes at2px tolerance. This motivates changing initial spatial allocation
and orientation, rather than assuming every random-volume point is useful.
The earlier100k and new5000-stage failures remain evidence, not overwritten
baselines. This initializer has not yet demonstrated image-quality gains.

Visual-hull initialization has precedent in Gaussian reconstruction, for example
[GaussianObject](https://chensjtu.github.io/papers/gaussianobject.html). Its sparse-view
reconstruction/diffusion-repair results are not evidence of OLAT tiny-highlight
recovery. Classical visual hulls are sensitive to calibration and silhouette
noise and cannot generally recover concavities; see the
[CMU visual-hull report](https://publications.ri.cmu.edu/visual-hull-construction-alignment-and-refinement-across-time).
Our95%-vote construction with dilation/smoothing is not a guaranteed conservative
visual hull, a complete reproduction of those methods, or geometry truth.

## Implemented construction and training contract

`surface_initialization.py` runs on CPU using the existing SciPy/NumPy/OpenCV
dependencies. No environment changes, mesh library or pretrained network.
The existing `prepare_surface_priors.py --kind surfel_init --device cpu` entry
creates `surfel_init.pt`, a report and source archive. It reads only the fit
images and stores all fit metadata plus excluded validation indices.

Fixed initial design, before any image-quality experiment:

- Same training-camera bounds as `train.py`, using its exact fit-index stride.
- All506 native512 fit alpha masks, alpha>.9, square dilation2pixels.
- Voxel consensus at least95% of fit masks; invalid/out-of-view/behind-near
  (.01) centers count as misses. Early rejection after too many misses is exact
  for this vote rule, not a camera-subset approximation.
- Coarse64-cubed camera-bound volume to estimate foreground bounds, expanded
  by two coarse cells into a cube; fine192-cubed carving in that cube. Empty
  or box-touching support fails explicitly. The coarse crop can miss small
  disconnected structures; it is a bootstrap, not a guaranteed enclosure.
- Signed EDT distance (negative inside), smoothed with sigma.75 fine voxels.
  Sample all axis-edge zero crossings. An edge along axis a receives approximate
  surface-area weight h²|n_a|; on a plane, crossing density|n_a|/h² makes the
  three axes sum to unit area density. Curved-grid area/sampling are approximate.
- Draw100000 points with local NumPy seed0, add tangent jitter within.45 voxel,
  project back with six Newton steps. Fail if residual exceeds.05 voxel or points
  leave the field. Normals are normalized interpolated field gradients. wxyz
  quaternions rotate local+Z onto those world normals, including stable-Z poles.
- Equal initial tangent sigma2sqrt(estimated area/N). With the unchanged
  opacity~.1 this suggests plane-average optical thickness~2.5, not guaranteed
  image alpha: finite disks, random overlap, curvature and occlusion matter.

`train.py --init-surfels <artifact>` is mutually exclusive with checkpoint and
geometry checkpoint initialization and supported only by `surface_reflectance`.
It validates full fit/validation lists, exact fit-frame metadata, scene,
resolution, count and camera bounds. Only means/quaternions/tangent scales change;
opacity, diffuse/material and light initialization remain fresh. Geometry is
subsequently freely optimized by the existing renderer; there is no continuing
SDF/hull constraint. The saved scene checkpoint contains its complete Gaussian
state, so evaluation does not need the external initializer artifact.

The treatment is an **initialization bundle** (allocation, normals and scales),
not a single-factor normal-only test. It must be compared to a fresh matched
random-volume control, not the previously trained25k/100k checkpoints.

## Feasibility and pending quality protocol

Four CPU checks passed first execution: signed-normal quaternion orientation
including near-negative-Z, independent sphere radius/normal/area sampling and
local RNG behavior, consensus vote invariance to a corrupted camera/order, and
geometry-only loading with strict split/calibration rejection. Source and log:
`runs/surface_initialization_smoke/{checked_source.tar,cpu_first.log}`.

The real fit-only seed construction completed in110.512s in
`runs/surface_initialization_smoke/seed` with the settings above. CUDA was not
initialized. Coarse/fine occupied nodes6510/1164553;96630 zero-crossing edges,
estimated area4.265515 world units squared, tangent sigma.01306218. All100000
projected seeds satisfy the field check (maximum residual.00019937 voxel).
These are constructed geometry statistics, not training or quality results.
Before a quality budget, inspect the constructed
seed geometry and actual native512 rendered alpha/normal/reflectance, run a
short canonical paired training/evaluation profile, and verify the initialization
and complete data lineage. No GPU training or formal quality budget has been
started at this documentation milestone. Previous validation images remain
development-observed, not a blind or pure fixed-camera lighting test.

The resource profile is fixed100 updates per fresh arm: `cube` versus
`mask_surface`, same native512/four64px cores/seed0/100000 points, intersection
shading and cutoff0 for both. All other original parameters/losses remain shared.
Cutoff0 retains the physically valid low-alpha/HDR gradient path in both arms;
the preceding cutoff-only recovery result is not being promoted as a quality win.
Each profile ends with one fit-image CLI evaluation. The initial snapshots are
then inspected at four predeclared evenly spaced fit frames0/184/370/561, with
actual alpha and world-normal composition. Geometric feasibility requires finite
outputs and, in each of those four training views, alpha>.5 versus GT alpha>.9
IoU>=.7 and foreground recall>=.8. This is a bootstrap coverage check, not a
geometry-GT or highlight gate; do not choose the training budget from profile RGB.

If these source/state/resource/initial-coverage checks pass, the quality pilot is
fixed10000 fresh updates per arm, reinitializing both with the same seed0 (never
continuing profile weights). Evaluate fit16, complete506fit and complete56val at
the fixed terminal step. No intermediate selection or official71. Preserve the
same eight validation gates and fixed GT-only crops/rings as earlier studies,
with original source evaluator scalars primary and CPU PNG analysis secondary.
All gates plus manual narrow-peak position/shape review are required; improved
initial alpha or whole-image PSNR alone cannot satisfy the highlight-recovery goal.

## Completed profile and initial coverage

`mask_surface_initialization_profile` completed both100-update training and
one-fit-frame CLI evaluations at2026-09-23 14:47:01UTC. Checked-idle GPUs0/1
were used. Cube/mask-surface training35.37/11.78s, peak allocated15.002/10.892GiB.
This short profile does not establish full-training speed or quality rankings.

The subsequent first integration execution passed: current production bytes
match launch source; both are fresh506fit/56val lineages, same frame/patch RNG
and budget, six finite updated parameter groups. Initial opacity/material/light
and camera bounds are byte-identical; only initializer/output config differs.
Treatment positions/quaternions/log-scales exactly load the seed artifact, and
stored normals agree with quaternion rotations. All fit metadata matches.

The actual INITIAL snapshots (before those100 updates) were rendered for
alpha/normals at fixed fit frames0/184/370/561, on freshly checked-idle GPU0.

| Initial fit view | Cube IoU | Mask-surface IoU | Mask-surface foreground recall |
| --- | ---: | ---: | ---: |
| 0 | .169250 | .856477 | .890259 |
| 184 | .275326 | .880958 | .912873 |
| 370 | .193790 | .897031 | .911104 |
| 561 | .157092 | .847157 | .877320 |

Mean initial IoU.198865→.870406. These silhouettes participated in construction;
this is training-mask self-consistency and a valid-coverage check, NOT independent
geometry accuracy or relighting generalization. The original cube covers nearly
the whole image, so its foreground recall1.0 is not good geometry. The treatment
passes the predeclared IoU/recall feasibility minimum in all four views.

Root viewed `integration/initial_geometry.png`: gross object/base support is
localized, but shapes are rounded/merged and concavities are missing. The saved
100-step fit frame shows a dark gray object, not learned red material/texture
or recovered small highlights. That RGB was inspected in response to the user's
status question; it did not select a different budget or hyperparameter.
All evidence is under `runs/mask_surface_initialization_profile/integration/`.

With feasibility passed, proceed to the already declared fixed10000 fresh
updates per arm. Keep full source metrics primary, fit/validation distinct,
and do not promote this initializer on the basis of initial silhouettes.


## Completed sampling and material diagnostics (before paired closeout)

CPU exact replay of the declared 10000 updates finds 163840000 core-pixel
samples, equivalent to only 1.23518 complete-image core samples per fit frame.
Of the 36659 tiny GT peak pixels, 7533 (20.5488%) never enter a loss core;
3890/19198 components (20.2625%) never enter a core. Even including SSIM
halos, 4426 pixels (12.0734%) never enter a queried patch. The completed
mask-surface checkpoint's frame/patch RNG and counts exactly match the replay.
This measures direct exposure, not effective gradients or shared multiview
learning. Most tiny pixels are interior and receive above-average expected
weight under uniform origins; border bias alone does not explain the failure.
Evidence: `runs/mask_surface_initialization_pilot/patch_coverage/`.

The mask arm's terminal parameter diagnostic finds 58649 points with opacity
at least 1/255. Their core GGX alpha median is .0991724 (initial .1), and only
one is below .01. Unsigned plane-normal change median is 11.07196 degrees;
center displacement/radius median .0121163, tangent anisotropy median1.18482.
Final shared light scale .0359505; mean F0 approximately [.04016,.03907,.04031].
These are uniform eligible-point statistics, not visibility-weighted material
or geometry error. Narrow lobes mostly have not been learned; the separate
ultranarrow-lobe pixel integration prototype is therefore not selected as
this closing phase's production intervention.
Evidence: `runs/mask_surface_initialization_pilot/mask_terminal_parameters.json`.

The user requests a bounded closeout: finish the initializer comparison,
then one inverse-PDF-compensated tiny-aware patch proposal comparison, then
freeze the final configuration and train/evaluate Cat and Pixiu official
splits. Cleanup and pause follow those results; they do not imply successful
highlight restoration. No production implementation was changed while the
initializer training/evaluation was running.


## Fixed10000 terminal results

All eight phases completed. Cube versus mask-surface:

| Scope | PSNR | SSIM | LPIPS | alpha L1 | tiny recall |
| --- | --- | --- | --- | --- | --- |
| Full506fit | 13.78755 → 21.78755 | .705911 → .857110 | .282246 → .173749 | .660730 → .023352 | 0 → 29/36659 |
| Full56validation | 14.36701 → 21.25033 | .621070 → .846899 | .301535 → .181661 | .666004 → .024226 | 0 → 4/3770 |

Validation tiny MAE .764796→.401182 and contrast/GT .000420832→.00666842.
This is substantial fixed-budget fitting/development-validation improvement,
but tiny contrast and recall gates still fail. It is not reliable highlight
recovery. The random-volume control's poor alpha support is part of this
initialization-bundle outcome; do not conflate with the earlier100k control.
Training2965.74/1135.23s, peak15.582/12.775GiB; final opacity>=1/255
24286/58649. CPU terminal source/state/lineage/RNG audit passed first execution.
Source means/component pools/fit16-versus-fullfit PNG and metric replay passed
112 checks. Saved-PNG analysis/manual gates are being finalized separately.


Terminal closeout: 4/8 numeric gates pass (tiny MAE/global PSNR/SSIM/LPIPS),
3 fail (tiny contrast/recall/fixed-ring overbrightness), and precision is
unevaluable because the control predicts no peaks. Root reviewed all8 fixed
crops and4 fullframes: restored gross color/support, but absent small peaks,
smoothed/misaligned edges and broad streaks. Manual gate false; no promotion.
All saved validation targets match canonical bytes, crops equal preregistration,
114 component comparisons and independent NumPy/CV2 rings pass. Strict scalar
cross-backend comparison has118 violations (retained in analysis_verification),
so do not call all numerical audits passed; source evaluate remains primary,
and primary/CPU gate decisions agree. No official test in this pilot.
