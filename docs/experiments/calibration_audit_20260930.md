# Real-scene calibration audit (Cat / Pixiu), 2026-09-30

Question: why does PORT trail GS3 / SSD-GS on real NRHints scenes, especially Pixiu,
and is the gap in the relighting model or somewhere else?

Answer: on Cat and Pixiu the official **fixed-calibration test score is dominated by camera
misalignment**, not by relighting quality. Two independent calibration defects exist:

1. **Training poses carry ~4-6 px per-frame error** (512 px images, f ~ 3100 px, i.e. about
   0.1-0.2 deg). PORT never corrected them, so every PORT model averaged misaligned
   observations: blurred texture, fur and highlights, and a training-fit ceiling of about
   24.6 dB (Pixiu) / 22.6 dB (Cat). GS3 and SSD-GS jointly refine training cameras.
2. **The official test poses are offset from the training-pose frame** by a nearly constant
   image shift: about (dy, dx) = (-3, -11) px on Pixiu and (+2, -1) px on Cat, while their
   per-view scatter is small (1-3 px). Any model is rendered at test time with this offset.

Consequently the fixed-calibration ranking of methods mostly reflects in which direction
each method's frame happened to drift. It does not measure relighting quality.

## Evidence 1: per-view best 2D shift of the rendered test images

`evaluate.py --shift-align 24` (PORT checkpoints) and `diagnose_image_errors.py
--external-renders DIR --scene SCENE` (saved GS3/SSD-GS/PORT PNGs) score each official test view
twice: as rendered with the original calibration (primary metric, unchanged), and after the
single global 2D translation of the render that best matches its GT view (integer search in
+-24 px, then 1/8 px, bilinear, requantized to uint8). The second number is a secondary,
calibration-compensated diagnostic, because the shift is fitted on GT.

All values: full official test (66 Cat / 71 Pixiu), 512 px, canonical PORT targets,
PSNR / SSIM / VGG-LPIPS. Mean shift is the translation applied to the render (+y down, +x right).

| Scene | Method | Fixed PSNR / SSIM / LPIPS | Shift-aligned PSNR / SSIM / LPIPS | Mean shift (dy, dx) px |
|---|---|---|---|---|
| Cat | GS3 (100k, train cam/light opt) | 19.12 / .7169 / .2319 | 26.47 / .8955 / .1696 | (+6.84, +0.13) |
| Cat | SSD-GS (100k, cam/light opt) | 18.00 / .7037 / .2482 | 26.06 / .8905 / .1734 | (+8.84, +0.04) |
| Cat | PORT default (directional_port512) | 21.54 / .7665 / .2304 | 24.19 / .8072 / .2235 | (+1.99, -1.24) |
| Cat | PORT DNA (2DGS) | 22.14 / .7764 / .2677 | 23.88 / .8021 / .2600 | (+2.45, -1.22) |
| Cat | PORT GGGS + default joint | 21.82 / .7702 / .2281 | 24.51 / .8113 / .2205 | (+2.25, -1.33) |
| Cat | PORT GGGS + DNA joint | 22.54 / .7743 / .2400 | 24.73 / .8099 / .2320 | (+2.63, -1.26) |
| Cat | PORT GGGS + local transport | 22.52 / .7746 / .2633 | 23.74 / .7928 / .2601 | (+2.49, -0.87) |
| Cat | PORT GGGS + neural material joint | 22.62 / .7797 / .2393 | 24.69 / .8132 / .2314 | (+2.53, -1.32) |
| Pixiu | GS3 | 23.72 / .8641 / .1173 | 30.75 / .9415 / .0931 | (+4.27, +0.83) |
| Pixiu | SSD-GS | 23.40 / .8621 / .1193 | 30.90 / .9431 / .0923 | (+4.74, +0.82) |
| Pixiu | PORT default (directional_port512) | 20.58 / .8456 / .1567 | 27.27 / .8968 / .1280 | (-2.88, -10.69) |
| Pixiu | PORT DNA (2DGS) | 21.50 / .8483 / .1670 | 26.80 / .8895 / .1457 | (-2.69, -9.46) |
| Pixiu | PORT GGGS + default joint | 20.54 / .8451 / .1560 | 27.41 / .8976 / .1268 | (-2.95, -10.78) |
| Pixiu | PORT GGGS + DNA joint | 21.18 / .8460 / .1581 | 27.35 / .8940 / .1328 | (-2.62, -9.99) |
| Pixiu | PORT GGGS + local transport | 21.90 / .8478 / .1742 | 25.60 / .8739 / .1618 | (-2.61, -8.36) |
| Pixiu | PORT GGGS + neural material joint | 21.57 / .8505 / .1580 | 26.75 / .8910 / .1381 | (-2.69, -9.19) |

Observations:

- Every method gains 2-8 dB once the global per-view offset is removed. Shift std across
  views is small for GS3/SSD-GS (~1-1.6 px) and 1.5-3.6 px for PORT, so the offset is mostly a
  constant per split, not random noise.
- GS3/SSD-GS lose Cat under fixed calibration because their training-camera optimization
  drifted the reconstruction frame by 7-9 px vertically. PORT loses Pixiu because it has no
  camera optimization and inherits the ~10 px horizontal train/test offset.
- Among PORT variants, fixed-calibration PSNR ranks follow the size of the accidental offset,
  not image quality: local transport has the best fixed Pixiu PSNR (21.90) and the worst
  shift-aligned PSNR (25.60); the older default is best aligned on Pixiu. Recent material
  choices were therefore selected on alignment noise.
- On calibration-compensated quality, PORT trails GS3/SSD-GS by ~1.7 dB (Cat) and ~3.4 dB
  (Pixiu), and by 0.05-0.06 LPIPS. The images show why: PORT is blurred everywhere
  (fur, carved relief, printed base pattern), which is the signature of averaging
  misaligned training views (defect 1), not of a missing BRDF lobe.

Artifacts: `runs/calib_camrot_20260930/baselines/<method>/<scene>/metrics.json` (per-view rows,
shift, both metric sets). PORT pair PNGs are verified byte-exact against canonical targets.

## Evidence 2: held-out train frames versus official test frames (model-independent of appearance)

`diagnose_image_errors.py CHECKPOINT --silhouette-audit` rasterizes the alpha of the GGGS
geometry that was trained **only on the internal fit split** (Cat fit470 / Pixiu fit506),
renders fit, held-out train (val52 / val56) and official test views at their original
calibration, and finds the best global shift of the predicted silhouette against GT alpha.
Silhouettes do not depend on lighting or the material model.

| Scene | Split | Frames | Shift dy (mean +- std) | Shift dx (mean +- std) | Silhouette IoU |
|---|---|---:|---|---|---:|
| Pixiu | fit (every 8th) | 64 | +0.34 +- 3.68 | +5.10 +- 3.50 | .9149 |
| Pixiu | held-out train | 56 | -0.61 +- 4.61 | +5.47 +- 4.24 | .9084 |
| Pixiu | official test | 71 | -2.73 +- 1.59 | -5.94 +- 3.32 | .9084 |
| Cat | fit (every 8th) | 59 | -0.21 +- 5.78 | -0.25 +- 5.02 | .9533 |
| Cat | held-out train | 52 | -2.62 +- 4.90 | +1.32 +- 3.37 | .9585 |
| Cat | official test | 66 | +1.55 +- 1.07 | -0.64 +- 1.17 | .9765 |

Held-out train frames behave like fit frames (same mean, large 4-6 px scatter): the
training calibration is noisy but internally unbiased. Official test frames have a much
smaller scatter but a systematic offset relative to the training frames (Pixiu about
(-3, -11) px). On Cat the fit-only model even predicts **test** silhouettes better than its
own training silhouettes (IoU .977 vs .953): the test poses are more precise than the
training poses. Reproduced with the project command (evidence:
`runs/calib_camrot_20260930/audit/silhouette_gggs_fit_only/<scene>/silhouette_audit.json`):
`python diagnose_image_errors.py runs/gggs_normal_depth_geometry/Real_NRHints/<scene>/last.pt
--silhouette-audit --output NEW_DIR`. (The +5 px Pixiu dx on fit views is a bias of this particular geometry/import,
common to all splits; only split differences are interpreted.)

Fitting the per-view test shifts of PORT GGGS+neural against camera azimuth,
`shift = c + a cos(az) + b sin(az)`, gives a constant of (dy, dx) = (-2.65, -9.26) px on Pixiu with
only 1.2 / 2.8 px sinusoidal amplitude (residual std 1.4 / 1.9 px), and (+2.47, -1.28) px on Cat.
A world-space translation of the scene would produce an azimuth-dependent shift; a constant
image-space shift instead indicates a common camera-orientation / principal-point difference
between the train and test calibrations.

## Evidence 3: what GS3's training-camera optimization learned

GS3 saves its optimized training poses (`point_cloud/iteration_100000/transforms_train.json`).
Relative to the official poses, camera centers moved < 0.01 units, but orientations rotated by
a median 0.18 deg (Cat) / 0.27 deg (Pixiu): a shared component equivalent to about (11, 6) px on
Pixiu plus per-view corrections with 4-6 px std. The per-view part matches evidence 2; the
shared part moved GS3's frame and explains its test offsets in the table above.

## Evidence 4: literature-style test-time calibration reproduces the shift-aligned numbers

`evaluate.py --calibrate-test-views 100` (scene frozen; per test view one camera rotation and one light offset
fitted on GT, as the released SSD-GS code does during training) on the previous PORT references gives
Pixiu 26.77 / 27.41 dB and Cat 24.75 / 24.65 dB (GGGS+neural / GGGS+default), within 0.1 dB of their
shift-aligned PSNR. The cheap model-agnostic 2D shift therefore captures nearly all of the calibration
compensation, and the previous PORT was 4-5 dB (Pixiu) and ~3 dB (Cat) below the published SSD-GS numbers
under a comparable protocol (SSD-GS paper: Pixiu 31.93, Cat 27.68; GS3 29.41 / 27.41; RNG 31.26 / 26.61).
Evidence: `runs/calib_camrot_20260930/baselines/PORT_*/<scene>_test_calibrated/metrics.json`.

## Consequences

- Correct the training calibration inside PORT (implemented: rotation-only fit-camera
  refinement about fixed calibrated centers, optional translation-gauge projection and
  per-frame light offsets; see `architecture/modules/cameras.md`).
- Report both metric sets. The primary score stays the fixed original-calibration test.
  Shift-aligned metrics isolate relighting quality and are the right basis for model choices.
- Published real-scene numbers (e.g. SSD-GS paper: Cat 27.68, Pixiu 31.93) are not comparable
  with fixed-calibration numbers: the released SSD-GS code also fits test-view camera/light
  parameters during training (see `SSD-GS/runs/real_fixed_calibration_20260913/*/protocol.json`).
