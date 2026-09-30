# Training-camera self-calibration for real scenes (calib_camrot_20260930)

Motivation: the [calibration audit](calibration_audit_20260930.md) showed 4-6 px per-frame error in the
official Real_NRHints training poses; every earlier PORT model averaged misaligned views.

## Protocol (round 1)

Canonical config `configs/validation.json`, name `calib_camrot_20260930`, launched with
`bash launch_validation.sh` on GPUs 0/1/2 in the `ssd-gs` environment (CUDA 12.1). Source snapshot,
manifest and resolved argv: `runs/calib_camrot_20260930/{source.tar,manifest.json,validation.json}`.

Identical to the completed references `gggs_neural_material_joint` and `gggs_default_joint` except for the
camera terms: original GGGS + normal/depth geometry initialization (geometry trainable), full official train
522/562, 30k steps, 512 px, seed 0, port/shadow start 5000, refinement until 25000, 400k cap, mask weight
.05, deep shadows. Variants:

- `neural`: neural_material, frozen shared procedural BRDF prior, scene light-scale fitting.
- `default`: directional_port_v1.

Camera terms (new): `--optimize-cameras --camera-mode rotation --camera-start 2000 --camera-lr 1e-3
--camera-lr-final 1e-5`. Every fit camera rotates about its fixed calibrated center; sparse per-view Adam;
no anchor camera; no gauge projection (round 1); lights fixed at calibration.

Evaluation per scene: full official test (original calibration, primary) with `--shift-align 24`
secondary metrics, all pairs saved, highlight proxies; full fit evaluation using saved corrections.
The literature-style test-time calibration (`--calibrate-test-views 100`) is run separately into
`test_calibrated/` (scene frozen; camera rotation + light offset per test view fitted on GT).

Startup check (queue started 2026-09-30 01:12 JST): scheduler plus three training children (PIDs 5474-5476) on GPUs 0-2,
history past step 500, GPU utilization confirmed; Pixiu-default queued behind the first finished job.
Short smoke checks before launch: 300-step Pixiu neural and 60-step Cat default with rotation cameras,
reload and fit evaluation of saved rotations; unit tests `test_methods.CameraCalibrationTests`.

## Results

### Pixiu, neural variant (complete)

All 71 official test views at original calibration (primary), plus secondary metrics; full fit 562.

| Pixiu | Fixed PSNR / SSIM / LPIPS | Shift-aligned PSNR / SSIM / LPIPS | Test offset (dy, dx) px | Fit PSNR |
|---|---|---|---|---:|
| Previous PORT GGGS + neural material (no camera fit) | 21.571 / .8505 / .1580 | 26.75 / .8910 / .1381 | (-2.69, -9.19) | 24.62 |
| **This run (rotation camera fit)** | **24.066 / .86409 / .11624** | **31.86 / .9426 / .0918** | (+4.15, +0.75) | 32.52 |
| GS3 (100k, train cam/light fit) | 23.724 / .86405 / .11734 | 30.75 / .9415 / .0931 | (+4.27, +0.83) | - |
| SSD-GS (100k) | 23.404 / .86212 / .11927 | 30.90 / .9431 / .0923 | (+4.74, +0.82) | - |

- Fixed calibration: +2.50 dB PSNR, +.0136 SSIM, -.0418 LPIPS over the previous PORT; now above GS3 and SSD-GS on
  all three metrics (the SSIM margin over GS3 is only 4e-5, i.e. a tie).
- Calibration-compensated quality: +5.1 dB over the previous PORT and +1.0 dB over the best baseline. Aligned MSE by
  region vs GS3/SSD-GS: silhouette band equal (.0060 vs .0060/.0062), highlights slightly lower (.0197 vs .0212/.0206),
  interior 45% lower (.0018 vs .0033/.0032). Per-view linear gain 1.028 +- .042 (previous PORT 1.059 +- .069, GS3
  .998 +- .126).
- Highlight proxies: neutral-peak recall@2px .213 (previous .013), 1-4 px component recall .096 (previous .013).
- Learned corrections: a rotation shared by all cameras equivalent to (dx, dy) = (+11.96, +6.71) px plus 4.3 / 4.1 px
  per-view std, consistent over the three capture sequences and matching GS3's independent estimate
  (~(11.2, 6.3) px shared, ~3.9 px per view). The remaining fixed-calibration error is mainly the +4.15 px vertical
  test offset: without a gauge constraint, the shared pitch correction moved the frame vertically (as for GS3).
  The translation-gauge projection targets exactly this drift (round 2).
- Cost: 3125 s training, peak allocated 11.8 GiB, 335064 final Gaussians; light scale .0365.

Visual check (`test/pair_*.png`, fixed frames 0/50/70): the carved relief, the printed floral base pattern and
head highlights are now resolved at the level of GS3/SSD-GS; the previous model was uniformly blurred.

Literature-style test-time calibration (`test_calibrated/`, scene frozen, per-view rotation + light offset on GT):
**32.05 / .9445 / .0902**. SSD-GS's own paper-protocol renders (`SSD-GS/output/real_nrhints_Pixiu_20260430`, test
cameras/lights fitted during its training; shuffled render order mapped back to the official order by exact GT
match), rescored with the same PORT metric code: 31.17 / .9450 / .0911 (its own evaluator: 31.17 / .9451 / .0800;
paper table 31.93 / .9637 / .0679 uses a different metric implementation).

### Cat, neural variant (complete): large appearance gain, but the frame drifted

| Cat | Fixed PSNR / SSIM / LPIPS | Shift-aligned PSNR / SSIM / LPIPS | Test offset (dy, dx) px | Fit PSNR |
|---|---|---|---|---:|
| Previous PORT GGGS + neural material | 22.623 / .7797 / .2393 | 24.69 / .8132 / .2314 | (+2.53, -1.32) | 22.55 |
| This run (rotation camera fit, no gauge) | 15.919 / .6705 / .2804 | **27.90 / .8932 / .1630** | (+14.71, -0.23) | 29.14 |
| GS3 | 19.124 / .7169 / .2319 | 26.47 / .8955 / .1696 | (+6.84, +0.13) | - |
| SSD-GS | 18.004 / .7037 / .2482 | 26.06 / .8905 / .1734 | (+8.84, +0.04) | - |
| SSD-GS paper protocol (test calibration fitted in training) | 27.249 / .9001 / .1661 | - | - | - |

Calibration-compensated quality improved by 3.2 dB and now exceeds GS3/SSD-GS in PSNR and LPIPS (SSIM .893 vs
.896/.891). Fit PSNR 22.55 -> 29.14. But the fixed-calibration score collapsed: the learned corrections contain a
rotation shared by all cameras equivalent to (dx, dy) = (+0.41, **+12.19**) px (per-view std 7.4 / 6.7 px), and the
test offset moved from +2.5 to +14.7 px, i.e. by exactly that shared vertical component. For cameras on rings
around the object a shared pitch is (to first order) the same as translating the scene vertically, so nothing in
the images pins it; it drifted. GS3/SSD-GS show the same mechanism with 7-9 px. This is the direct motivation for
the translation-gauge projection tested in round 2 (`calib_gauge_20260930`). Cost: 4578 s, 13.3 GiB, 399786
Gaussians. Neutral-peak recall stays low on Cat (.006; fur has few specular peaks).

### Cat, default variant (complete): best calibration-compensated Cat so far

Fixed 17.449 / .69218 / .24949; shift-aligned **29.12 / .9096 / .1506** (test offset (+10.62, +0.05) px, std 1.5 / 1.8).
Aligned it exceeds the neural variant by 1.2 dB and even SSD-GS's own paper-protocol renders (27.25 / .9001 / .1661)
on all three metrics. Shared correction (dx, dy) = (+0.79, +8.25) px, per-view std 7.0 / 6.6 px; the test offset again
equals the uncorrected offset (+2.5 px) plus the shared vertical correction. The default representation is the stronger
Cat base once calibration is fixed; round 2b (`calib_gauge_default_20260930`) tests it with the gauge projection.

### Pixiu, default variant (complete): best Pixiu result of round 1

Fixed **24.699 / .86973 / .11111** (vs GS3 23.724 / .86405 / .11734 and SSD-GS 23.404 / .86212 / .11927: +0.98 dB,
+.0057 SSIM, -.0062 LPIPS over the better baseline); shift-aligned **32.19 / .9450 / .0892** (test offset
(+3.46, +0.78) px); neutral-peak recall@2px .349 (neural .213, previous PORT .013). Shared correction
(+11.98, +6.06) px, per-view std 4.2 / 4.1 px, again matching GS3. 3362 s, 12.2 GiB, 353186 Gaussians.

Literature-style test-time calibration of the round-1 default models (`test_calibrated/`): Pixiu **32.365 / .9469 /
.0881**, Cat **29.422 / .9165 / .1466**, against SSD-GS's own paper-protocol renders rescored with the same code
(Pixiu 31.174 / .9450 / .0911, Cat 27.249 / .9001 / .1661): +1.19 / +2.17 dB with better SSIM and LPIPS on both scenes.

Round 1 decision: with calibration fixed, the default directional_port_v1 base is better than the frozen neural
BRDF on both scenes (aligned Cat 29.12 vs 27.90, Pixiu 32.19 vs 31.86). The queued neural `gauge_light` jobs of
round 2 were cancelled before starting (existing-output guard, see `runs/calib_gauge_20260930/gauge_light/CANCELLED.txt`)
and light-offset fitting is tested on the default base instead (`calib_gauge_default_light_20260930`).

Round 2 startup check: `calib_gauge_20260930` scheduler with Cat-gauge (PID 14480, GPU0) and Pixiu-gauge (PID 14481, GPU1);
at step 4400 / 2600 the camera RMS is 0.19 / 0.10 deg and the logged `scene_gauge_shift` accumulates mainly along the
vertical axis (Cat [0.0004, -0.0070, 0.0056] after 2.4k camera steps), i.e. the projection is absorbing the shared pitch.

Remaining weakness by lighting geometry (shift-aligned PSNR vs GS3, binned by the angle between camera and light
directions seen from the object): Cat +1.66 / +1.56 / +1.41 / -0.24 dB and Pixiu +2.03 / +1.04 / +1.05 / +0.38 dB for
0-30 / 30-60 / 60-90 / >90 deg. PORT's advantage vanishes under side and back lighting, consistent with the
dark reddish blotch on the lit fur of side-lit Cat view 43 (also present before calibration). Shadow/visibility
estimation and light transmitted through fur and translucent material are the next appearance bottleneck.

## Round 2: translation-gauge projection (calib_gauge_20260930, neural base)

Same as round-1 `neural` plus `--camera-gauge translation`.

| Pixiu | Fixed PSNR / SSIM / LPIPS | Shift-aligned PSNR / SSIM / LPIPS | Test offset (dy, dx) px |
|---|---|---|---|
| round 1 neural (no gauge) | 24.066 / .8641 / .1162 | 31.86 / .9426 / .0918 | (+4.15, +0.75) |
| **round 2 neural + gauge** | **26.271 / .8803 / .1061** | 31.94 / .9430 / .0917 | (-2.56, +0.65) |

The projection works as designed: the shared vertical correction fell from +6.71 px to +0.22 px while the observable
shared yaw stayed (+11.95 px); the scene absorbed the vertical part as a translation of (-0.0088, -0.0376, +0.0054)
world units. Calibration-compensated quality is unchanged (31.94 vs 31.86), but the fixed-calibration test offset
returns to the uncorrected train/test difference (-2.6 px) instead of drifting to +4.2 px, adding 2.2 dB to the primary
metric: +4.70 dB over the previous PORT and +2.55 dB / +.0162 SSIM / -.0112 LPIPS over GS3.

| Cat | Fixed PSNR / SSIM / LPIPS | Shift-aligned PSNR / SSIM / LPIPS | Test offset (dy, dx) px |
|---|---|---|---|
| previous PORT GGGS + neural | 22.623 / .7797 / .2393 | 24.69 / .8132 / .2314 | (+2.53, -1.32) |
| round 1 neural (no gauge) | 15.919 / .6705 / .2804 | 27.90 / .8932 / .1630 | (+14.71, -0.23) |
| **round 2 neural + gauge** | **24.849 / .8142 / .1767** | 28.15 / .8964 / .1611 | (+2.01, +0.32) |
| GS3 | 19.124 / .7169 / .2319 | 26.47 / .8955 / .1696 | (+6.84, +0.13) |
| SSD-GS | 18.004 / .7037 / .2482 | 26.06 / .8905 / .1734 | (+8.84, +0.04) |

On Cat the shared vertical correction fell from +12.19 px to +0.04 px (scene translation (0.002, -0.024, 0.0073)), the
test offset returned from +14.7 px to +2.0 px, and the primary metric rose by 8.9 dB over round 1: +2.23 dB / +.0345
SSIM / -.0626 LPIPS over the previous PORT and +5.72 dB / +.0973 / -.0552 over GS3. With the gauge projection the
neural base now leads GS3 and SSD-GS on all three fixed-calibration metrics on both scenes.

## Rounds 2b, 2c and 3: final configuration and ablations (all complete)

Runs: `calib_gauge_default_20260930` (default + rotation camera fit + translation gauge), `calib_gauge_default_light_20260930`
(same + per-frame light offsets from step 10000), `calib_noports_20260930` (same as 2b with `--port-start 30001`).
All: full official test at original calibration (primary), shift-aligned and test-time-calibrated secondary metrics, full fit.

| Cat | Fixed PSNR / SSIM / LPIPS | Shift-aligned | Test-time calibrated | Fit PSNR |
|---|---|---|---|---:|
| previous PORT GGGS + neural | 22.623 / .7797 / .2393 | 24.69 / .8132 / .2314 | 24.75 / .8150 / .2275 | 22.55 |
| R2 neural + gauge | 24.849 / .8142 / .1767 | 28.15 / .8964 / .1611 | 28.31 / .9017 / .1578 | 29.19 |
| **R2b default + gauge** | **25.203 / .8201 / .1674** | 29.18 / .9106 / .1503 | 29.40 / .9169 / .1464 | 30.99 |
| R2c default + gauge + lights | 25.194 / .8198 / .1677 | 29.20 / .9103 / .1505 | 29.46 / .9168 / .1467 | 31.08 |
| R3 default + gauge, no ports | 25.191 / .8183 / .1677 | 29.01 / .9072 / .1519 | 29.25 / .9139 / .1470 | 30.97 |
| GS3 | 19.124 / .7169 / .2319 | 26.47 / .8955 / .1696 | - | - |
| SSD-GS | 18.004 / .7037 / .2482 | 26.06 / .8905 / .1734 | - | - |
| SSD-GS own paper protocol | - | - | 27.249 / .9001 / .1661 | - |

| Pixiu | Fixed PSNR / SSIM / LPIPS | Shift-aligned | Test-time calibrated | Fit PSNR |
|---|---|---|---|---:|
| previous PORT GGGS + neural | 21.571 / .8505 / .1580 | 26.75 / .8910 / .1381 | 26.77 / .8915 / .1361 | 24.62 |
| R2 neural + gauge | 26.271 / .8803 / .1061 | 31.94 / .9430 / .0917 | 32.04 / .9446 / .0903 | 32.52 |
| **R2b default + gauge** | **26.264 / .8811 / .1041** | 32.31 / .9451 / .0891 | 32.43 / .9469 / .0879 | 33.46 |
| R2c default + gauge + lights | 26.298 / .8812 / .1041 | 32.35 / .9452 / .0890 | 32.47 / .9470 / .0879 | 33.54 |
| R3 default + gauge, no ports | 26.181 / .8806 / .1073 | 31.59 / .9421 / .0928 | 31.71 / .9441 / .0920 | 33.05 |
| GS3 | 23.724 / .8641 / .1173 | 30.75 / .9415 / .0931 | - | - |
| SSD-GS | 23.404 / .8621 / .1193 | 30.90 / .9431 / .0923 | - | - |
| SSD-GS own paper protocol | - | - | 31.174 / .9450 / .0911 | - |

Test offsets with the gauge: Cat (+2.0, +0.4) px, Pixiu (-2.6, +0.7) px for every gauge run; this constant is the
raw train/test calibration difference along the translation gauge and caps the fixed-calibration score.

Conclusions:
- **Recommended configuration: R2b** (directional_port_v1, GGGS init, `--optimize-cameras --camera-mode rotation
  --camera-start 2000 --camera-lr 1e-3 --camera-lr-final 1e-5 --camera-gauge translation`). Versus the previous PORT on the
  primary metric: Cat +2.58 dB / +.0404 SSIM / -.0719 LPIPS, Pixiu +4.69 dB / +.0306 / -.0539. Versus GS3: Cat +6.08 dB /
  +.1032 / -.0645, Pixiu +2.54 dB / +.0170 / -.0132. It leads GS3 and SSD-GS on all three metrics, all three protocols,
  both scenes; under the literature-style protocol it exceeds SSD-GS's own paper-protocol renders by +2.15 dB (Cat) and
  +1.26 dB (Pixiu) with better SSIM and LPIPS.
- Per-frame light offsets (R2c) change nothing beyond run-to-run noise (|dPSNR| <= .06 dB on every protocol); not adopted.
- **Ports ablation (R3 vs R2b):** removing PORT's nonlocal directional ports costs 0.72 dB shift-aligned / test-time-
  calibrated PSNR and +.0037 LPIPS on Pixiu (translucent jade) and 0.17 dB / +.0016 LPIPS on Cat. With calibration fixed,
  the ports are a measurable contribution, larger for the translucent object. Fixed-calibration differences are smaller
  (.08 / .01 dB) because the constant test offset dominates that metric.

