# Current research results — 2026-09-12–13

## Round 1: fixed-last Cat validation, followed by a second-round decision

`runs/research_20260912/cat_r1_s0/last.pt`, 30,000 steps, seed 0, GPU 0,
512px black background, original held-out calibration, 470 fit / 52 validation.
Final metrics use uint8 observation RGB, standard VGG LPIPS input [-1,1].
Training took 1,470.625 seconds. First-round weights are reproduced with
`runs/research_20260912/cat_r1_source.tar`.

| Checkpoint | Validation PSNR | SSIM | Standard LPIPS | Gaussians |
| --- | ---: | ---: | ---: | ---: |
| Historical `cat_shadow_gradient_r2_s0`, 30k | 22.201852 | .770023 | .253591 | 156667 |
| Current `cat_r1_s0`, 30k | 22.540632 | .771293 | .247135 | 399358 |

PSNR increases .338780 dB; this combined representation/geometry change does not
isolate either component's effect. Quantized fit32 PSNR is 25.856165 and SSIM
.845761. The intermediate 20k unquantized validation value is 22.682633 dB;
30k training validation is 22.541187. Both monitor values are distinct from the
fixed-last quantized score used in the table.

The shared diagnosis covers 32 fit and all 52 validation frames. Validation fine
detail energy is .268290 of GT versus approximately .271 historically; fit32 is
.279299. Mean visible support radius falls from about 21.81px to 17.338px on
validation, while 78.62% of squared error remains in the object interior.
The increased point count and smaller support have not resolved blur. Mean
source exchange fraction is .154824; this statistic describes learned parameters
and does not measure a causal image-quality contribution.

Difficult validation [frame 233](../../runs/research_20260912/cat_r1_s0/diagnosis/hard_233.png)
reaches 14.698247 dB and
[frame 158](../../runs/research_20260912/cat_r1_s0/diagnosis/hard_158.png) reaches
17.055780 dB. The first round improves average metrics, while the remaining
texture and difficult-view errors motivate a second round instead of acceptance.

Evidence: [analysis summary](../../runs/research_20260912/cat_r1_s0/analysis_summary.json),
[validation metrics](../../runs/research_20260912/cat_r1_s0/validation/metrics.json),
[fit32 metrics](../../runs/research_20260912/cat_r1_s0/fit32/metrics.json),
[diagnosis](../../runs/research_20260912/cat_r1_s0/diagnosis/metrics.json),
[training history](../../runs/research_20260912/cat_r1_s0/history.jsonl).

## Round 2: accepted for perceptual and detail gains

`cat_r2_s0/last.pt`, fresh 30k, seed 0, GPU 0, the same 470/52 split and standard
uint8 evaluation. Training completed in 1,612.392 seconds with 398,791 Gaussians.
GPU operator and receiver-rendering audits passed their documented checks.

| Validation measure | Round one | Accepted round two |
| --- | ---: | ---: |
| PSNR | 22.540632 | 22.316405 |
| SSIM | .771293 | .778143 |
| Standard LPIPS | .247135 | .228932 |
| Fine-detail energy / GT | .268290 | .339026 |
| Mean visible support radius | 17.338px | 12.646px |
| Silhouette IoU | .957555 | .954375 |

Detail energy increases 26.4%; PSNR decreases .224227 dB and silhouette IoU
also decreases. Compared with historical 30k matched validation
22.201852/.770023/.253591, the accepted model remains +.114553 dB in PSNR with
better SSIM/LPIPS; historical detail energy was about .2710 of GT. Acceptance
prioritizes perceptual/detail gains while retaining the PSNR and outline tradeoff.
The remaining detail energy is still well below GT. Independent paired review
finds LPIPS improvement on 52/52 validation frames, SSIM improvement on 35/52,
and PSNR improvement on 22/52. Alpha L1 increases .020662 → .023228 and mean
silhouette distance increases 3.840 → 4.217px; these costs accompany acceptance.

Fit32 is 25.819267 dB / .847134 SSIM, with detail energy .350311 of GT and
12.755px support. Validation hard
[frame 233](../../runs/research_20260912/cat_r2_s0/diagnosis/hard_233.png) reaches
14.948138 dB (+.249891 over round one), and
[frame 158](../../runs/research_20260912/cat_r2_s0/diagnosis/hard_158.png) reaches
17.324608 dB (+.268827). These remain challenging views.

Structural iteration ends after two rounds; first-round startup corrections are
engineering restarts. The selected source is frozen in `cat_r2_source.tar`, and
round-one weights remain reproducible with `cat_r1_source.tar`.
Fresh Cat full fit and then Translucent/Bunny full fits are authorized on GPU 0
in sequence, using frozen settings. Cat full fitting and official testing have completed; results follow below.
Selection was completed before official test evaluation. See
[frozen cross-data protocol](comparison_20260912.md).

Evidence: [round-two summary](../../runs/research_20260912/cat_r2_s0/analysis_summary.json),
[validation](../../runs/research_20260912/cat_r2_s0/validation/metrics.json),
[fit32](../../runs/research_20260912/cat_r2_s0/fit32/metrics.json),
[receiver operator audit](../../runs/research_20260912/receiver_operator_audit.json),
[receiver rendering audit](../../runs/research_20260912/receiver_rendering_audit.json).

## Frozen full Cat: complete official test, above 20 dB

Fresh training used all 522 train frames for 30k steps on GPU 0, seed 0, 512px
black background; the fixed last checkpoint contains 398,851 Gaussians.
All 66 official test frames were evaluated once with original camera/light
calibration and zero test fitting. The second-round selection and source freeze
preceded test evaluation. Recorded training time is 1579.640 seconds, launcher
wall time 1621.451 seconds, and full test evaluation 6.326 seconds.

| Complete Cat official test | PSNR | SSIM | Standard LPIPS |
| --- | ---: | ---: | ---: |
| Historical September 11 full fit | 22.000071 | .767526 | .249252 |
| Frozen September 12 second-round full fit | 21.501174 | .766281 | .227281 |
| Current minus historical | -.498897 | -.001245 | -.021971 |

The >20 dB real-Cat target is met. LPIPS improves about 8.81%; PSNR is lower
than historical full fitting. Independent paired review finds LPIPS improved
on 65/66 frames, PSNR improved on 22/66 and SSIM improved on 31/66. Alpha L1
increases .013970 → .019071 (+.005101), with all 66 frames worse in alpha error.
These paired statistics preserve the perception/pixel/outline tradeoff.
This is consistent with the perceptual/detail
selection priority, while broad lighting and position errors remain. Inspection
of [current frame 41](../../runs/research_20260912/cat_full_s0/test/hard_41.png)
and [historical frame 41](../../runs/cat_refinement_full_s0/test/inspection_041.png)
shows clearer texture alongside unresolved broad errors.

The two lowest-PSNR official frames and the retained historical difficult frame are
[33](../../runs/research_20260912/cat_full_s0/test/hard_33.png): 17.230951 dB,
[41](../../runs/research_20260912/cat_full_s0/test/hard_41.png): 18.207802 dB, and
[64](../../runs/research_20260912/cat_full_s0/test/hard_64.png): 20.542339 dB
(the historical difficult frame, not the third-lowest frame).
All 66 frames contribute to the reported aggregate.

Evidence: [run result](../../runs/research_20260912/cat_full_s0/result.json),
[full test metrics](../../runs/research_20260912/cat_full_s0/test/metrics.json),
[aggregate full results](../../runs/research_20260912/full_results.json).
The final aggregate is **3/3 complete**. External results follow below; test
scores triggered no method changes or further tuning.

## Frozen Translucent: complete 400-frame official test

The accepted second-round source `cat_r2_source.tar` trained a fresh model on
all 2,000 GS³ Translucent train frames for 30k steps, seed 0, GPU 0, 512px white
background and gamma 2.2. The fixed-last checkpoint has 399,830 Gaussians.
Training took 1539.787 seconds (1576.255 seconds launcher wall time), and the
complete 400-frame test took 26.346 seconds. Original test metadata and the
frozen settings were retained throughout.

| Translucent result | PSNR | SSIM | LPIPS | Training/protocol scope |
| --- | ---: | ---: | ---: | --- |
| PORT frozen round two | 28.304924 | .960368 | .051791 | 30k, complete 400 test, standard VGG LPIPS |
| GS³ paper reference | 32.34 | .9740 | .0318 | Official script 100k; final paper metric implementation incompletely specified |

The GS³ values are published references. The HDR white-background/gamma export
path matches, but budget and metric evidence differ; this table is not a matched
SOTA ranking. No external-data tuning, branch ablations or other-method training
were performed. See [protocol audit](../research/related_work_20260912.md).

Difficult-view outputs include
[frame 330](../../runs/research_20260912/translucent_full_s0/test/hard_330.png)
(25.081852 dB) and
[frame 118](../../runs/research_20260912/translucent_full_s0/test/hard_118.png)
(25.106934 dB). Main-task inspection of frame 330 finds the checkerboard and
overall object shape retained, with softer predicted cast shadows, some
boundary/position mismatch and local material-brightness differences. This
supports approximate visibility as a remaining limitation; one image does not
identify a unique cause. The frozen experiment protocol remains unchanged.
Evidence: [run result](../../runs/research_20260912/translucent_full_s0/result.json),
[complete test metrics](../../runs/research_20260912/translucent_full_s0/test/metrics.json).

## Frozen Bunny small: complete 500-frame test with failure views

The final frozen run used all 500 `bunny_small` training images, a fresh 30k
model, seed 0 and GPU 0 at 256px black background with gamma 2.2 and explicit
`unit_light_intensity=1`. The final checkpoint has 354,592 Gaussians. Training
took 2328.689 seconds (2341.946 seconds launcher wall time), and all 500 official
test frames were evaluated in 19.113 seconds.

| Complete Bunny small test | PSNR | SSIM | Standard LPIPS |
| --- | ---: | ---: | ---: |
| Frozen PORT round two | 37.600658 | .986484 | .019684 |

The average hides severe failures: [frame 332](../../runs/research_20260912/bunny_full_s0/test/hard_332.png)
is 12.143550 dB and [frame 98](../../runs/research_20260912/bunny_full_s0/test/hard_98.png)
is 12.436795 dB. Both remain in the 500-frame aggregate. Median PSNR is
38.047665 dB, with 7/500 frames below 20 dB and 9/500 below 25 dB. Main-task
inspection of frame 332 finds a clear side-view bunny in GT but severe predicted
view/outline and shape errors. This is a substantial view-specific failure;
the image alone does not establish a unique cause or a dataset bug. The SSS-GS paper's
35.01 dB small-data result is a synthetic-scene aggregate, not a verified Bunny
single-scene reference, and its metric defaults differ. The present 37.60 dB
therefore does not establish a win over SSS-GS. The unit-light setting remains
an explicit equal-power assumption.

Evidence: [Bunny result](../../runs/research_20260912/bunny_full_s0/result.json),
[complete test metrics](../../runs/research_20260912/bunny_full_s0/test/metrics.json),
[protocol boundaries](../research/related_work_20260912.md).

## Completed full-fit summary — September 13 JST

| Frozen 30k full fit | Train / test | PSNR | SSIM | LPIPS | Points | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cat | 522 / 66 | 21.501174 | .766281 | .227281 | 398851 | 1579.640 |
| Translucent | 2000 / 400 | 28.304924 | .960368 | .051791 | 399830 | 1539.787 |
| Bunny small | 500 / 500 | 37.600658 | .986484 | .019684 | 354592 | 2328.689 |

[Final aggregate](../../runs/research_20260912/full_results.json) records **3/3
complete**. Two structural rounds preceded the three fresh full fits. All ran
serially on GPU 0, which is released. The accepted source and parameters stayed
fixed; there was no external-data tuning, ablation or other-method training.
Complete existing data were reused under `/workspace/datasets/`, with no duplicate
downloads. Source archives and all earlier outputs remain intact.

Final code accounting: eight production modules 2009 → 1531 lines (23.79%
reduction); all top-level Python including tests 4133 → 2242 (45.75%). These are
physical-line counts with matched scope, recorded in
[cleanup statistics](../../runs/research_20260912/cleanup_files.json).
Folders retain the September 12 launch date; completion is September 13 JST.

The following results are the completed September 11 research cycle.

# Real-scene repair results — 2026-09-11

Two structural rounds are complete. The accepted implementation fixes opacity
reset scheduling and propagates shadow derivatives to geometry/opacity at fixed
per-forward sampling settings. Independent source, derivative and result reviews
are complete. Both frozen full fits and their complete official tests are
finished. All training/evaluation processes have exited. [Setup](setup.md) records budgets, exact commands and data membership.

## Cat: fixed 30k training-derived validation

470 fit / 52 validation frames, seed 0, 512px, uint8 metrics, fixed last checkpoint:

| Run | PSNR | SSIM | Standard LPIPS | Gaussians |
|---|---:|---:|---:|---:|
| Historical `cat_localized_s0` | 21.699086 | .770378 | .255818 | 318381 |
| `cat_refinement_r1_s0` | 21.646310 | .770942 | .257096 | 216456 |
| `cat_shadow_gradient_r2_s0` | **22.201852** | .770023 | **.253591** | **156667** |

Round 1 corrects a verified schedule bug and reduces model size, but leaves Cat
image quality slightly lower. After actual image/data diagnosis, round 2 adds
conditional shadow derivatives. Its configuration matches round 1 except for
the output directory; the forward rendering function remains numerically equal
for fixed weights. R2−R1 PSNR is +.555542 dB, with median +.391666 dB and 33/52
frames improved. LPIPS improves .003506 and 31/52 frames; SSIM improves on 23/52.
Alpha L1 increases .019837 → .020715.

Relative to the historical localized run, final PSNR improves **.502767 dB** and
point count falls **50.79%**. Fit32 improves from R1's 24.503358 / .822503 /
.229579 to R2's **25.575004 / .833277 / .221392** (PSNR / SSIM / standard LPIPS).

### What the pictures and error data establish

The same 84-frame diagnostic covers fit32 and validation52. Validation
quantized MSE falls .00801867 → .00704586 (**12.13%**); 16x16 block-averaged
error falls .00453249 → .00362905 (**19.93%**). Two-pixel detail energy grows
26.20% → 27.10% of GT, while substantial blur remains. Silhouette IoU slightly
falls .95787 → .95647. Mean camera-weighted support radius falls 23.61 → 21.81px.

Hard frames 233 and 158 remain poor and slightly worsen: 14.8659 → 14.7876 and
16.8408 → 16.3586 dB. Their broad lighting and alignment errors remain visible.
These results support an overall RGB/LPIPS improvement, with clear unresolved
failure modes. Model visibility-bin membership changes between runs; bucket
averages describe each model and do not identify a same-pixel causal effect.

See [image diagnosis](cat_image_diagnosis.md),
[paired frame changes](../../runs/cat_round2_paired_validation.json), and the
[R2 diagnostic data](../../runs/cat_shadow_gradient_r2_image_diagnosis/metrics.json).

## Pixiu: first-round validation and full fitting

Same historical 32 validation frames, 30k steps, unquantized training metrics:

| Metric | Historical `pixiu_radiometric` | `pixiu_refinement_r1_s0` |
|---|---:|---:|
| PSNR | 23.465475 | **23.998248** |
| SSIM | .861398 | **.864636** |
| Alpha L1 | .025812 | .028809 |
| Gaussians | 93443 | **38413** |

PSNR improves .532774 dB and point count falls 58.89%; alpha error grows slightly.
The old config omits later-added defaults, including transport measure. The
rerun resolves them with the current parser and records the resulting values,
so this is a repaired-project outcome with limited single-factor attribution.

Full 56-frame uint8 validation is 23.634690 / .868399 / .156508; fit32 is
25.969370 / .894024 / .138293. These use a different metric/sample protocol from
the 32-frame unquantized table above.

## Frozen full fits and official tests

Each full fit is fresh, uses all official train frames, has empty validation,
and evaluates its fixed final checkpoint once. No test-time parameter fitting.

| Scene | Steps / train frames | Official test | PSNR | SSIM | Standard LPIPS |
|---|---|---|---:|---:|---:|
| Pixiu | 30k / 562 | 71 frames complete | **20.972036** | **.847689** | **.163980** |
| Cat | 30k / 522 | 66 frames complete | **22.000071** | **.767526** | **.249252** |

Pixiu final point count is 37965, test alpha L1 .025962 and separately labeled
LPIPS[0,1] .159615. Its output is
[`pixiu_refinement_full_s0`](../../runs/pixiu_refinement_full_s0/test/metrics.json).
Cat used the accepted r2 configuration in `runs/cat_refinement_full_s0`, GPU 1.
Its final point count is 167692, test alpha L1 .013970, and LPIPS[0,1] .226048.
The complete report is [Cat test metrics](../../runs/cat_refinement_full_s0/test/metrics.json).
The new Cat full fit uses 30k steps; the historical 100k candidate also has
different optional angular/compositing settings, so its 18.8050 dB test score is
a historical project reference rather than a single-factor comparison.

## Verification and attribution

- Independent review identified the installed gsplat expression
  `step % self.reset_every == 0 & step > 0` as always false. PORT owns its corrected
  callback while preserving the shared dependency. Actual runs log opacity resets
  at 3000, 6000, 9000 and 12000.
- The point budget constrains growth while scheduled pruning/reset continue.
  Real-frame regression covered reset boundaries and opacity Adam-state clearing.
- Existing GPU checks cover transport linearity, positivity, normalization,
  reciprocity, coordinate rotations and deferred source subsets. Cleanup preserves
  RGB/alpha exactly on the tested existing Cat/Pixiu frames.
- Shadow opacity finite differences converge: at step .001, weighted 128-logit
  relative error is .551%, single-logit error .0163%; a real receiver's checks
  stay below .066%. Geometry gradients pass finite-value checks. This does not
  constitute exhaustive geometric finite differences or exact physical visibility.
- Fixed-camera image diagnosis and direction statistics preceded choosing round 2.
  The accepted gain does not explain every calibration, silhouette or material
  error. A third structural round is unnecessary for this cycle's measured gain.

Review evidence: [code review](../project/code_review.md),
[cleanup parity](../../runs/cleanup_verification.log),
[derivative audit](../../runs/shadow_derivative_audit.json).
All old outputs, failed-diagnostic evidence and user research documents remain
at their original paths. Commands and outputs are recorded within PORT-GS.


### Final Cat project outcome

Historical official-test PSNR 18.805021 → **22.000071 dB**, a **3.195050 dB**
increase; 61/66 corresponding frames improve. SSIM .742850 → .767526 and
standard LPIPS .295151 → .249252. The historical model uses 100k steps and
different optional Gaussian-frame/alpha settings; this reports project progress,
not a single-factor attribution. The same-config r1/r2 internal validation above
provides the narrower evidence for shadow-gradient utility.

The previous worst test frame 64 improves from 10.9224 to 21.4234 dB. The current
worst is frame 41 at 18.4222 dB. Final image inspection still shows blurred fur,
soft paper folds and some alignment/brightness errors. Both the worst current
frame and the historical failure frame were inspected with original cameras and
fixed weights; inspection did not modify metrics or select another model.

[Final frame changes](../../runs/cat_final_test_summary.json) ·
[Current difficult test frame](../../runs/cat_refinement_full_s0/test/inspection_041.png) ·
[Historical failure frame, current prediction](../../runs/cat_refinement_full_s0/test/inspection_064.png)

Independent final review confirmed fresh full fitting, empty validation,
`best_validation_psnr=null`, completed 30k logs, fixed `last.pt`, `limit=0`, and
complete 66/71-frame test membership. The cycle is complete after two structural
rounds, with measured improvements and the remaining limitations stated above.
