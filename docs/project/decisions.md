# Material decisions — completed 2026-09-13

## Resume research under the new user instruction

The September 12 request restores PORT-GS development and supersedes the earlier
retirement arrangement. At most three rounds may review, improve and retrain the
method. The target is Cat above 20 dB under original test calibration, with better
quality where possible. The historical 22.000071 dB score remains a previous
method result. SSD-GS contributes only metric references in this cycle.

## Accept round two and freeze after two structural rounds

The second-round fixed 30k checkpoint gives 22.316405 dB / .778143 SSIM /
.228932 standard LPIPS on the same 52 validation frames. Relative to round one,
PSNR decreases .224227 dB while detail energy increases .268290 → .339026
(+26.4%) and LPIPS improves .247135 → .228932. Mean visible support decreases
17.338 → 12.646px. Silhouette IoU decreases .957555 → .954375; edge accuracy
remains a limitation. Difficult frames 233 and 158 improve .249891/.268827 dB.
Relative to historical matched 30k validation, PSNR remains +.114553 dB.

Independent paired review finds LPIPS improvement on all 52 held-out frames,
SSIM improvement on 35/52 and PSNR improvement on 22/52. Alpha L1 increases
.020662 → .023228 and mean silhouette distance increases 3.840 → 4.217px.
The decision prioritizes perceptual/detail gains while recording these PSNR and
outline tradeoffs. Structural iteration ends after two rounds. The two
first-round startup corrections were engineering restarts within round one.
GPU operator/receiver audits and training are complete. Archive
`cat_r2_source.tar` freezes the selected code, and its resolved training settings
are the basis for all subsequent runs.

Fresh 30k full fits completed serially on GPU 0: Cat 522/66 train/test,
GS³ Translucent 2000/400 at 512px white background, and SSS-GS Bunny 500/500 at
256px black background with unit intensity 1. Final test used original metadata,
and the source/configuration freeze preceded testing. All three results are
complete and GPU 0 is released. External runs used frozen settings without new
tuning, ablations, other-method training or duplicate data downloads.

Cat reaches 21.501174 dB, meeting the >20 dB target. Compared with historical
full Cat, PSNR decreases .498897 dB and LPIPS improves .021971; LPIPS improves on
65/66 frames, while alpha L1 worsens on every frame. This retains a measured
perception/pixel/outline tradeoff. Translucent reaches 28.304924 dB with qualified
GS³ paper references. Bunny reaches 37.600658 dB, but its weakest views are
12.143550/12.436795 dB and the SSS-GS aggregate is not a Bunny-specific comparator.
These final test outcomes did not trigger additional method selection.

Matched-scope source counts are 2009 → 1531 physical lines across eight production
modules (-23.79%) and 4133 → 2242 across top-level Python including tests (-45.75%).
All earlier source/checkpoint archives remain intact. See
[final results](../experiments/results.md) and
[cleanup accounting](../../runs/research_20260912/cleanup_files.json).
[Round-two evidence](../../runs/research_20260912/cat_r2_s0/analysis_summary.json).

The following candidate decisions record how this selection was reached.

## Advance to pixel receivers after first-round validation

The first-round fixed 30k checkpoint reaches 22.540632 dB on 52 Cat validation
frames, +.338780 dB over historical 22.201852. It uses 399,358 Gaussians; detail
energy remains .268290 of GT versus roughly .271 previously. Mean visible support
falls to 17.338px, so smaller support alone has not restored detail. Fit32 remains
blurred as well. These measurements motivate a second round rather than candidate
acceptance. The 20k monitor peak of 22.682633 dB is unquantized and belongs to an
earlier checkpoint; reporting uses the fixed-last 30k quantized result.

The second-round renderer replaces constant per-Gaussian shaded RGB with
pixel-receiver shading. Rasterization blends base/features/visibility and expected
camera-Z; pixel-center back-projection reconstructs world receivers. The material
network gains 51 spatial channels from eight-band world-position encoding. Every
Gaussian remains in the source integral, while pixel receivers query its same
partition and feature-conditioned exchange fraction.

The renderer sets `channel_chunk=attributes.shape[-1]+1`, so the default 36
attributes and depth share one rasterization pass. The dependency's default
32-channel chunks overwrite `means2d.absgrad` across passes, losing part of the
geometry-growth signal. This explicit pass width supplies the complete attribute
and depth contribution to absolute projected gradients.

Source-node conservation and reversibility retain their precise discrete scope.
The continuous receiver extension and expected-depth mixing introduce separate
approximations. GPU audits and fresh 30k training were scheduled on the same 470/52 split,
seed 0 and fixed-last selection; their completion and acceptance are recorded above.
There are at most three rounds in total. First-round weights use the archived
`cat_r1_source.tar`; second-round changes are not a first-round improvement claim.
[First-round evidence](../../runs/research_20260912/cat_r1_s0/analysis_summary.json).

## Replace additive RGB transport with conservative irradiance exchange

One material-conditioned exchange fraction controls both source pooling and
receiver redistribution. The receiver's shared angular response acts on the
resulting irradiance. This directly replaces the weakly coupled additive branch
and the historical representation options. The exchange preserves a discrete
quadrature-weighted irradiance sum and detailed balance at fixed geometry.
These guarantees stop before local material response and image formation.
The candidate's empirical benefit and publication novelty remain open.

## Split broad image supports directly

Historical Cat diagnosis found camera-contribution-weighted support near 22px
and detail energy around 27% of GT. The new training configuration adds split
candidates when projected radius exceeds 3% of the image's long edge (15.36px at
512px), alongside the gradient-based criterion. Duplication excludes split
candidates. `refine_scale2d_stop_iter=refine_stop` keeps this rule active during
refinement and `grow_scale2d=0.03` defines the threshold.

The dependency also couples this switch to screen-radius pruning, whose parent
radii become stale after splitting. Explicit `prune_scale2d=inf` keeps pruning
based on opacity/world size while avoiding that invalid child decision. A first
startup was interrupted before validation; its logs remain identified as
`interrupted_screen_pruning`, and the first structural round restarts with this
correction. Geometry budget control is established engineering rather than a
novelty claim. Exchange and geometry changes are evaluated together, so a result
cannot isolate their individual effects.

## Factor exchange normalization for efficient GPU execution

The expanded `N x R x RGB` tensor and dimension-0 softmax were an implementation
bottleneck as the first Cat run approached 400k points and about 0.5 seconds per
training step. The replacement stores contiguous `R x N` source weights and
normalizes along the final dimension, then handles RGB exchange fractions through
two `R x 3` matrix products. Cancellation of the common source normalization
preserves the exact exchange formula and `O(NR)` complexity.

On 167,692 real Cat Gaussians with rank 32 and GPU 0, CUDA-event median operator
forward/backward time decreased from 126.353 to 0.982 ms (128.7x); forward time
decreased from 75.648 to 0.246 ms. These timings cover the isolated operator.
Complete training also includes material response, shadows, rasterization,
refinement and optimization, whose throughput is measured separately.

Float64 maximum relative differences are 3.40e-15 for outputs and at most
1.76e-13 for gradients. Float32 output error against the float64 reference is
1.52e-4 for the archived expanded layout and 2.13e-6 for the factored layout;
checked gradient accuracy also improves. The
[factorization audit](../../runs/research_20260912/operator_factorization_audit.json)
and [operator audit](../../runs/research_20260912/operator_audit.json) preserve
inputs, measurements and numerical checks.

That first-round implementation pause and restart were recorded under
`interrupted_tensor_expansion`. The restarted `cat_r1_s0` has now completed;
its quality evidence is recorded separately. These layout timings remain a
first-round operator measurement as the second-round pixel path is introduced.

## Preserve history through source and experiment provenance

The original source is archived in `runs/research_20260912/source_before.tar`.
Previous logs, checkpoints, data and research documents remain intact. Historical
checkpoints use their archived code; the active implementation uses one selected
representation. Current checkpoints omit optimizer state because initialization
starts fresh optimization rather than resuming an interrupted run.

## Keep selection in training-derived validation

Use Cat's 470/52 train-light split, 512px, seed 0 and 30k steps on GPU 0. Run at
most three structural rounds and stop early on accepted improvement. Freeze the
accepted configuration before fresh full fitting of 522 official train frames
and one evaluation of the 66 official test frames. Training-camera corrections,
when enabled, belong to fitted frames; held-out evaluation uses original camera
and light metadata. Fit evaluation restores its learned camera offsets.

## Reuse complete external data first

GS³ Translucent is the first additional dataset candidate. Existing GS³ and
SSS-GS small data are complete with respect to their local JSON references.
Reuse shared datasets; record actual execution separately. Environment-light
ReCap data requires a different input path. Published metrics become matched
comparisons only after their data and observation protocols are aligned.

The following decisions describe the completed September 11 work. They preserve
historical reasoning; current scheduling and method choices are defined above.

# Historical material decisions — 2026-09-11

## Retire PORT under the revised research objective

The user's September 11 instruction superseded the earlier decision to stop after any
measured improvement. Two repair rounds leave Cat at 22.000 dB and Pixiu at
20.972 dB, well below the local SSD-GS references. PORT development ends here.
This is a research investment decision based on measured quality and images;
different training budgets leave the theoretical capacity of PORT unresolved.
All historical decisions below remain a record of the completed repair cycle.
See [retirement](retirement.md) for evidence and the test-image correspondence audit.

## Own the refinement schedule

Independent review found a deterministic opacity-reset bug in the installed
gsplat dependency. PORT's replacement invokes existing geometry/statistics
operations with the correct reset condition. Reusing the shared environment
preserves the SSD-GS stack. The point budget limits growth while pruning and
opacity resets remain scheduled through the configured refinement phase.

## Repair verified behavior before adding a new appearance model

Existing records already found little benefit from radiance moments and a shared
camera response, and losses from the camera/object learning-rate change. Current
work first repairs verified optimization behavior. Shadow-model limitations are
recorded as hypotheses until rerun evidence establishes their practical effect.

## Preserve experiment membership and provenance

Initialized weights retain their checkpoint's fit/validation membership;
`--fit-all` explicitly consumes all train frames. Exact launch arguments, GPU,
seed, metrics and outputs are recorded within PORT-GS. User data, previous logs,
checkpoints and research documents stay intact. Existing controls are historical
references, and this cycle launches only the repaired method.

## Simplify source/query integration

Forward Gaussians and deferred pixels now share one transport evaluation. Source
weights have one implementation; identical source/target bases are reused within
the call. Observation rendering is also shared by training and evaluation.
GPU bounds eliminate unnecessary whole-image transfers. Temporary formatting
tools reside under `/tmp` and do not alter the training environment.


## Select round 2 from actual Cat images and error data

Round 1 reduces Cat's point count but leaves PSNR slightly lower. Inspection of
84 images shows blurred fit texture and broad validation illumination/structure
errors, especially in partial-shadow regions. Direction-dependent deterioration
and low correlation with alpha changes prioritize the coupling of photometric
error and shadow geometry as the next hypothesis. See the image-diagnosis report
for limitations. The candidate's unchanged forward values, convergent opacity
finite differences and finite geometry gradients justify one same-budget rerun;
quality remains to be established from its fixed-last validation and images.


## Accept measured round-2 improvement and freeze

The final 30k Cat result improves PSNR and LPIPS with about half the historical
Gaussian count. Matched-frame images show lower coarse RGB error and slightly
higher detail energy, while several difficult frames and outline errors remain.
This is sufficient for the user's early-stop option after improvement; it is
not a claim of complete real-scene accuracy. A fresh full fit uses the same
frozen 30k budget and is evaluated once on official test. A third structural
change is outside the accepted scope of this cycle.
