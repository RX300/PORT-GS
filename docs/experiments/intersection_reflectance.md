# Ray–surfel intersection reflectance

## Terminal result: completed, gates failed (2026-09-23)

Both fixed 100000-step arms and all twelve scheduled phases completed at
13:33:54 UTC. Checkpoint/source/RNG and source-metric audits completed on CPU;
the cached CPU VGG saved-PNG comparison and fixed-image manual review are also
complete. Intersection passes only **2/8 numerical gates (PSNR and SSIM)**;
all six other numerical gates and the manual gate fail. No official71 evaluation,
default promotion, budget extension, or new training was added. The complete
research goal remains unfinished.

Source-evaluator metrics remain primary; the full506 fit and development-observed
56-frame validation are distinct. Recall and precision below are fractions.

| Split / mode | PSNR | SSIM | LPIPS | Tiny MAE | Tiny contrast/GT | Tiny recall | Peak precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation56 / aggregate |20.710851|0.835748|0.189349|0.412307|0.004432|0.000265|0.050584|
| validation56 / intersection |20.756309|0.841830|0.194711|0.417327|0.001574|0.000265|0.038462|
| fit506 / aggregate |21.517662|0.844556|0.183445|0.375719|0.005352|0.000164|0.042975|
| fit506 / intersection |21.411010|0.849832|0.188207|0.384767|0.003849|0.000109|0.097187|

On validation, tiny contrast/GT falls .004432305→.001573793; recall is unchanged
at one matched tiny pixel out of3770 in both arms. Tiny RGB MAE rises
.412306658→.417327029, global precision drops .050583658→.038461538, and
LPIPS rises .189348605→.194711062. PSNR increases .045458dB and SSIM .006082,
which do not compensate for the failed required conditions.

The independently reconstructed **complete-frame** fixed-four ring has19039
pixels. Its neutral positive error is .008547779742 for aggregate versus
.024173499343 for intersection: **+182.804425%**, failing the5% limit. The full56
ring has335132 pixels and respective errors .023312582403/.031167514861;
this full56 value is descriptive and does not replace the frozen fixed-four gate.

All eight preregistered GT crops fail narrow-peak position/shape recovery for both
arms: broad smooth or faceted shading replaces the small white points. Fullframes
32/148/329/409 show blurred/streaked contours and lost fine structure. Intersection
also produces a large false white/cyan patch on the frame148 front/base and a
cyan/white patch on the frame409 red body. These are direct image observations,
not a diagnosis of true geometry or proof of a specific failure cause. Root independently
reviewed all eight crops and all four fullframes and concurred with the failed
manual gate; the earlier optional second-subagent attempt hit the thread limit
and remains recorded, with no additional subagent created.

The three fit16 checkpoints were evaluated after training, without early selection:

| Mode / fit16 checkpoint | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| aggregate / 25k |17.730924|0.762603|0.239578|
| aggregate / 50k |21.784414|0.850811|0.175611|
| aggregate / 100k |22.156758|0.852122|0.174215|
| intersection / 25k |15.242400|0.800528|0.251439|
| intersection / 50k |21.573245|0.855450|0.180544|
| intersection / 100k |22.086715|0.857495|0.178140|

The source audit checks all1220 saved pairs across the ten evaluation reports,
their original-frame mappings, canonical target bytes, raw component pools and
source scalar pooling. Terminal fit16/fullfit overlap replays exactly in this run.
Validation GT is17920 neutral peak pixels and2039 tiny components/3770 pixels;
fit GT is194957 peak pixels and19198 tiny components/36659 pixels. Terminal tiny
fit recall is only6/36659 aggregate and4/36659 intersection, so the problem is
also present in observed fit images.

All16 frozen production/config/launcher files and the eight common initial tensors
match byte-for-byte. Each arm has1001 logged sample records; dedicated frame/patch
RNG, checkpoint counts at25k/50k/100k, the506/56 lineage, finite state/light and
all six changed geometry/material groups were verified. Both used400000 patch
draws; center/radius remained fixed. Individual zero-gradient group rows
(30 intersection,35 aggregate) are retained and allowed, not nonfinite failures.

| Opacity>=1/255 eligibility among stored100000 points | 25k | 50k | 100k |
| --- | ---: | ---: | ---: |
| intersection |5047|557|511|
| aggregate |6160|947|844|

This is a CPU-decoded opacity support bound, not visibility-weighted contribution
or true surface area. At100k, eligible-population mean diffuse/F0 is
.303865/.249519 for intersection and .251087/.180436 for aggregate; fractions
with core GGX alpha<.01 are37.9648%/21.8009%. The final positive light scales
are .0443875566/.0498010404, relative gains3.1489169/2.8066226. Narrow material
parameters alone do not establish narrow image highlights or identified reflectance.

Logged training time is16630.04/16415.13s and peak own allocated GPU memory
13.709668/10.276782GiB (intersection/aggregate). These long-run timings are
distinct from the100-step profile launched on checked-idle GPUs. The separate
resource-sharing record includes a foreign GPU0 process observed at10:34:39UTC
and absent at the later root observation; its continuous duration was not measured.
No global or future GPU-idle claim follows these records.

The CPU comparison's structure, exact canonical GT, component rows/pools, peak
counts, fixed crops, and independent ring reconstruction pass. **Strict scalar
backend checks do not all pass** at the unchanged rtol1e-6/atol1e-8:106 frame
LPIPS plus both model means,10 frame SSIM, and49 frame plus one pooled global
contrast-ratio checks fail (168 checks total). Maximum absolute differences are
1.062453e-5 LPIPS,1.490116e-6 SSIM,3.814697e-6 PSNR and2.700835e-8 global
contrast ratio. Source metrics were not replaced, no tolerance was expanded,
and all eight primary/secondary gate decisions agree. The numerical discrepancy
record remains a limitation; this is not an all-checks-passed claim.

This single-seed unoccluded implementation fails its fixed matched study. It
does not reject all surface foundations or establish an optimization/representation
upper bound. Validation excludes these56 frames from the fresh lineage but was
historically development-observed; camera and light both vary. Fit, validation,
state/replay self-consistency and true geometry must remain separate. The existing
population-lifecycle and pixel-footprint synthetic diagnostics are separate research
evidence; neither is a selected production remedy or a new quality result.

[primary gate](../../runs/intersection_reflectance_pilot/validation_gate.json), [manual review](../../runs/intersection_reflectance_pilot/manual_review.json), [checkpoint audit](../../runs/intersection_reflectance_pilot/terminal_checkpoint_audit.json), [source metrics audit](../../runs/intersection_reflectance_pilot/terminal_metrics_audit.json), [CPU comparison verification](../../runs/intersection_reflectance_pilot/analysis_verification.json).
The launch/source archive, postlaunch analysis archive, earlier integration/startup
audit failures and the root phase-observer issue remain preserved. Final terminal
records are `completion.json`, `verification.json` and `docs.tar` under the run.


Postterminal provenance: after the completed source/metric audits and failed primary/manual gates, root began the separate optional `fragment_alpha_min` / fixed-checkpoint gradient diagnostic implementation (default cutoff remains1/255). The final closeout check failed because it still required the current worktree to equal the old launch source; freeze release had not been explicitly coordinated with archival. The first log/source and `closeout_provenance_issue.json` are retained. This result is governed by immutable `source.tar` and the earlier successful terminal audit, not the subsequently edited worktree. In a later update root reported that the four-fit-frame gradient probe completed (gradients restored, all four initial patch losses worsened), and the separate `surface_opacity_profile` cut/uncut100-step resource stages launched from the same intersection25k source. Root verified their actual processes/GPU activity; this terminal auditor did not perform that live check. The new canonical/core/profile are separate later work and provide no new quality claim here.

## Purpose and launch record (historical)

The paired RGB-residual study and the cross-foundation audit are complete and
negative for reliable tiny-highlight recovery. This study changes the rendering
foundation: evaluate nonlinear reflectance at individual ray–surfel hits before
alpha composition, jointly optimizing geometry and local material from scratch.
It does not add another residual to the old full-562-frame model.

Implementation and feasibility are complete. No real-scene quality result exists
yet. Canonical `configs/validation.json` now fixes `intersection_reflectance_pilot`,
100000 updates per mode; launched 2026-09-23 08:50:31 UTC after final protocol,
resolved CLI/source checks and a fresh GPU0/1 idle check. Scheduler239375;
training239381(intersection/GPU0),239382(aggregate/GPU1) were observed live with
actual step1 logs and GPU activity. Current progress is `status.json`/history,
not an assumption that these startup PIDs remain alive. No terminal quality
result exists yet; source/config snapshots and `launch_record.json` are in the run.
The 100-step resource profile is complete and preserved under its own run name.
The preceding foundation configuration remains exactly preserved in
`runs/foundation_comparison_audit/analysis/manifest.json`; do not rerun it.

## Matched rendering modes

- `intersection`: sum `T_i alpha_i L(x_i, n_i, material_i)` over true ordered hits.
- `aggregate`: use exactly the same hits, weights and alpha; average hit position,
  view-facing geometric normal and **decoded physical** material parameters by
  normalized weights, then evaluate `alpha_total L(mean attributes)`.

Both use the existing analytic two-lobe GGX response, per-surfel diffuse RGB,
F0 RGB, two physical roughness values and one mixture weight. Neither uses PORT
exchange, a neural material decoder, free RGB correction, nor a pretrained
geometry/material checkpoint. The directional-albedo fit is approximate; learned
geometry/material are not thereby certified physically correct. Disks are two-sided.

The reference backend uses finite disk support `rho <= 9`, alpha threshold
`1/255`, a 0.01 camera near plane, conservative perspective bounds, actual hit-depth
ordering and no center-order early termination. Native gsplat tile indexing is
only candidate acceleration. Native screen-space minimum filtering is absent.
These are shared differences from previous PORT/GS3 rendering, not isolated
evidence for the shading-order hypothesis. The matched two modes isolate that
order within this new backend.

## Feasibility gates, before a quality budget

1. Exhaustive independent plane intersections, front/crossing/behind support,
   center-order reversals, deterministic ties, chunking, and geometry gradients.
   Completed first attempt: five backend tests passed, saved in
   `runs/intersection_reflectance_smoke/backend/` and `backend_first.log`.
   Synthetic peak memory 151.35 MiB and time 1.824 s are not real-scene costs.
2. Moving narrow analytic highlight within one large surfel; overlapping surfaces
   with distinct normals/materials; all geometry/material gradients; state reload;
   and empty coverage. Completed: a single surfel covering the whole native512
   view produces a 3x2-pixel half-max spot moving 29 pixels with the light;
   single-layer modes agree exactly. Two-layer modes match their respective
   independent references (maximum absolute RGB error about 4.8e-6) and differ
   by up to .095608. All geometry/material groups and all six feature columns
   receive finite nonzero RGB-only gradients. Six material-only CPU tests passed.
   The initial exact-replay assertion failed at one RGB value by 2.33e-10.
   A bounded probe localized repeated-render drift to CUDA prefix transmittance,
   with identical state, hits, alpha inputs, physical materials and radiance.
   Production arithmetic was retained; only the two remaining checks were run
   with explicit replay rtol=1e-6/atol=1e-7 and passed (RGB drift <=1.86e-9).
   First failure, probe and both source snapshots remain. `shading_summary.json`
   combines the first three checks and final two; the original partial report's
   `running` field is its pre-exception snapshot, not an active process.
3. Existing `train.py`/`evaluate.py` integration, fresh real-scene 3+3-stage
   training/reload checks, followed by a fixed 100-step resource profile for both
   modes. Use only checked-idle GPUs, at most two. No quality selection from this
   smoke test; preserve failures and source snapshots. Registry/patch integration
   CPU checks passed first attempt (4 tests, .750 s): SSIM halo losses/gradients
   match full-image windows including boundaries, duplicate draws retain weight,
   patch RNG is independent, and generic registry serialization/linearity pass.

The first real `intersection` child completed three finite-gradient optimization
steps (1.65 s logged training, peak 10.78544 GiB), then its CPU sampling audit
failed. Expected dedicated-seed frame draws were [478,219,431], while actual draws
were [478,431,502]: a first-render library path consumed the global Python RNG.
The first source/checkpoints/logs remain under `real/`; this is an integration
control failure, not evidence about image quality. `train.py` now uses a dedicated
`random.Random(seed)` for the new method's frame stream as well as its independent
patch generator. The coherent two-mode 3+3 retry uses `real_dedicated_rng/`; other
methods' historical RNG behavior is unchanged. Optimizers/RNG/counters reset at
each weight-initialized stage, so 3+3 is not an uninterrupted six-step run.

The retry completed all eight checks: byte-identical fresh initial states, 506/56
split, exact independent frame/patch streams and stage counters, fixed 100000
points/light buffer, changes to all six parameter groups, finite gradients/states,
and CLI evaluation plus sparse/full/reload rendering within declared tolerances.
The shared-light-scale branch then passed all six fresh three-step checks on its
first attempt (`light_scale/`, 27.39 s helper wall time), including the parameterless
transport and exact frozen checkpoint scalar on reload/evaluation. Neither smoke
used validation updates or measured reconstruction quality.

## Completed fixed resource profile

`runs/intersection_reflectance_profile/` ran on freshly checked-idle GPUs 0/1,
started 2026-09-23 08:38:37 UTC and completed both training and one-fit-frame CLI
evaluations at 08:39:19 UTC. Reused `launch_validation.sh`/canonical configuration.
Each arm used fresh100k points, native512, 100 updates, four64² patches, seed0,
fixed light scale, no priors/camera fitting/shadows/refinement. All failures above
are separate preceding integration checks; the profile itself completed first attempt.

| Mode | Logged training time | Mean update after first | Peak allocated GPU memory | Step100 candidate / retained fragments |
| --- | ---: | ---: | ---: | ---: |
| intersection | 30.23 s | .295051 s | 13.21697 GiB | 25.42M / 6.70M |
| aggregate | 25.00 s | .241515 s | 9.87526 GiB | 25.02M / 6.34M |

CPU audit verified eight byte-identical initial tensors, only mode/output config
differences, both exact100-step RNG streams, 400 patch draws and 93 sampled fit
frames, finite final states, all groups changed, and fixed center/radius/light.
Launch-source provenance is preserved; postrun test/manifest edits are identified
separately. `startup_audit.json`, `resource_audit.json`, `verification.json` retain
the evidence. The initial live GPU observation is explicitly root-reported;
the auditor did not independently see live startup. No quality conclusion follows
from the single fit-frame evaluation. Naive100k cost is 8.20/6.71 hours, excluding
setup/evaluation and future support changes; memory is feasible on the allocated
49GB cards. This supports retaining the proposed100k budget, not cutting it based
on intermediate quality.

The first real-scene feasibility and matched composition study are explicitly
unoccluded: production dispatch rejects shadow=True. The visibility callback
accepts actual shaded positions, but a shadow method has not been implemented.
Do not substitute primitive-center visibility and describe it as hit-point tracing.
This study tests local reflectance/composition, not the complete relighting system;
failure cannot by itself reject a surface foundation with proper cast shadows.
Resolve resource costs before fixing a long run.

## Fixed quality study (frozen launch protocol; completed)

Use Pixiu native 512, seed 0, fresh random 100000 surfels from **fit-only** camera
bounds. Existing `split_train_lights` gives 506 fit and 56 validation frames.
Audit the complete new initialization/optimization lineage. No full-562 checkpoint,
camera correction, normal/depth prior bank or SDF teacher may enter this lineage.
The 56 frames have been observed during earlier development and are not a blind
test, although they are excluded from this new model's training. Camera and lamp
both vary, so they do not establish pure fixed-view light generalization.

Before any new predictions, the GT-only validation population was recorded in
`runs/intersection_reflectance_smoke/validation_population.json`: 17920 neutral
peak pixels, 2039 tiny (1–4px) components containing 3770 pixels. Fixed validation
ordinals 0/14/28/42 map to original training metadata indices 32/148/329/409.
The first two tiny component labels in each frame give eight predetermined crops;
selection uses only quantized canonical GT, never a model prediction.

Proposed optimization uses fixed primitive count, trainable positions/quaternions/
scales/opacities/materials, fixed cameras, four uniformly placed native 64x64 core
patches per step and a 5-pixel SSIM halo. Keep real neighboring pixels inside the
image and zero padding only outside its boundary; never apply a full-frame SSIM
loss to a sparsely filled image. Overlapping patch draws retain their loss weight.
Shared objective: 0.8 L1 + 0.2 (1-SSIM), plus 0.05 alpha-mask L1. No peak quota,
paired loss or adaptive selection. Native densification statistics are unavailable
for this renderer; do not fabricate them or silently densify with unrelated gradients.
Uniform patch origins are not uniform pixel inclusion near image borders. This
is the explicitly shared patch objective, not an unbiased full-image loss.

The resource probe kept the initial light normalization fixed. Both quality arms
fit one positive **scene-wide** irradiance
normalization using fit506 only, and freeze it for validation. The current
fit-center median normalization is arbitrary, not radiometric calibration:
unit irradiance with unit Lambertian diffuse gives at most `1/pi` linear RGB
(about 0.594 at gamma 2.2). A 0.9 diffuse observation requires about 2.492 times
that irradiance, even before the directional-albedo attenuation. Specular radiance
can exceed this diffuse bound, so this is not a total-image expressivity bound.
It explains why fixed normalization could force diffuse brightness into incorrect
specular/material/geometry compensation. This is shared scalar calibration, not
per-frame or RGB exposure, and does not identify absolute reflectance. Historical
frozen-neural light-scale experiments were negative; they do not calibrate this
fresh bounded-material foundation. No new scalar-ablation study is proposed.

The fixed budget is 100000 steps per arm, with 25000/50000 checkpoints for
fit diagnostics and terminal full-506-fit plus full-56-validation evaluation.
There is no intermediate image-quality budget adjustment or checkpoint selection.
Gaussian learning rates reuse `Gaussians.optimizers`: position starts at
`1.6e-4*fit_radius` and exponentially decays by .01 over100k; scales .005,
quaternions .001, opacity .05, features/base .0025 remain fixed. Shared log-light
scale uses Adam .001. These are matched training choices, not a claim of optimal
optimization for arbitrarily narrow peaks. One seed tests this implementation;
it does not measure variance across seeds or establish a representation upper bound.
Both arms must have equal initial tensors and sample draws, geometry support,
BRDF, loss, optimizer schedule and observation transform.

The existing scheduler now supports explicit `eval_validation` and `eval_save_all`.
It evaluates fit16 at25k/50k/100k only after training has completed, then complete
fit506 and validation56 using the terminal model; all pairs are saved. The new
manifest checks passed first attempt (2 tests): full validation/fit do not inherit
the fit16 limit, and no official test is added. `fit/metrics.json` is the limited
fit diagnostic; primary conclusions must read `validation/metrics.json`, with
`fit_full/metrics.json` reported separately.

On **all56 validation frames**, intersection relative to aggregate must satisfy:

- Tiny 1–4px component contrast/GT distance to one decreases by at least .03;
  recall increases at least .03; pooled RGB MAE decreases.
- Whole neutral-peak precision decreases by at most .01; pooled neutral
  overbrightness in the 11px rings of the four fixed validation frames increases
  by at most5%, using the existing ring-domain definition over the **complete
  frames and all GT-neutral peaks**, not just the eight displayed64px crops.
- PSNR decreases by at most .1dB, LPIPS increases by at most .002, and SSIM
  decreases by at most .002.
- The eight fixed GT-only crops support accurate narrow-peak position/shape
  without compensating false bright/dark patches. Every numeric and manual gate
  is required; retain failures rather than lowering thresholds.

Fit506 evidence is reported separately. An old full562-trained checkpoint cannot
be a clean validation control for these56 frames; aggregate is the matched control.
GT connected components are a bright-neutral proxy, not proof of physical specular
identity. No automatic official-71 test expansion or adoption follows a single
relative gain. A successful new split study would require a matched full-train
refit and original-GS quality gates; official test images are development-observed.

## Related evidence

- [Completed foundation comparison](foundation_comparison.md)
- [Completed paired-loss study](gs_residual_paired_loss.md)
- [Backend and material contract](../architecture/modules/surface_reflectance.md)

Environment remains the shared `ssd-gs` environment, PyTorch 2.4.1, CUDA 12.1 and
installed gsplat 1.5.3. Existing source/data/outputs in PORT-GS, SSD-GS and GS3 remain.
