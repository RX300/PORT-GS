# Surface opacity cutoff: gradient and recovery study

## Motivation and evidence boundary

The completed `intersection_reflectance_pilot` did not recover tiny highlights.
Its intersection and aggregate arms ended with only511 and844 of100000 stored
surfels having opacity>=1/255. Since fragment alpha<=primitive opacity, all
remaining points have no retained fragments under that rule and receive no
direct rendering gradient. This is a demonstrated support/gradient limitation,
not yet a demonstrated cause of poor image quality. Low opacity can also be a
useful rejection of incorrectly placed random initial geometry.

This follow-up isolates the cutoff. It does not add a BRDF, density strategy,
opacity reset, RGB residual, prior, camera correction or extra loss. The current
surface material exposes `--fragment-alpha-min`, default1/255. Zero retains all
positive representable fragment alphas within the same radius-three disk and
near-plane support. The old default is unchanged and old checkpoint configs
resolve to it. Evaluation restores the saved cutoff. The sphere/disk bounds,
actual intersection depth order and no-early-termination contract are unchanged.
Zero-alpha fragments remain excluded to make aggregate normalization well-defined.

An opacity threshold alone does not bound the error of an HDR radiance
contribution; the synthetic probe tests this with a fixed known radiance. It
does not establish that every discarded real-scene point contains a useful glint.

## Ordered experiment, declared before recovery training

1. Run existing six material CPU checks and five synthetic CUDA fragment checks.
2. In the existing `test_method_integration.py` entrypoint, run
   `--surface-opacity-checkpoint <intersection step_025000.pt>`: a synthetic
   one-ray weak-HDR closed-form composition/opacity-gradient check, then the
   same four uniform64px-core patches per native512 training image under both
   cutoffs. Four images are evenly spaced in the saved506-fit list, independent
   of GT or prediction content. The patch generator has seed4. No optimization,
   validation frame, state mutation, or relighting-quality claim is involved.
3. If the diagnostic passes, use canonical `configs/validation.json` and
   `launch_validation.sh` for a100-update resource/integration profile, both
   from the SAME earliest retained25k intersection checkpoint. The only method
   difference is cutoff1/255 versus0. Each ends with one fit-frame CLI evaluation.
   Verify initial state/split equality, saved cutoff roundtrip, finite updates,
   identical frame/patch sampling and resource use. Do not select quality from
   this tiny profile or reuse its trained state for the recovery pilot.
4. If feasible, run a fixed5000-update recovery pilot from the SAME original25k
   checkpoint, not the profile output. Retain both arms and complete full506fit
   plus56validation evaluation at the fixed final step, in independent outputs.
   Both reset optimizers and frame/patch RNG using the existing weight-stage
   `--init-checkpoint` contract; neither is an exact continuation of original
   Adam or a comparison to uninterrupted30k. Position LR uses the same100k
   decay horizon restarted for both arms. Six Gaussian parameter groups and
   one shared scene-wide positive light scale learn in both; fixed cameras,
   no refinement, same512 resolution/four64px cores, seed0 and original loss.

The earliest retained25k snapshot is chosen to test recoverability before the
later most severe support loss, not by selecting the best checkpoint metric.
Full training lineage remains the original506 fit frames; the excluded56 were
historically development-observed and combine camera/light variation. They are
not a blind test or isolated unseen-light protocol. Official71 is not evaluated.

## Interpretation fixed before the pilot

Primary mechanistic evidence is rendering-gradient access and movement of the
initially below1/255 points. Report counts at the SAME reference threshold in
both arms even when the treatment threshold is zero; do not call an increased
retained-fragment count a capacity or quality success by definition. The logged
population counts refer to the parameter/gradient state BEFORE that update.
After training, separately count initially below-threshold points that crossed
the reference threshold, and report raw learned opacity/geometry/material state.

Primary quality metrics remain the source evaluator's complete506-fit and
complete56-validation PSNR/SSIM/LPIPS, GT tiny-component MAE, contrast ratio,
2px recall and global peak precision. Reuse the prior GT-only fixed eight crops
and four full-frame11px peak rings. The previous eight numerical gates and
manual position/shape/no-false-bright-patch gate remain prerequisites for a
positive validation conclusion. Fit improvement alone proves only fitting;
gradient restoration alone proves only optimization access. No default adoption
or new test run follows automatically, even if the mechanism works.

## Provenance and current state

The previous study's source remains its immutable launch `source.tar` and
earlier completed terminal audits. Root began this next implementation after
old training/evaluation and primary/manual gate completion but before the
old final archive's current-worktree recheck. That recheck correctly failed
because current source had changed; its first failure and agent's closeout
explanation are retained. No old weights, renders, metrics or source archives
were changed, and no old training/evaluation was rerun. Current worktree must
not be represented as the unchanged source of that previous experiment.

Initial checks/diagnostic outputs: `runs/surface_opacity_smoke/`.

## Completed feasibility results

Six CPU material tests and the five existing synthetic CUDA backend tests passed
on their first executions. The new weak-HDR probe gives linear RGB2 at opacity
.002 with cutoff0, versus0 at the old cutoff; its independently derived opacity
logit gradient matches5.988. Exact zero alpha yields an empty ray. These are
constructed operator properties, not reconstructed real GT peaks.

The unchanged25k checkpoint has94953 points below1/255. For fixed training
frames0/184/370/561 and the same patches, cut/uncut loss is respectively
.026375/.135622, .168499/.206590, .115520/.148192, .042998/.071189. Thus the
instantaneous switch worsens all four observed patch losses. The old cutoff
gives zero rendering gradients to every below-threshold point. Cutoff0 gives
nonzero opacity gradients to7107/9225/12316/10901 such points, of which
433/2348/4068/2483 have a gradient descent direction increasing opacity.
These counts overlap across frames. All Gaussian states remain byte-identical
to the loaded source; no updates were made. Peak GPU allocation was7.788GiB.

`surface_opacity_profile` completed both100-update stages and one-fit-image CLI
evaluations at2026-09-23 13:51:56UTC. Cut/uncut logged times16.82/23.50s and
peak allocated memory7.939/10.846GiB demonstrate feasible cost on the checked
GPUs0/1. The CPU audit (`audit/report.json`) passed its first execution: both
initial states equal each other AND the original25k source byte-for-byte,
only cutoff/output configs differ, all six parameter groups update finitely,
506/56 lineage and100 frame/patch draws/RNG states are exact, saved cutoffs
are restored by CLI evaluation. CUDA was not initialized by the audit.

At the same reference1/255 threshold, final point counts are4612/8245; the
uncut arm moved3960 initially below-threshold points across that reference,
the cut arm moved0. This proves short-stage reactivation, not improved geometry
or highlight recovery. No profile quality metric was used to choose a budget.

Canonical config launched `surface_opacity_recovery_pilot`: both fixed5000
updates from the original25k source, followed by fit16, complete506fit and
complete56validation (eight total phases). Startup evidence and later status
belong to that separate run. No intermediate quality selection or officialtest.

## Terminal recovery outcome: no tiny-highlight recovery

Both5000-update stages and all eight phases completed at2026-09-23
14:15:08.982666UTC. No training/evaluation was restarted. The source/state/
budget/RNG audit passed first execution (`audit/report.json`): initial Gaussian
tensors and light equal both the paired arm and original25k source byte-for-byte.
Cut/uncut training times801.05/737.80s, peak allocation8.498/11.013GiB. These
times are run measurements, not an invariant renderer-speed ranking.

| Full population | cut | uncut |
| --- | ---: | ---: |
| 506fit PSNR | 17.080990 | 17.352340 |
| 506fit SSIM | .819355 | .821868 |
| 506fit LPIPS | .245428 | .234647 |
| 506fit tiny RGB MAE | .536993 | .541358 |
| 506fit tiny contrast/GT | .003859562 | .003987392 |
| 506fit tiny matched pixels | 0/36659 | 0/36659 |
| 56validation PSNR | 17.483615 | 17.708159 |
| 56validation SSIM | .800828 | .802003 |
| 56validation LPIPS | .256468 | .246412 |
| 56validation tiny RGB MAE | .552859 | .552898 |
| 56validation tiny contrast/GT | .002442075 | .002980685 |
| 56validation tiny matched pixels | 0/3770 | 0/3770 |
| Fixed four full-frame ring overbrightness | .022728375 | .015190698 |

Neither arm predicts a peak under the preregistered GT-interior detector, so
global peak precision is **undefined**, not zero or a passed gate. The eight
primary gates have four passes (PSNR, SSIM, LPIPS, ring), three failures (tiny
contrast gain, recall gain, MAE), and one unevaluable required precision gate.
All required conditions therefore fail. Root viewed all eight fixed GT-only
crops and four full frames: both show dim/blurred/streaked structure, lost narrow
white GT points, inaccurate head/outline/base detail, and a bright streak near
the upper-left border in frame409. The manual gate fails. Reduced ring
overbrightness does not make a dim blurred reconstruction a peak-recovery result.

Cutoff0 provides a modest paired whole-image fitting/validation gain but no
tiny-highlight benefit. This is a25k-initialized5000-step recovery test, not a
comparison to the previous fresh100k endpoint, nor proof that a fresh longer
uncut run could never behave differently. No budget extension, official71,
default change or claim of true geometry improvement follows.

## Mechanism after5000 updates

StoredN stays100000. Final opacity>=1/255 counts3137/3293; initially below
points crossing the same reference0/623, versus0/3960 in the100-step profile.
That early reactivation is not a persistent large increase in reference-eligible
population. Initially eligible points dropping below number1910/2377.

Median opacity.003542/.0000232404; summed primitive opacity below the reference
307.794/5.02274. These unweighted parameter sums are not accumulated image alpha,
visible area, radiance contribution or physical mass. Removing the cutoff allows
optimization to suppress many weak points as well as increase others. Restored
gradient access is real but insufficient as the sole recovery intervention here.
All states remain finite; `training_population.json` retains full statistics.

## Terminal verification and preserved limitations

`source_metrics_audit.json` independently checked458 source row means,
component raw-pool formulas and fit16/full506 metric/PNG replays: zero
discrepancies. Saved validation GT bytes exactly match canonical GT; all
source/common component and integer peak counts agree. Fixed crops equal the
prior GT preregistration. Independent NumPy/CV2 full-frame ring construction
confirms all19039 ring pixels and aggregate values.

Strict source/common CPU scalar checks retain rtol1e-6, atol1e-8:108 LPIPS
violations (max absolute1.4945865e-5), no PSNR/SSIM violations. Source evaluator
scalars stay primary, no tolerance widening; primary/secondary decisions agree.
This is not a blanket all-numerical-checks-passed result. See
`analysis_verification.json`, `validation_gate.json`, `manual_review.json` and
`analysis/`. Manual review is by root; no second independent reviewer is claimed.

A root observer call failed when a terminated observer cell's stored command
was unavailable; the tool rejected missing `cmd` before execution. An explicit
read-only command then rechecked the same live evaluations. `observer_issue.json`
preserves it; no training/evaluation failure or restart. The previous study's
separate final-source-recheck coordination issue remains recorded in that run.

## Next research decision

Do not extend this cutoff-only recovery budget. The independent all506fit-mask
initial-support diagnostic finds only2683 of100000 original random centers
inside at least95% of masks with2px dilation. Many mask-consistent centers also
become transparent, while later survivors become more mask-consistent. Initial
allocation and maintaining useful surface coverage deserve a controlled change;
blanket point revival or more stored tensors alone is not justified. See
[population research](../research/surfel_population_lifecycle.md).

Next priority: a fit-only foreground-conditioned surface initialization and
allocation study, with explicit calibration/visual-hull limits and fresh matched
budgets. No initializer or new training is implemented/launched by this closeout.
The complete research goal remains active and unfulfilled.
