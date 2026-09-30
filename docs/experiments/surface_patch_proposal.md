# Inverse-probability tiny-peak patch proposal

## Hypothesis and bounded protocol

The completed exposure diagnostic found that 20.55% of training tiny-peak
pixels never enter a direct loss core in 10000 uniform-origin updates.
This intervention increases their sampling probability without changing the
expected original uniform-origin loss. It is not the earlier uncorrected
peak quota or an added highlight loss. Less sampling variance is a hypothesis;
importance weighting can also increase variance elsewhere.

For M valid core origins, let c(o) be the number of canonical uint8 GT tiny
peak pixels inside the core. Tiny components have 1–4 pixels under the existing
8-connected detector. Use q(o) = .5/M + .5 c(o)/sum(c).
If no tiny pixels exist, q is uniform. Sample four origins independently with
replacement and weight EACH complete patch loss by 1/(M q(o)); divide by four,
not the sum of sampled weights. Weight L1, 1-SSIM and mask loss together.
Thus E_q[w loss] equals the previous uniform-origin expectation, including its
existing edge inclusion bias. Target-defined q is fixed with respect to learned
parameters; all origins retain positive probability and weights are at most2.
Only fit RGB/alpha contribute. No validation/test proposals, extra loss,
geometry/shader changes, or extra ray budget.

After current initialization experiment closeout, implement and check exhaustive
small-image loss/gradient expectation, support/probability, no-peak behavior,
independent RNG, and a minimal real saved-checkpoint training/evaluation smoke.
Then compare uniform versus mixture-.5, each exactly5000 new updates from the
same mask-surface10000 checkpoint, resetting Adam/frame/patch RNG equally.
Same100000 points/native512/four64px cores/5px halos/intersection/cutoff0/loss
and lr horizon100000. This is a matched weight stage, not exact optimizer resume.
Full506fit and56development-validation follow; same fixed crops and eight
highlight/global-quality gates, original evaluate metrics primary. No midrun
selection/budget extension. Adopt the proposal only if all numerical and manual
highlight gates pass; otherwise retain uniform sampling for the final method.

## User-requested final test and pause

After this one improvement, freeze the selected configuration. Fresh30000
updates from full-train-only surface seeds for Cat522 and Pixiu562, seed0,
lr horizon100000, same fixed geometry budget/render/loss; no source checkpoint.
Evaluate every official test frame (Cat66, Pixiu71) at native512 with original
test calibration and no exposure/pose fitting. Initialization uses only each
scene's full training frames. Test scenes have been observed during prior
research, so this is a development-stage joint view/light test, not a blind
or pure-light generalization claim. Do not tune after final results.

Preserve failed results. Then inventory registered methods and results, remove
verified redundant artifacts with a deletion manifest while retaining final
models/metrics/source/config and essential failed evidence, and pause as the
user requested. Negative results do not justify an unbounded new method search
before that pause.


## Implementation and feasibility

Implemented in surface_sampling.py and train.py, with --surface-patch-proposal
uniform/tiny. Fit-only tiny coordinates are stored in memory; each frame's
origin counts use rectangle differences/cumulative sums, without a large PDF
cache. CPU torch multinomial uses the dedicated seed4 generator. Weighted
L1, 1-SSIM and mask divide by the fixed patch count; the uniform branch preserves
its previous arithmetic. Sample weights are logged, and surface_proposal.json
records the fit/excluded split and tiny counts. Four CPU checks passed first
execution (0.359s), including exhaustive expected loss/gradient equivalence,
border coverage, empty-target uniformity, sampling frequencies and local RNG.

The100-update profile completed and CPU source/initial-state/RNG/finite-update/
fit-only proposal audits passed first execution. Uniform/tiny time11.47/20.70s,
peak9.618/12.664GiB, direct tiny-core draws651/2539; never-seen tiny pixels
36048/34507 out of36659. Each checkpoint reloaded via existing CLI evaluation.
Coverage is not quality. Profile weights are not used for the fixed5000 pilot.
Evidence: runs/surface_patch_proposal_profile/ and surface_patch_proposal_smoke/.


Stage interpretation limitation: resetting the stage RNG to seed0/patchseed4
means the uniform arm repeats the first5000 patch draws of its10000-step source.
Both arms have the same frame sequence/reset, but the tiny proposal changes
the actual patch draws. This is a controlled proposal-stage comparison, not an
independent-seed estimate of gradient variance; increased novel exposure and
proposal concentration are not isolated from one another. Do not silently
change the predeclared running seed/budget. The eventual fresh full-train test
uses no source weights and does not have this repeated-stage sampling history.

Full-train initializer preparation also completed on CPU, with no CUDA:
Cat522 126.404s, Pixiu562 122.025s, each100000 seeds; no validation indices.
Exact seed metadata equals every official train frame, and resolved train/test
image paths are disjoint (Cat522/66, Pixiu562/71). The metadata audit read no test
pixels; final model lineage remains to be checked after training.
Evidence: runs/surface_reflectance_final_seeds/lineage_audit.json and per-scene
surfel_init_report.json/source archive. These tests have historical development
exposure and must not be described as untouched blind sets.


Final comparison is preregistered in
runs/surface_reflectance_final_seeds/final_comparison_plan.json: reuse the
already-exported full official-test images of port_default and port_neural,
add the final surface method without re-rendering old models. Preserve the
previous GT-only fixed test crop frames (Cat0/16/33/49, Pixiu0/17/35/53).
The primary baseline is port_default; neural is additional context. These
are same30000-update practical comparisons, not equal pixel work, priors,
geometry/transport or wall time. Original evaluator reports remain primary;
common PNG analysis and manual review check details. No test-based selection.
The existing integration entry now includes --surface-final-run for terminal
full-train/source/seed/RNG/complete137-frame test verification. It is pending
actual final outputs, not a completed check.


## Terminal controlled results

Both fixed5000 stages and all eight evaluation phases completed. The first
terminal CPU audit passed: exact shared source weights, same full506/56 lineage,
only proposal/output config differences, finite parameter updates, all logged
origins/weights and complete RNG/counters reproduced exactly, and fit-only tiny
counts agree. Source means/raw component pools plus fit16/fullfit metric and PNG
replay passed112 checks. No profile continuation or midrun modification.

| Scope | Uniform PSNR / SSIM / LPIPS | Tiny proposal PSNR / SSIM / LPIPS |
| --- | --- | --- |
| Full506fit | 21.932463 / .857884 / .172535 | 21.996165 / .857838 / .173550 |
| Full56validation | 21.389876 / .847856 / .179838 | 21.370032 / .847426 / .180911 |

Validation tiny MAE .39030046→.39096618 (worse), contrast/GT
.00599695→.00867992, recall0→10/3770 (.2653%, below the required+3percentage
points). Training tiny recall39→26/36659 despite tiny MAE.359678→.356500.
This does not establish reliable tiny-highlight recovery. Partial local
contrast gains must not be described as an overall quality improvement.

Actual stage tiny-core draws29962→120258 (4.014x); never-in-core tiny pixels
16492→3476 of36659 (44.99%→9.48%). These counts concern this stage alone,
not union with the source10000. They establish sampling exposure, not effective
gradients. Training564.65→978.73s (1.733x), peak12.205→14.246GiB. Same nominal
patch count does not imply same fragment cost. Overall validation does not
improve, so the predeclared selection rule retains uniform sampling for the
final fresh full-train method. Manual and strict cross-backend checks follow
in the terminal records; do not call them completed until their files exist.


Terminal closeout completed:5/8 numerical gates pass; all three tiny-target
improvement gates fail (contrast, recall, MAE). Root viewed all8 fixed crops and
4 fullframes, manual false: white GT micropeaks remain absent, surfaces/edges
blurred and streaked, false broad bright spots remain (f148base, f409body).
Saved canonical GT and fixed crops match exactly;114 component comparisons,
independent float64 NumPy/CV2 rings, and source metrics/replay audit pass.
Strict source/CPU scalar comparison has128 violations (retained, no tolerance
change); primary source and secondary CPU gate decisions agree. No promotion,
no additional proposal budget. Select uniform for final full-train testing.
Complete records: runs/surface_patch_proposal_pilot/{audit,analysis,
source_metrics_audit.json,analysis_verification.json,validation_gate.json,
manual_review.json,completion.json}. No GPU/CPU work remains for this pilot.
