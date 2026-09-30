# GGGS initialization + default directional relighting, joint optimization

2026-09-27. User requests the default relighting method on the selected geometry,
then explicitly permits continued geometry optimization. This supersedes the
initial frozen-geometry proposal before formal training starts.

## Fixed protocol

Source: completed30k GGGS + StableNormal/DA3 checkpoints at
`runs/gggs_normal_depth_geometry/Real_NRHints/{Cat,Pixiu}/last.pt`.
Import the filtered world covariance and compensated opacity, preserving source
point positions/orientations. No source SH or previous material weights transfer.

Use existing `directional_port_v1`, feature32, width128, rank512, four directional
basis terms. Full-image L1(.8)+SSIM(.2), mask.05, feature penalty1e-5; existing
Adam rates/epsilon and decay. Default port/shadow activation5000, deep shadows,
geometric refinement through25000, cap400000 points. All geometry parameters
and material parameters are trainable. Original source weights remain untouched.
No camera/light calibration fitting, no SDF, no teacher losses during relighting.
The imported GGGS3D filter is baked into scales/opacity; it is not recomputed as
an author GGGS filter during subsequent default-geometry refinement.

Cat/Pixiu each30000 joint steps,512px, seed0, GPU0/1. Source geometry fit470/506;
material/joint stage uses official train522/562, including old internal validation.
Evaluate complete official test66/71 and complete fit522/562; no test fitting or
validation-based selection. Existing tested reference images are historically
observed, not a fresh blind benchmark.

## Important rendering distinction

This uses the default gsplat EWA projection, center-Z sorting and expected
center depth, followed by interpolated attributes and deferred directional
transport. It is neither the GGGS continuous-depth renderer nor `local_transport`
per-contribution shading nor `distribution_material` (DNA). Consequently, the
comparison with the prior local pipeline changes rendering, geometric freedom,
material representation, point count and pixel budget together. It is a practical
pipeline comparison, not an isolated material ablation.

The original default's per-step geometry projection also applies to the joint
run. For optional frozen GGGS import that projection is explicitly disabled;
a100-step preflight exposed that behavior before the user changed the scope to
joint optimization. See `gggs_default_preflight_rejected.json`. No formal failed
training was used as a result.

## Checks and outputs

The focused handoff regression checks world units, filtered covariance,
opacity compensation and rotations. Both600-step preflights activate ports and
shadows fromstep1 and cross the first population refinement. Those are smoke
checks, not final scores. Final protocol retains the original5000-step warmup.

Use canonical `configs/validation.json` and `bash launch_validation.sh`.
Output: `runs/gggs_default_joint/`; each scene keeps only final `last.pt`.
Source/config/manifest/logs are archived by the existing launcher.
Evaluate with normal `evaluate.py` (no new renderer flag).

`diagnose_image_errors.py CHECKPOINT --gggs-default-review --output NEW_DIR`
compares three fixed official-train views: index0, floor(N/2), N-1. Columns show
observed RGB, source continuous-depth clay, imported geometry in the default
renderer, and final default geometry. This separates conversion artifacts from
optimization changes. Boundary/roughness/IoU are proxies; there is no geometric GT.

Both600-step checks completed and reloaded successfully. Cat/Pixiu populations changed128596→175979 /57741→68739; all saved parameters finite. See [preflight record](gggs_default_joint_preflight.json). Formal GPU0/1 workers and step100 logs were confirmed; both full30k joint runs and complete test/fit evaluation are finished.

## Pixiu result

Pixiu30000 joint steps completed,141368 final Gaussians (source57741;
old default final138131). Test71frames: PSNR20.54358, SSIM.845065, LPIPS.155993.
Old default: PSNR20.57825, SSIM.845556, LPIPS.156690. Differences are small,
not a convincing overall quality improvement. Prior GGGS-fixed/local:21.90448 /
.847826 /.174197; joint default losesPSNR but improvesLPIPS against that pipeline.
Full fit562frames:27.49756 /.908982 /.118330. The large fit/test gap is evidence
of stronger training-image fitting without a corresponding held-out improvement;
different camera/light composition prevents attributing all of the gap to a
single cause.

Fixed geometry review was visually inspected. Relative to imported geometry in
the SAME default renderer, final expected-depth surfaces become markedly rougher
and outlines deteriorate. Three-view mean silhouetteIoU: import.89976→final.86389;
boundary roughnessp95 mean19.13°→74.50°. These are train-view proxies withoutGT
surface geometry. Comparing source GGGS continuous depth to default expected
depth alone would conflate rendering and optimization; the intermediate import
column is included to expose that distinction.

[Pixiu geometry](../../runs/gggs_default_joint/geometry_review/Pixiu/geometry.png) /
[relight preview](../../runs/gggs_default_joint/relight_preview/Pixiu/relighting.png).
Camera0/light0 preview exactly replays the test prediction, byte-for-byte.
Novel light combinations show changing illumination but still miss sharp highlights;
they have no pairedGT and are qualitative only. Final weights/losses are finite,
all training indices and71 test pairs checked. Training time38.53min, peak9.22GiB.

## Complete two-scene outcome

Both full30000-step models and alltest66/71 /fit522/562 evaluations completed.

| Scene | Old default testPSNR | GGGS+default testPSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Cat |21.53618|21.81926|.770223|.228127|
| Pixiu |20.57825|20.54358|.845065|.155993|

Cat improves+.28308dB, SSIM+.003758, LPIPS-.002278 against olddefault.
Pixiu changes-.03466dB, SSIM-.000491, LPIPS-.000697. These are single-seed
observations, not a statistically established overall improvement.
Cat fullfit:24.81346 /.834513 /.208892; Pixiu fullfit above.
Cat final399118 points from128596; source geometry stayed in its original file.
Cat training76.58min, peak13.56GiB. Both keep only terminal material/geometry models.

Cat fixed geometry review, SAME default renderer import→final: meanIoU
.96082→.94453, boundaryF1 .37457→.19541, roughnessp95 mean27.77°→45.21°.
The gray model is visibly noisier. There is no depth/meshGT; these are limited
three-view shape proxies, not a3D accuracy measurement. RGB improvement does
not demonstrate geometry improvement. The default reference is unchanged.

[Cat full RGB comparison](../../runs/gggs_default_joint/comparison/Cat/full_images.png) /
[Cat crops](../../runs/gggs_default_joint/comparison/Cat/crops.png) /
[Cat geometry](../../runs/gggs_default_joint/geometry_review/Cat/geometry.png) /
[Cat relight preview](../../runs/gggs_default_joint/relight_preview/Cat/relighting.png).
[Pixiu full RGB comparison](../../runs/gggs_default_joint/comparison/Pixiu/full_images.png) /
[Pixiu crops](../../runs/gggs_default_joint/comparison/Pixiu/crops.png).
[Full results and audit](gggs_default_joint_results.json).

All3 comparison methods' saved targets match canonical GT exactly on every66/71
image. Full/crop figures were manually inspected; no promotion. The CPU comparison
tool reported exit143 after writing all final artifacts; row counts, both-scene
summary, target equality and primaryPSNR agreement were separately verified.
Original evaluator metrics above remain primary; CPU recomputation has tiny
SSIM/LPIPS floating-point differences. Primary training/test/fit exited normally.

Final audit: all parameters/losses finite, full train indices,3D scales retained,
geometry/topology changed, and exact zero/double-intensity linear render response.
Both original-light previews reproduce testframe0 byte-for-byte. Novel light
combinations are unpaired qualitative previews, not accuracy measurements.
The shared evaluation-source overlay records conditional DNA3D support added
while default training was in progress; default forward equality against archived
training renderer passed bitwise. Running training itself was not changed.

The user-queued DNA experiment was launched only after both default jobs reached
completed, including complete fit evaluation. It uses the original GGGS source,
not these jointly updated models. See [DNA protocol](gggs_dna_joint.md).

## Same-GGGS continuous-depth cross-check

Final geometry was additionally replayed through the SAME GGGS continuous-depth
backend in the original source numerical coordinates, with no duplicate3D filter.
Physical initial covariance/opacity/rotation/position agreement passes. Source
roundtrip alpha mean errors are below4.4e-8, p99 below7.7e-7; rare maximum changes
up to.00349 remain belowone byte. Depthp99 error/radius is below1.4e-6. No exact
bitwise geometric replay claim is made. See `diagnostic_protocol.json` for the
initial overly strict max-alpha check and revised explicit precision bounds.

Three fixed-view nativeGGGS source→final: CatIoU .96417→.94923, boundaryF1
.42526→.23493, depth-normal roughnessp95 mean15.74°→55.76°; PixiuIoU
.90694→.88551, roughness13.31°→108.25°. The qualitative geometric deterioration
is therefore not solely an artifact of switching to expected center depth.
There is still no geometricGT; these are image-space shape proxies.
The updated figure includes the final continuous-depth panel and method/step title.
