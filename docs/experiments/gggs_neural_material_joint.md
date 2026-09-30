# GGGS geometry + neural material, joint optimization

Requested by user after completing default and DNA joint studies. Completed on2026-09-27; full train/test evaluation and checkpoint audits passed. Not promoted.

## Fixed protocol

Fresh `neural_material`, `material_model=neural`; this is the existing frozen
shared neural BRDF decoder, not the optional analytic GGX replacement and not DNA.
The old prior was retired. Recreate it with canonical `configs/material.json`:
50,000 procedural-only steps, batch4096, seed0, final-only checkpoint at
`runs/neural_material_highlight_prior/last.pt`. No scene RGB or geometry enters
this prior. Both scenes use exactly the same frozen weights, embedded in their
final scene checkpoints. Optimize six material codes, three tangent normal
correction codes, diffuse base and nonlocal transport; keep the decoder frozen.

Initialize independently from original
`runs/gggs_normal_depth_geometry/Real_NRHints/{Cat,Pixiu}/last.pt`, not either
previous relighting result. Preserve compensated opacity and all three filtered
world-scale axes at import. Rasterize with gsplat3D and supply differentiable
shortest-covariance-axis normals, face-forward and interpolated to receivers.
This proxy is not native GGGS continuous-depth surface normal. All geometry
parameters continue optimizing; densification/pruning also remains enabled.
Native GGGS continuous-depth and StableNormal/DA3 losses do not continue in this
stage. Explicitly zero 2D surface/prior losses; no SDF or camera fitting.

Cat/Pixiu each30,000 steps, full512px images, seed0, feature32, width128, rank512,
port/shadow start5000, refinement until25000, cap400000. Positive scene-wide light
scale optimized as in DNA; default joint kept it fixed. No per-frame exposure.
Official train522/562 (including original internal val52/56), test66/71 and full
fit evaluation. All test observations have been seen in prior research; no test
fitting or tuning in this run. Pipeline comparisons are observational, not an
isolated material ablation (DNA architecture/schedule/loss also differ).

## Validation and evidence

Targeted test: 3D render and reload, finite nonzero gradients through geometry
and material, frozen decoder through training-stage activation, old DNA adapter
regression. Real600-step smoke for each scene activates shadow/transport at1 to
exercise both plus density refinement, then independently reloads two fit views.
Smoke results are not final-quality evidence. Preserve logs/numerical audit and
remove only temporary smoke weights after completion.

Canonical config: `configs/validation.json`, name `gggs_neural_material_joint`.
Launcher/source snapshot/manifest remain the existing entrypoints. Use GPUs0/1
only in ssd-gs. Final-only scene checkpoints; complete test/fit metrics, fixed
comparison images, native GGGS geometry replay and fixed-camera relighting.
Fixed test frames Cat[0,21,43,65], Pixiu[0,23,46,70]; same GT-derived highlight
crops and targets as completed default/DNA comparisons. Geometry views use train
Cat[0,261,521], Pixiu[0,281,561], no geometry GT. Final metrics and promotion
status pending; do not infer a quality improvement from successful training.

## Preflight results

Three focused tests passed. Both real600-step smokes completed with finite losses
and parameters. Decoder tensors bitwise equal to procedural prior after scene
optimization. Cat128596→172676, Pixiu57741→69663 confirms density refinement.
Independent two-fit-view reload passed (Cat PSNR19.6498, Pixiu21.8960), not a
quality claim. Full configs/logs/commands: `gggs_neural_material_joint_preflight.json`.
Temporary smoke weights removed after this evidence was preserved.

Prior completed50000 steps in378.95s; unseen procedural validation32768 samples:
relative BRDF L2 .136746, log1p RMSE .143720, transmission MAE .028231.
Direct encoded-material lobe checks show very narrow roughness.001 peaks
underestimated (dielectric ratio.405,metal.568), widths.258° vs teacher.148°.
This measures encoder+decoder for these procedural examples; it is not proof of
a global decoder fitting limit and not a scene geometry measurement. Prior
source/config/logs/metrics/highlight_curves and final weights are retained.

## Final results

All official test66/71 and fit522/562 evaluated; all images exported. Scene steps
30000 each, original GGGS initialization, geometry continuously trainable.
Shared50k procedural prior remains bitwise unchanged in both final checkpoints.

| Method / test | Cat PSNR↑ | SSIM↑ | LPIPS↓ | Pixiu PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---:|---:|---:|---:|---:|---:|
| GGGS + default joint |21.8193|.770223|.228127|20.5436|.845065|.155993|
| GGGS + DNA joint |22.5446|.774314|.240020|21.1848|.846024|.158111|
| GGGS + neural material joint |22.6227|.779711|.239291|21.5711|.850532|.157998|

Cat/Pixiu PSNR gains vs DNA are .0781/.3863dB; LPIPS differences are tiny
(−.000729/−.000113). Both still have worse LPIPS than default joint. One seed,
different pipeline choices; do not claim statistically significant superiority.
Full fit: Cat22.5523/.799162/.238547, Pixiu24.6165/.885621/.140417. Fit metrics are
worse than default and DNA, consistent with restricted fitting capacity or
optimization mismatch; this trial does not isolate their cause.

Final points399410/240557; scene training4487.82/2640.94s; peak allocated GPU
memory13.270/9.293GiB. Prior training378.95s additional shared cost. Decoder26695
parameters frozen, total material transport55926 parameters. Test/fit counts,
finite weights/losses, shared decoder equality, zero/double-light linearity,
independent reload and byte-exact test0/preview replay all passed. Source code,
launch configuration, hardware/process startup, logs and final checkpoints are
retained in the run; no scene intermediate checkpoints or smoke weights remain.

## Geometry and visual interpretation

Fixed three TRAIN views per scene, same native GGGS renderer at source and final;
initial filter-baked replay is checked against source physical tensors and depth/
alpha. No geometry GT. Roughness below is the mean of each view's boundary-depth
normal p95 proxy, not a reconstruction accuracy metric. Tiny Gaussian replay
without a new Mip filter can alias. The middle gsplat-depth panels use a different
depth definition and should not be treated as pure geometry-change measurements.

| Scene | Source→final silhouette IoU | Source→final boundaryF1 | Source→final roughness |
|---|---:|---:|---:|
|Cat|.964173→.959280|.425261→.400497|15.74°→47.77°|
|Pixiu|.906939→.910434|.379584→.393799|13.31°→68.31°|

Final roughness is lower than default joint55.76°/108.25° and DNA50.34°/100.70°,
but both scenes clearly lose the initial smooth surface. Pixiu contour proxies
improve slightly despite much noisier surface normals. Cat remains blurred in
fur/face details; Pixiu has soft boundaries, blurry base pattern and missing
specular dots. Tiny1–4px Pixiu peak recall is1.296% (DNA1.315%), contrast
ratio−.000218: this does not solve small-highlight recovery. Four-light previews
respond to light changes; only camera0/light0 has pairedGT, other combinations
are qualitative.

Decision: retain the trial and reference models, **do not promote**. Merely
switching the material branch while removing GGGS surface constraints has not
preserved geometric quality. A constrained geometry update with appropriate
3D/native-GGGS supervision is a supported next hypothesis, not a validated fix.
The frozen decoder's procedural lobe error and deferred mixing of normals/codes
before nonlinear shading are additional plausible limits; no ablation here
establishes either as the unique cause. No follow-up redesign was run in this trial.

## Artifacts

- [Machine-readable results](gggs_neural_material_joint_results.json)
- [Cat full comparison](../../runs/gggs_neural_material_joint/comparison/Cat/full_images.png), [crops](../../runs/gggs_neural_material_joint/comparison/Cat/crops.png)
- [Pixiu full comparison](../../runs/gggs_neural_material_joint/comparison/Pixiu/full_images.png), [crops](../../runs/gggs_neural_material_joint/comparison/Pixiu/crops.png)
- [Cat geometry](../../runs/gggs_neural_material_joint/geometry_review/Cat/geometry.png), [Pixiu geometry](../../runs/gggs_neural_material_joint/geometry_review/Pixiu/geometry.png)
- [Cat relighting](../../runs/gggs_neural_material_joint/relight_preview/Cat/relighting.png), [Pixiu relighting](../../runs/gggs_neural_material_joint/relight_preview/Pixiu/relighting.png)
- Final models: `runs/gggs_neural_material_joint/Real_NRHints/{Cat,Pixiu}/last.pt`
- Shared prior: `runs/neural_material_highlight_prior/last.pt`

Comparison completed with exit0: all137 test frames, all three saved targets byte-exact against canonicalGT; recomputed PSNR agrees within1e-5dB. SSIM/LPIPS backend roundoff is retained; tables use primary evaluator numbers. Full sheets, GT-fixed crops, geometry and relighting previews manually reviewed; no promotion. Cat side-lit frame43 has dark blotches and over-smoothed hair; Pixiu retains thin boundary streaks in some views.
