# GGGS initialization + PORT-DNA, joint geometry/material optimization

Completed on2026-09-27, explicitly AFTER completing the default
`gggs_default_joint` experiment. Start each scene from the same ORIGINAL
GGGS+StableNormal/DA3 final geometry, not the default run's optimized result.

## Geometry-preserving material adaptation

The existing `distribution_material` remains the DNA material implementation:
GGX direct response plus32-component spatial/vMF transport, feature24, width64,
positive receiver energy and image-marginal KL. It is8DNA-inspired, not a path-
supervised8DNA reproduction. It is unrelated to the new `local_transport` shader.

The original DNA used2DGS. With explicit `--init-geometry-format gggs`, preserve
all three Gaussian scales, positions, rotations and compensated opacity at
initialization, use3DGS/gsplat deferred receivers, and provide face-forward
shortest-covariance-axis normals interpolated into pixels. These are geometric
proxies; they are not GGGS's continuous-depth surface normals. Existing bounded
material normal offsets still apply. Rotation receives gradients through both
rasterization and this normal proxy. Axis selection/facing signs are discrete.

The existing quadrature measure supports3D scales via the sum of three projected
pairwise areas times opacity, normalized and detached. Spatial probabilities and
vMF solid-angle densities remain normalized. This is not an exact8D boundary
surface model or a calibrated physical BSSRDF.

Legacy2DGS training/checkpoints keep their existing behavior. The new format is
recorded in checkpoint config and controls saved-model geometry restoration.
No new method registry entry or decoder architecture is introduced.

## Training protocol

Two fresh scene models, Cat/Pixiu each30000 joint steps,512px, seed0, complete
train522/562; full test66/71 and full fit. Same initial GGGS weights, world units,
400000-point cap and refinement through25000 as the current default experiment.
Use default3DGS geometry optimizer/schedule. All four geometry arrays and the
material are trainable. Original source and previous final models stay intact.

Keep DNA's GGX/mixture/image loss: .8L1+.2SSIM+.05mask+1e-5feature+.03imageKL,
port/shadow start1000 and its positive scene-wide light-scale optimization.
Use full512px images in the3D pipeline. No surface-patch sampling, SDF, camera
fitting, or teacher losses during relighting. The old2DGS surface distortion and
surfel/depth-normal consistency losses are not defined for this import and are
explicitly disabled; do not silently call them equivalent3D losses.

Comparisons are observational pipelines, not equal-budget isolated material
ablations: legacy DNA had40000 disks, different geometry training and sample
budgets; the priorlocal model froze geometry and used sparse patches.
Even against the new default, material activation/photometric scale fitting differ.
No test-image training or tuning. Fixed comparison views follow the first run.

## Integration checks

Five focused tests passed: PDF normalization/subdivision, light linearity,
image KL,3D normals and nonzero finite geometry/material gradients, model
save/load with all three scales, and registry config roundtrip. Legacy default
renderer output is checked against the ongoing run's archived renderer.
Logs: `logs/gggs_dna_adapter_tests.log`; regression record:
[default renderer check](gggs_dna_default_regression.json).

Both600-step real-scene joint checks and independent2-fit-view reload evaluations
passed. All saved weights were finite; populations128596→172881 /57741→69388.
See [preflight record](gggs_dna_joint_preflight.json). These short scores are not
final quality evidence. Formal training launched only AFTER the default scheduler
reported completed for both scenes, including all test/fit evaluation. GPU workers
and step100 logs confirmed. Both full30000-step runs and complete test/fit evaluations are now finished.

## Pixiu result

Pixiu30000 steps,249373 final Gaussians, alltest71 andfit562 completed.
Test: PSNR21.18482, SSIM.846024, LPIPS.158111. Compared with new default joint:
+.64124dB butLPIPS+.002118 (worse). Compared with legacyDNA: PSNR-.31813dB,
LPIPS-.008911 (better); not an overall victory. Fit:25.69037 /.892313 /.132815.
Tiny1–4px test peak recall is only1.31%, with contrast ratio-.00525. The original
sharp highlights remain largely missing. Fixed-camera relighting responds to
new lights but has unpairedGT exceptlight0; it is not a physical-accuracy proof.

Geometry review additionally replays final arrays through GGGS continuous depth,
in source normalized coordinates and without applying the initial3D filter twice.
The initial replay passes physical-array and depth/alpha precision checks.
Fixed3-view native source→DNA: meanIoU.90694→.89199, boundaryF1.37958→.30132,
roughnessp95 mean13.31°→100.70°. The final gray model remains very noisy.
These are image-space proxies, not a3D ground-truth metric. Replay uses the learned
covariance as stored, without recomputing a new Mip filter; tiny learned Gaussians
can add aliasing. Both original expected-depth and native-depth diagnostics are
kept to avoid attributing all roughness to one renderer.

[Pixiu geometry](../../runs/gggs_dna_joint/geometry_review/Pixiu/geometry.png) /
[Pixiu relighting](../../runs/gggs_dna_joint/relight_preview/Pixiu/relighting.png).
The completed default study remains a separate comparison condition.

## Final two-scene comparison

Both materials start independently from the SAME original GGGS normal/depth
checkpoints. Both allow position/scale/rotation/opacity updates and population
refinement. Each scene completes30000 additional joint steps at512px, then full
test66/71 and fullfit522/562. These are completed runs, not early-stop results.

| Scene | Relighting on GGGS initialization | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---:|---:|---:|
| Cat | Default directional ports |21.81926|.770223|.228127|
| Cat | DNA3D adaptation |22.54456|.774314|.240020|
| Pixiu | Default directional ports |20.54358|.845065|.155993|
| Pixiu | DNA3D adaptation |21.18482|.846024|.158111|

DNA gains+.72530/+.64124dB and SSIM against the new default, but LPIPS worsens
by.011893/.002118. Against legacyDNA, Cat gains+.40288dB and improvesLPIPS by.027694;
Pixiu loses.31813dB while improvingLPIPS by.008911. Compared with fixedlocal,
CatPSNR is nearly unchanged (+.02779dB) and Pixiu is lower(-.71966dB); LPIPS is
better for both. No method wins all quality criteria and geometry remains poor.
Single-seed and unequal pipeline budgets do not establish a causal architecture
improvement. Default reference and old final checkpoints remain unchanged.

Cat fullfit:23.05869 /.806233 /.233954. Cat final399243 Gaussians, Pixiu249373;
material+geometry remain finite. Cat/Pixiu training51.21/33.19min, peak4.83/4.06GiB.
Their fitted radiometric scales are.346717/.043537; these are scene normalizations,
not measured physical albedo or camera exposure.

Cat's same-GGGS geometry source→DNA: IoU.96417→.95468, boundaryF1.42526→.33893,
roughnessp95 mean15.74°→50.34°. DNA retains somewhat better geometry proxies than
newdefault but fails to preserve the initial geometry in either scene. Grayscale
surfaces remain noisy. There is noGT mesh/depth; do not equate these proxies with
true3D reconstruction error. Native replay does not rebuild a filter after training.

The most useful next hypothesis is to retain GGGS geometric constraints and use
restricted geometry refinement during relighting. These experiments do not verify
that hypothesis, and no additional training was launched under it.

## Final artifacts and checks

- [Cat full image comparison](../../runs/gggs_dna_joint/comparison/Cat/full_images.png) /
  [Cat detail crops](../../runs/gggs_dna_joint/comparison/Cat/crops.png).
- [Pixiu full image comparison](../../runs/gggs_dna_joint/comparison/Pixiu/full_images.png) /
  [Pixiu detail crops](../../runs/gggs_dna_joint/comparison/Pixiu/crops.png).
- [Cat geometry](../../runs/gggs_dna_joint/geometry_review/Cat/geometry.png) /
  [Pixiu geometry](../../runs/gggs_dna_joint/geometry_review/Pixiu/geometry.png).
- [Cat changing-light preview](../../runs/gggs_dna_joint/relight_preview/Cat/relighting.png) /
  [Pixiu changing-light preview](../../runs/gggs_dna_joint/relight_preview/Pixiu/relighting.png).
- [Exact results, reference deltas and audit](gggs_dna_joint_results.json) /
  [Default experiment](gggs_default_joint.md).

Final model paths: `runs/gggs_dna_joint/Real_NRHints/{Cat,Pixiu}/last.pt` and
`runs/gggs_default_joint/Real_NRHints/{Cat,Pixiu}/last.pt`.
Each retains only one final checkpoint; all temporary short-training models were removed.
Source/config/commands are in each run's source archive, manifest and validation config.
The diagnostic-source overlay records the added continuous-depth geometry cross-check;
it did not alter training or RGB evaluation. All jobs have completed.

All three methods' saved targets in the DNA comparison match canonicalGT exactly
on every66/71 frame, with verified metadata mapping. Full and cropped images were
manually reviewed; neither new joint variant is promoted. CPU SSIM recomputation
can differ slightly from originalGPU evaluation; the table uses original metrics.
Final checks confirm finite weights/losses, complete official-train indices,3D
geometry retained and optimized, exact zero/double-light linear response and exact
byte replay of camera0/light0. Other preview camera/light combinations lack pairedGT.
Geometry roundtrip precision bounds and renderer limitations are recorded in
`diagnostic_protocol.json`; no exact native geometric bitwise replay is claimed.
