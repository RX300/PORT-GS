# Neural material: 1M Gaussian cap ablation

User requested2026-09-28. Completed and reviewed; control is completed
`gggs_neural_material_joint` (400000 cap).

Change only `max-points`400000→1000000. Fresh initialization from exactly the
same original GGGS+StableNormal/DA3 geometry and same frozen shared procedural
neural BRDF decoder. Keep30k steps,512px,seed0, full official train522/562,
port/shadow5000, refine_stop25000, density thresholds and optimizers unchanged.
Continue geometry and material updates throughout; no new geometry supervision.
The higher cap permits additional growth; it does not force either scene to1M.
Use GPUs0/1, ssd-gs environment and canonical launch_validation.sh entrypoint.
Prior training and existing600-step integration tests need not be repeated.

Validate memory using two forward/backward/Adam steps on1M replicated Cat
Gaussians with the existing shader and a training camera. This artificial fixture
is a resource check, not a scene initialization or quality result. Retain only
its numerical record/log. Existing400k executable Python sources are checked
byte-for-byte against their actual archived training snapshot.

Full test66/71 and fullfit522/562, saved uint8 pairs and standard PSNR/SSIM/LPIPS.
Compare directly with400k in same fixed test views Cat[0,21,43,65],
Pixiu[0,23,46,70], GT-derived crops; no test tuning/early selection.
Also fixed-camera relighting and native GGGS depth/normal review at three train
views per scene. Record max/final counts, compute/memory and geometric proxies.
A single seed does not establish significance; renderer nondeterminism may
produce differences even if the point cap is inactive. Existing final models
remain preserved; final-only checkpoints for the new condition.

Resource preflight passed:1M Gaussians, two full512px steps including shadows, nonlocal transport, all geometry gradients and Adam state. Peak allocated26.848GiB, reserved40.619GiB on48GiBGPU. This is one representative view, not a guarantee for every dynamic topology/view.48 Python source files match the400k snapshot; the only training option difference is max-points. No preflight model saved.

Launch confirmed: scheduler and both train children active, checkpoints configured1M/30k, geometry unfrozen; `startup_audit.json` records actual progress/PIDs. Existing benchmark automatically performs full test and fit after each scene completes. Final comparison is complete;1M is not promoted. Outputs: `runs/gggs_neural_material_1m/`.

## Completed results

Both scenes completed30000 steps, full test66/71 and fit522/562. No failed or
partial result substituted. Only max_points/output differ in checkpoint configs;
shared decoder tensors are bitwise identical to the retained prior, all model
weights finite. Both runs use the same executable Python sources. Common
comparison re-scored all137 test images; saved targets byte-exact, PSNR agrees
with primary evaluator within1e-5dB. Figures manually reviewed. No test tuning.

| Scene | Cap | Final Gaussians | Test PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---:|---:|---:|---:|---:|
|Cat|400000|399410|22.622656|0.779711|0.239291|
|Cat|1000000|747578|22.520350|0.776848|0.238933|
|Pixiu|400000|240557|21.571101|0.850532|0.157998|
|Pixiu|1000000|248502|21.535612|0.850536|0.157801|

Cat PSNR−.102306dB, SSIM−.002862, LPIPS−.000359; Pixiu PSNR−.035489dB,
SSIM+.000004, LPIPS−.000197. Tiny LPIPS changes do not establish a meaningful gain.
Cat full-fit PSNR22.5523→22.5177; Pixiu24.6165→24.6274. A single seed does not
establish statistical significance. Pixiu stays below even400k in both runs;
its small changes cannot be taken as evidence of an active-cap benefit and are
consistent with nondeterministic training variation.

Cat final399410→747578 (+87.2%);1M candidate maximum LOGGED count848262, not a
forced1M population. Pixiu240557→248502; candidate max logged255567. Logs sample
training periodically, so max logged is not an exact per-step peak.
Cat time4487.82→7150.81s (+59.3%), peak allocated memory13.270→25.062GiB.
Pixiu2640.94→2658.73s,9.293→9.418GiB. These are measured runs, not controlled
hardware throughput benchmarks.

Fixed native-GGGS three-view geometry proxies,40万→100万:
- Cat silhouetteIoU .959280→.958464, boundaryF1 .400497→.391562,
  boundary depth-normal roughness47.77°→49.40°.
- Pixiu IoU .910434→.910516, boundaryF1 .393799→.397890,
  roughness68.31°→69.02°.
No geometryGT; same-renderer replay and initial roundtrip validation pass.
These proxies do not establish actual3D error. Both remain much rougher than
initial GGGS surfaces. Images still show blurred Cat hair, side-light dark
blotches and blurry paper, plus soft Pixiu boundaries/missing highlights.
Pixiu tiny1–4px peak recall1.296%→1.052%, no recovery of sharp highlights.

Decision: **do not promote1M**. Extra points have not resolved the current
pipeline's quality limitations. Constrained geometry/material optimization and
rendering/representation mismatch are more useful next hypotheses than another
cap increase, but no new method or experiment was launched by this review.
Keep both final conditions and the requested1M canonical config for provenance.

## Outputs

- [Full metrics/audits](gggs_neural_material_1m_results.json)
- [Cat comparison](../../runs/gggs_neural_material_1m/comparison/Cat/full_images.png), [crops](../../runs/gggs_neural_material_1m/comparison/Cat/crops.png)
- [Pixiu comparison](../../runs/gggs_neural_material_1m/comparison/Pixiu/full_images.png), [crops](../../runs/gggs_neural_material_1m/comparison/Pixiu/crops.png)
- [Cat geometry](../../runs/gggs_neural_material_1m/geometry_review/Cat/geometry.png), [Pixiu geometry](../../runs/gggs_neural_material_1m/geometry_review/Pixiu/geometry.png)
- [Cat relighting](../../runs/gggs_neural_material_1m/relight_preview/Cat/relighting.png), [Pixiu relighting](../../runs/gggs_neural_material_1m/relight_preview/Pixiu/relighting.png)
