# GGGS geometry + local per-contribution neural material

2026-09-27 user explicitly requests continuing to relighting with the current geometry.
This overrides the earlier geometry-pass prerequisite. Current method:
[implementation](../architecture/modules/local_transport.md).

## Protocol fixed before official test

Reuse the completed GGGS+normal/depth geometry checkpoints (Cat128596/Pixiu57741 points).
Freeze geometry, fit fresh material parameters for30000 steps at512px, seed0, GPU0/1.
Two views/step, two16px cores per view with5px halo:1024 supervised core pixels and2704
rendered ray samples per step, three quadrature nodes per Gaussian contribution.
Train all522/562 images; evaluate all66/71 official test and complete522/562 fit images.
No training on test views. Existing DNA/default test scores are observational references:
geometry design, prior use, source-geometry fit coverage and pixel budgets differ.

The per-contribution renderer preserves original GGGS alpha/EWA sorting; it does not
silently replace this geometry with gsplat's different pixel footprints or thin surfels.
The source geometry has known shape errors; the user authorizes measuring the resulting
material/relighting quality anyway. Geometry is frozen during this first material stage.

## Preflight

CUDA synthetic checks cover original-rasterizer colors/alpha, no contribution truncation,
finite material/local-slope gradients, zero-light/linear intensity, quadrature and reload.
The PyTorch MLP implementation completed200 real Cat steps and independent2-view fit
reload evaluation. It measured roughly.11s/step and11.6GiB peak memory; it is a short
implementation check, not a final quality result. Logs local_material_preflight*.log.
Actual source geometry handoff checks are recorded in local_transport_geometry_handoff.json.
Final shader remains PyTorch SiLU. A same-ray two-frame chunk profile measured.1100s/step at32768 fragments versus.0779s at131072; the latter is selected without changing the mathematical model. See local_transport_chunk_profile.json.

## Environment

Reuse ssd-gs/PyTorch2.4.1/CUDA12.1. Author GGGS binding adds read-only access to projection
buffers; original rasterization equations unchanged. Local sparse traversal builds into
third_party/local_fragments_build. The system-installed tinycudann binary requires unavailable
GLIBC2.33, so a project-local build was tested from existing tiny-cuda-nn source
48d6989c95def307a40baf176b2d6015dada19f9; shared environment and binary remained unchanged. Its isolated MLP backward benchmark was not faster than PyTorch, so this unused local build and object files were removed. The final implementation does not depend on tinycudann.
Teacher inference and material fitting run as local GPU jobs.

## 30k results and interpretation

Both scenes completed30000 material steps. The completed source geometry also had30000 steps;
it was reused, not retrained again. The current stage freezes all geometric buffers and trains
only a fresh local total-response field. It does not implement joint geometry refinement.

Official test,512px, full-image mean; PSNR/SSIM higher and LPIPS lower are better:

| Scene | Method | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| Cat | Default reference |21.53618|.766465|.230405|
| Cat | PORT-DNA reference |22.14168|.776381|.267714|
| Cat | GGGS + local |22.51677|.774560|.263330|
| Pixiu | Default reference |20.57825|.845556|.156690|
| Pixiu | PORT-DNA reference |21.50296|.848252|.167022|
| Pixiu | GGGS + local |21.90448|.847826|.174197|

Against DNA: Cat gains.37509dB with a small LPIPS improvement; Pixiu gains.40152dB
but LPIPS worsens. Both SSIM scores decrease and both LPIPS scores remain worse than
Default. This is mixed evidence, not an overall quality improvement or a fair-budget
architecture ablation. Test images were historically observed during development;
this is not a newly blind benchmark. No test RGB was fitted in this experiment.

Fixed-image review: Cat recovers some spatial variation compared with DNA, but individual
fur strands, facial detail and paper folds remain blurred. Pixiu retains smeared base texture,
head/edge streaks and missing small specular peaks. On Pixiu test, GT1–4px peak recall is0,
and contrast ratio is.00108; even fit images have only.000346 pixel recall for these tiny
peaks. Failure on fit images indicates the problem is not solely held-out generalization.
The loss/pixel budget, frozen footprints and response representation remain competing
explanations; this experiment does not isolate their causal contributions. More iterations
alone are not established as a solution. Geometry errors were inherited unchanged.

Fixed-camera, four-light previews show changing illumination, but only the originalcamera0/
light0 pair has paired GT. The other combinations are qualitative renders, not evidence of
physical accuracy. This response field is not an identified albedo/roughness BRDF.

## Artifacts and final checks

Run: `runs/gggs_local_transport_final/`; final weights:
`Real_NRHints/{Cat,Pixiu}/last.pt`. Keep only one final material checkpoint per scene.
Commands/config/environment/source provenance are in `manifest.json`, `validation.json`, scene `config.json`,
`source.tar` and the logs. `evaluation_source.tar` / `evaluation_protocol.json` record
explicit TF32 evaluation matching training and the added preview command; training math
was not changed after launch.

- [Cat full comparison](../../runs/gggs_local_transport_final/comparison/Cat/full_images.png) /
  [detail crops](../../runs/gggs_local_transport_final/comparison/Cat/crops.png).
- [Pixiu full comparison](../../runs/gggs_local_transport_final/comparison/Pixiu/full_images.png) /
  [detail crops](../../runs/gggs_local_transport_final/comparison/Pixiu/crops.png).
- [Cat changing-light preview](../../runs/gggs_local_transport_final/relight_preview/Cat/relighting.png) /
  [Pixiu changing-light preview](../../runs/gggs_local_transport_final/relight_preview/Pixiu/relighting.png).
- [Model audit](../../runs/gggs_local_transport_final/model_audit.json) and
  [full results](local_transport_results.json).

Final audit confirms all seven geometry buffers are bitwise identical to the source conversion;
finite weights/losses, complete official-train indices, positive learned local slopes,
exact zero/double-light linear response on real contributions, and exact byte replay of
previewlight0 against testframe0. All three methods' saved targets match canonical GT
exactly on every66/71 test image. Common comparison recomputation uses CPU and has tiny
SSIM/LPIPS floating-point differences; the table above uses original evaluator metrics.
Cat/Pixiu have8,626,111/3,878,826 trainable material parameters and128,596/57,741 frozen Gaussians.
Training took40.56/33.28min, peak allocated memory12.01/8.89GiB, using GPU0/1.
Test rendering averaged2.06/1.02sec per512px image; this implementation is not real time.

The new method remains an experimental branch; Default is unchanged. No successful
high-frequency geometry/material reconstruction is claimed.

Full official train replay also completed: Cat522frames PSNR21.31597, SSIM.771268, LPIPS.269440;
Pixiu562frames PSNR23.63143, SSIM.864910, LPIPS.166752. Cat's fit/test averages have
different camera/light composition; higher test PSNR is not evidence of test fitting.
All scheduled jobs have exited successfully. Temporary200-step smoke weights/images were
removed after their metrics, configuration, logs and audit evidence were archived.
