> 2026-09-27更新：后续修复和重训已完成，见[最终修复报告](native_2dgs_topology_repair.md)。本文涉及的前轮native_2dgs_normalized_geometry已归档至最终run的`controls/prior_native/`；旧模型权重已按final-only要求清理。

> 2026-09-27补充：用户质疑后发现具体训练链路问题，见[参数与剪枝审计](native_2dgs_parameter_audit.md)。以下记录保留原实验结果，但数据/标定的原因排序需重新验证；原生控制采用median depth1，并非作者全部通用默认。

# Geometry-first native 2DGS: Cat / Pixiu

2026-09-26: user authorized trying the redesigned approach, with native 2DGS reconstruction first. This first stage establishes whether explicit geometry is usable before introducing a new relighting network. No SDF or pretrained geometry supervision.

## Fixed first experiment

The first canonical configuration was `native_2dgs_geometry` (the current canonical is the final normalized profile):

- Cat470 fit /52 internal validation, Pixiu506/56, determined from official train only by existing light-group split; original official test is not evaluated in this stage.
- Author 2DGS model/CUDA, fresh30000, native512, SH3,40000 fit-mask occupancy initial points, author uncapped dynamic point population.
- Author Adam/LR, clone/split/prune schedule through15000, opacity reset3000/cull.05. Normal weight.05 after7000; author default distortion0; median depth.
- Fixed original calibration including principal-point offsets; no color/gamma/exposure calibration. SH fits observed PNG RGB directly. No point-light/material/PORT branch.
- GPUs0/1, seed0, existing ssd-gs/CUDA12.1; extensions compiled locally. One completed model per scene.
- Complete internal validation and complete fit evaluation, LPIPS/RGB and geometry proxies. Only four RGB pairs plus four predetermined geometry views per split are exported, rather than recreating all intermediate experiment images.

## Review before relighting

Inspect silhouette/coverage, world-normal and median-depth maps, geometry-only clay, oriented surface points, and nearest-view depth consistency. Report active opacity counts and model size. Require no gross collapse, disconnected sheets or wrong large-scale shape before using these weights as a geometry initializer.
For a first coarse screening, seek validation silhouette IoU around.9 or higher, useful foreground coverage and consistent neighboring depths; all ratios/counts and rejected occlusions must be shown. Self consistency and low RGB error do not prove actual geometry accuracy. Cat fur and Pixiu translucent regions may remain ambiguous without fixed-light observations.
If the stock reconstruction produces gross geometry failures, do not move a new material head onto it merely because training finished. Diagnose initialization/illumination/calibration next. If geometry is usable, transfer only geometry and keep the exact fit/validation lineage when training fresh relighting material.

## Build and run

The author clone and its submodules are under `third_party/2d-gaussian-splatting` (versions in the module document). Build each CUDA extension in its own source directory:

```bash
CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=8 \
 /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python setup.py build_ext --inplace
```

No pip install or dependency-stack upgrade is needed. With the canonical configuration:

```bash
bash launch_validation.sh
python evaluate.py runs/native_2dgs_normalized_geometry/Real_NRHints/Cat/last.pt \
 --native-2dgs --split validation --output runs/new_geometry_evaluation --lpips
```

The retained final models and run metadata live in `runs/native_2dgs_normalized_geometry/`. Each scene additionally archives author model/CUDA source in `upstream_source.tar`. The two development-control reports/images/logs/source snapshots are consolidated under its `controls/`; their intermediate weights and geometry exports have been removed under the user's final-only retention rule. Original absolute paths inside immutable archives describe historical execution, and `retention.json` records the new directory mapping. The dataset and previous five final relighting runs remain intact.

## Preflight

Full-K/pixel-center projection and analytic plane depth normals pass; actual author CUDA projection peak, depth and finite gradients pass, including geometric regularizers. Cat random1000/128px five-step train, checkpoint reload, two validation frames, normal/depth/clay exports and cross-view reports pass. This smoke is interface validation, not a quality result. Initial shared-simple_knn import failure is preserved in logs; fixed by project-local compilation.

[Implementation and limitations](../architecture/modules/native_reconstruction.md).

## Completed control and next profile

The 30000-step native control completed, with all52/56 validation and470/506 fit views evaluated:

| Scene | Val PSNR / SSIM / LPIPS | Silhouette IoU | Alpha L1 | Final / opacity-eligible points |
| --- | --- | ---: | ---: | ---: |
| Cat | 14.4243 / .71621 / .31876 | .44402 | .54030 | 105079 /100836 |
| Pixiu | 18.1954 / .84387 / .16551 | .20713 | .78412 | 74312 /68599 |

Geometry panels reveal opaque background sheets. Black SH color can reproduce the black background without empty space, and the unmasked normal term rewards opacity there. This profile fails the geometry gate despite smoother normals and improved self-consistency. These RGB metrics are view-dependent reconstruction under changing lights, not relighting benchmarks.

The second canonical profile `native_2dgs_mask_geometry` keeps the split, seed, initialization, original calibration and30000 steps. It adds supplied-alpha L1 weight.2 and uses the author's DTU depth-distortion setting1000 (after3000), median depth1. It retains original unmasked normal loss and all author model/CUDA/topology code. This is an explicit mask-supervised data adaptation. Both changes are tested together as a reconstruction recipe; no single-factor causal claim will be made. At an aligned opaque black background surface the alpha penalty outweighs the normal term's opacity incentive (.2 versus at most.1 derivative). No distance field, learned prior, light input or external points are introduced.

The diagnostic entrypoint `diagnose_image_errors.py CHECKPOINT --native-reconstruction --reference-geometry OLD.pt --output NEW_DIR` compares both geometries through the same native renderer. In the first four-view audit, only Cat frame36 had sufficient nearest-camera SIFT matches:31 common supported features yielded median model reprojection3.576px for the previous geometry and3.701px for the native control. The other seven pairs had insufficient matches. Expanding to eight nearest fit cameras yields a few more correspondences, but Cat frame36/406 calibrated epipolar median3.205/1.382px suggests correspondence or camera-calibration error in addition to depth error. These sparse, potentially false matches do not establish geometry accuracy. The old PORT-DNA reference used all train images, so this is observational geometry comparison, not a matched validation trial.


## Mask profile completed; scale diagnostic

Mask profile finished with validation Cat52:PSNR14.5399, SSIM.71813, LPIPS.31271, IoU.94982, alphaL1.01974; Pixiu56:18.2888/.84537/.16511, IoU.90450, alphaL1.01839. Background sheets largely disappear, but Cat stays oversmooth and Pixiu shows sharp sheets/terraces and spikes. It passes a coarse silhouette check, not the geometry quality gate.

The native distortion is variance of `m = far/(far-near)*(1-near/z)` with CUDA constants near.2/far100. Under world length scaling it changes quadratically. Fixed Cat36/Pixiu32 median depths are5.050/14.176 with camera extents6.065/14.738. Raw foreground distortion means6.27e-9/2.18e-9 have negative pixel fractions47.9%/49.4% (roundoff; the true pairwise variance is nonnegative). Evaluating the same geometry in extent-normalized coordinates changes RGB mean absolute error only2.64e-6/4.38e-7, but substantially rescales distortion. This shows an objective-scale issue, not proof it is the only cause of artifacts. Audit: `runs/native_2dgs_normalized_geometry/controls/native_2dgs_mask_geometry/coordinate_scale_audit.json`.

A final controlled geometry profile `native_2dgs_normalized_geometry` changes only training coordinates to `(world-center)/camera_extent`. Cameras, kNN scales, optimizer spatial LR and topology extent are transformed together; SH directions and image projection remain equivalent. Default camera/lighting metadata remain untouched. Canonical geometry and PLY are exported back in world coordinates. Author capture/optimizer remain in training coordinates, and evaluation explicitly restores world geometry. Native near clipping is still.2; the selected scene depths after normalization are around.83/.96, above it. Same30k, fit/validation, seed, mask.2, distortion1000; no new supervision. This is a numerical/data adaptation and not a claim of exact original preprocessing.


## Final result and geometry decision

All three profiles completed Cat/Pixiu30k and full internal validation/fit evaluation. These are six fresh reconstructions, not checkpoint continuations. No official test evaluation or relighting training was performed in this stage.

| Profile | Scene | Val PSNR | SSIM | LPIPS | Silhouette IoU | Alpha L1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Native control | Cat | 14.4243 | 0.71621 | 0.31876 | 0.44402 | 0.54030 |
| Native control | Pixiu | 18.1954 | 0.84387 | 0.16551 | 0.20713 | 0.78412 |
| Mask + distortion | Cat | 14.5399 | 0.71813 | 0.31271 | 0.94982 | 0.01974 |
| Mask + distortion | Pixiu | 18.2888 | 0.84537 | 0.16511 | 0.90450 | 0.01839 |
| + coordinate normalization | Cat | 14.5374 | 0.71802 | 0.31405 | 0.94959 | 0.01968 |
| + coordinate normalization | Pixiu | 18.2816 | 0.84538 | 0.16509 | 0.90494 | 0.01846 |

The final normalized profile has98412/54180 stored points and96884/49916 points with opacity at least1/255. This is an eligibility upper bound, not a per-view visible-contribution count. It corrects the preceding fixed40k/no-refinement bottleneck, but does not establish correct detailed geometry.

Manual review of all eight fixed validation geometry panels: Cat's coarse body/support is localized, but fur-scale shape is absent and there are flat regions/creases. Pixiu retains large planar/terraced patches and sharp edge spikes. Normalization does not materially improve these artifacts over the mask profile. The numerical-scale issue is real, but this one controlled experiment does **not** support it as the dominant quality bottleneck. **Geometry gate: failed for both scenes.** The surviving model is a reproducible coarse reconstruction reference, not an accepted geometry teacher.

[Cat geometry comparison](../../runs/native_2dgs_repaired_geometry/controls/prior_native/review/Cat/geometry_comparison.png) · [Pixiu geometry comparison](../../runs/native_2dgs_repaired_geometry/controls/prior_native/review/Pixiu/geometry_comparison.png). Reference is the previous PORT-DNA model, rendered through the same author backend with material discarded. That reference trained all official train frames, so this comparison is observational; its validation overlap is reported in each review JSON. Gray panels use fixed-world two-sided shading from predicted normals, not GT geometry or learned material.

The richer geometry metrics still do not provide ground-truth geometry. Nearest-view depth self-consistency can be high for an incorrect smooth shape. Sparse SIFT matches show camera/correspondence discrepancies and are mostly unavailable for Pixiu. No calibrated subpixel geometric-accuracy claim is justified.

### Consequence for the next stage

Keep the user's order: geometry first, then local per-contribution neural light transport. The main unresolved issue is obtaining geometry evidence that is consistent across views under changing captured illumination. Native SH does not take lamp position as input, so it cannot generally explain these camera/light pairs with stable appearance. This is an assumption mismatch supported by the capture metadata, not a proof that every artifact comes from illumination.

If fixed-world-light or diffuse natural-light multiview images of the same unchanged objects are available, those are the preferred native geometry input. With only the current images, the next meaningful geometry task is fit-only classical correspondence/calibration auditing and robust local illumination-invariant multiview constraints, excluding unreliable translucent/specular regions; their feasibility must be demonstrated before using them as supervision. More iterations or a larger neural material alone is not a validated remedy. Masks only constrain outlines, not internal concavities.

The geometric payload/PLY is back in original coordinates and can initialize the future relighting stage with its original cameras/lights and the same fit/validation split, but it is not promoted to that stage now. The approved local neural transport redesign remains unimplemented pending the geometry prerequisite. No SDF, pretrained geometry teacher, or independently supplied point cloud was used. LPIPS's pretrained network is used only for image evaluation.

### Verification and retention

Full-K/pixel-center projection, finite author-CUDA geometry/SH/alpha gradients, similarity projection invariance, five-step train/reload/export and original-world PLY roundtrip passed. Final saved geometry is finite and exactly corresponds to the documented coordinate transform. Core code bytes match the final launch source archive; each later geometry review has its own source archive. Source tests and both preflight reports are under `logs/` and this experiment directory.

Only the final Cat/Pixiu checkpoints remain for this reconstruction method, alongside the previous18 relighting checkpoints. Control metrics/figures/logs/source and the numerical audit remain under `controls/`, while smoke runs and intermediate weights/point exports are removed. [Machine-readable full metrics](native_2dgs_geometry_results.json) · [retention record](../../runs/native_2dgs_repaired_geometry/controls/prior_native/retention.json).
