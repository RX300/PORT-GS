> **2026-09-30 方法清理更新：** 当前只保留 `directional_port_v1`、`surface_attention`、`neural_material`。
> 其余方法和独立几何训练入口已删除；已有 GGGS 几何读取/诊断和三种方法的公共依赖保留。
> 以下涉及已删除方法的内容仅作历史记录，不是可运行入口。清理详情见 `docs/experiments/method_retirement_20260930.md`（项目根目录相对路径）。

# Gaussian Wrapping adapter

`wrapping_reconstruction.py` implements the existing scene-data/experiment
interface around `third_party/GaussianWrapping`, author revision
`11e3b6fb5ca6f0a54e2b5587a09693488f3655af`.
Entry: `train.py --representation gaussian_wrapping`; evaluation: `evaluate.py --wrapping`.

The model, CUDA continuous-depth rasterizer and RaDeGS population routines come
from the author code. Four per-Gaussian normal parameters store a direction and
tanh sign. At step20001 they are initialized from the shortest covariance axis.
Alignment uses the author's configured .05×.6 median-depth normal loss.
At steps21999/22999/23999/24999/25999, every fit camera contributes a count-normalized
normal error through the dominant contributor map. The top5% errors are cloned
with flipped normal signs. Tangent-frame scales preserve combined volume and
parent/child centers shift ±.1 standard deviations along the normal. This follows
the actual released code's dominant-contributor accumulation, rather than
claiming an exact all-contributions attribution.

MVS uses the author's slow PatchMatch path: render both views, reprojection
consistency and differentiable warped patch NCC. We choose this supplied path
to support original off-center intrinsics; its NCC/geometry weights are .6/.02,
starting at step7000. Neighbor selection uses author camera-angle/baseline rules.
Exposure affine parameters apply to training L1 as in the official ours launcher;
evaluation reports raw RGB, never fitting validation exposures.

Adaptations:

- Existing fit-only silhouette occupancy seeds, 40000 points, no old geometry checkpoint.
- Full-K centered virtual canvas with output resampling, already used/tested by GGGS.
- Supplied alpha L1 .2; depth-normal and shell-error terms restricted to foreground.
- The Python ours renderer now returns its computed CUDA alpha instead of dropping it.
- Author MVS divided by zero for background depth even inside a masked `where`;
  denominators are protected before division. A synthetic plane isolated the
  nonfinite gradient to geometric reprojection; it passes after this fix.
- Mip opacity determinant compensation is computed through per-axis ratios in
  a project subclass, avoiding underflow of the product of six small scales.
  Opacity reset uses its algebraically equivalent stable expression.
- Source-level upstream adjustments are saved in
  `docs/experiments/wrapping_upstream_patch.diff`; CUDA algorithms are unchanged.

Shell densification is adapted locally from the author implementation to avoid
importing inactive SDF/mesh-optimization dependencies. The optional learned SDF,
depth-order teacher, and MILo branches are disabled. This is an NRHints adaptation,
not a claim to reproduce official benchmark numbers or the complete textured mesh pipeline.

StableNormal is optional through `wrapping_normal_priors` / `wrapping_prior_weight`.
This follows the user's explicit 2026-09-27 authorization. Fit-only camera-space
normals are transformed to world coordinates and supervise BOTH rasterized
geometric normals and depth-derived normals; independent learned orientation
vectors alone cannot satisfy the added loss. No SDF or pretrained depth is used.

Checks: `test_methods.WrappingGeometryTests` exercises real CUDA RGB/alpha/depth,
normal features, MVS derivatives, shell cloning, save/reload and tiny-scale Mip gradients.
`BoundaryGeometryTests` checks that removing outward error by shrinking a silhouette
does not pass the boundary recall criterion. Synthetic geometry plus an800-step
real-scene smoke precede the full experiment.

Boundary measurements are silhouette F1 at2px, fraction/p95 of predicted boundary
outside the supplied silhouette, and depth-normal adjacent-angle p95 within8px of
the GT silhouette. They are image/surface consistency diagnostics, not ground-truth
3D shape accuracy. Their interpretation must include the fixed-view depth/clay images.
