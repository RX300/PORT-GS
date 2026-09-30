> 2026-09-27：本文描述历史方案。surface_reflectance和directional_surfel的独立方法入口已按用户要求删除；共享2DGS几何、GGX和初始化模块仍保留给DNA等方法。历史复现使用原实验源码归档。

# Intersection reflectance prototype

`methods/surface_reflectance.py` decodes nine local physical material values from
Gaussian base RGB and six feature logits. The transport has no trainable network
parameters: its only state is `light_scale`. Training must optimize Gaussian
geometry/material without constructing an Adam optimizer over an empty parameter list.
The shader evaluates two-lobe GGX at world-space receiving points using the
independent world-space camera and point-light positions and inverse-square falloff.
Alpha composition and the existing observation transform remain outside the BRDF.

`surface_fragments.py` builds differentiable planar homographies, conservative
finite-support projected bounds, and gsplat tile indices. Querying sorted unique
native pixel IDs enumerates all tile candidates, solves true ray–plane hits,
filters finite support/alpha/near depth, then stably sorts by pixel, hit depth and
Gaussian ID. Log-transmittance scans use float64; returned attributes retain
geometry/opacity gradients. Selection, support and ordering are piecewise discrete.
CUDA's global prefix sum is not bitwise repeatable: synthetic replay found
sub-ULP differences in a few weights/alpha/RGB values with identical state and
hit inputs. The initial exact-replay failure and bounded probe remain recorded;
replay uses rtol=1e-6/atol=1e-7, external ray batches rtol=2e-6/atol=2e-6.
No native center-depth early termination or minimum screen-space filter is reused.

`fragment_alpha_min` is saved in the method configuration and defaults to1/255,
including when loading an older checkpoint without the field. The optional0
setting disables that opacity cutoff while retaining rho<=9, near-plane and
positive-representable-alpha requirements. It does not change opacity parameters,
reset points or refine geometry. Training weight stages may explicitly change
the cutoff; evaluation restores the saved value. The cutoff recovery study and
its separate gradient/quality evidence are in
[surface_opacity_cutoff.md](../../experiments/surface_opacity_cutoff.md).

The returned `weights`, `ray_lengths`, `covered_query_indices` and `alpha` define
one composition contract. `compose_fragments` combines arbitrary hit attributes.
`shade_surface_fragments` either shades every hit then combines RGB, or combines
physical attributes first and shades covered rays. Faceforward each disk normal
before averaging. Never average the encoded material logits. Empty rays return
background without evaluating BRDFs at an invented zero point.

An optional visibility callback receives actual points being shaded and returns
one scalar per point. None means unoccluded illumination; it is not a shadow
approximation or exact light-ray tracing. Registered production dispatch rejects
shadow=True, absgrad, geometry-only warmup and legacy auxiliary fields. Sparse
`pixel_indices` queries return [Q,3]/[Q,1]; a full no-grad image uses external
4096-ray query/shade batches. These bound live fragments per batch, not tile-index
memory or candidate count. All training graphs remain live until backward.

`surface_sampling.py` draws uniformly placed native core patches, reconstructs
real neighboring pixels with a 5px halo, and restricts L1/mask/SSIM averages to
cores. Duplicate draws retain weight despite deduplicated rendering. Only outside
image boundaries use zero padding. This is a patch-placement objective; border
pixels have different inclusion frequency. CPU loss/gradient equivalence tests
cover the boundaries and repeated patches.

`train.py` bypasses native refinement hooks while updating every Gaussian
parameter group, rejects unsupported auxiliary modes, and handles the empty
transport optimizer explicitly. Frame and patch RNGs are independent of render
execution. The first real test caught global Python RNG consumption during an
initial library path; its failed sampling audit is retained and motivated the
dedicated frame stream before launching any matched quality experiment.
Initial state, fit/validation membership, counters and final RNG states are saved.
`--init-checkpoint` starts a new optimization stage with fresh optimizers/RNG,
not uninterrupted continuation. Scene/resolution/split mode/material mode and
patch/observation contracts must match. The optional existing positive scene-wide
light-scale parameter is fitted only on fit frames; no per-frame exposure exists.

This module represents a new foundation under test. Neither synthetic correctness
nor fitted normals/material values establish real geometry accuracy or unseen-light
quality. See the [experiment protocol](../../experiments/intersection_reflectance.md).


## Closing implementation and outcome (2026-09-24 JST)

The current branch additionally supports train-mask-only geometry initialization
through prepare_surface_priors.py --kind surfel_init / train.py --init-surfels.
Geometry remains freely optimized after initialization; this is not a persistent
SDF constraint or geometry truth. The optional --surface-patch-proposal tiny
uses half-uniform GT-tiny-origin sampling with inverse-PDF compensation of the
entire patch objective. Its controlled study failed quality gates; uniform
remains the final selected sampler and default. No dependency upgrades.

Fresh30000 full-train Cat/Pixiu and all137 official test frames are complete.
Tiny highlights were not reliably recovered, so this method is not promoted.
[Final results](../../experiments/surface_reflectance_final.md) and
[sampling protocol](../../experiments/surface_patch_proposal.md) distinguish
implementation correctness, fit, development testing and missing geometry truth.
Further research is paused at the user's request after cleanup.
