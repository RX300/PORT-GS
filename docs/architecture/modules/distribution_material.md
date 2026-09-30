> **2026-09-30 方法清理更新：** 当前只保留 `directional_port_v1`、`surface_attention`、`neural_material`。
> 其余方法和独立几何训练入口已删除；已有 GGGS 几何读取/诊断和三种方法的公共依赖保留。
> 以下涉及已删除方法的内容仅作历史记录，不是可运行入口。清理详情见 `docs/experiments/method_retirement_20260930.md`（项目根目录相对路径）。

> 2026-09-27 extension: an explicit GGGS import now preserves3D geometry and adapts
the SAME DNA material to gsplat deferred3D receivers using covariance-axis normal
proxies. Legacy2D behavior is unchanged. See [3D joint protocol](../../experiments/gggs_dna_joint.md).

# PORT-DNA-2DGS: distribution neural material

## Sources and adaptation

Read on 2026-09-24: [8DNA paper](https://arxiv.org/html/2604.25129v1),
[official implementation](https://github.com/lwwu2/8dna26), especially
`models/eight_dna.py` and `train.py`. HF paper endpoint returned 404; used arXiv.
8DNA decomposes full boundary transport F(xo,wo,xi,wi)=alpha(xo,wo)p(xi,wi|xo,wo).
Its RGB autoregressive rational-quadratic flows are trained with path throughput
weighted negative log likelihood and an independent throughput/albedo MSE.
It separates analytic direct scattering, and uses a box-ray parameterization
with Jacobians. It assumes a known renderable asset and online path samples.

Our photos have neither path tuples nor known internal scattering. Therefore
this is a new inverse-rendering adaptation, **not a reproduction of 8DNA**, and
we do not claim its low-variance path-training result. No official weights or
Mitsuba dependencies are used. Existing PyTorch/gsplat/CUDA stack is unchanged.

## Representation

Each 2D Gaussian stores geometry, diffuse RGB, six GGX parameters, three bounded
shading-normal offsets and further learned material features (24 total).
The disk rasterizer produces surface receivers, using native expected center-Z
and alpha-normalized material interpolation. This has the usual deferred-shading
approximation; it is not exact per-fragment intersection shading.

For source surfel j and component k, q_j is the normalized, detached
opacity-times-tangent-area measure; spatial probability is
s_jk = softmax_j(log q_j - ||x_j-a_k||²/(2 sigma_k²)).
Source material features predict an axis offset and concentration offset;
together with per-component parameters they define normalized S2 vMF densities
h_jk(wi). Concentrations are bounded to [.05,32.05]. These distributions are
conditioned on the asset, never on light intensity or position.
An outgoing-position/view/material MLP predicts RGB mixture probabilities
pi_kc(xo,wo), normalized over k, and independent sigmoid energy A_c(xo,wo).
Thus P_c(j,wi|xo,wo)=sum_k pi_kc s_jk h_jk(wi), and
sum_j integral_S2 P_c dwi = 1 for every color and outgoing query.
Position retains two effective dimensions on the learned surface, plus two
incident-direction dimensions and four outgoing dimensions: a discrete
approximation of an 8D boundary transport, with a finite-mixture restriction.
This replaces 8DNA's full autoregressive flow with a tractable 32-component
spatial/angular mixture. RGB mixtures differ, but component PDFs are shared.

L_out = L_GGX_direct + A * sum_k pi_k * sum_j s_jk h_jk(w_light,j) E_j.
E_j uses the actual point-light distance squared, RGB intensity, shared fitted
scene scale and geometry-derived deep shadow visibility. The neural density
conceptually includes angular transport weighting; no additional incident cosine
is multiplied into that branch. Direct GGX includes its own cosine once.
Direct and indirect terms are nonnegative and linear in light intensity before
the observation transform. A<=1 bounds indirect integrated response, but the
sum with direct GGX is **not guaranteed globally energy conserving or reciprocal**.
The discrete area surrogate and shadow approximation do not certify a physical
BSSRDF or uniquely separated materials. No ray-space box Jacobian is used because
we operate directly on surfel quadrature, not a box-parameterized flow.

## Training and geometry constraints

Only official training RGB, masks, cameras and point lights. Fresh geometry is
sampled by area from a smoothed occupancy isosurface obtained from all training
silhouettes (95% consensus, two-pixel mask padding); this code path constructs
**no signed distance field**. The occupancy volume is discarded after initialization.
Train independent disk positions, tangent scales, orientations, opacity and
material/network parameters. No SDF loss/field, pretrained normal/depth model,
pretrained material, previous checkpoint or camera optimization.
Photometric objective is .8 L1 + .2(1-SSIM), mask L1 .05, feature L2 1e-5;
from step1000, depth-derived normal self-consistency .01 and depth distortion .01.
The self-consistency normals come from the model's own rendered depth.

An additional .03 RGB-channel spatial KL compares normalized nonnegative
predicted/target image values. RGB regression supplies brightness information.
This is a likelihood objective for the **observed image marginal**, in the
existing camera response space. It is not path likelihood and does not recover
hidden scattering paths from photographs. A 1e-6 positive image floor supports
logs; zero target channels contribute zero. We make no calibrated photon-count
interpretation. An ablation would be needed to attribute improvements to KL.

## Implementation and checks

`methods/distribution_material.py` owns the neural material; registered as
`distribution_material`. `train.py` adds initialization and image KL. Existing
renderer, evaluator, light convention, deep shadows and checkpoint system are
reused. `prepare_surface_priors.py --surfel-surface-mode occupancy` selects the
SDF-free initialization branch. Tests cover S2 and discrete normalization,
source subdivision invariance, light linearity/zero input, gradients, checkpoint
roundtrip, image KL, and finite occupancy-derived surfels. A real-scene smoke
must pass before final experiments.

Chinese method proposal: [PORT-DNA-2DGS](../../research/port_dna_proposal.md).

## Interpretation of energy and evaluation

The shared learned scene light scale is radiometric normalization, not a measured
exposure calibration. Consequently A and the diffuse/GGX values must not be
reported as calibrated material albedo. LPIPS uses VGG only during evaluation;
no VGG or other pretrained model contributes any training/geometry gradient.
Native center-depth and interpolated material shading remain approximations;
there is no ground-truth geometry evaluation in these experiments.
