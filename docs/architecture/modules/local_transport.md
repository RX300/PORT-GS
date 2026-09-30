> **2026-09-30 方法清理更新：** 当前只保留 `directional_port_v1`、`surface_attention`、`neural_material`。
> 其余方法和独立几何训练入口已删除；已有 GGGS 几何读取/诊断和三种方法的公共依赖保留。
> 以下涉及已删除方法的内容仅作历史记录，不是可运行入口。清理详情见 `docs/experiments/method_retirement_20260930.md`（项目根目录相对路径）。

# Local contribution neural light transport

`local_transport` is a separate stage-two relighting pipeline, dispatched by the existing
train/evaluate/manifest entrypoints. It implements the local total-response direction in
[the redesign](../../research/local_transport_redesign_20260926.md), with the user's new
instruction to use the current GGGS geometry despite its failed quality gate.

## Geometry contract

Reuse `gggs_normal_depth_geometry/Real_NRHints/{Cat,Pixiu}/last.pt`: 30k GGGS +
StableNormal/DA3 geometry. Keep all 128596/57741 finite-thickness 3D Gaussians,
orientations, filtered scales and opacity compensation. No 3D-to-2D flattening.
The current material experiment freezes this geometry. It does not implement or
claim residual-driven population changes or joint geometry refinement from the broader proposal.

Author GGGS renders a centered virtual sensor; a small read-only binding exports
its actual means2D, conic-opacity, tile ranges and radial-depth ordering.
`csrc/local_fragments.cu` traverses these lists for requested rays, with the author's
alpha cap .99, cutoff1/255 and transmittance termination1e-4. There is no top-K cap.
Original off-center cameras use the same four-neighbor output interpolation as GGGS.
Tests compare constant/random per-Gaussian colors and alpha to the author renderer.
Frozen projection/visibility buffers are cached in memory; no material values are cached.

## Material and composition

For each contributing Gaussian and camera ray, analytically condition its 3D Gaussian
on that ray. Evaluate three Gauss-Hermite nodes at mean and mean±sqrt(3)*sigma with
weights2/3,1/6,1/6. These are finite-thickness quadrature samples, not surface intersections
or Gaussian centers mislabeled as intersection points. EWA alpha remains an image-space
Gaussian coverage approximation, so this is not exact participating-media ray tracing.

Each Gaussian has a feature vector, local linear feature coefficients, and RGB response bias.
Local coordinates and incident/view/half/reflection directions are computed before alpha
composition. Direction Fourier features, six GGX roughness kernels and a geometry-derived
deep-shadow visibility hint condition a positive neural total response. The GGX cue uses
the face-forward shortest covariance axis; it is a normal proxy, not a claim to reproduce
the continuous-depth surface normal. Full local directions are also network inputs. Visibility is an
input, not a hard multiplier; this permits learned transmission/indirect response.

Radiance is response * RGB light intensity / distance² / fixed scene light scale.
Intensity and frame IDs are not network inputs. Thus zero input gives zero radiance,
and source intensities add linearly before the observation transform. Nonnegative output
does not prove reciprocity, global energy conservation, or uniquely recovered physical BRDFs.
The shared scene scale is radiometric normalization, not calibrated albedo.

Per-fragment quadrature colors are weighted by the exact author alpha/transmittance,
then resampled from the virtual sensor. Material attributes are never averaged into one
pixel receiver before nonlinear decoding. PNG foregrounds are gamma2.2 encoded after
linear compositing using the existing observation function.

## Training and evaluation

Material training uses all official train522/562 views, with no test image fitting.
Source geometry used fit470/506; teacher inference and geometry fitting in this pipeline did not use official test images.
Source geometry validation frames are now material-training frames; they cannot serve
as material validation. The final benchmark is official test66/71, historically observed
by project development, not a new blind test.

Each step accumulates two independent camera/light conditions. Per condition sample two
16×16 cores with5px SSIM halo (26×26 rendered patches): one foreground-centered and one
uniform. This is an explicitly foreground-emphasized patch objective, not an unbiased
whole-image loss estimator. Loss is .8 L1(core)+.2 DSSIM(core with halo), no image KL,
pretrained material, frame exposure fitting, or SDF. Only neural parameters/codes update.
Two-view accumulation provides no claim of guaranteed visible overlap.

Whole-image evaluation uses the same fragment evaluator, complete official test and train,
quantized PSNR/SSIM/LPIPS and existing neutral-peak image proxies. Learned codes are a
neural material representation; previews are relighting reconstructions, not GT material maps.
