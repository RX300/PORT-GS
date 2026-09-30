# Shared neural material with directional transport

## Research question

`neural_material` replaces the position-conditioned material MLP with a shared,
pretrained and frozen BRDF decoder. Each Gaussian stores a six-dimensional
non-diffuse material code and a separate diffuse color. The experiment asks
whether a scene-independent material prior improves relighting of Real_NRHints
Cat and Pixiu. It does not assume an improvement before the official test.

## Sources and reconstruction boundary

- Yu et al., [Toward Richer Material Generation via Procedural Data Enhancement](https://arxiv.org/html/2606.14988v1),
  SIGGRAPH 2026, sections 3–4; [author supplemental](https://blaire9989.github.io/assets/4_DataEnhance/supplemental.pdf),
  pages 2–9 and 16–18.
- [Author project](https://blaire9989.github.io/assets/4_DataEnhance/project.html),
  inspected 2026-09-22: Code/Data labels have no download links. This is an
  independent PyTorch implementation, not an official checkpoint or full reproduction.
- [MaterialX microfacet sources](https://github.com/AcademySoftwareFoundation/MaterialX/tree/main/libraries/pbrlib/genglsl/lib)
  supply analytic directional-albedo fits. Attribution and license are in
  [materials/NOTICE.md](../../../materials/NOTICE.md).

The paper's useful separation is retained: pretrain a universal material model,
freeze it, then fit material codes. No diffusion generator, video model, text
conditioning, or generated Cat/Pixiu training images are used.

## Procedural pretraining

`materials/procedural.py` samples 22 coupled controls for dielectric/conductor
core and haze, dust, clearcoat, inner scatter and subcutaneous scatter. Four
equally weighted themes plus 20% infill provide isolated and combined effects.
BRDF evaluation uses exact dielectric/conductor Fresnel, isotropic GGX, and
Imageworks sheen. Reflection and transmission are accumulated separately so
metal absorption is not mislabeled reflected energy. The printed core/haze
weight interchange in supplemental equation S.2 is resolved by using consistent
weights for reflection and attenuation.

`MaterialEncoder` maps the controls to six sigmoid-bounded latent channels.
`MaterialDecoder` contains separate four-hidden-layer, 64-wide ReLU MLPs for
RGB non-diffuse BRDF and incident-only RGB transmission/scalar reflectance.
BRDF outputs use exp; energy outputs use sigmoid. Pretraining perturbs latent
codes uniformly by ±0.005. The canonical protocol is 50k Adam updates with cosine LR decay from 0.001 to 0.00001,
batch 4096, seed 0; log(f+0.01) squared BRDF error and albedo L1 weights are .95/.04/.01.
The original run used target-relative squared error; highlight diagnostics found
severe narrow-peak underestimation, motivating the current log-space objective.
Held-out procedural samples use a separate seed and never contain scene images.

Explicit adaptations:

- Online sampling replaces the original large parameter grids; this is a
  smaller pretraining budget than the paper's 500k BRDF/albedo updates.
- Isotropic angular invariants and one log half-angle coordinate replace an
  unspecified original direction embedding. Encoder roughness controls use log
  coordinates. Anisotropy is not represented by the procedural teacher.
- MaterialX directional-albedo fits use Schlick F0 approximations although the
  evaluated BRDF uses exact Fresnel. Energy conservation is therefore approximate.
- The current 50/50 direction mixture uses valid Rusinkiewicz coordinates and GGX
  half-vector proposals; it is not the paper's full reference-BRDF sampler.
- No sampling decoder is needed for this point-light rasterization workflow.

## Inverse-rendering interface

```
z(x) = sigmoid(interpolated Gaussian features[0:6])
c(x) = sigmoid(interpolated Gaussian base)
n_shade = normalize(n_geometry + tangent_projection(0.5*tanh(features[6:9])))
(f_non, T, R) = frozen_decoder(z, incoming, outgoing, n_shade)
cosine = max(dot(n_shade, incoming), 0)
response = (f_non + T * c / pi) * cosine
E = light_intensity * visibility / (light_scale * distance^2)
L_direct = E * response
L = E * f_non * cosine + (1-a) * E * T * c / pi * cosine
    + a * directional_port_exchange
```

The first six of the existing 32 feature channels are material logits; all
feature channels remain available to transport, but the first nine are detached
on that branch: transport may condition on material without fitting its codes or shading normal
through a separate RGB response. Cloning, splitting, pruning
and alpha-normalized interpolation inherit the current Gaussian implementation.
The material decoder receives no position, RGB light intensity or scene ID.
The old local shading MLP is removed for this method. Remaining transport
features, port matrices and geometry are optimized per scene.

Native 2DGS normals provide the local surface orientation and use the existing
StableNormal/DA3 training supervision. Renderer normals face the current camera;
this is a two-sided surface convention, not recovered globally oriented topology.
Features 6:9 provide a zero-initialized bounded tangent correction for the material
shading normal, separate from the supervised geometry normal. Its angular change
is at most atan(sqrt(3)/2), about 40.9 degrees. This models appearance detail without
requiring the monocular geometry prior to explain every narrow specular peak.
The material BRDF includes the incident cosine exactly once. Indirect port
transport is retained from `directional_surfel` and is not a physical BRDF-based
multi-bounce solver; the method does not establish exact material/transport
separation, reciprocity, or global energy conservation.

The inherited light_scale is the median training-camera illumination at the
estimated scene center. It normalizes a scene's radiometric scale rather than
calibrating absolute irradiance. A frozen BRDF and bounded diffuse color cannot
absorb arbitrary scale changes as freely as the old MLP. Fitted diffuse colors
must therefore not be interpreted as calibrated reflectance; saturation and
port contribution are useful diagnostics if image quality degrades.

Fresh training loads `--material-decoder` once and initializes material codes
from the pretrained encoder's neutral dielectric. Decoder weights are embedded
in every scene checkpoint; evaluation does not need the pretraining file.
`requires_grad_` preserves decoder freezing across geometry/relighting stage
transitions while retaining gradients to material codes and normals.
The direct non-diffuse response is never attenuated or recolored by the RGB
exchange gate; that gate mixes only direct diffuse and learned nonlocal output.
The first completed run gated the entire direct response. Reproduce its weights
with its archived source; the current canonical formula intentionally changes.
For a material-repair run, `--init-checkpoint ... --reset-material
--material-decoder ...` preserves scene geometry/transport while replacing the
frozen decoder and initializing its six codes with the new neutral material and
its three shading-normal offsets with zero.

## Original 30k experiment protocol

Only Cat/Pixiu, 512 px black background, full official train, fresh geometry,
30k steps, seed 0. Surface supervision starts at 1000, shadows/ports at 5000,
refinement ends at 25000, no RGB warmup or camera optimization. Only final
`last.pt` is evaluated, on all 66/71 official test frames using original
calibration. No test-based model or checkpoint selection.

Compare to preserved default 3DGS and attention checkpoints and the recorded
surfel results. Report per-scene PSNR/SSIM/LPIPS, training time, peak allocated
memory, point counts and the decoder pretraining cost separately. These runs
test a material-prior hypothesis and do not certify physical material recovery.

Use `pretrain_material.py --config configs/material.json` for the prior and
the existing `configs/validation.json` / `launch_validation.sh` for scenes.
Current execution state and actual output paths are recorded in
[status.md](../../project/status.md) and [results.md](../../experiments/results.md).

The original two 30k runs and all 137 official test frames completed on 2026-09-22.
Mean PSNR/SSIM are 21.978903/0.816073, higher than the three recorded references,
but LPIPS is worse for both scenes (mean 0.208648). This supports a pixel-error
improvement on these scenes, not an overall quality or physical-recovery claim.
Both scene checkpoints preserve all 20 decoder tensors exactly and load without
the external prior file. The default method remains `directional_port_v1`.

The current highlight repair and its additional training budget are tracked
separately in [highlight_recovery.md](../../experiments/highlight_recovery.md).

## Free-code lobe diagnostic

`python pretrain_material.py --evaluate runs/neural_material_highlight_prior/last.pt \
  --output runs/<fresh-output> --fit-lobes 1000` fits only six material logits on CPU.
The frozen decoder is evaluated on analytic dielectric/conductor GGX cuts, using
15/45/70-degree incident angles for fitting and 30/60 degrees for evaluation.
This distinguishes encoder initialization error from one local fit of the decoder
manifold. It uses no scene images and establishes neither relighting quality nor
a global capacity bound. Results and limitations are recorded in the research log.


## 直接解析材质对照（2026-09-23，真实场景检查通过）

`neural_material --material-model ggx`保留同一2DGS、逐像素着色、有界法线修正和PORT传输，
将6维材质码解释为RGB F0、两项GGX alpha、窄/宽叶片混合系数；alpha以对数映射覆盖.001到1。
高光用Schlick Fresnel乘相关Smith-GGX；漫反射透过率用同一MaterialX方向反照率近似的1-albedo。
近似能量积分不等于精确守恒；不包含原神经先验的sheen、多层涂层或一般透射材质。
解析材质无需预训练数据，也没有decoder权重。代码`materials/ggx.py`复用已有teacher的GGX计算。
默认仍是`material_model=neural`，旧checkpoint通过方法默认值明确恢复原神经模型。

已有神经模型转换时必须显式`--reset-material --material-model ggx`。仅重置features[:6]，
保留已拟合的法线features[6:9]、漫反射底色和传输权重；初始F0=.04、alpha=.1/.3、混合.9。
神经模型自身`--reset-material`同样保留法线，且必须提供新prior路径。
这改变的是显式材质重置的行为，旧实验应使用各自source.tar重现。
不带reset的跨材质加载被拒绝，避免将神经latent误解释为物理参数。

当前23项CPU检查通过，包括法向入射峰值、互易性、背面无高光、方向反照率不依赖出射方向、
材质梯度数值检查、状态保存与恢复。真实512px场景3+3步转换/续训、冻结几何/相机、逐位渲染重载与CLI评价也已通过，证据runs/analytic_material_smoke。
此候选受SpecGloss-GS的显式F0/roughness设计启发，但两叶片和近场点光协议是本项目实验，
不是论文完整复现：https://gkouros.github.io/projects/SpecGloss-GS/ 。

解析材质CPU窄峰拟合记录在`runs/analytic_material_lobe_fit`：从同一常量材质初始化，
仅优化6维参数1200步，训练入射15/45/70度，检查30/60度；没有场景图像。
在30度、alpha=.005时，介质/导体半高宽为teacher的.989/.957，峰值1.074/1.101；
alpha=.001时宽度为1.158、峰值.798/.760。最窄峰仍有局部优化和数值采样误差。
60度下峰值也有明显偏差，候选Schlick和teacher精确Fresnel并不相同。
这不是场景效果，也不是与此前神经拟合相同学习率/步数的严格公平对照；只证明此初始化可以拟合窄峰。


## 场景光强尺度校准（可选，尚未证明改善）

`train.py --optimize-light-scale`只添加一个全场景共享的训练标量：优化log(light_scale)，
每步用exp恢复正值，Adam lr=.001。沿用point_light的intensity/light_scale，直接项和非局部源能量同倍缩放。
不改变各帧光位置/相对强度、相机、图像响应gamma，也不提供逐帧曝光或测试时优化。
默认关闭，原中位数归一化行为保留。

训练标量只在训练器中存在；每步更新后的正值保存到既有transport.light_scale buffer，
评价/重载使用checkpoint中的同一数值，无需新模型权重键或外部文件。续训重新建立log参数和optimizer，
与当前其余优化器重建的契约一致。history记录light_scale和相对此次初始化的relative_light_gain。
光强、反照率、非局部传输仍有尺度歧义，不将拟合值称为真实曝光或绝对光度标定。
`runs/light_scale_smoke`已通过真实3+3步正值变化、几何/相机固定、逐位渲染重载和CLI评价。

## GGGS 3D initialization

Explicit `--init-geometry-format gggs` preserves all3D scales and enables the shared covariance-normal adapter. `requires_normals=True` requests these receiver normals; legacy2D rendering is unchanged. Explicit zero2D/prior regularization weights are required. Geometry and codes are trainable, shared neural decoder remains frozen across stage activation and restores from embedded checkpoint tensors. See [protocol](../../experiments/gggs_neural_material_joint.md).
