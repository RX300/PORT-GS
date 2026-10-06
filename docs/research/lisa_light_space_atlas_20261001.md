# LiSA-GS: Light-Space Neural Transport Atlases for Relightable Gaussian Splatting

> 2026-10-06：最终版的新颖性定位请结合[相关工作审计](lisa_prior_art_audit_20261006.md)。
> 新核查的Gaussian Shadow Maps、NeuMIP、RNG及PG2026矩阴影工作限制了下文若干宽泛贡献表述；本提案保留为历史记录。

Proposal written 2026-10-01 (JST) on branch `feature_claude_neural_relighting`, target venue CVPR 2027.
Code name in PORT-GS: representation `light_atlas`. This is a fresh design: it does not reuse the
ports, the attention exchange or the frozen neural BRDF prior of the three existing PORT methods
(`directional_port_v1`, `surface_attention`, `neural_material`). It reuses only shared infrastructure
(data loader, gsplat camera rasterization, densification, training-camera calibration, evaluator).

## 1. Problem and evidence from this project

Task: multi-view images of an object, each lit by one calibrated point light; render novel views under
novel point-light positions. Six-scene panel (fixed test calibration, full official test, PSNR):

| Scene | SSD-GS | GS³ | RNG | PORT default | PORT attention | PORT neural |
|---|---:|---:|---:|---:|---:|---:|
| Cat | 18.00 | 19.12 | 20.99 | 25.26 | 25.14 | 24.22 |
| Pixiu | 23.40 | 23.72 | 19.95 | 26.32 | 26.39 | 26.14 |
| AnisoMetal | 28.00 | 27.51 | 24.32 | 28.26 | 28.43 | 23.37 |
| Translucent | 31.99 | 32.76 | 29.29 | 29.32 | 28.64 | 22.88 |
| bunny_small | 38.48 | 34.42 | 37.87 | 38.17 | 36.57 | 26.64 |
| dragon_small | 37.38 | 36.11 | 35.26 | 36.14 | 34.67 | 32.40 |

Sources: `runs/refactored_six_scene_20261001/RESULTS.md`, `runs/surface_attention_six_scene_20261001/RESULTS.md`,
`../benchmarks/six_scene_baselines/RESULTS.md` (baselines 100k iterations; PORT 30k; real-scene PORT uses
training-camera self-calibration).

Inspection of the worst test views of the strongest PORT model (`directional_port_v1`):

1. **Cast shadows on large receivers are missing or smeared** (AnisoMetal view 308, Translucent view 45: the
   dice shadows on the floor). All three PORT methods, GS³ and SSD-GS compute *one visibility per Gaussian*
   and splat it as an attribute. A large floor Gaussian has one visibility value, so a shadow boundary
   crossing it cannot be represented; the 64-bin deep-shadow volume adds depth quantization.
2. **Translucency is under-expressed** (Translucent vase/dice glow, Pixiu resin, dragon/bunny thin parts).
   Local per-pixel or per-Gaussian shading has no access to light deposited elsewhere on the object; the
   SSS modules of SSD-GS and SSS-GS are also *local* (per-Gaussian MLPs predicting dipole parameters or
   SSS radiance from light/view directions).
3. The two worst bunny and dragon test views (bunny 332/98 at ~12 dB, dragon 130/54) show a different object
   pose than the render — a data artifact; LiSA scores them identically. (Correction after the first LiSA run:
   bunny views 413/441/96/231, 13.7–15.4 dB for the default, are *model* failures — the front-lit bunny is
   rendered red instead of white — and LiSA renders them at 35.9–38.4 dB.)

## 2. Literature survey (what each line contributes)

**Classical shadows.** Shadow maps (Williams 1978); PCF (Reeves et al. 1987); PCSS (Fernando 2005);
variance / exponential / moment shadow maps (Donnelly & Lauritzen 2006; Annen et al. 2008; Peters & Klein
2015) filter *statistics of depth* so that soft visibility comes from pre-filtered maps; deep and opacity
shadow maps (Lokovic & Veach 2000; Kim & Neumann 2001) store transmittance functions for hair/fur.

**Classical subsurface scattering.** Dipole diffusion (Jensen et al. 2001). Translucent shadow maps
(Dachsbacher & Stamminger 2003) store irradiance and position in light space and integrate the BSSRDF
around a receiver's light-space projection. Sum-of-Gaussians diffusion profiles (d'Eon & Luebke 2007)
make that integral a set of separable blurs, and Gaussian kernels factor across lateral distance and
depth, which gives cheap thickness-dependent transmission. Screen-space separable SSS (Jimenez et al.
2015) and pre-integrated skin (Penner 2011) are cheaper, view-dependent variants.

**Classical global illumination.** Reflective shadow maps (Dachsbacher & Stamminger 2005) treat light-space
texels as virtual point lights; PRT (Sloan et al. 2002) makes outgoing light a linear function of
lighting coefficients; light propagation volumes and voxel cone tracing (Kaplanyan 2010; Crassin et al.
2011) use pre-filtered multi-scale light representations.

**Classical materials.** Microfacet GGX (Walter et al. 2007), Disney principled BRDF (Burley 2012),
anisotropic spherical Gaussians (Xu et al. 2013), LTC (Heitz et al. 2016).

**Neural rendering and shading.** Deep Shading (Nalbach et al. 2017) and Deferred Neural Rendering (Thies
et al. 2019) decode G-buffers with CNNs; Deferred Neural Lighting (Gao et al. 2020) feeds radiance cues
rendered with basis materials; Neural Light Transport (Zhang et al. 2021) works in texture space; NRHints
(Zeng et al. 2023) feeds ray-traced shadow and Blinn–Phong highlight hints; Neural Shadow Mapping (Datta
et al. 2022) and kernel-predicting neural shadow maps (2025) learn soft shadows from shadow-map buffers in
screen space for *known* geometry; RenderFormer (2025), DiffusionRenderer (2025) and TRON (2026) are
pretrained neural renderers driven by G-buffers or ray-traced guidance.

**Neural materials.** Latent-code BRDF decoders: NBRDF (Sztrajman et al. 2021), NeuMIP (Kuznetsov et al.
2021), neural layered BRDFs (Fan et al. 2022), real-time neural appearance models (Zeltner et al. 2024),
the procedural neural BRDF prior of Yu et al. (2026, used by PORT `neural_material`). Neural transport for
known assets: NeuPreSS (2024), 8DNA (Wu et al. 2026).

**Relightable Gaussians.** Point light / OLAT: GS³ (per-Gaussian angular Gaussians, per-Gaussian shadow
splatting + MLP refinement, residual MLP), RNG (neural Gaussian features, *depth-only* shadow map giving a
scalar per-pixel cue |PQ|, depth refinement MLP), BiGS (bidirectional SH per Gaussian), SSS-GS (per-Gaussian
SSS MLP and learned incident light field), SSD-GS (ICLR 2026: per-Gaussian shadow splatting + refinement,
per-Gaussian dipole-parameter MLP, ASG specular), F-RNG (2026, feed-forward). Avatars: RGCA (Saito et
al. 2024, learned radiance transfer), Relightable full-body GCA (Wang et al. 2025, shadow CNN in mesh UV
space), BecomingLit (2025), Deep Gaussian Shadow Maps (Mir et al. 2026, classical, no learning).
Global illumination with Gaussians: radiosity on surfels (Jiang et al. 2025), GI-GS (2024).

## 3. Key insight

For an isotropic point light of intensity I, the irradiance-weighted area element of the lit surface is
the light's solid-angle element:

    E(x) dA(x) = I cos(theta_i) / r^2 dA = I dw.

Every light-space texel therefore carries the same incident flux `I dw`, independent of surface normal and
distance. Splatting the Gaussians *from the light* with ordinary alpha compositing deposits flux
`I dw T_i alpha_i` on Gaussian i: exactly the light-space compositing weight. Consequently

- **visibility** of any receiver is a function of the depth distribution of that deposition along its
  light ray (shadow / deep-shadow maps),
- **subsurface transport** `L(x_o) = ∫ R(x_o, x_i) E(x_i) dA(x_i) = I ∫ R(x_o, x(w)) dw` is a convolution
  of the light-space deposition with a receiver-dependent kernel (translucent shadow maps), and with
  Gaussian kernels the lateral part is a light-space blur and the depth part a thickness attenuation,
- **low-order indirect light** from nearby lit surfaces is a larger-scale gather of the same deposition
  (reflective shadow maps).

All three are gathers from one light-space map. Classical methods fix the kernels (PCF, dipole) and the
stored quantity (depth, irradiance). We make the stored quantity a **learned flux feature** of the Gaussians
and the gather a **learned, receiver-conditioned kernel over a multi-scale pyramid**, trained end to end
through differentiable splatting. Because the map is re-rendered for every light and the network consumes
only light-relative quantities (depth offsets, thickness, deposited flux), the transport operator is
equivariant to light motion; generalization to unseen light positions is structural rather than
interpolated from a light-position input.

## 4. Method

### 4.1 Representation
3D Gaussians (means, scales, rotations, opacities) carry a base color parameter `b_i` and a latent
material code `f_i` (32-D). Geometry is native 3DGS with gsplat densification; real scenes reuse the
existing training-camera rotation calibration with translation gauge (unchanged protocol).

### 4.2 Light pass: neural flux atlas
For light position `p`, a perspective light camera looks at the scene center; its field of view covers the
splatted extent of 99.9% of the Gaussians (robust to floaters). Each Gaussian emits light-space channels

    c_i = [ phi_i , z_i , z_i^2 ],  phi_i = softplus(E_theta(f_i, b_i, |n_i . l_i|)) in R^C (C = 8),

with `z_i` the light-space depth normalized by the object radius and `n_i` the shortest covariance axis.
One gsplat rasterization (512^2) gives `A = sum_i T_i alpha_i c_i` and `M0 = sum_i T_i alpha_i`:
deposited flux features `Phi`, and the first two depth moments `M1, M2` of the deposition.
A 7-level Gaussian pyramid (binomial blur + 2x decimation) provides world-scale kernels from about one
texel to roughly a third of the object.

### 4.3 Deferred receiver gather
Camera rasterization yields per-pixel receivers: world point from expected depth, material code, base and
normal. Each receiver projects into light space (u, v, z) and reads every pyramid level bilinearly. Per
level k it forms: coverage `M0_k`, deposition mean `mu_k`, spread `s_k = sqrt(var_k + tau_k^2)` (`tau_k` = level
texel size), depth offset `delta_k = z - mu_k` (thickness behind the lit surface), normalized offset
`t_k = delta_k / s_k`, the moment shadow test `V_k = 1 - M0_k Phi_N(t_k - 3)`, and the flux features `Phi_k`.

### 4.4 Neural deferred shading
Outgoing radiance at a receiver with incident irradiance scale `E = I / r^2`:

    L = E * [ V * rho  +  sum_k sum_c W_{k,c}(x) Phi_{k,c} ].

- `rho` — **local neural material**: an MLP on the latent code, normal, world-frame directional encodings of
  light/view/half vectors, their cosines and a bank of spherical-Gaussian highlight hints
  `exp(kappa_j (n.h - 1))`, `kappa in {8, 32, 128, 512}`; output `softplus(b + MLP)`.
- `V` — **moment visibility with neural refinement**: `sigmoid(logit(V_0) + g(stats, f, n.l))`, `g` zero-initialized,
  so training starts from the classical test and learns penumbrae, bias and fur transmittance from
  multi-scale statistics.
- `W` — **light-space transfer kernels**: an MLP predicts nonnegative weights `W in R^{K x 3 x C}` from the
  receiver code and per-level statistics. Radiance is *linear in the gathered flux* (light-space PRT /
  neural BSSRDF); the sum-of-Gaussians TSM `w_k exp(-delta^2 / 2 sigma_k^2)` is the special case with one
  flux channel and a fixed attenuation.

### 4.5 Training
Loss, schedule and densification are the PORT defaults (0.8 L1 + 0.2 D-SSIM in the display domain, 0.05
alpha mask L1, 30k steps, seed 0). The light pass starts at step 5000 (`--shadow-start`, `--port-start`),
before which `L = E * rho`. Gradients from the light pass reach Gaussian geometry, opacity and codes.

### 4.6 Cost
One extra rasterization (512^2, 10 channels), a 7-level pyramid, 7 bilinear lookups and three small MLPs per
covered pixel; it replaces the 64-channel deep-shadow volume and the all-Gaussian port exchange of the
default PORT method.

## 5. Novelty review (closest work, checked 2026-10-01)

| Work | What it does | Difference |
|---|---|---|
| RNG (CVPR 2025) | depth-only shadow map; scalar per-pixel cue; MLP decoder | we splat learned flux features and depth moments, gather multi-scale, model translucency/bounce |
| GS³ (SIGGRAPH Asia 2024), SSD-GS (ICLR 2026) | per-Gaussian shadow from splatting toward the light + MLP refinement | per-pixel (deferred) visibility; one Gaussian can hold a shadow edge |
| SSD-GS, SSS-GS (NeurIPS 2024) | local per-Gaussian SSS (dipole parameters / MLP of light & view directions) | nonlocal, thickness-aware light-space transport, linear in deposited flux |
| Translucent / reflective shadow maps (2003/2005), d'Eon & Luebke (2007) | analytic kernels, stored irradiance, known geometry | learned flux features and kernels, inverse rendering through Gaussians |
| Neural shadow mapping (2022), KPNSM (2025) | screen-space CNN on shadow-map buffers, known geometry | light-space gather, no screen-space CNN, joint geometry/material optimization |
| Full-body GCA (2025) | shadow CNN in mesh UV space on precomputed irradiance | light-space flux atlas of Gaussians, no mesh |
| Deep Gaussian Shadow Maps (2026) | classical transmittance atlases for Gaussians | learned features, SSS/bounce, neural refinement |
| RenderFormer, DiffusionRenderer, TRON | pretrained neural renderers | per-scene, physically structured, no pretraining |

Claimed contributions: (1) the light-space reparameterization of point-light transport for Gaussian
splats with a learned, normal-free flux atlas; (2) multi-scale receiver gathering with light-space
transfer kernels linear in flux (a learned translucent shadow map / light-space PRT); (3) moment-based
deferred Gaussian shadows with neural refinement. Not claimed: physically exact SSS parameters, global
illumination beyond what a light-space gather can see (e.g. light reaching a receiver only after bounces
in regions invisible from the light), environment-light relighting.

## 6. Feasibility review

- Fits the existing pipeline: one new transport class, a renderer hook to skip per-Gaussian deep shadows,
  no new CUDA code (gsplat rasterization, `grid_sample`, MLPs).
- All train/test lights lie outside the objects (checked on the six scenes: half-angle needed to cover the
  Gaussians is 9–51 degrees), so a single perspective atlas per light suffices.
- Main risks and mitigations: self-shadow acne (spread-normalized test with a 3-sigma bias, then learned
  residual); light-pass gradients destabilizing geometry (light pass starts at 5000 steps; can be detached);
  calibration noise of real lights (soft visibility is learnable; camera calibration protocol unchanged).
- Budget: six scenes x 30k steps on two RTX 6000 Ada GPUs, about one hour per scene.

## 7. Validation plan

Same six scenes and protocol as `refactored_six_scene_20261001` / `surface_attention_six_scene_20261001`
(30k steps, seed 0, full official train, 512 px Real/GS3 and 256 px SSS, GS3 white background, gamma 2.2,
SSS unit light 1, real-scene training-camera calibration, original test calibration, PSNR/SSIM/LPIPS on
every official test frame, last checkpoint, no test-time fitting). Success criteria: higher six-scene mean
PSNR than `directional_port_v1` (30.58) and gains concentrated where the diagnosis predicts (GS3 shadows,
translucent scenes). Ablations on the same protocol: no light-space transport (`--light-transport none`)
and classical moment visibility only (`--visibility-model moment`).

## 8. Validation status (2026-10-02, complete for this round)

Implemented as `light_atlas` ([module](../architecture/modules/light_atlas.md)); full record and provenance in
[the experiment log](../experiments/development.md). Final configuration: `--material-head svbrdf`
(local positional capacity only through light-independent coefficients; adopted on held-out lights by a pre-registered rule).

| Six-scene test mean (PSNR / SSIM / LPIPS) | |
|---|---|
| **LiSA, final (svbrdf head)** | **31.367 / 0.9353 / 0.0630** |
| LiSA, spatial head | 31.459 / 0.9354 / 0.0622 |
| LiSA, compact head | 31.211 / 0.9350 / 0.0632 |
| PORT directional_port_v1 | 30.578 / 0.9316 / 0.0704 |
| SSD-GS (100k steps) | 29.541 / 0.9115 / 0.0787 |
| GS³ (100k steps) | 28.939 / 0.9106 / 0.0856 |

Evidence for the claims:

- **Light-space transport (claim 2)**: removing it costs 0.66 dB on average and 2.67 dB on bunny; probes show it carrying
  30–49% of radiance on subsurface/fur materials and ~9% on metal.
- **Deferred light-space visibility (claims 1, 3)**: +0.11 dB over per-Gaussian deep shadows with ~1.35x faster training;
  the learned residual over the moment test adds +0.18 dB (+0.87 on bunny). Crisp, correctly placed floor shadows on GS3.
- **Light generalization**: on held-out light-angle groups (novelty up to ~16°) LiSA beats `directional_port_v1` by
  +1.6–1.8 dB (dragon +2.76, bunny +2.53, Translucent +1.55); the official test lights are within 0–2° of training lights,
  so this split, not the test table, carries the generalization evidence.
- **SSS scenes**: bunny/dragon beat the 100k-step SSD-GS (+1.68/+0.55 dB final configuration).

Weaknesses to address before submission: brushed-metal anisotropic highlights (AnisoMetal −0.67 dB vs the default PORT on
test; tied on held-out lights); side-lit long shadows on Translucent (per-Gaussian shadows win at 30–90°); real-scene scores
dominated by test-pose calibration; single seed; six scenes. Next steps: anisotropic lobe hints in a Gaussian tangent frame,
grazing-light atlas resampling (e.g. warped/cascaded light maps), the full 18-scene benchmark with multiple seeds, and
equal-budget (100k-step) comparisons with SSD-GS/GS³.
