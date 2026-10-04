# `light_atlas` — LiSA light-space transport atlas

Added 2026-10-01 on branch `feature_claude_neural_relighting`. Implementation:
[`methods/light_atlas.py`](../../../methods/light_atlas.py); renderer hook in
[`renderer.py`](../../../renderer.py) (`light_space` attribute). Design rationale, survey and novelty review:
[proposal](../../research/lisa_light_space_atlas_20261001.md).

## Contract

`LightAtlasTransport` is a `TransportBase` registered as `light_atlas`. Geometry is native 3DGS; it sets
`requires_normals = True` (the renderer splats face-forward covariance normals as a receiver attribute) and
`light_space = True`, which makes the renderer

- skip the per-Gaussian deep-shadow volume (`visibility_hint`), leaving the splatted visibility channel at 1;
- pass `shadow=step >= --shadow-start` to `forward` in addition to `port_active=step >= --port-start`.

`forward(...)` returns linear foreground RGB for covered pixels, like every other method; alpha
composition and the observation transform stay in the renderer/evaluator.

## Data flow per training/evaluation image

1. **Local material** (always): `rho = softplus(b + MLP_m(f, n, enc(l), enc(v), enc(h), [enc(r)], cosines, hints, [enc(x)]))`;
   `r = 2 (n.v) n - v` and the positional encoding `enc(x)` of `(x - c) / R` exist only in the `reflect`/`spatial` heads;
   the `svbrdf` head instead uses `rho = softplus(b + m_0 + sum_j s_j(f, enc(x)) m_j)` where `m_j` come from the
   direction MLP (no explicit position) and `s_j` from a light-independent positional MLP. This limits the explicit
   positional interaction to eight basis coefficients; spatially varying latent codes and the final softplus
   still permit local appearance flexibility, so it does not guarantee a unique physical decomposition
   with `enc` 3-band Fourier encodings, cosines `(n.l, n.v, n.h, v.h, l.v)` and spherical-Gaussian highlight
   hints `exp(kappa (n.h - 1))`, `kappa in {8, 32, 128, 512}`. If neither switch is active the output is
   `E * rho`, `E = I / light_scale / r^2`.
2. **Light pass** (`light_atlas`): look-at camera from the light to `gaussians.center`; the half-angle covers
   the splatted extent of `atlas_coverage` (99.9%) of the Gaussians in front of the light, x1.05, capped at
   75 degrees. Per-Gaussian flux features `phi = softplus(MLP_f(f, b, |n.l|))` (`flux_dim` channels) and the
   normalized light depth `z = (z_light - |c - p|) / R` and `z^2` are alpha-composited at
   `atlas_resolution^2` by gsplat. Channels: `[Phi (C), M1, M2, M0]`.
3. **Pyramid**: `atlas_levels` levels, each a separable binomial blur + 2x average pooling (zero padding:
   outside the frustum is empty).
4. **Gather** (`gather`): receivers (world points from expected camera depth) are projected into light space;
   each level is sampled bilinearly. Per level k: coverage `M0`, mean `mu = M1/M0`, spread
   `s = sqrt(var + tau_k^2)` with `tau_k` the level texel size (radius units), offset `delta = z - mu`,
   `t = delta / s`, moment visibility `V_k = 1 - M0 * Phi_N(t - 3)`. Statistics vector per level:
   `[M0, clamp(t/4), clamp(4 delta), log s, V_k]`.
5. **Visibility** (`receiver_visibility`): `sigmoid(logit(V_0) + MLP_v(stats, f, n.l))`; the residual MLP is
   zero-initialized. `visibility_model=moment` uses `V_0` only; `none` uses 1. With `visibility_bound=B > 0` the
   residual is `B tanh(r / B)` (default 0: unbounded, as in all runs before 2026-10-04).
6. **Transfer** (`transfer`): `W = softplus(MLP_k(stats, f, n.l, n.v))` reshaped to `[levels, 3, flux_dim]`;
   transport radiance `sum_k sum_c W[k,:,c] Phi_k[c]`, linear in the gathered flux. `light_transport=none`
   disables it.
7. **Specular lobes** (optional, `specular=lobes`, added 2026-10-04): `s = clamp(n.l, 0) * sum_j softplus(MLP_s(f, enc(x)))_j
   * exp(kappa_j (n.h - 1))` over the head's sharpness values (RGB weight per lobe, light independent). They are shaded by
   the geometric level-0 moment test `V_0`, not by the learned visibility, so highlights keep a gradient path when the
   learned visibility or the diffuse local term collapses. Built last and only when enabled (no parameters otherwise).
8. Output `E * (V * rho + V_0 * s + transfer)` (`s = 0` unless enabled).

## Options (CLI fields)

| Option | Default | Meaning |
|---|---|---|
| `--feature-dim` | 32 | per-Gaussian latent material code |
| `--width` | 128 | width of material and transfer MLPs |
| `--flux-dim` | 8 | flux feature channels splatted from the light |
| `--atlas-resolution` | 512 | light atlas side; divisible by `2^(levels-1)` |
| `--atlas-levels` | 7 | pyramid levels (512 ... 8) |
| `--light-transport` | `atlas` | `none` = ablation without light-space transfer |
| `--visibility-model` | `neural` | `moment` = classical test only; `gaussian` = ablation with the renderer's per-Gaussian deep-shadow visibility splatted as an attribute (atlas only feeds transfer); `none` = no shadows |
| `--atlas-coverage` | 0.999 | fraction of Gaussians the light frustum must contain |
| `--visibility-bound` | 0 | bound `B` of the learned logit residual (`B tanh(r/B)`); 0 = unbounded |
| `--specular` | `none` | `lobes` adds the moment-shaded specular lobe bank (step 7) |
| `--material-head` | `compact` | local material head: `compact` (first run; default so its checkpoints load), `reflect` (+ reflection-vector encoding, 4 direction bands, 2048 hint, 4 hidden layers), `spatial` (`reflect` + 8-band positional encoding of the receiver), `svbrdf` (**final**: `reflect` whose MLP outputs 1+8 light-dependent RGB responses without position; an 8-band positional MLP gives 8 light-independent coefficients that scale the 8 basis responses, zero-initialized) |

Training-loop switches: `--shadow-start` enables visibility, `--port-start` enables transfer (both light
pass); the six-scene protocol uses 2000 for both.

## Initialization

Flux encoder output bias `softplus^-1(1)`; material head last layer N(0, 1e-3), zero bias (so `rho` starts at
`softplus(b)`); visibility residual zero; transfer head bias -7 (transport starts near zero); specular lobe weights
zero weight, bias -6 (each lobe starts at softplus(-6) = 0.0025).

## Training diagnostics

In training mode `forward` stores detached means of the last pass in `diagnostics` (`visibility`,
`visibility_residual`, `rho`, `local` = linear `E * (V rho + V_0 s)`, `specular`, `transport`); the trainer writes them
every 100 steps as `transport_stats`. These statistics exposed the early local-branch saturation documented in
[the v2 root-cause record](../../experiments/lisa_v2_root_causes_20261004.md).

## Tests

- Generic `test_methods.MethodTests` round-trip/linearity/gradient checks run `light_atlas` on CUDA float32.

- `test_methods.LightAtlasTests.test_svbrdf_head_limits_position_to_light_independent_coefficients`: no positional
  input in the direction MLP, coefficient MLP shapes, zero-initialized coefficients, finite non-negative output.
- `test_methods.LightAtlasTests.test_material_heads_and_per_gaussian_visibility_ablation`: compact layout unchanged
  (first-run checkpoints load), spatial layout, per-Gaussian visibility ablation multiplies by the splatted visibility.

- `test_methods.LightAtlasTests.test_specular_lobes_use_geometric_shadow_and_bounded_visibility`: defaults add no
  parameters (old checkpoints load), lobes vanish in the cast shadow and are unchanged when the learned visibility is
  forced to zero, and the bounded residual equals `B tanh(r/B)`.
- `test_methods.AppearanceWeightTests`: with appearance weight 0 the transport receives no gradient while opacities do.
- `test_methods.LightAtlasTests`: a floor and an occluding square under an overhead light. Checks pyramid
  sizes, statistic shapes, moment visibility in the cast shadow (< 0.05) and on lit floor/occluder (> 0.95),
  occluder-position gradients, local-only output before the light pass and linearity of the transfer in flux.
- `test_method_integration.py --methods light_atlas --optimize-cameras`: CLI train, checkpoint, reload,
  bitwise-identical re-render and CLI evaluation on a 4-frame Cat subset.

## Limitations

Lights must lie outside the scene (one perspective atlas; true for all six benchmark scenes). Light that
reaches a receiver only through regions invisible from the light is not represented. The 3-sigma bias
of the classical test leaves some acne on grazing lit surfaces, which the learned residual must absorb.
Receivers use alpha-normalized expected depth, so mixed-depth silhouette pixels gather from an average point.

## Actual atlas visualizations (final `svbrdf` checkpoints)

`diagnose_image_errors.py --light-atlas-preview --split test --atlas-frames <indices> --output <new-directory>`
exports the actual learned channels, depth statistics, seven-level pyramid and camera-space shading decomposition.
It uses the ordinary evaluator and checks that the recomputed linear local and transfer terms reconstruct its
output (`rtol=1e-5`, `atol=1e-6`). Checkpoints and datasets are read only; no fitting occurs.

Examples in `runs/light_atlas_visualization_svbrdf/` use 30k-step, seed-0 final checkpoints from
`runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/`:

- `Translucent/frame_045_atlas.png`: coverage, mean depth, depth standard deviation and all eight learned features.
- `Translucent/frame_045_pyramid.png`: coverage and feature 0 at all scales, sharing a color range across levels.
- `Translucent/frame_045_decomposition.png`: GT, final render, receiver visibility, local and transfer contributions.
- `Translucent/frame_300_*.png`: a second light/view pair (both camera and light change).
- `Pixiu/frame_018_*.png`: the real translucent-object example.

Each frame also has raw `*_buffers.npz`; commands, checkpoint, light position and visualization ranges are in
`preview.json`. Environment: existing `ssd-gs`, RTX 6000 Ada, GPUs 2/3. Commands from the project root:

```bash
CUDA_VISIBLE_DEVICES=2 /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python diagnose_image_errors.py \
  runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Synthetic_GS3/Translucent/last.pt \
  --light-atlas-preview --split test --atlas-frames 45 300 --output runs/light_atlas_visualization_svbrdf/Translucent
CUDA_VISIBLE_DEVICES=3 /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python diagnose_image_errors.py \
  runs/light_atlas_svbrdf_six_scene_20261002/lisa_svbrdf/Real_NRHints/Pixiu/last.pt \
  --light-atlas-preview --split test --atlas-frames 18 --output runs/light_atlas_visualization_svbrdf/Pixiu
```

All three exported GT/render pairs match the existing final-test PNGs byte for byte; maximum absolute
decomposition error is 2.4e-7. Flux colors are scalar heatmaps with separate 99th-percentile ranges per channel,
not object RGB or calibrated physical irradiance. `M0` is light-view coverage, not receiver visibility;
depth mean/spread are derived from `M1/M0` and `M2/M0`, masking uncovered texels. Display-encoded component
images do not add numerically: only their underlying linear radiance does. The nonlocal term is a learned
approximation and is not ground-truth isolated subsurface scattering or a full multi-bounce solution.
