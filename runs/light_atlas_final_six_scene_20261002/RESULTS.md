# LiSA six-scene results (spatial material head) and ablations

Note (2026-10-02 11:20 JST): the final configuration is the `svbrdf` head, adopted on held-out lights; see
`../light_atlas_svbrdf_six_scene_20261002/RESULTS.md`. The ablations below (spatial head) remain the component
evidence.

Config `light_atlas_final_six_scene_20261002` (variants `lisa`, `no_transport`, `gaussian_visibility`,
`moment_visibility`; 24 jobs on GPUs 2/3, started 2026-10-02 00:28 JST). Protocol, provenance and preflight:
`docs/experiments/light_atlas_six_scene_20261001.md`. 30,000 steps, seed 0, full official train and test, last
checkpoint, original test calibration, no test-time fitting. Metrics: uint8-quantized PSNR, SSIM 11x11 σ1.5, VGG LPIPS
on [-1,1]. The material head was selected on held-out training lights (`runs/light_atlas_head_selection_20261001`),
never on test frames. Exact values: `results.csv`.

## Main result (`lisa`): 6/6 completed, 0 failed, 1,937 test frames

| Method | Cat | Pixiu | AnisoMetal | Translucent | bunny_small | dragon_small | Mean PSNR | Mean SSIM | Mean LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **LiSA (spatial head)** | 25.231 / 0.8197 / 0.1645 | 26.188 / 0.8811 / **0.1019** | 27.302 / 0.9598 / **0.0365** | 31.155 / **0.9767** / 0.0321 | **40.861 / 0.9919 / 0.0144** | **38.017 / 0.9833** / 0.0235 | **31.4588** | **0.93542** | **0.06215** |
| LiSA (compact head, first run) | **25.300 / 0.8215** / 0.1659 | 26.247 / 0.8808 / 0.1024 | 26.610 / 0.9566 / 0.0383 | 31.072 / 0.9767 / 0.0319 | 40.132 / 0.9913 / 0.0167 | 37.902 / 0.9830 / 0.0238 | 31.2105 | 0.93500 | 0.06316 |
| PORT directional_port_v1 | 25.258 / 0.8207 / 0.1673 | 26.320 / 0.8812 / 0.1056 | 28.264 / 0.9581 / 0.0453 | 29.320 / 0.9651 / 0.0513 | 38.170 / 0.9879 / 0.0186 | 36.138 / 0.9766 / 0.0342 | 30.5783 | 0.93162 | 0.07038 |
| PORT surface_attention | 25.144 / 0.8208 / 0.1647 | **26.388 / 0.8832** / 0.1064 | **28.430 / 0.9602** / 0.0401 | 28.637 / 0.9666 / 0.0458 | 36.569 / 0.9855 / 0.0225 | 34.674 / 0.9706 / 0.0411 | 29.9735 | 0.93115 | 0.07009 |
| PORT neural_material | 24.217 / 0.8082 / 0.1944 | 26.138 / 0.8806 / 0.1095 | 23.365 / 0.9302 / 0.0573 | 22.878 / 0.9313 / 0.0720 | 26.639 / 0.9493 / 0.0689 | 32.401 / 0.9572 / 0.0532 | 25.9396 | 0.90948 | 0.09254 |
| SSD-GS (100k) | 18.001 / 0.7039 / 0.2481 | 23.402 / 0.8622 / 0.1189 | 28.001 / 0.9566 / 0.0394 | 31.986 / 0.9751 / **0.0302** | 38.480 / 0.9891 / 0.0148 | 37.377 / 0.9818 / **0.0208** | 29.5412 | 0.91145 | 0.07871 |
| GS³ (100k) | 19.120 / 0.7171 / 0.2318 | 23.722 / 0.8642 / 0.1170 | 27.513 / 0.9502 / 0.0585 | **32.756** / 0.9752 / 0.0417 | 34.418 / 0.9798 / 0.0319 | 36.107 / 0.9768 / 0.0324 | 28.9393 | 0.91055 | 0.08555 |
| RNG (30k+70k) | 20.990 / 0.7731 / 0.2866 | 19.952 / 0.8480 / 0.1718 | 24.316 / 0.9230 / 0.0724 | 29.292 / 0.9607 / 0.0518 | 37.870 / 0.9879 / 0.0202 | 35.264 / 0.9726 / 0.0423 | 27.9474 | 0.91088 | 0.10753 |

Cells are PSNR / SSIM / LPIPS; bold = best per column. Baselines: `../benchmarks/six_scene_baselines/results.csv`
(their real-scene numbers use fixed calibration without PORT's training-camera self-calibration; 100k-step budgets).
Single seed; no significance claim.

Per-view comparison with `directional_port_v1`: Translucent +1.835 dB (83% of views better), bunny +2.691 (95%),
dragon +1.879 (98%), Cat −0.028 (44%), Pixiu −0.132 (28%), AnisoMetal −0.962 (7%).

Highlight image proxies (`--highlights`; neutral bright local peaks of the GT, 2 px matching; image proxies, not
physical specular labels), default → LiSA: recall@2px Cat 0.458 → 0.524, Pixiu 0.343 → 0.603, AnisoMetal 0.545 → 0.445,
Translucent 0.734 → 0.876, bunny 0.881 → 0.913, dragon 0.008 → 0.578; 1–4 px peak-component hit fraction Pixiu
0.191 → 0.397, Translucent 0.318 → 0.520, dragon 0.007 → 0.448 (AnisoMetal 0.306 → 0.230).

Training time (30k steps, RTX 6000 Ada, one job per GPU): Cat 1,489 s, Pixiu 1,213, AnisoMetal 1,399, Translucent 1,469,
bunny 1,112, dragon 1,047.

## Ablations (same protocol, spatial head)

| Variant | Cat | Pixiu | AnisoMetal | Translucent | bunny_small | dragon_small | Mean PSNR | Mean SSIM | Mean LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full LiSA | 25.231 / 0.8197 / 0.1645 | 26.188 / 0.8811 / 0.1019 | 27.302 / 0.9598 / 0.0365 | 31.155 / 0.9767 / 0.0321 | 40.861 / 0.9919 / 0.0144 | 38.017 / 0.9833 / 0.0235 | 31.4588 | 0.93542 | 0.06215 |
| no light-space transport | 25.088 / 0.8153 / 0.1671 | 26.169 / 0.8811 / 0.1029 | 26.585 / 0.9533 / 0.0455 | 31.468 / 0.9755 / 0.0361 | 38.187 / 0.9881 / 0.0235 | 37.286 / 0.9800 / 0.0290 | 30.7970 | 0.93221 | 0.06736 |
| per-Gaussian visibility (`gaussian_visibility`) | 25.282 / 0.8207 / 0.1644 | 26.284 / 0.8807 / 0.1020 | 27.181 / 0.9578 / 0.0389 | 30.865 / 0.9750 / 0.0352 | 40.353 / 0.9913 / 0.0149 | 38.141 / 0.9837 / 0.0228 | 31.3510 | 0.93485 | 0.06303 |
| moment test only, no learned residual (`moment_visibility`) | 25.217 / 0.8188 / 0.1653 | 26.209 / 0.8802 / 0.1028 | 27.265 / 0.9578 / 0.0386 | 31.106 / 0.9747 / 0.0335 | 39.992 / 0.9907 / 0.0165 | 37.908 / 0.9827 / 0.0246 | 31.2829 | 0.93417 | 0.06354 |

Transport: +0.66 dB mean (bunny +2.67 with 98% of views better, dragon +0.73, AnisoMetal +0.72, Cat +0.14, Pixiu +0.02,
Translucent −0.31 dB PSNR while SSIM/LPIPS still favour transport). Without transport the remaining system (deferred
light-space visibility + spatial local head) still exceeds `directional_port_v1` (30.80 vs 30.58). With the spatial
head the local term can absorb part of the translucent appearance (see the decomposition trade-off in the experiment log),
so this ablation understates what the operator does under a structurally constrained local term.

Deferred light-space visibility vs the renderer's per-Gaussian 64-bin deep shadow (transfer unchanged): +0.11 dB mean
(bunny +0.51 with 70% of views better, Translucent +0.29, AnisoMetal +0.12 with 68%; Cat −0.05, Pixiu −0.10, dragon −0.12),
better mean SSIM and LPIPS, and faster: training 1,047–1,489 s with the atlas vs 1,538–2,017 s with per-Gaussian deep
shadows. Translucent by lighting angle: +0.89 dB for camera–light angles 0–30°, but −0.80/−0.63 dB at 30–60°/60–90° —
side lighting is a weak spot of the single light-space atlas (grazing light stretches the floor's texel footprint).

Learned visibility residual vs the classical level-0 moment test alone: +0.18 dB mean (bunny +0.87, dragon +0.11,
Translucent +0.05, AnisoMetal +0.04, Cat +0.01, Pixiu −0.02) and better SSIM/LPIPS on every synthetic scene.

Summary (six-scene mean PSNR): full 31.459; without light-space transport 30.797 (−0.66); per-Gaussian instead of
deferred light-space visibility 31.351 (−0.11, and ~1.35x slower training); moment test without learned residual
31.283 (−0.18). All 24 jobs completed, 0 failed (2026-10-02 00:28–05:20 JST).
