# LiSA (`light_atlas`, compact material head) six-scene results

6/6 jobs completed, 0 failed; 30,000 steps each; 1,937 full official test frames (66/71/400/400/500/500).
Queue ran 2026-10-01T13:10–14:18 UTC (22:10–23:18 JST) on GPUs 2/3. Protocol, provenance and preflight:
`docs/experiments/light_atlas_six_scene_20261001.md`. Metrics: uint8-quantized PSNR, SSIM 11x11 σ1.5,
VGG LPIPS on [-1,1]; last checkpoint; original test calibration; no test-time fitting. Exact values in `results.csv`.

| Method | Cat | Pixiu | AnisoMetal | Translucent | bunny_small | dragon_small | Mean PSNR | Mean SSIM | Mean LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **LiSA (this run)** | 25.300 / 0.8215 / 0.1659 | 26.247 / 0.8808 / 0.1024 | 26.610 / 0.9566 / 0.0383 | 31.072 / 0.9767 / 0.0319 | **40.132 / 0.9913** / 0.0167 | **37.902 / 0.9830** / 0.0238 | **31.2105** | **0.93500** | **0.06316** |
| PORT directional_port_v1 | 25.258 / 0.8207 / 0.1673 | 26.320 / 0.8812 / 0.1056 | 28.264 / 0.9581 / 0.0453 | 29.320 / 0.9651 / 0.0513 | 38.170 / 0.9879 / 0.0186 | 36.138 / 0.9766 / 0.0342 | 30.5783 | 0.93162 | 0.07038 |
| PORT surface_attention | 25.144 / 0.8208 / 0.1647 | 26.388 / 0.8832 / 0.1064 | 28.430 / 0.9602 / 0.0401 | 28.637 / 0.9666 / 0.0458 | 36.569 / 0.9855 / 0.0225 | 34.674 / 0.9706 / 0.0411 | 29.9735 | 0.93115 | 0.07009 |
| PORT neural_material | 24.217 / 0.8082 / 0.1944 | 26.138 / 0.8806 / 0.1095 | 23.365 / 0.9302 / 0.0573 | 22.878 / 0.9313 / 0.0720 | 26.639 / 0.9493 / 0.0689 | 32.401 / 0.9572 / 0.0532 | 25.9396 | 0.90948 | 0.09254 |
| SSD-GS (100k) | 18.001 / 0.7039 / 0.2481 | 23.402 / 0.8622 / 0.1189 | 28.001 / 0.9566 / 0.0394 | 31.986 / 0.9751 / 0.0302 | 38.480 / 0.9891 / 0.0148 | 37.377 / 0.9818 / 0.0208 | 29.5412 | 0.91145 | 0.07871 |
| GS³ (100k) | 19.120 / 0.7171 / 0.2318 | 23.722 / 0.8642 / 0.1170 | 27.513 / 0.9502 / 0.0585 | 32.756 / 0.9752 / 0.0417 | 34.418 / 0.9798 / 0.0319 | 36.107 / 0.9768 / 0.0324 | 28.9393 | 0.91055 | 0.08555 |
| RNG (30k+70k) | 20.990 / 0.7731 / 0.2866 | 19.952 / 0.8480 / 0.1718 | 24.316 / 0.9230 / 0.0724 | 29.292 / 0.9607 / 0.0518 | 37.870 / 0.9879 / 0.0202 | 35.264 / 0.9726 / 0.0423 | 27.9474 | 0.91088 | 0.10753 |

Cells are PSNR / SSIM / LPIPS. Baselines come from `../benchmarks/six_scene_baselines/results.csv`; their
real-scene numbers use fixed test calibration without PORT's training-camera self-calibration, and their
training budgets differ (100k steps). Single seed; no significance claim.

## Per-view comparison with directional_port_v1 (same 30k protocol)

| Scene | ΔPSNR | views improved | largest-gain lighting bin (camera–light angle) |
|---|---:|---:|---|
| Cat | +0.042 | 55% | 90–120°: +0.248 (n=3) |
| Pixiu | −0.073 | 31% | uniform −0.03…−0.10 |
| AnisoMetal | −1.654 | 4% | losses largest at 0–30° (−1.86): glossy metal reflections |
| Translucent | +1.752 | 80% | 0–30°: +2.57 |
| bunny_small | +1.962 | 90% | 0–30°: +2.68; 120–180°: +1.94 |
| dragon_small | +1.764 | 97% | 30–60°: +2.09 |

Diagnosis of the AnisoMetal regression: lower L1 than the default but higher MSE; squared error concentrates on
near-saturated metal reflections in front-lit views while cast shadows on the floor improve. The light-space
transport carries only ~7% of metal radiance (vs ~50% on Pixiu resin and Cat fur), so the deficit is the compact
local material head. Follow-up: `material_head` selection on held-out training lights.

## Cost

Training 1,071–1,442 s per scene (Cat 1,393, Pixiu 1,220, AnisoMetal 1,337, Translucent 1,442, bunny 1,156,
dragon 1,071), peak allocated 1.2–10.7 GiB (GS3 includes all 2,000 training images on the GPU); final
Gaussians 200k–400k.
