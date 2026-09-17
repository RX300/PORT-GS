# Directional port six-scene validation — 2026-09-15

Fresh training uses `configs/validation.json`: directional_port_v1, rank64,
dir_dim4, dir_width32, feature_dim32, direct width128, 30k steps and seed0.
Selection and evaluation match the previous rank512 experiment: Cat/Pixiu,
AnisoMetal/Translucent, bunny_small/dragon_small. All official train frames are
used; fixed last.pt is evaluated on all official test frames with quantized
PSNR/SSIM and standard VGG LPIPS. This six-scene validation suite uses official
test splits, not the training-light holdout. No test-based model selection.
GPU workers 0/1, existing ssd-gs environment and CUDA12.1. Outputs, exact command
arrays, resolved config, source archive and logs are under
`runs/directional_port_validation_20260915/`. Source archive includes untracked
new modules; source_revision identifies the base Git revision plus that archive.

Preflight formula/gradient and real CUDA renderer checks passed. Training started at 08:37:37 UTC. Cat PID194812 on GPU0 and Pixiu PID194811
on GPU1 both passed step100 and showed 65% GPU utilization at startup.
All six jobs completed successfully; final results are below. Compare against `runs/rank512_validation_20260914/`.

Launch: `bash launch_validation.sh` from PORT-GS.

Checkpoint check: the new Pixiu step10000 checkpoint and legacy rank512 Pixiu
step30000 checkpoint both loaded via evaluate.load_model and rendered a real
64px training frame with finite RGB/alpha on GPU2. Evidence:
`runs/directional_port_validation_20260915/checkpoint_reload_check.json`.

## Final results


Completed: 6/6.

| Scene | Frames | PSNR ↑ | Δ PSNR | SSIM ↑ | Δ SSIM | LPIPS ↓ | Δ LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cat | 66 | 21.603310 | +0.027430 | 0.765444 | -0.001235 | 0.230311 | +0.003518 |
| Pixiu | 71 | 20.491304 | -0.049635 | 0.845282 | -0.000062 | 0.156564 | +0.001768 |
| AnisoMetal | 400 | 27.419796 | +0.533654 | 0.954502 | +0.000843 | 0.047118 | -0.000070 |
| Translucent | 400 | 28.265111 | -0.002979 | 0.960398 | +0.000296 | 0.051435 | -0.000790 |
| bunny_small | 500 | 35.849408 | -1.625855 | 0.984052 | -0.002823 | 0.023331 | +0.004092 |
| dragon_small | 500 | 35.806004 | +0.005657 | 0.974763 | -0.000257 | 0.035398 | +0.001854 |
| Scene mean | — | 28.239156 | -0.185288 | 0.914073 | -0.000540 | 0.090693 | +0.001729 |

All scenes: fresh 30k, seed0, fixed last.pt, full official test, original camera calibration.
Δ is new minus old. Higher PSNR/SSIM and lower LPIPS are better.
The old method used 512 ports; the supplied new architecture uses 64 ports and four direction channels.
This compares the requested complete architectures, not an isolated direction-only ablation.

The scene-mean metrics are PSNR28.239156, SSIM0.914073, LPIPS0.090693.
Relative to legacy rank512: PSNR−0.185288dB, SSIM−0.000540, LPIPS+0.001729.
This run does not show an overall quality improvement. AnisoMetal improves
+0.533654dB, while bunny_small regresses −1.625855dB. Other scenes are close
in PSNR, with mixed SSIM/LPIPS changes. These are single-seed comparisons of
the requested complete architectures, including the change from 512 to64 ports;
they do not isolate a cause or establish statistical significance.

Bunny's PSNR worsens in367/500 matched views; median paired delta is−1.156927dB.
Mean alpha L1 rises from0.001320607 to0.002537183. The regression covers most
views and accompanies greater alpha error; this evidence does not identify a
specific causal mechanism. No tuning or extra training was performed after test.
Details: `runs/directional_port_validation_20260915/bunny_frame_comparison.json`.

All1937 official test frames were evaluated, all six trainings reached30000,
and all six loss plots/checkpoints are present. The completion audit verifies
matching training protocol fields against the prior rank512 run and unchanged
production code relative to the launch source snapshot. Reports:
`comparison.json`, `comparison.md`, `completion_audit.json`, and per-scene
`test/metrics.json` under the run directory. Preflight and both legacy/new
checkpoint reload/render checks passed. No experiment retries were required.
