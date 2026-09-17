# Directional rank512 restart — 2026-09-15

## User-requested restart

The previous in-progress rank512 experiment was stopped: scheduler199671,
Cat199677, Pixiu199678 and collector199874. Its output directory
runs/directional_port512_validation_20260915/ was deleted at the user's explicit
request. Completed directional rank64 and legacy experiment directories remain.
This experiment starts from fresh initialization in the same output path.

## Current protocol

- Representation: directional_port_v1, rank512, dir_dim4, dir_width32.
- Six scenes: Real_NRHints Cat/Pixiu, Synthetic_GS3 AnisoMetal/Translucent,
  Synthetic_SSS-GS bunny_small/dragon_small.
- 30000 steps, seed0, full official train, fixed final last.pt, full official test
  with PSNR/SSIM/standard LPIPS. GPU workers0/1, existing ssd-gs/CUDA12.1.
- shadow_start=5000 and port_start=5000.
- refine_stop=25000: stop topology refinement and opacity resets at25000;
  Gaussian geometry remains trainable through30000.
- validate_every=0: no periodic validation or intermediate checkpoints; save
  only final last.pt. Training history and final loss.png remain enabled.
- Other settings follow configs/validation.json and its family overrides.

The comparison against directional rank64 now includes changes in port count,
shadow-start timing and refinement duration. It is not a rank-only comparison.
The reporting/checkpoint schedule also differs. Test evaluation remains enabled.

Exact commands, source snapshot, resolved settings and results live in the run
folder. Launch from PORT-GS: bash launch_validation.sh.

Status: completed. Preflight passed: validate_every=0 selects only step30000;
positive intervals still work; a real3-step CUDA training run saved exactly one
final last.pt and produced the loss curve. Evidence: preflight.json in the run folder.

## Restart startup evidence

Fresh queue started at2026-09-15T13:24:24 UTC, scheduler200404. Cat PID200410
onGPU0 and Pixiu PID200411 onGPU1 passed step100 and reached800/1000 at the
startup check; GPU utilization was69%/40%. Both saved configs match the new
schedule and neither scene had any .pt file at startup. Evidence:
startup_evidence.json. The results pane runs the updated collector, which
checks final-only checkpoints and explicitly records the schedule differences
against rank64 before writing comparison.json/comparison.md. Final queue results follow below.

## Final comparison

# 512方向端口新训练计划与原64方向端口对照

已完成：6/6；队列状态：completed。

| 场景 | 帧数 | PSNR ↑ | ΔPSNR | SSIM ↑ | ΔSSIM | LPIPS ↓ | ΔLPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cat | 66 | 21.536184 | -0.067126 | 0.766465 | +0.001022 | 0.230405 | +0.000094 |
| Pixiu | 71 | 20.578249 | +0.086945 | 0.845556 | +0.000274 | 0.156690 | +0.000126 |
| AnisoMetal | 400 | 28.297312 | +0.877517 | 0.958092 | +0.003590 | 0.045509 | -0.001609 |
| Translucent | 400 | 29.011980 | +0.746869 | 0.964224 | +0.003826 | 0.052495 | +0.001060 |
| bunny_small | 500 | 39.045105 | +3.195697 | 0.988939 | +0.004888 | 0.018330 | -0.005001 |
| dragon_small | 500 | 36.071024 | +0.265020 | 0.976341 | +0.001578 | 0.034777 | -0.000621 |
| 场景均值 | — | 29.089976 | +0.850820 | 0.916603 | +0.002530 | 0.089701 | -0.000992 |

差值为512方向端口减64方向端口。每场景从头训练30k步、seed0，固定last.pt，完整官方test。
汇总前核对了相同测试帧、训练配置（rank、shadow_start、refine_stop、validate_every及output不同）、最终步数、有限loss/指标及checkpoint/曲线文件。
这是单seed的模型与训练计划共同变化对比，不能将差异单独归因于端口数量；没有根据测试结果调整模型。
