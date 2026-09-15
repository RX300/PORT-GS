# Residual HashGrid validation — 2026-09-15

## Architecture change

The user requested a residual network after the queried HashGrid features.
Replace the plain RGB MLP in the canonical direct-query implementation with:

```text
175 inputs -> Linear(175,128) -> SiLU
           -> ResidualBlock(128) -> ResidualBlock(128)
           -> Linear(128,3)

ResidualBlock(h) = SiLU(h + Linear2(SiLU(Linear1(h))))
RGB = softplus(base + decoder_output) * intensity / light_scale / distance^2
```

Each residual branch contains two 128→128 affine layers with an identity skip.
The configured decoder has one input projection, four branch affine layers and
an RGB output. The former plain decoder had four hidden affine layers total;
this experiment changes decoder depth as well as adding skip connections.
It is not a parameter-count-matched skip-only ablation.

The 175 inputs retain queried NVIDIA HashGrid features, material features,
encoded light/view geometry, distance and visibility. HashGrid uses 16 levels ×
2 features, base16 to nominal2048 per axis, table cap2^19, and the same FP32
NVIDIA tiny-cuda-nn build. There are no 512-channel mixtures, global source
pooling or delayed port activation. Visibility remains an input, while intensity
and inverse-square falloff remain explicit output factors. No conservation or
reversibility claim is made for this direct learned response.

The block count is `train.residual-blocks` in `configs/validation.json` (default2),
and width remains128. Checkpoints save the block count and
`representation=residual_hashgrid_rgb`; the evaluator restores those exact
settings. Older plain/pooled/anchor checkpoints use their archived source.

## Protocol and baseline

Reuse the exact preceding six-scene protocol: fresh training, seed0, 30000steps,
same refinement/shadow schedules, losses, environment, original calibration and
complete official test split with LPIPS.

| Family | Scenes | Resolution/background |
| --- | --- | --- |
| Real_NRHints | Cat, Pixiu | 512 / black |
| Synthetic_GS3 | AnisoMetal, Translucent | 512 / white |
| Synthetic_SSS-GS | bunny_small, dragon_small | 256 / black, unit light1 |

The immediately preceding plain direct-query run completed at 30k/seed0.
Its scene-mean PSNR/SSIM/LPIPS was 27.264570 / 0.904588 / 0.097787.
Pixiu regressed to13.157520dB and its loss plateaued near0.1. This motivates
testing the residual decoder; it does not establish the cause of the regression
or guarantee that residual connections will resolve it.

## Run and outputs

From PORT-GS, after the existing local tiny-cuda-nn setup:

```bash
bash launch_validation.sh
```

The launcher freezes `configs/validation.json`, `configs/hashgrid.json`, a source
snapshot, Git revision and full commands into
`runs/residual_hashgrid_validation_20260915/`. GPU workers0/1 run Cat/Pixiu first,
then the four synthetic scenes. tmux socket:
`port-validation-residual_hashgrid_validation_20260915`, session `validation`.

Each scene retains `history.jsonl`, `last.pt`, `test/metrics.json`, and `loss.png`.
Total and weighted component losses are logged at step1, every100steps and the
final step. Training automatically writes the full loss curve at completion.
To redraw a running or completed curve:

```bash
/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python plot_loss.py \
  runs/residual_hashgrid_validation_20260915/Real_NRHints/Cat/history.jsonl
```

Luna Max monitoring keeps the prior cadence: host checks every60seconds, model
inspection every1800seconds and on anomaly/completion, using `port_validation`.

## Verification and results

Evidence is written to `runs/residual_hashgrid_preflight/`: the query audit checks
both affine layers in each residual branch, native HashGrid/input gradients,
independence from query batch composition, intensity scaling, zero-light behavior
and checkpoint roundtrip. A 150-step real Cat run checks training and loss plots;
full receiver gradients and evaluator reload are checked before the long run.
All preflight checks passed, including both branch layers in both blocks, full
receiver derivatives, evaluator/LPIPS reload, and valid loss.png output. Loss
records at steps1/100/150 summed correctly across weighted terms. A six-scene
argv audit found only the added residual-blocks argument and new output/config
paths; the frozen NVIDIA encoding JSON also matched the preceding run exactly.
Formal six-scene results are pending the new experiment.

## Launch evidence

Source revision `cdba5d2`. Queue PID180749 started Cat PID180755 on GPU0 and
Pixiu PID180756 on GPU1. Both emitted step100 startup confirmation and continued
to step300 with finite total/component losses. Four synthetic scenes remain
queued. Both use the saved residual_blocks=2 configuration from fresh training.
Host verification at 50seconds showed both training children alive, GPU0
79%/5693MiB and GPU1 69%/5737MiB. Monitor PID181035 is live in `validation:monitor`
on the same socket; its first Luna Max call completed successfully at
2026-09-15 02:49:18 UTC with no runtime anomalies. It polls the host every60seconds
and invokes Luna every1800seconds and on anomaly/completion. Latest report:
`runs/residual_hashgrid_validation_20260915/luna_latest.txt`.
