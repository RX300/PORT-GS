# Direct HashGrid RGB validation — 2026-09-14

## User-requested correction

The preceding HashGrid experiment still projected features to 512 spatial
weights and pooled all Gaussian illumination into 512 RGB channels. The user
explicitly requested direct location queries instead. This experiment removes
that projection, source integral, quadrature, and material exchange fraction.
There is no rank/anchor/port-count parameter or port-start schedule.

The model queries NVIDIA `tinycudann.Encoding` only at covered pixel receivers.
It concatenates spatial features (32), interpolated material features (32),
light/view/half-direction encodings and dot product (82), encoded normalized
light position (27), log1p normalized light distance (1), and visibility (1).
The resulting 175 inputs feed four width-128 SiLU layers and a 3-channel output.

```text
receiver position -> NVIDIA HashGrid -> 32 features
material + light/view geometry + visibility -> concatenate -> RGB decoder
RGB = softplus(base + decoder_output) * light_intensity / light_scale / distance^2
```

Inverse-square distance and light intensity are explicit radiometric factors;
the decoder learns the positive position/light/view-dependent response.
Visibility is a network input rather than a hard multiplier, allowing a nonzero
learned response in shadow. This is a direct neural radiance model, not a solved
multiple-scattering integral, and it does not claim discrete energy conservation
or reversibility. It remains linear in point-light intensity before gamma.

HashGrid is active from step 1. Its NVIDIA config remains 16 levels × 2 features,
base resolution 16, final nominal resolution 2048, log2 table cap 19, FP32.
The material decoder's previous world-position Fourier input is replaced by the
queried hash features; no exchange weights or all-source mixing are computed.
The source library and local dependency path are unchanged from
[the preceding build](hashgrid_validation_20260914.md#dependency-setup).

## Protocol

The same six scenes are trained fresh at 30,000 steps, seed 0, followed by full
official test evaluation with LPIPS. Cat/Pixiu and AnisoMetal/Translucent use
512px; bunny_small/dragon_small use 256px and unit light intensity 1. Original
calibration, image losses, Gaussian initialization/refinement, and shadow start
at step 1500 remain unchanged. The architecture now trains directly throughout;
the removed step-5000 port activation is not retained as a misleading warmup.

Edit `configs/validation.json` for run/scene/training settings and
`configs/hashgrid.json` for NVIDIA encoding settings. Start from PORT-GS:

```bash
bash launch_validation.sh
```

The output root is `runs/direct_hashgrid_validation_20260914/`. The launcher uses
GPU0/1 with two workers, tmux socket
`port-validation-direct_hashgrid_validation_20260914`, session `validation`.
The manifest freezes source, Git revision, both configs and exact train/eval argv.
Historical anchor and pooled-HashGrid checkpoints use their saved source;
the direct decoder checkpoint architecture replaces them in place.

## Loss history and plots

Each scene writes `history.jsonl` at step 1, every 100 steps, and the final step.
Every loss record includes raw total `loss`, unweighted `l1`, and `loss_terms`
containing the actual weighted L1, SSIM, mask, feature regularization and camera
regularization contributions. Their sum equals total loss up to floating-point
roundoff. Images/loss weights are unchanged; camera regularization is zero in
this protocol because camera fitting is disabled.

After training, `train.py` writes `loss.png` next to `last.pt`. The upper panel
shows raw total loss and a trailing mean over 10 recorded samples (normally
about 1000 steps); the lower panel shows weighted components. These are sampled
training-frame losses, not dataset-wide means or validation losses.

To redraw during or after this run:

```bash
/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python plot_loss.py \
  runs/direct_hashgrid_validation_20260914/Real_NRHints/Cat/history.jsonl
```

`plot_loss.py` expects the new loss-terms history format. Older runs retain their
original logs. Luna Max monitoring continues with 60-second host samples and
1800-second model inspection, plus anomaly/completion inspection.

## Checks and results

Preflight evidence lives in `runs/direct_hashgrid_preflight/`. The query audit
uses actual saved Cat geometry and training lighting, checking batch-independent
queries, light-intensity scaling, zero-light behavior, nonzero shadow responses,
native HashGrid/decoder/input gradients, an optimizer update and checkpoint
roundtrip. Full-renderer derivatives and a short train/plot/reload check are
performed before the six-scene launch. All passed: the 150-step smoke run logged
steps 1/100/150, the weighted terms matched the total, loss.png was a valid PNG,
and the trained checkpoint reloaded through the full evaluation/LPIPS entrypoint.
The scene-by-scene argv comparison matched the preceding protocol except for
output/config paths and removal of rank/port-start. These smoke metrics are not
experiment results. Final metrics are pending this new run.
