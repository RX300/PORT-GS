# 60k experiment cancelled — 2026-09-16

The user explicitly requested stopping the current experiment, deleting its
output directory, and restoring the training budget to30000 steps.

Stopped the60k scheduler, result collector and active training processes on
GPU1/3. Deleted `runs/directional_port512_60k_validation_20260916/`, including
its models, logs, metrics, source snapshot and run-local scheduling helpers.
The dedicated tmux session was removed. This was a cancellation, not completion
of the six-scene60k experiment.

The completed30k outputs in `runs/directional_port512_validation_20260915/`
and earlier experiments are preserved. Both train.py's default and
configs/validation.json now use30000 steps. Other settings remain unchanged:
rank512, shadow_start=port_start=5000, refine_stop=25000, validate_every=0.
No replacement experiment was launched. The next configured experiment name is
`directional_port512_validation_20260916`, with up to two free GPUs (configured1/3).
