# PORT-GS rank-512 validation — 2026-09-14

This run changes the learned spatial exchange-node count from 32 to 512. The
per-Gaussian `feature_dim` remains 32. Existing rank-32 checkpoints, manifests,
logs and metrics are historical artifacts and are not overwritten.

The fixed six-scene subset has exactly two scenes from each supported dataset
family:

| Family | Scenes | Resolution / background |
| --- | --- | --- |
| `Real_NRHints` | `Cat`, `Pixiu` | 512 / black |
| `Synthetic_GS3` | `AnisoMetal`, `Translucent` | 512 / white |
| `Synthetic_SSS-GS` | `bunny_small`, `dragon_small` | 256 / black |

The selection is declared before training in
[`manifest.json`](../../runs/rank512_validation_20260914/manifest.json). The
run-local source snapshot is
[`port_gs_source_rank512.tar`](../../runs/rank512_validation_20260914/port_gs_source_rank512.tar);
the inherited rank-32 archive is retained as `protocol_source_archive` for
historical context. The selection keeps two materially different scenes per family and has complete train/test
metadata plus prior PORT-GS references for comparison. Each job trains from a
fresh initialization on all official train frames for 30,000 steps with seed 0,
then evaluates the fixed `last.pt` on the complete official test split using
original calibration and LPIPS.

The generator and launcher are:

```bash
/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python \
  make_validation_manifest.py
./launch_rank512_validation.sh
```

The launcher uses the existing `ssd-gs` environment, CUDA 12.1, GPU workers 0
and 1, and tmux socket `port-rank512-validation` / session `rank512`. Fresh
outputs and phase logs are under
`runs/rank512_validation_20260914/<family>/<scene>/`; scheduler state is in
`status.json` and the queue log is `queue.log`.

The requested half-hour Luna Max monitor should use this manifest and its own
run-specific record paths:

```bash
/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python \
  hourly_monitor.py \
  --manifest runs/rank512_validation_20260914/manifest.json \
  --record runs/rank512_validation_20260914/hourly_monitor.jsonl \
  --latest-report runs/rank512_validation_20260914/luna_latest.txt \
  --action-required runs/rank512_validation_20260914/action_required.md \
  --interval 60 --luna-interval 1800 \
  --profile port_nodes512_validation \
  --codex /home/wenxiao-z/.local/bin/codex
```

The monitor is separate from the scheduler; local state is sampled every 60
seconds, while Luna Max runs at startup and every 1,800 seconds (30 minutes),
or immediately when an anomaly or completion transition occurs. The profile
is read-only and never changes the manifest, algorithm, outputs, retries, or
processes. The active pane is `rank512:monitor` on tmux socket
`port-rank512-validation`.

## Launch evidence

The manifest passed a six-job structural check before launch: each selected
train command contains `--rank 512` and `--feature-dim 32`. A small CUDA
forward/backward check also produced a finite loss (`0.3574461`) and a nonzero
gradient on the 512 anchor centers. At 2026-09-14 06:12:29 UTC, the scheduler
started Cat on GPU 0 (PID 142716) and Pixiu on GPU 1 (PID 142717). Both printed
their real step-100 startup row and continued writing progress; the saved
configs resolve `rank: 512`. The queue is therefore running from fresh
initializations, rather than being inferred from a launcher process alone.
