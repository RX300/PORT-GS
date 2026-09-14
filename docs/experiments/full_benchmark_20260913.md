# PORT full benchmark queue — 2026-09-13

The running queue manifest contains 43 jobs: 18 PORT-GS jobs, 13 GS3 jobs,
seven SSD evaluation jobs, and five SSS-GS jobs. SSD and SSS entries were
appended after launch and are discovered at the next job boundary. PORT uses seed 0, fresh `--fit-all` models, 30,000 training steps,
fixed `last.pt`, and one complete official test render with standard LPIPS.
Each method keeps its native dataset family and evaluation contract.

The frozen source is
`runs/research_20260912/cat_r2_source.tar`. Shared data are read from
`/workspace/datasets/SSD-GS/data/` and outputs for fresh jobs are owned by
`runs/full_benchmark_20260913/<family>/<scene>/`.

## Scene inventory

| Family | Scenes | Resolution / background / gamma | Unit light | Queue state |
| --- | --- | --- | ---: | --- |
| `Real_NRHints` | Cat, CatSmall, CupFabric, Fish, FurScene, Pikachu, Pixiu | 512 / 0 / 2.2 | — | Cat reused; six fresh |
| `Synthetic_GS3` | AnisoMetal, Drums, FurBall, Hotdog, Lego, Translucent | 512 / 1 / 2.2 | — | Translucent reused; five fresh |
| `Synthetic_SSS-GS` | bunny_small, candle_small, dragon_small, soap_small, statue_small | 256 / 0 / 2.2 | 1 | bunny_small reused; four fresh |

The three reused entries point to the already confirmed artifacts in
`runs/research_20260912/full_results.json`; they do not start another train or
evaluation process:

| Scene | Existing run | PSNR | SSIM | standard LPIPS |
| --- | --- | ---: | ---: | ---: |
| Real_NRHints/Cat | `runs/research_20260912/cat_full_s0` | 21.501174 | 0.766281 | 0.227281 |
| Synthetic_GS3/Translucent | `runs/research_20260912/translucent_full_s0` | 28.304924 | 0.960368 | 0.051791 |
| Synthetic_SSS-GS/bunny_small | `runs/research_20260912/bunny_full_s0` | 37.600658 | 0.986484 | 0.019684 |

PORT fresh entries have `train` followed by `eval`; GS3 entries have
`train→render→eval`. PORT training progress is `<output>/history.jsonl`, GS3
progress is `<output>/progress.jsonl`, and each method's final metrics path is
declared explicitly. The manifest records every phase's full argv, working
directory, log path, output path, and result path using the current method
interfaces.

## Two-GPU scheduler

Start the queue only after the main task has checked GPU availability:

```bash
export PATH="/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin:/usr/local/cuda-12.1/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.1
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=8
/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python \
  /workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS/run_benchmark.py \
  --manifest /workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS/runs/full_benchmark_20260913/manifest.json
```

The scheduler starts exactly one serial worker for each manifest GPU, sets that
worker's child environment to `CUDA_VISIBLE_DEVICES=0` or `1`, and draws fresh
jobs from one shared queue. A child is launched with the manifest's exact
`argv` and `cwd`; stdout and stderr are combined into that phase's `logpath`.
GS3 phases also receive their method-owned `TORCH_EXTENSIONS_DIR` from the
manifest. The fixed budgets are PORT 30k, GS3 100k, and SSS-GS 60k; SSD jobs
only evaluate existing native checkpoints.
When a training child prints its first JSON progress row at or beyond step 100,
the scheduler records the actual child PID and row in `status.json` and emits a
`startup_confirmed` event. This is the startup checkpoint for the main task; it
does not add a smoke phase or a periodic metrics-reading loop.

The scheduler writes `status.json` at phase start, phase finish, job finish,
startup confirmation, and worker/queue termination. A nonzero child exit,
missing declared result path, malformed manifest phase, or other phase
exception is propagated: that GPU worker stops and the overall queue is marked
`failed`. The queue has no retry or fallback path. Existing configs, splits,
histories, checkpoints, and logs are preserved; fresh train outputs are
created by their owning method.

For the one-shot view used by the hourly monitor, read the status file and the
last JSON line of every declared training progress file:

```bash
/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python \
  /workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS/run_benchmark.py \
  --manifest /workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS/runs/full_benchmark_20260913/manifest.json \
  --status
```

The `--status` command performs one read and exits. Hourly sampling is active through `hourly_monitor.py`, which calls
`gpt-5.6-luna` at `max` effort and saves its analysis locally. The first
model call succeeded; subsequent samples occur every 3600 seconds.

## Appending verified jobs

The scheduler re-reads the manifest under its process lock at every job
boundary. While at least one worker is still processing the queue, the main
task may append a new job with a new `id` and `state: "pending"`; the next
boundary adds it to the same shared queue, preserving the two-GPU limit. Each
new job must carry its own method-owned output directory and an ordered `steps`
array. Each step must provide `name`, `state`, full `argv`, absolute `cwd`,
`logpath`, `output`, and `resultpath`; a train step should also provide
`progress_path` when its JSON progress is available.

The final manifest contains all 43 jobs. PORT covers its declared full family
set at 30k; GS3 covers Real_NRHints and Synthetic_GS3 at 100k; SSS-GS covers
Synthetic_SSS-GS at 60k; SSD jobs evaluate only existing SSD checkpoints in
their matching native family. MATE-GS, STRATA-GS, maniflow_gs, bakedmat-gs,
and other local projects remain outside this paper-method queue.

Each appended job carries its native method output directory and explicit
phases. An SSD comparison job is eval-only with original calibration and does
not retrain SSD or write under PORT-GS source/output paths.

The queue discovers appended jobs by `id`; it does not rewrite existing job
definitions or infer commands. If the initial queue has already drained, start
one unified run after adding the verified jobs so all methods share this one
two-worker process.

The complete machine-readable list is
`runs/full_benchmark_20260913/manifest.json`; scheduler state is written next
to it as `status.json` when the queue starts.

## Active handoff

`launch_full_benchmark.sh` supplies the required environment. The active
training children are GS3 Cat on GPU 0 and PORT CatSmall on GPU 1. Both
passed the real training startup checkpoint. The first failed CUDA PATH
attempt is retained in `failed_startup_cuda_path/`.

Monitoring output: `runs/full_benchmark_20260913/luna_latest.txt` and
`hourly_monitor.jsonl`. The read-only monitor runs independently of the chat;
it records completion or failure and then stops. It uses the documented
[Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)
with `--skip-git-repo-check` for this multi-project workspace.
