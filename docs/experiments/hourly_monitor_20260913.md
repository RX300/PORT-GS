# PORT-GS monitor — 2026-09-13

[hourly_monitor.py](/workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS/hourly_monitor.py)
is a persistent monitor for the manifest named
`runs/full_benchmark_20260913/manifest.json`. It performs a local snapshot
every 60 seconds and invokes `gpt-5.6-luna` immediately, every 1,800 seconds,
or sooner when a new failure state appears. The local poll checks the
scheduler PID, worker and child PIDs, job failure states, the latest progress
rows, non-finite progress values, GPU 0/1 utilization, and the tail of
manifest-declared logs for `Traceback`, `NaN`, infinity, OOM, and CUDA/runtime
error signals.

Each local snapshot records the manifest, status, queue log, active output and
result paths, active progress paths, the current running/failed phase logs,
their failure tails, and queue.log as scheduler context. queue.log signals are
not failure alerts by themselves; scheduler failure comes from status/PID.
The Luna view aggregates completed, pending, running,
and failed counts by method, then gives detailed paths, progress, and current
phase log tails only for running/failed jobs. The monitor does not mutate the scheduler, manifest,
training code, or GPU processes during a local poll. An unchanged anomaly is
represented by the same in-memory state tuple, so a persistent failure does
not cause a Luna call every 60 seconds. A changed anomaly or the regular
interval creates a new model call.

The Luna call runs with workspace write access and automatic approval:

```text
codex -C /workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting exec \
  -m gpt-5.6-luna -c model_reasoning_effort="max" \
  --approve-for-me --ephemeral \
  --skip-git-repo-check -o <luna_latest.txt> <prompt>
```

The prompt limits automatic work to a proven environment, startup, or metric
implementation fault. It requires the fixed GPU 0/1 allocation, seed,
training budget, dataset split, original calibration and metric protocol to be
preserved. It forbids blind reruns, numerical-method changes, `nan_to_num`,
defensive patches, budget or metric changes, killing healthy jobs, or
restarting the whole queue. It scopes work to jobs listed in the current
manifest: PORT-GS and the official GS3, SSS-GS, and SSD-GS entries. SSD-GS is
evaluation-only on its original test calibration; it is never retrained or
fit to the test set. Other local method projects are outside this monitor.

For a proven operational repair to a failed job, Luna must first preserve the
failure evidence and apply the smallest repair. A failed train may archive the
whole training output; a failed render/eval must preserve its training
checkpoint and archive only that failed phase output and log. Luna only then
sets `retry_requested:true` on that manifest job. The
updated scheduler consumes that marker at the next job boundary. An
unresolved numerical NaN or method problem is diagnostic evidence for the
main thread and cannot receive a retry marker. If the coordinator exits while
pending jobs remain, Luna must first verify that no training child is active
before reusing the pane's saved `python --resume` command with
`tmux -L port-full-benchmark respawn-pane -t full:queue`; healthy work is left alone and only repaired,
explicitly marked jobs are eligible for retry.

The default files are:

- `runs/full_benchmark_20260913/hourly_monitor.jsonl`: one record per local
  poll, including every model call and its result or failure.
- `runs/full_benchmark_20260913/luna_latest.txt`: the latest Luna response
  or failed-call report.
- `runs/full_benchmark_20260913/action_required.md`: current anomalies, the
  latest model-call status, and items that require main-thread judgment.

If a Luna subprocess returns a nonzero exit status, the monitor records the
CalledProcessError in the JSONL and action_required.md, then raises it. It does
not replace the last successful luna_latest.txt report.

Run it only after the parent scheduler has been launched:

```bash
/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python \
  /workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS/hourly_monitor.py \
  --manifest /workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS/runs/full_benchmark_20260913/manifest.json
```

`--interval` defaults to 60 seconds, `--luna-interval` to 1,800 seconds,
`--log-tail-lines` to 40, and `--codex` to `codex`. These options are for
inspection and deployment configuration; this implementation does not start
or stop the monitor as part of code validation.

The persistent CLI can invoke `codex exec`, but it has no direct
`collaboration.send_message` or thread-wakeup capability. The monitor
therefore makes no claim that it can wake the current chat. The main thread
reads `action_required.md` and the JSONL records; this subagent reports
implementation progress and known faults directly to the parent agent.

## Verified deployment

The active monitor passed its first Luna call at 2026-09-13T05:24:20Z and
subsequent 60-second polls without extra model calls. Routine checks use the
host snapshot only; model-side sandbox process listings can hide host PIDs.
Any process-management diagnosis requires an escalated, read-only host check.
The CLI uses `--approve-for-me`, which selects workspace-write itself; the
mutually exclusive `--sandbox` flag is omitted.

Healthy training remains with scheduler PID 82369. Per-job failure isolation
and explicit `retry_requested` consumption are staged for its next safe
resume. Latest deployment details are in the run directory `handoff.json`.
