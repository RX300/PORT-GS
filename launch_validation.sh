#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

PROJECT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
PYTHON=/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python
CONFIG="${1:-$PROJECT_DIR/configs/validation.json}"
if [[ $# -gt 0 ]]; then shift; fi
RUN_NAME=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$CONFIG")
RUN_DIR="$PROJECT_DIR/runs/$RUN_NAME"
SOCKET="port-validation-$RUN_NAME"
SESSION=validation

export PATH="/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin:/usr/local/cuda-12.1/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.1
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=8
export OMP_NUM_THREADS=8

if [[ ! -f "$RUN_DIR/manifest.json" ]]; then
  "$PYTHON" "$PROJECT_DIR/make_validation_manifest.py" --config "$CONFIG"
fi
if tmux -L "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
  printf 'session already exists: %s:%s\n' "$SOCKET" "$SESSION" >&2
  exit 1
fi

printf -v QUEUE_COMMAND '%q ' "$PYTHON" -u "$PROJECT_DIR/run_benchmark.py" \
  --manifest "$RUN_DIR/manifest.json" "$@"
printf -v QUEUE_LOG '%q' "$RUN_DIR/queue.log"
tmux -L "$SOCKET" new-session -d -s "$SESSION" -n queue -c "$PROJECT_DIR" \
  "exec $QUEUE_COMMAND >> $QUEUE_LOG 2>&1"
tmux -L "$SOCKET" set-option -g remain-on-exit on
tmux -L "$SOCKET" list-panes -a -F '#{session_name}:#{window_name} pid=#{pane_pid} exited=#{pane_dead} exit_status=#{pane_dead_status}'
