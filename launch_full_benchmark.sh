#!/usr/bin/env bash
set -e
PROJECT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
RUN_DIR="$PROJECT_DIR/runs/full_benchmark_20260913"
PYTHON=/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python
export PATH="/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin:/usr/local/cuda-12.1/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.1
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=8
export OMP_NUM_THREADS=8
printf -v QUEUE_COMMAND '%q ' "$PYTHON" -u "$PROJECT_DIR/run_benchmark.py" "$@"
printf -v QUEUE_LOG '%q' "$RUN_DIR/queue.log"
printf -v MONITOR_COMMAND '%q ' "$PYTHON" -u "$PROJECT_DIR/hourly_monitor.py" --codex /home/wenxiao-z/.local/bin/codex
printf -v MONITOR_LOG '%q' "$RUN_DIR/hourly_monitor.log"
tmux -L port-full-benchmark new-session -d -s full -n queue -c "$PROJECT_DIR" "exec $QUEUE_COMMAND >> $QUEUE_LOG 2>&1"
tmux -L port-full-benchmark set-option -g remain-on-exit on
tmux -L port-full-benchmark new-window -d -t full -n monitor -c "$PROJECT_DIR" "sleep 60; exec $MONITOR_COMMAND >> $MONITOR_LOG 2>&1"
tmux -L port-full-benchmark list-panes -a -F '#{session_name}:#{window_name} pid=#{pane_pid} exited=#{pane_dead} exit_status=#{pane_dead_status}'
