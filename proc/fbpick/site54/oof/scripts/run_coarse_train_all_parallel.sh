#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 '0 1 2' [smoke|full]" >&2
  exit 2
fi
read -r -a GPU_LIST <<< "$1"
MODE=${2:-full}
OOF_ROOT=${OOF_ROOT:-/workspace/proc/fbpick/site54/oof}
RUN_ID=${RUN_ID:-baseline_physical_center}
RUN_ROOT=${RUN_ROOT:-$OOF_ROOT/runs/$RUN_ID}
LAUNCHER_LOG_DIR="$RUN_ROOT/logs/launcher"
mkdir -p "$LAUNCHER_LOG_DIR"
pids=()
for i in 0 1 2 3 4 5; do
  fold=$(printf 'fold%02d' "$i")
  gpu=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}
  echo "[launch] $fold on GPU $gpu"
  "$OOF_ROOT/scripts/run_coarse_train_fold.sh" "$fold" "$gpu" "$MODE" > "$LAUNCHER_LOG_DIR/launcher_${fold}_${MODE}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
