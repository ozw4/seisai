#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: RUN_ID=baseline_physical_center $0 fold00 [gpu_id]" >&2
  exit 2
fi
FOLD=$1
GPU=${2:-}
REPO_ROOT=${REPO_ROOT:-/workspace}
OOF_ROOT=${OOF_ROOT:-/workspace/proc/fbpick/site54/oof}
RUN_ID=${RUN_ID:-baseline_physical_center}
RUN_ROOT=${RUN_ROOT:-$OOF_ROOT/runs/$RUN_ID}
CONFIG_ROOT=${CONFIG_ROOT:-$RUN_ROOT/configs}
CONFIG="$CONFIG_ROOT/$FOLD/02_coarse_infer.yaml"
cd "$REPO_ROOT"
LOG_DIR="$RUN_ROOT/logs/$FOLD"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/02_coarse_infer.log"
echo "[run] run_id=$RUN_ID fold=$FOLD config=$CONFIG log=$LOG"
if [[ -n "$GPU" && "$GPU" != "cpu" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python cli/run_fbpick_coarse_infer.py --config "$CONFIG" 2>&1 | tee "$LOG"
else
  python cli/run_fbpick_coarse_infer.py --config "$CONFIG" 2>&1 | tee "$LOG"
fi
