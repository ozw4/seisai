#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: RUN_ID=baseline_physical_center $0 fold00 [gpu_id] [smoke|full]" >&2
  exit 2
fi
FOLD=$1
GPU=${2:-}
MODE=${3:-full}
REPO_ROOT=${REPO_ROOT:-/workspace}
OOF_ROOT=${OOF_ROOT:-/workspace/proc/fbpick/site54/oof}
RUN_ID=${RUN_ID:-baseline_physical_center}
RUN_ROOT=${RUN_ROOT:-$OOF_ROOT/runs/$RUN_ID}
CONFIG_ROOT=${CONFIG_ROOT:-$OOF_ROOT/configs}
if [[ "$MODE" == "smoke" ]]; then
  CONFIG="$CONFIG_ROOT/$RUN_ID/$FOLD/06_fine_train_smoke.yaml"
  LOG_NAME="06_fine_train_smoke.log"
else
  CONFIG="$CONFIG_ROOT/$RUN_ID/$FOLD/06_fine_train.yaml"
  LOG_NAME="06_fine_train.log"
fi
cd "$REPO_ROOT"
LOG_DIR="$RUN_ROOT/logs/$FOLD"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$LOG_NAME"
echo "[run] run_id=$RUN_ID fold=$FOLD mode=$MODE config=$CONFIG log=$LOG"
if [[ -n "$GPU" && "$GPU" != "cpu" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python cli/run_fbpick_fine_train.py --config "$CONFIG" 2>&1 | tee "$LOG"
else
  python cli/run_fbpick_fine_train.py --config "$CONFIG" 2>&1 | tee "$LOG"
fi
