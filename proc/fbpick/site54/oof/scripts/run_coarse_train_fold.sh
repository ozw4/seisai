#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 fold00 [gpu_id] [smoke|full]" >&2
  exit 2
fi
FOLD=$1
GPU=${2:-}
MODE=${3:-full}
REPO_ROOT=${REPO_ROOT:-/workspace}
OOF_ROOT=${OOF_ROOT:-/workspace/proc/fbpick/site54/oof}
CONFIG_DIR=${CONFIG_DIR:-$OOF_ROOT/configs}
if [[ "$MODE" == "smoke" ]]; then
  CONFIG="$CONFIG_DIR/config_train_fbpick_coarse_${FOLD}_smoke.yaml"
else
  CONFIG="$CONFIG_DIR/config_train_fbpick_coarse_${FOLD}.yaml"
fi
cd "$REPO_ROOT"
mkdir -p "$OOF_ROOT/logs"
LOG="$OOF_ROOT/logs/train_${FOLD}_${MODE}.log"
echo "[run] fold=$FOLD mode=$MODE config=$CONFIG log=$LOG"
if [[ -n "$GPU" && "$GPU" != "cpu" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python cli/run_fbpick_coarse_train.py --config "$CONFIG" 2>&1 | tee "$LOG"
else
  python cli/run_fbpick_coarse_train.py --config "$CONFIG" 2>&1 | tee "$LOG"
fi
