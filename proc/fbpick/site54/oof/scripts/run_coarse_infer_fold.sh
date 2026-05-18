#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 fold00 [gpu_id]" >&2
  exit 2
fi
FOLD=$1
GPU=${2:-}
REPO_ROOT=${REPO_ROOT:-/workspace}
OOF_ROOT=${OOF_ROOT:-/workspace/proc/fbpick/site54/oof}
CONFIG_DIR=${CONFIG_DIR:-$OOF_ROOT/configs}
CONFIG="$CONFIG_DIR/config_infer_fbpick_coarse_${FOLD}_heldout.yaml"
cd "$REPO_ROOT"
mkdir -p "$OOF_ROOT/logs"
LOG="$OOF_ROOT/logs/infer_${FOLD}_heldout.log"
echo "[run] fold=$FOLD config=$CONFIG log=$LOG"
if [[ -n "$GPU" && "$GPU" != "cpu" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python cli/run_fbpick_coarse_infer.py --config "$CONFIG" 2>&1 | tee "$LOG"
else
  python cli/run_fbpick_coarse_infer.py --config "$CONFIG" 2>&1 | tee "$LOG"
fi
