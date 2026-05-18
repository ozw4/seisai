#!/usr/bin/env bash
set -euo pipefail
FOLD="${1:?usage: run_physics_fold.sh foldXX}"
OOF_ROOT="${OOF_ROOT:-/workspace/proc/fbpick/site54/oof}"
CONFIG="$OOF_ROOT/configs/config_run_fbpick_physics_${FOLD}_heldout.yaml"
LOG_DIR="$OOF_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/physics_${FOLD}.log"
if [ ! -f "$CONFIG" ]; then
  echo "missing config: $CONFIG" >&2
  exit 2
fi
cd /workspace
echo "[run] fold=$FOLD config=$CONFIG log=$LOG"
python cli/run_fbpick_physics_batch.py --config "$CONFIG" 2>&1 | tee "$LOG"
