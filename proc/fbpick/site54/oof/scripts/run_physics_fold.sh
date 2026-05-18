#!/usr/bin/env bash
set -euo pipefail
FOLD="${1:?usage: RUN_ID=baseline_physical_center run_physics_fold.sh foldXX}"
REPO_ROOT="${REPO_ROOT:-/workspace}"
OOF_ROOT="${OOF_ROOT:-/workspace/proc/fbpick/site54/oof}"
RUN_ID="${RUN_ID:-baseline_physical_center}"
RUN_ROOT="${RUN_ROOT:-$OOF_ROOT/runs/$RUN_ID}"
CONFIG_ROOT="${CONFIG_ROOT:-$OOF_ROOT/configs}"
PHYSICS_CONFIG="$CONFIG_ROOT/$RUN_ID/$FOLD/03_physics.yaml"
QC_CONFIG="$CONFIG_ROOT/$RUN_ID/$FOLD/04_physics_qc.yaml"
LOG_DIR="$RUN_ROOT/logs/$FOLD"
mkdir -p "$LOG_DIR"
PHYSICS_LOG="$LOG_DIR/03_physics.log"
QC_LOG="$LOG_DIR/04_physics_qc.log"
if [ ! -f "$PHYSICS_CONFIG" ]; then
  echo "missing config: $PHYSICS_CONFIG" >&2
  exit 2
fi
if [ ! -f "$QC_CONFIG" ]; then
  echo "missing config: $QC_CONFIG" >&2
  exit 2
fi
cd "$REPO_ROOT"
echo "[run] run_id=$RUN_ID fold=$FOLD config=$PHYSICS_CONFIG log=$PHYSICS_LOG"
python cli/run_fbpick_physics_batch.py --config "$PHYSICS_CONFIG" 2>&1 | tee "$PHYSICS_LOG"
echo "[run] run_id=$RUN_ID fold=$FOLD config=$QC_CONFIG log=$QC_LOG"
python cli/run_fbpick_physics_qc.py --config "$QC_CONFIG" 2>&1 | tee "$QC_LOG"
