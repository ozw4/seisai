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
cd "$REPO_ROOT"
LOG_DIR="$RUN_ROOT/logs/$FOLD"
mkdir -p "$LOG_DIR"
if [[ -n "${CONFIG_PATH:-}" ]]; then
  CONFIGS=("$CONFIG_PATH")
else
  FOLD_CONFIG_DIR="$CONFIG_ROOT/$FOLD"
  if [[ ! -d "$FOLD_CONFIG_DIR" ]]; then
    echo "Fine infer config directory does not exist: $FOLD_CONFIG_DIR" >&2
    exit 1
  fi
  CONFIGS=()
  if [[ -f "$FOLD_CONFIG_DIR/07_fine_infer.yaml" ]]; then
    CONFIGS+=("$FOLD_CONFIG_DIR/07_fine_infer.yaml")
  fi
  while IFS= read -r config; do
    CONFIGS+=("$config")
  done < <(find "$FOLD_CONFIG_DIR" -maxdepth 1 -type f -name '07_fine_infer_*.yaml' | sort)
fi
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No fine infer configs found for $FOLD under $CONFIG_ROOT" >&2
  exit 1
fi
for CONFIG in "${CONFIGS[@]}"; do
  STEM=$(basename "$CONFIG" .yaml)
  LOG="$LOG_DIR/$STEM.log"
  echo "[run] run_id=$RUN_ID fold=$FOLD config=$CONFIG log=$LOG"
  if [[ -n "$GPU" && "$GPU" != "cpu" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" python cli/run_fbpick_fine_infer.py --config "$CONFIG" 2>&1 | tee "$LOG"
  else
    python cli/run_fbpick_fine_infer.py --config "$CONFIG" 2>&1 | tee "$LOG"
  fi
done
