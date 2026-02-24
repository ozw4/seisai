#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

LOG_DIR="proc/mae/logs_opt_v2"
mkdir -p "$LOG_DIR"

GPU0=${GPU0:-0}
GPU1=${GPU1:-1}

CONFIGS=(
  proc/mae/configs/opt_v2_T00_adamw_torch_lr5e-4_wd1e-2.yaml
  proc/mae/configs/opt_v2_T01_adamw_timm_fbbnF_lr5e-4_wd1e-2.yaml
  proc/mae/configs/opt_v2_T02_adamw_timm_fbbnT_lr5e-4_wd1e-2.yaml
  proc/mae/configs/opt_v2_T03_lion_lr2p5e-4_wd1e-2.yaml
  proc/mae/configs/opt_v2_T04_lion_lr5e-4_wd1e-2.yaml
  proc/mae/configs/opt_v2_T05_lion_lr1e-3_wd1e-2.yaml
  proc/mae/configs/opt_v2_T06_adan_lr2p5e-4_wd1e-2.yaml
  proc/mae/configs/opt_v2_T07_adan_lr5e-4_wd1e-2.yaml
  proc/mae/configs/opt_v2_T08_adan_lr1e-3_wd1e-2.yaml
  proc/mae/configs/opt_v2_T09_lamb_lr2p5e-4_wd1e-2.yaml
  proc/mae/configs/opt_v2_T10_lamb_lr5e-4_wd1e-2.yaml
  proc/mae/configs/opt_v2_T11_lamb_lr1e-3_wd1e-2.yaml
  proc/mae/configs/opt_v2_T12_radam_lr2p5e-4_wd1e-2.yaml
  proc/mae/configs/opt_v2_T13_radam_lr5e-4_wd1e-2.yaml
  proc/mae/configs/opt_v2_T14_radam_lr1e-3_wd1e-2.yaml
  proc/mae/configs/opt_v2_T15_lion_lr5e-4_wd0.yaml
  proc/mae/configs/opt_v2_T16_lion_lr5e-4_wd5e-3.yaml
  proc/mae/configs/opt_v2_T17_adan_lr5e-4_wd0.yaml
  proc/mae/configs/opt_v2_T18_adan_lr5e-4_wd5e-3.yaml
)

run_one() {
  local gpu="$1"
  local cfg="$2"
  local name
  name=$(basename "$cfg" .yaml)
  local log_path="$LOG_DIR/${name}.log"

  echo "[GPU${gpu}] ${name}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python cli/run_blindtrace_train.py --config "$cfg" \
    >"$log_path" 2>&1
}

trap 'jobs -p | xargs -r kill; exit 130' INT TERM

for ((i = 0; i < ${#CONFIGS[@]}; i += 2)); do
  cfg0="${CONFIGS[$i]}"
  (run_one "$GPU0" "$cfg0") &
  pid0=$!

  if ((i + 1 < ${#CONFIGS[@]})); then
    cfg1="${CONFIGS[$i + 1]}"
    (run_one "$GPU1" "$cfg1") &
    pid1=$!

    st0=0
    st1=0
    wait "$pid0" || st0=$?
    wait "$pid1" || st1=$?

    if ((st0 != 0 || st1 != 0)); then
      echo "ERROR: job failed (st0=${st0}, st1=${st1})"
      exit 1
    fi
  else
    wait "$pid0"
  fi
done

echo "done"