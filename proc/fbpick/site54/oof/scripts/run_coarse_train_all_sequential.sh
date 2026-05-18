#!/usr/bin/env bash
set -euo pipefail
GPU=${1:-}
MODE=${2:-full}
OOF_ROOT=${OOF_ROOT:-/workspace/proc/fbpick/site54/oof}
for i in 0 1 2 3 4 5; do
  fold=$(printf 'fold%02d' "$i")
  "$OOF_ROOT/scripts/run_coarse_train_fold.sh" "$fold" "$GPU" "$MODE"
done
