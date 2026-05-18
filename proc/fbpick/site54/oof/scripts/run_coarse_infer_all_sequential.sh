#!/usr/bin/env bash
set -euo pipefail
GPU=${1:-}
OOF_ROOT=${OOF_ROOT:-/workspace/proc/fbpick/site54/oof}
for i in 0 1 2 3 4 5; do
  fold=$(printf 'fold%02d' "$i")
  "$OOF_ROOT/scripts/run_coarse_infer_fold.sh" "$fold" "$GPU"
done
