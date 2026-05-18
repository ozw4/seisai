#!/usr/bin/env bash
set -euo pipefail
OOF_ROOT="${OOF_ROOT:-/workspace/proc/fbpick/site54/oof}"
for i in 0 1 2 3 4 5; do
  FOLD=$(printf 'fold%02d' "$i")
  bash "$OOF_ROOT/scripts/run_physics_fold.sh" "$FOLD"
done
