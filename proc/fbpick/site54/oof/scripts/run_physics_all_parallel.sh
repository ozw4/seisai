#!/usr/bin/env bash
set -euo pipefail
# Physics is CPU/IO oriented. Use modest parallelism to avoid saturating storage.
JOBS="${1:-2}"
OOF_ROOT="${OOF_ROOT:-/workspace/proc/fbpick/site54/oof}"
mkdir -p "$OOF_ROOT/logs"

run_one() {
  local fold="$1"
  bash "$OOF_ROOT/scripts/run_physics_fold.sh" "$fold"
}
export OOF_ROOT
export -f run_one
printf 'fold%02d\n' 0 1 2 3 4 5 | xargs -I{} -P "$JOBS" bash -lc 'run_one "$@"' _ {}
