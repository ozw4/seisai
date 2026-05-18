#!/usr/bin/env bash
set -euo pipefail
# Physics is CPU/IO oriented. Use modest parallelism to avoid saturating storage.
JOBS="${1:-2}"
OOF_ROOT="${OOF_ROOT:-/workspace/proc/fbpick/site54/oof}"
RUN_ID="${RUN_ID:-baseline_physical_center}"
RUN_ROOT="${RUN_ROOT:-$OOF_ROOT/runs/$RUN_ID}"
LAUNCHER_LOG_DIR="$RUN_ROOT/logs/launcher"
mkdir -p "$LAUNCHER_LOG_DIR"

run_one() {
  local fold="$1"
  bash "$OOF_ROOT/scripts/run_physics_fold.sh" "$fold" > "$LAUNCHER_LOG_DIR/launcher_${fold}_full.log" 2>&1
}
export OOF_ROOT RUN_ID RUN_ROOT LAUNCHER_LOG_DIR
export -f run_one
printf 'fold%02d\n' 0 1 2 3 4 5 | xargs -I{} -P "$JOBS" bash -lc 'run_one "$@"' _ {}
