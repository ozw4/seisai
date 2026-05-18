#!/usr/bin/env bash
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_ROOT="/workspace/proc/fbpick/site54/oof"
RUN_ID="${RUN_ID:-baseline_physical_center}"
RUN_ROOT="${RUN_ROOT:-$DEST_ROOT/runs/$RUN_ID}"
mkdir -p "$DEST_ROOT/scripts" "$RUN_ROOT/logs/launcher"

if [ ! -d "$DEST_ROOT/fold_lists" ]; then
  cp -R "$SRC_DIR/fold_lists" "$DEST_ROOT/fold_lists"
else
  echo "[install] fold lists already exist: $DEST_ROOT/fold_lists"
fi
cp "$SRC_DIR"/scripts/*.py "$DEST_ROOT/scripts/"
cp "$SRC_DIR"/scripts/*.sh "$DEST_ROOT/scripts/"
chmod +x "$DEST_ROOT/scripts"/*.sh

echo "[install] installed physics OOF runner to $DEST_ROOT"
