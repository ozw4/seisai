#!/usr/bin/env python3
"""Compatibility entry point for the canonical fine OOF evaluator."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    target = (
        Path(__file__).resolve().parents[1]
        / "oof"
        / "scripts"
        / "evaluate_fine_oof.py"
    )
    print(
        "WARNING: This path is deprecated. Use "
        "proc/fbpick/site54/oof/scripts/evaluate_fine_oof.py instead.",
        file=sys.stderr,
    )
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
