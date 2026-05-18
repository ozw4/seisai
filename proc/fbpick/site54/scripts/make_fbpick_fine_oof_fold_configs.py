#!/usr/bin/env python3
"""Compatibility entry point for the canonical fine OOF config generator."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    target = (
        Path(__file__).resolve().parents[1]
        / "oof"
        / "scripts"
        / "make_fine_fold_configs.py"
    )
    print(
        "WARNING: proc/fbpick/site54/scripts/make_fbpick_fine_oof_fold_configs.py "
        "is deprecated; use proc/fbpick/site54/oof/scripts/make_fine_fold_configs.py",
        file=sys.stderr,
    )
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
