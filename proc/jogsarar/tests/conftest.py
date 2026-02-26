from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    # Make sibling imports work (e.g. `import jogsarar_shared`).
    jogsarar_dir = Path(__file__).resolve().parents[1]  # proc/jogsarar
    p = str(jogsarar_dir)
    if p not in sys.path:
        sys.path.insert(0, p)


def _ensure_jogsarar_on_syspath() -> None:
    """Allow tests to import proc/jogsarar modules via file-execution style imports.

    The jogsarar scripts use sibling imports (e.g. `import jogsarar_shared`).
    Adding `proc/jogsarar` to sys.path mirrors the expected runtime environment.
    """
    repo_root = Path(__file__).resolve().parents[1]
    jogsarar_dir = repo_root / 'proc' / 'jogsarar'
    if not jogsarar_dir.is_dir():
        return

    p = str(jogsarar_dir)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_jogsarar_on_syspath()
