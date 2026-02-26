from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

__all__ = ['validate_files_exist']


def validate_files_exist(files: Iterable[str | Path]) -> None:
    for p in files:
        if p is None:
            msg = 'file path must be non-empty'
            raise ValueError(msg)
        if isinstance(p, str) and not p.strip():
            msg = 'file path must be non-empty'
            raise ValueError(msg)

        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            msg = f'expected file: {path}'
            raise ValueError(msg)
