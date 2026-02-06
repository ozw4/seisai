from __future__ import annotations

from pathlib import Path
from typing import Iterable

__all__ = ['validate_files_exist']


def validate_files_exist(files: Iterable[str | Path]) -> None:
	for p in files:
		if p is None:
			raise ValueError('file path must be non-empty')
		if isinstance(p, str) and not p.strip():
			raise ValueError('file path must be non-empty')

		path = Path(p)
		if not path.exists():
			raise FileNotFoundError(path)
		if not path.is_file():
			raise ValueError(f'expected file: {path}')
