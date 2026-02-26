from __future__ import annotations

from collections.abc import Sequence

__all__ = ['split_key_path']


def split_key_path(key_path: str | Sequence[str]) -> list[str]:
    parts = key_path.split('.') if isinstance(key_path, str) else list(key_path)
    if not parts or not all(isinstance(p, str) and p for p in parts):
        msg = 'key_path must be non-empty str or sequence[str]'
        raise ValueError(msg)
    return parts
