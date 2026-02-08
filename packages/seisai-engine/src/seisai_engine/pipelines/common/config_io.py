from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from seisai_utils.config import load_config as _load_config

__all__ = [
    'load_config',
    'resolve_cfg_paths',
    'resolve_relpath',
]


def load_config(path: str | Path) -> dict:
    """Load a YAML config using seisai_utils.config.load_config."""
    return _load_config(path)


def resolve_relpath(base_dir: str | Path, p: str) -> str:
    """Resolve a path relative to base_dir if it is not absolute."""
    if not isinstance(p, str):
        msg = 'path value must be str'
        raise ValueError(msg)
    pp = Path(p).expanduser()
    if not pp.is_absolute():
        pp = Path(base_dir) / pp
    return str(pp.resolve())


def _split_key_path(key_path: str | Sequence[str]) -> list[str]:
    if isinstance(key_path, str):
        parts = key_path.split('.')
    else:
        parts = list(key_path)
    if not parts or not all(isinstance(p, str) and p for p in parts):
        msg = 'key_path must be non-empty str or sequence[str]'
        raise ValueError(msg)
    return parts


def _resolve_value(base_dir: str | Path, value: Any) -> Any:
    if isinstance(value, str):
        return resolve_relpath(base_dir, value)
    if isinstance(value, list):
        if not all(isinstance(v, str) for v in value):
            msg = 'path list items must be str'
            raise ValueError(msg)
        return [resolve_relpath(base_dir, v) for v in value]
    msg = 'path value must be str or list[str]'
    raise ValueError(msg)


def resolve_cfg_paths(
    cfg: dict,
    base_dir: str | Path,
    *,
    keys: Iterable[str | Sequence[str]],
) -> dict:
    """Resolve explicit path keys inside cfg relative to base_dir.

    Only the provided keys are resolved; no heuristics are applied.
    """
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)
    for key_path in keys:
        parts = _split_key_path(key_path)
        cur: Any = cfg
        for key in parts[:-1]:
            if not isinstance(cur, dict):
                raise TypeError(f'config[{key}] must be dict')
            if key not in cur:
                raise KeyError(f'config missing key: {".".join(parts)}')
            cur = cur[key]
        if not isinstance(cur, dict):
            raise TypeError(f'config parent must be dict for {".".join(parts)}')
        last = parts[-1]
        if last not in cur:
            raise KeyError(f'config missing key: {".".join(parts)}')
        cur[last] = _resolve_value(base_dir, cur[last])
    return cfg
