from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config_io import _split_key_path
from .validate_files import validate_files_exist

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = ['expand_cfg_listfiles', 'load_path_listfile']


def _normalize_listfile_path(listfile: str | Path) -> Path:
    if isinstance(listfile, Path):
        raw = str(listfile)
    elif isinstance(listfile, str):
        raw = listfile
    else:
        msg = 'listfile must be str or Path'
        raise TypeError(msg)

    expanded = os.path.expandvars(raw)
    path = Path(expanded).expanduser()
    return path.resolve()


def load_path_listfile(listfile: str | Path) -> list[str]:
    """Load a listfile (1 path per line) and return absolute paths."""
    listfile_path = _normalize_listfile_path(listfile)
    if not listfile_path.exists():
        raise FileNotFoundError(listfile_path)
    if not listfile_path.is_file():
        msg = f'expected file: {listfile_path}'
        raise ValueError(msg)

    lines = listfile_path.read_text(encoding='utf-8').splitlines()
    base_dir = listfile_path.parent
    paths: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            continue
        expanded = os.path.expandvars(stripped)
        path = Path(expanded).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        paths.append(str(path.resolve()))

    if len(paths) == 0:
        msg = f'listfile is empty: {listfile_path}'
        raise ValueError(msg)

    validate_files_exist(paths)
    return paths


def _expand_value(value: Any, *, key_path: str) -> list[str]:
    if isinstance(value, list):
        if not all(isinstance(v, str) for v in value):
            msg = f'config.{key_path} must be list[str]'
            raise TypeError(msg)
        if len(value) == 0:
            msg = f'config.{key_path} must be non-empty'
            raise ValueError(msg)
        return list(value)
    if isinstance(value, (str, Path)):
        return load_path_listfile(value)
    msg = f'config.{key_path} must be list[str] or str'
    raise TypeError(msg)


def expand_cfg_listfiles(
    cfg: dict, *, keys: Iterable[str | Sequence[str]]
) -> dict:
    """Expand listfile values (str) into list[str] for specific cfg keys."""
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)
    for key_path in keys:
        parts = _split_key_path(key_path)
        cur: Any = cfg
        for key in parts[:-1]:
            if not isinstance(cur, dict):
                msg = f'config[{key}] must be dict'
                raise TypeError(msg)
            if key not in cur:
                msg = f'config missing key: {".".join(parts)}'
                raise KeyError(msg)
            cur = cur[key]
        if not isinstance(cur, dict):
            msg = f'config parent must be dict for {".".join(parts)}'
            raise TypeError(msg)
        last = parts[-1]
        if last not in cur:
            msg = f'config missing key: {".".join(parts)}'
            raise KeyError(msg)
        key_path_str = '.'.join(parts)
        cur[last] = _expand_value(cur[last], key_path=key_path_str)
    return cfg
