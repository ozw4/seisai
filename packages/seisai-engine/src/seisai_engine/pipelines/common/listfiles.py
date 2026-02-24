from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config_io import _split_key_path
from .validate_files import validate_files_exist

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    'expand_cfg_listfiles',
    'get_cfg_listfile_meta',
    'load_path_listfile',
    'load_path_listfile_with_meta',
]


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


def _parse_listfile_line(
    stripped: str,
    *,
    listfile_path: Path,
    line_no: int,
) -> tuple[str, dict[str, Any] | None]:
    if '\t' not in stripped:
        return stripped, None

    raw_path, raw_meta = stripped.split('\t', 1)
    path_token = raw_path.strip()
    meta_token = raw_meta.strip()
    if not path_token:
        msg = f'empty path token in {listfile_path}:{line_no}'
        raise ValueError(msg)
    if not meta_token:
        msg = f'empty metadata token in {listfile_path}:{line_no}'
        raise ValueError(msg)
    try:
        meta = json.loads(meta_token)
    except json.JSONDecodeError as exc:
        msg = f'invalid metadata json in {listfile_path}:{line_no}: {exc.msg}'
        raise ValueError(msg) from exc
    if not isinstance(meta, dict):
        msg = f'metadata json must be object in {listfile_path}:{line_no}'
        raise ValueError(msg)
    return path_token, meta


def load_path_listfile_with_meta(
    listfile: str | Path,
) -> tuple[list[str], list[dict[str, Any] | None]]:
    """Load a listfile and return absolute paths with optional per-line metadata."""
    listfile_path = _normalize_listfile_path(listfile)
    if not listfile_path.exists():
        raise FileNotFoundError(listfile_path)
    if not listfile_path.is_file():
        msg = f'expected file: {listfile_path}'
        raise ValueError(msg)

    lines = listfile_path.read_text(encoding='utf-8').splitlines()
    base_dir = listfile_path.parent
    paths: list[str] = []
    metas: list[dict[str, Any] | None] = []
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            continue
        raw_path, meta = _parse_listfile_line(
            stripped,
            listfile_path=listfile_path,
            line_no=int(line_no),
        )
        expanded = os.path.expandvars(raw_path)
        path = Path(expanded).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        paths.append(str(path.resolve()))
        metas.append(meta)

    if len(paths) == 0:
        msg = f'listfile is empty: {listfile_path}'
        raise ValueError(msg)

    validate_files_exist(paths)
    return paths, metas


def load_path_listfile(listfile: str | Path) -> list[str]:
    """Load a listfile (1 path per line) and return absolute paths."""
    paths, _metas = load_path_listfile_with_meta(listfile)
    return paths


def _expand_value(
    value: Any, *, key_path: str
) -> tuple[list[str], list[dict[str, Any] | None] | None]:
    if isinstance(value, list):
        if not all(isinstance(v, str) for v in value):
            msg = f'config.{key_path} must be list[str]'
            raise TypeError(msg)
        if len(value) == 0:
            msg = f'config.{key_path} must be non-empty'
            raise ValueError(msg)
        return list(value), None
    if isinstance(value, (str, Path)):
        return load_path_listfile_with_meta(value)
    msg = f'config.{key_path} must be list[str] or str'
    raise TypeError(msg)


def _set_cfg_listfile_meta(
    cfg: dict,
    *,
    key_path: str,
    metas: list[dict[str, Any] | None] | None,
) -> None:
    existing = cfg.get('_listfile_meta')
    if metas is None and existing is None:
        return
    if existing is None:
        meta_root: dict[str, list[dict[str, Any] | None]] = {}
        cfg['_listfile_meta'] = meta_root
    elif isinstance(existing, dict):
        meta_root = existing
    else:
        msg = 'config._listfile_meta must be dict when present'
        raise TypeError(msg)

    if metas is None:
        meta_root.pop(key_path, None)
    else:
        meta_root[key_path] = list(metas)


def get_cfg_listfile_meta(
    cfg: dict,
    *,
    key_path: str | Sequence[str],
) -> list[dict[str, Any] | None] | None:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)
    parts = _split_key_path(key_path)
    key_path_str = '.'.join(parts)
    meta_root = cfg.get('_listfile_meta')
    if meta_root is None:
        return None
    if not isinstance(meta_root, dict):
        msg = 'config._listfile_meta must be dict'
        raise TypeError(msg)
    metas = meta_root.get(key_path_str)
    if metas is None:
        return None
    if not isinstance(metas, list):
        msg = f'config._listfile_meta[{key_path_str!r}] must be list'
        raise TypeError(msg)
    out: list[dict[str, Any] | None] = []
    for item in metas:
        if item is None:
            out.append(None)
            continue
        if not isinstance(item, dict):
            msg = f'config._listfile_meta[{key_path_str!r}] must contain dict or None'
            raise TypeError(msg)
        out.append(dict(item))
    return out


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
        expanded, metas = _expand_value(cur[last], key_path=key_path_str)
        cur[last] = expanded
        _set_cfg_listfile_meta(cfg, key_path=key_path_str, metas=metas)
    return cfg
