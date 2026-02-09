from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

__all__ = [
    'build_data_manifest',
    'calc_data_id',
]


def _normalize_manifest(manifest: dict) -> str:
    return json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def _collect_files_from_paths(paths: dict) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for key, value in paths.items():
        if not isinstance(key, str) or not key.endswith('_files'):
            continue
        if not isinstance(value, list):
            continue
        if not all(isinstance(v, str) and v.strip() for v in value):
            msg = f'paths.{key} must be list[str] with non-empty entries'
            raise TypeError(msg)

        for raw in value:
            raw = raw.strip()
            expanded = os.path.expandvars(raw)
            path = Path(expanded).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            if not path.is_file():
                msg = f'expected file: {path}'
                raise ValueError(msg)
            stat = path.stat()
            files.append(
                {
                    'key': key,
                    'path': str(path),
                    'size_bytes': int(stat.st_size),
                    'mtime': float(stat.st_mtime),
                }
            )
    files.sort(key=lambda item: (item['path'], item['key']))
    return files


def build_data_manifest(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)
    paths = cfg.get('paths')
    if not isinstance(paths, dict):
        msg = 'cfg.paths must be dict'
        raise TypeError(msg)

    files = _collect_files_from_paths(paths)
    if len(files) == 0:
        msg = 'no *_files list entries found in cfg.paths'
        raise ValueError(msg)

    return {'files': files}


def calc_data_id(manifest: dict) -> str:
    if not isinstance(manifest, dict):
        msg = 'manifest must be dict'
        raise TypeError(msg)
    if 'files' not in manifest:
        msg = 'manifest must include "files"'
        raise KeyError(msg)

    normalized = _normalize_manifest(manifest)
    digest = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    return digest[:12]
