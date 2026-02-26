from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np


def normalize_for_json(x: object) -> object:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, tuple):
        return [normalize_for_json(v) for v in x]
    if isinstance(x, list):
        return [normalize_for_json(v) for v in x]
    if isinstance(x, dict):
        return {str(k): normalize_for_json(v) for k, v in x.items()}
    if isinstance(x, np.generic):
        return x.item()
    return x


def cfg_hash(cfg: object) -> str:
    if not is_dataclass(cfg):
        msg = f'cfg must be a dataclass instance, got {type(cfg).__name__}'
        raise TypeError(msg)
    normalized = normalize_for_json(asdict(cfg))
    dumped = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(dumped.encode('utf-8')).hexdigest()


def read_git_sha(repo_root: Path) -> str:
    root = Path(repo_root).expanduser().resolve()
    head_path = root / '.git' / 'HEAD'
    if not head_path.is_file():
        return ''

    head_text = head_path.read_text(encoding='utf-8').strip()
    if head_text.startswith('ref: '):
        ref_rel = head_text[5:].strip()
        if ref_rel == '':
            return ''
        ref_path = root / '.git' / ref_rel
        if not ref_path.is_file():
            return ''
        sha = ref_path.read_text(encoding='utf-8').strip()
        if sha == '':
            return ''
        return sha[:12]

    if head_text == '':
        return ''
    return head_text[:12]


def lineage_npz_payload(
    *,
    iter_id: int | None,
    source_model_id: str | None,
    cfg_hash: str,
    git_sha: str,
    seed_kind: str | None = None,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        'iter_id': np.asarray(-1 if iter_id is None else int(iter_id), dtype=np.int32),
        'source_model_id': np.asarray('' if source_model_id is None else str(source_model_id)),
        'cfg_hash': np.asarray(str(cfg_hash)),
        'git_sha': np.asarray(str(git_sha)),
    }
    if seed_kind is not None:
        payload['seed_kind'] = np.asarray(str(seed_kind))
    return payload


__all__ = [
    'cfg_hash',
    'lineage_npz_payload',
    'normalize_for_json',
    'read_git_sha',
]
