from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from seisai_engine.infer.segy2segy_cli_common import cfg_hash

__all__ = ['build_lineage_payload', 'read_git_sha']


def _resolve_git_dir(repo_root: Path) -> Path:
    git_path = repo_root / '.git'
    if git_path.is_dir():
        return git_path
    if git_path.is_file():
        text = git_path.read_text(encoding='utf-8').strip()
        prefix = 'gitdir:'
        if not text.startswith(prefix):
            msg = f'unsupported .git file format: {git_path}'
            raise ValueError(msg)
        rel = text[len(prefix) :].strip()
        git_dir = Path(rel)
        if not git_dir.is_absolute():
            git_dir = (repo_root / git_dir).resolve()
        if not git_dir.is_dir():
            msg = f'git dir not found: {git_dir}'
            raise FileNotFoundError(msg)
        return git_dir
    msg = f'.git not found under repo root: {repo_root}'
    raise FileNotFoundError(msg)


def _lookup_packed_ref(*, git_dir: Path, ref_name: str) -> str:
    packed_refs = git_dir / 'packed-refs'
    if not packed_refs.is_file():
        msg = f'git ref not found: {ref_name}'
        raise FileNotFoundError(msg)

    for line in packed_refs.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('^'):
            continue
        sha, name = stripped.split(' ', maxsplit=1)
        if name == ref_name:
            return sha

    msg = f'git ref not found: {ref_name}'
    raise FileNotFoundError(msg)


def read_git_sha(repo_root: Path) -> str:
    if not isinstance(repo_root, Path):
        msg = 'repo_root must be Path'
        raise TypeError(msg)

    git_dir = _resolve_git_dir(repo_root.resolve())
    head_path = git_dir / 'HEAD'
    if not head_path.is_file():
        msg = f'git HEAD not found: {head_path}'
        raise FileNotFoundError(msg)

    head = head_path.read_text(encoding='utf-8').strip()
    if not head:
        msg = f'git HEAD is empty: {head_path}'
        raise ValueError(msg)

    if head.startswith('ref:'):
        ref_name = head[len('ref:') :].strip()
        ref_path = git_dir / ref_name
        if ref_path.is_file():
            sha = ref_path.read_text(encoding='utf-8').strip()
        else:
            sha = _lookup_packed_ref(git_dir=git_dir, ref_name=ref_name)
    else:
        sha = head

    if len(sha) < 7:
        msg = f'invalid git sha: {sha!r}'
        raise ValueError(msg)
    return sha


def build_lineage_payload(
    cfg: dict[str, Any],
    *,
    repo_root: Path,
    source_model_id: str | None,
    iter_id: int | None,
) -> np.ndarray:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    payload = {
        'iter_id': None if iter_id is None else int(iter_id),
        'source_model_id': source_model_id,
        'cfg_hash': cfg_hash(cfg),
        'git_sha': read_git_sha(repo_root),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
    )
    return np.asarray(encoded)
