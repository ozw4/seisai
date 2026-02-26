from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.lineage import cfg_hash, lineage_npz_payload, normalize_for_json, read_git_sha


@dataclass(frozen=True)
class _DummyCfg:
    out_root: Path
    seed_values: tuple[int, int]
    options: dict[str, object]


def test_normalize_for_json_converts_path_tuple_numpy() -> None:
    raw = {
        'p': Path('/tmp/a'),
        't': (1, 2),
        'n': np.asarray(7, dtype=np.int32)[()],
    }
    normalized = normalize_for_json(raw)

    assert normalized == {'p': '/tmp/a', 't': [1, 2], 'n': 7}


def test_cfg_hash_is_stable_for_same_dataclass() -> None:
    cfg = _DummyCfg(
        out_root=Path('/tmp/out'),
        seed_values=(1, 2),
        options={'alpha': 0.1, 'flag': True},
    )
    assert cfg_hash(cfg) == cfg_hash(cfg)


def test_cfg_hash_changes_when_cfg_changes() -> None:
    cfg_a = _DummyCfg(
        out_root=Path('/tmp/out'),
        seed_values=(1, 2),
        options={'alpha': 0.1},
    )
    cfg_b = _DummyCfg(
        out_root=Path('/tmp/out2'),
        seed_values=(1, 2),
        options={'alpha': 0.1},
    )
    assert cfg_hash(cfg_a) != cfg_hash(cfg_b)


def test_read_git_sha_returns_empty_without_git(tmp_path: Path) -> None:
    assert read_git_sha(tmp_path) == ''


def test_read_git_sha_resolves_ref_head(tmp_path: Path) -> None:
    git_dir = tmp_path / '.git'
    refs_dir = git_dir / 'refs' / 'heads'
    refs_dir.mkdir(parents=True, exist_ok=True)
    (git_dir / 'HEAD').write_text('ref: refs/heads/main\n', encoding='utf-8')
    (refs_dir / 'main').write_text(
        '1234567890abcdef1234567890abcdef12345678\n',
        encoding='utf-8',
    )

    assert read_git_sha(tmp_path) == '1234567890ab'


def test_read_git_sha_returns_direct_head_prefix(tmp_path: Path) -> None:
    git_dir = tmp_path / '.git'
    git_dir.mkdir(parents=True, exist_ok=True)
    (git_dir / 'HEAD').write_text(
        'abcdefabcdefabcdefabcdefabcdefabcdefabcd\n',
        encoding='utf-8',
    )

    assert read_git_sha(tmp_path) == 'abcdefabcdef'


def test_lineage_npz_payload_shapes_and_defaults() -> None:
    payload = lineage_npz_payload(
        iter_id=None,
        source_model_id=None,
        cfg_hash='abc123',
        git_sha='',
        seed_kind='stage1',
    )

    assert set(payload.keys()) == {
        'iter_id',
        'source_model_id',
        'cfg_hash',
        'git_sha',
        'seed_kind',
    }
    assert int(np.asarray(payload['iter_id']).item()) == -1
    assert str(np.asarray(payload['source_model_id']).item()) == ''
    assert str(np.asarray(payload['cfg_hash']).item()) == 'abc123'
    assert str(np.asarray(payload['seed_kind']).item()) == 'stage1'
