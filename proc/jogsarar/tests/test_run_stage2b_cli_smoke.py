from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import run_stage2b
from common.iter_layout import resolve_iter_layout


def test_iter_layout_stage2b_paths_can_be_resolved(tmp_path: Path) -> None:
    out_root = tmp_path / 'out'
    in_layout = resolve_iter_layout(out_root, iter_id=0)
    out_layout = resolve_iter_layout(out_root, iter_id=1)

    assert in_layout.stage4_out == out_root.resolve() / 'iter00' / 'stage4_pred'
    assert out_layout.stage2_out == out_root.resolve() / 'iter01' / 'stage2_win512'


def test_stage2b_yaml_loader_rejects_unknown_keys(tmp_path: Path) -> None:
    cfg_path = tmp_path / 'run_stage2b.yaml'
    cfg_path.write_text('in_path: /tmp\nunknown_key: 1\n', encoding='utf-8')

    with pytest.raises(ValueError):
        run_stage2b._load_yaml_defaults(cfg_path)


def test_stage2b_cli_validation_rejects_negative_iters() -> None:
    with pytest.raises(ValueError):
        run_stage2b._validate_args(
            argparse.Namespace(
                in_path=Path('/tmp'),
                out_root=Path('/tmp/out'),
                iter_in=-1,
                iter_out=1,
            )
        )

    with pytest.raises(ValueError):
        run_stage2b._validate_args(
            argparse.Namespace(
                in_path=Path('/tmp'),
                out_root=Path('/tmp/out'),
                iter_in=0,
                iter_out=-2,
            )
        )
