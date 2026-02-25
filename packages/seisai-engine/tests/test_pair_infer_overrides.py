from __future__ import annotations

import pytest

from seisai_engine.infer.segy2segy_cli_common import (
    merge_with_precedence as merge_with_precedence_common,
)
from seisai_engine.pipelines.pair.infer_segy2segy import (
    apply_unknown_overrides,
    merge_with_precedence,
)


def _base_cfg() -> dict:
    return {
        'paths': {
            'segy_files': ['/tmp/in.sgy'],
            'out_dir': '/tmp/out',
        },
        'infer': {
            'ckpt_path': '/tmp/best.pt',
            'allow_unsafe_override': False,
            'standardize_eps': None,
        },
        'tile': {
            'tile_h': 64,
            'overlap_h': 16,
            'tile_w': 512,
            'overlap_w': 64,
            'tiles_per_batch': 4,
            'amp': True,
            'use_tqdm': False,
        },
        'tta': [],
    }


@pytest.mark.parametrize(
    'override_token',
    [
        'model.backbone=resnet18',
        'model.in_chans=1',
    ],
)
def test_apply_unknown_overrides_rejects_unsafe_key_by_default(
    override_token: str,
) -> None:
    cfg = _base_cfg()
    with pytest.raises(ValueError, match='unsafe override key'):
        apply_unknown_overrides(
            cfg=cfg,
            unknown_overrides=[override_token],
        )


def test_apply_unknown_overrides_allows_unsafe_keys_when_enabled() -> None:
    cfg = _base_cfg()
    out = apply_unknown_overrides(
        cfg=cfg,
        unknown_overrides=[
            'infer.allow_unsafe_override=true',
            'model.backbone=resnet18',
            'model.in_chans=1',
        ],
    )
    assert out['infer']['allow_unsafe_override'] is True
    assert out['model']['backbone'] == 'resnet18'
    assert out['model']['in_chans'] == 1


def test_merge_precedence_default_then_ckpt_then_infer_then_unknown() -> None:
    default_cfg = _base_cfg()
    ckpt_cfg = {
        'tile': {
            'tile_h': 96,
            'tiles_per_batch': 8,
        },
        'infer': {
            'standardize_eps': 1.0e-6,
        },
        'tta': ['hflip'],
    }
    infer_cfg = {
        'tile': {
            'tile_h': 128,
        },
        'infer': {
            'standardize_eps': 1.0e-5,
        },
        'tta': ['hflip', 'vflip'],
    }

    merged_pair = merge_with_precedence(
        default_cfg=default_cfg,
        ckpt_cfg=ckpt_cfg,
        infer_cfg=infer_cfg,
    )
    merged_common = merge_with_precedence_common(default_cfg, ckpt_cfg, infer_cfg)
    assert merged_pair == merged_common

    out = apply_unknown_overrides(
        cfg=merged_pair,
        unknown_overrides=['tile.tile_h=256'],
    )

    assert out['tile']['tile_h'] == 256
    assert out['tile']['tiles_per_batch'] == 8
    assert out['infer']['standardize_eps'] == pytest.approx(1.0e-5)
    assert out['tta'] == ['hflip', 'vflip']
