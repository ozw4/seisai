from __future__ import annotations

import pytest

from seisai_engine.pipelines.blindtrace.infer_segy2segy import (
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
        'cover': {
            'mask_ratio': 0.5,
            'band_width': 1,
            'noise_std': 1.0,
            'mask_noise_mode': 'replace',
            'use_amp': True,
            'offsets': [0],
            'passes_batch': 4,
        },
    }


def test_apply_unknown_overrides_rejects_unsafe_key_by_default() -> None:
    cfg = _base_cfg()
    with pytest.raises(ValueError, match='unsafe override key'):
        apply_unknown_overrides(
            cfg=cfg,
            unknown_overrides=['model.backbone=resnet18'],
        )


def test_apply_unknown_overrides_allows_unsafe_key_when_enabled() -> None:
    cfg = _base_cfg()
    out = apply_unknown_overrides(
        cfg=cfg,
        unknown_overrides=[
            'infer.allow_unsafe_override=true',
            'model.backbone=resnet18',
        ],
    )
    assert out['infer']['allow_unsafe_override'] is True
    assert out['model']['backbone'] == 'resnet18'


def test_merge_precedence_default_then_ckpt_then_infer_then_unknown() -> None:
    default_cfg = _base_cfg()
    ckpt_cfg = {
        'tile': {
            'tile_h': 96,
            'tiles_per_batch': 8,
        },
        'cover': {
            'mask_ratio': 0.25,
        },
    }
    infer_cfg = {
        'tile': {
            'tile_h': 128,
        },
        'cover': {
            'mask_ratio': 0.75,
        },
    }
    merged = merge_with_precedence(
        default_cfg=default_cfg,
        ckpt_cfg=ckpt_cfg,
        infer_cfg=infer_cfg,
    )
    out = apply_unknown_overrides(
        cfg=merged,
        unknown_overrides=['tile.tile_h=256'],
    )

    assert out['tile']['tile_h'] == 256
    assert out['tile']['tiles_per_batch'] == 8
    assert out['cover']['mask_ratio'] == 0.75
