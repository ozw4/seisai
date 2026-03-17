from __future__ import annotations

import pytest

from seisai_engine.pipelines.common.encdec2d_cfg import build_encdec2d_kwargs
from seisai_engine.pipelines.pair.config import load_pair_train_config


def _make_pair_cfg() -> dict:
    return {
        'paths': {
            'input_segy_files': ['in.sgy'],
            'target_segy_files': ['tg.sgy'],
            'infer_input_segy_files': ['in.sgy'],
            'infer_target_segy_files': ['tg.sgy'],
            'out_dir': './out',
        },
        'dataset': {
            'max_trials': 8,
            'use_header_cache': True,
            'verbose': False,
            'primary_keys': ['ffid'],
            'secondary_key_fixed': False,
        },
        'train': {
            'seed': 42,
            'batch_size': 1,
            'epochs': 1,
            'samples_per_epoch': 1,
            'num_workers': 0,
            'use_amp': False,
            'max_norm': 1.0,
            'lr': 1.0e-4,
            'subset_traces': 8,
            'loss_scope': 'all',
            'losses': [
                {
                    'kind': 'l1',
                    'weight': 1.0,
                    'scope': 'all',
                    'params': {},
                }
            ],
        },
        'transform': {
            'time_len': 32,
        },
        'infer': {
            'seed': 43,
            'batch_size': 1,
            'max_batches': 1,
            'subset_traces': 8,
            'num_workers': 0,
        },
        'tile': {
            'tile_h': 8,
            'overlap_h': 4,
            'tiles_per_batch': 1,
            'amp': False,
            'use_tqdm': False,
        },
        'vis': {
            'out_subdir': 'vis',
            'n': 1,
            'cmap': 'seismic',
            'vmin': -3.0,
            'vmax': 3.0,
            'transpose_for_trace_time': True,
            'per_trace_norm': True,
            'per_trace_eps': 1.0e-8,
            'figsize': [8.0, 6.0],
            'dpi': 120,
        },
        'ckpt': {
            'save_best_only': True,
            'metric': 'infer_loss',
            'mode': 'min',
        },
        'model': {
            'backbone': 'resnet18',
            'in_chans': 1,
            'out_chans': 1,
            'disable_prestage_skip_indices': [],
            'disable_backbone_skip_indices': [],
        },
    }


def test_build_encdec2d_kwargs_accepts_empty_disable_skip_lists() -> None:
    model_cfg = {
        'backbone': 'resnet18',
        'disable_prestage_skip_indices': [],
        'disable_backbone_skip_indices': [],
    }

    kwargs = build_encdec2d_kwargs(model_cfg, in_chans=1, out_chans=1)

    assert kwargs['disable_prestage_skip_indices'] == ()
    assert kwargs['disable_backbone_skip_indices'] == ()


def test_load_pair_train_config_preserves_empty_disable_skip_lists() -> None:
    typed = load_pair_train_config(_make_pair_cfg())

    assert typed.model.disable_prestage_skip_indices == ()
    assert typed.model.disable_backbone_skip_indices == ()


@pytest.mark.parametrize(
    ('key', 'value', 'error_type', 'match'),
    [
        (
            'disable_prestage_skip_indices',
            [-1],
            ValueError,
            r'disable_prestage_skip_indices\[0\] must be >= 0',
        ),
        (
            'disable_backbone_skip_indices',
            [0, 0],
            ValueError,
            r'disable_backbone_skip_indices contains duplicate index 0',
        ),
        (
            'disable_prestage_skip_indices',
            ['0'],
            TypeError,
            r'disable_prestage_skip_indices\[0\] must be int',
        ),
    ],
)
def test_build_encdec2d_kwargs_rejects_invalid_disable_skip_lists(
    key: str,
    value: object,
    error_type: type[Exception],
    match: str,
) -> None:
    model_cfg = {
        'backbone': 'resnet18',
        key: value,
    }

    with pytest.raises(error_type, match=match):
        build_encdec2d_kwargs(model_cfg, in_chans=1, out_chans=1)
