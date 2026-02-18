from __future__ import annotations

import pytest

from seisai_engine.pipelines.blindtrace.train import (
    _raise_if_deprecated_time_len_keys as validate_blindtrace_time_len_keys,
)
from seisai_engine.pipelines.pair.config import load_pair_train_config
from seisai_engine.pipelines.psn.build_dataset import (
    build_infer_transform,
    build_train_transform,
)


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
        },
    }


def test_psn_rejects_deprecated_transform_target_key() -> None:
    cfg = {'transform': {'target_len': 32}}
    with pytest.raises(ValueError, match='deprecated key'):
        build_train_transform(cfg)


def test_psn_rejects_deprecated_train_time_key() -> None:
    cfg = {
        'train': {'time_len': 32},
        'transform': {'time_len': 32},
    }
    with pytest.raises(ValueError, match='transform.time_len'):
        build_infer_transform(cfg)


def test_pair_rejects_deprecated_train_time_key() -> None:
    cfg = _make_pair_cfg()
    cfg['train']['time_len'] = 32
    with pytest.raises(ValueError, match='transform.time_len'):
        load_pair_train_config(cfg)


def test_pair_rejects_deprecated_transform_target_key() -> None:
    cfg = _make_pair_cfg()
    cfg['transform'] = {'target_len': 32}
    with pytest.raises(ValueError, match='transform.time_len'):
        load_pair_train_config(cfg)


def test_blindtrace_rejects_deprecated_time_keys() -> None:
    with pytest.raises(ValueError, match='transform.time_len'):
        validate_blindtrace_time_len_keys(
            train_cfg={'time_len': 32},
            transform_cfg={'time_len': 32},
        )
    with pytest.raises(ValueError, match='transform.time_len'):
        validate_blindtrace_time_len_keys(
            train_cfg={},
            transform_cfg={'target_len': 32},
        )
