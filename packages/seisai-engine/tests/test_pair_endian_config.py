from __future__ import annotations

import pytest

from seisai_engine.pipelines.pair.config import _load_dataset_cfg


def _base_dataset_cfg() -> dict:
    return {
        'max_trials': 16,
        'use_header_cache': True,
        'verbose': False,
        'primary_keys': ['ffid'],
        'secondary_key_fixed': False,
        'waveform_mode': 'eager',
    }


def test_pair_dataset_cfg_defaults_endian_to_big() -> None:
    cfg = _load_dataset_cfg(_base_dataset_cfg())
    assert cfg.train_input_endian == 'big'
    assert cfg.train_target_endian == 'big'
    assert cfg.infer_input_endian == 'big'
    assert cfg.infer_target_endian == 'big'


def test_pair_dataset_cfg_accepts_little_endian() -> None:
    raw = _base_dataset_cfg()
    raw['train_input_endian'] = 'little'
    raw['train_target_endian'] = 'little'
    raw['infer_input_endian'] = 'little'
    raw['infer_target_endian'] = 'little'
    cfg = _load_dataset_cfg(raw)
    assert cfg.train_input_endian == 'little'
    assert cfg.train_target_endian == 'little'
    assert cfg.infer_input_endian == 'little'
    assert cfg.infer_target_endian == 'little'


def test_pair_dataset_cfg_rejects_invalid_endian() -> None:
    raw = _base_dataset_cfg()
    raw['infer_target_endian'] = 'auto'
    with pytest.raises(
        ValueError, match='dataset.infer_target_endian must be "big" or "little"'
    ):
        _load_dataset_cfg(raw)
