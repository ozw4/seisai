from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from seisai_transforms import AdditiveNoiseMix, PerTraceStandardize, RandomCropOrPad

from seisai_engine.pipelines.blindtrace.build_dataset import (
    build_train_transform as build_blindtrace_train_transform,
)
from seisai_engine.pipelines.common import noise_add as noise_add_module
from seisai_engine.pipelines.common.noise_add import (
    NoiseTraceSubsetProvider,
    maybe_build_noise_add_op,
)
from seisai_engine.pipelines.pair.build_dataset import (
    build_train_transform as build_pair_train_transform,
)
from seisai_engine.pipelines.psn.build_dataset import (
    build_train_transform as build_psn_train_transform,
)


def _noise_provider_ctx() -> dict[str, object]:
    return {
        'subset_traces': 8,
        'primary_keys': ('ffid',),
        'secondary_key_fixed': False,
        'waveform_mode': 'eager',
        'segy_endian': 'big',
        'header_cache_dir': None,
        'use_header_cache': True,
    }


def _noise_augment_cfg() -> dict:
    return {
        'noise_add': {
            'segy_files': [str(Path(__file__).resolve())],
            'prob': 0.5,
            'gain_range': [0.2, 0.8],
        }
    }


def test_maybe_build_noise_add_op_none_when_unset() -> None:
    op = maybe_build_noise_add_op(
        augment_cfg={},
        subset_traces=8,
        primary_keys=('ffid',),
        secondary_key_fixed=False,
        waveform_mode='eager',
        segy_endian='big',
        header_cache_dir=None,
        use_header_cache=True,
    )
    assert op is None


def test_maybe_build_noise_add_op_rejects_gain_and_snr_together() -> None:
    with pytest.raises(ValueError, match='cannot be set at the same time'):
        maybe_build_noise_add_op(
            augment_cfg={
                'noise_add': {
                    'segy_files': [str(Path(__file__).resolve())],
                    'gain_range': [0.2, 0.8],
                    'snr_db_range': [6.0, 18.0],
                }
            },
            subset_traces=8,
            primary_keys=('ffid',),
            secondary_key_fixed=False,
            waveform_mode='eager',
            segy_endian='big',
            header_cache_dir=None,
            use_header_cache=True,
        )


def test_maybe_build_noise_add_op_rejects_missing_noise_file() -> None:
    with pytest.raises(FileNotFoundError):
        maybe_build_noise_add_op(
            augment_cfg={
                'noise_add': {
                    'segy_files': ['/path/that/does/not/exist.sgy'],
                    'gain_range': [0.2, 0.8],
                }
            },
            subset_traces=8,
            primary_keys=('ffid',),
            secondary_key_fixed=False,
            waveform_mode='eager',
            segy_endian='big',
            header_cache_dir=None,
            use_header_cache=True,
        )


def test_psn_train_transform_inserts_noise_op_before_standardize() -> None:
    cfg = {
        'transform': {'time_len': 64},
        'augment': _noise_augment_cfg(),
    }

    transform = build_psn_train_transform(cfg, noise_provider_ctx=_noise_provider_ctx())
    noise_indices = [
        i for i, op in enumerate(transform.ops) if isinstance(op, AdditiveNoiseMix)
    ]
    std_indices = [
        i
        for i, op in enumerate(transform.ops)
        if isinstance(op, PerTraceStandardize)
    ]

    assert len(noise_indices) == 1
    assert len(std_indices) == 1
    assert noise_indices[0] < std_indices[0]


def test_psn_train_transform_without_noise_add_keeps_previous_behavior() -> None:
    cfg = {'transform': {'time_len': 64}, 'augment': {}}
    transform = build_psn_train_transform(cfg)
    assert not any(isinstance(op, AdditiveNoiseMix) for op in transform.ops)


def test_blindtrace_train_transform_inserts_noise_op_before_standardize() -> None:
    transform = build_blindtrace_train_transform(
        time_len=64,
        per_trace_standardize=True,
        augment_cfg=_noise_augment_cfg(),
        noise_provider_ctx=_noise_provider_ctx(),
    )
    noise_indices = [
        i for i, op in enumerate(transform.ops) if isinstance(op, AdditiveNoiseMix)
    ]
    std_indices = [
        i
        for i, op in enumerate(transform.ops)
        if isinstance(op, PerTraceStandardize)
    ]

    assert len(noise_indices) == 1
    assert len(std_indices) == 1
    assert noise_indices[0] < std_indices[0]


def test_blindtrace_train_transform_requires_ctx_when_noise_add_is_set() -> None:
    with pytest.raises(ValueError, match='noise_provider_ctx is required'):
        build_blindtrace_train_transform(
            time_len=64,
            per_trace_standardize=True,
            augment_cfg=_noise_augment_cfg(),
            noise_provider_ctx=None,
        )


def test_pair_train_transform_inserts_noise_op_input_only() -> None:
    input_transform, target_transform = build_pair_train_transform(
        time_len=64,
        augment_cfg=_noise_augment_cfg(),
        noise_provider_ctx=_noise_provider_ctx(),
    )

    input_noise_indices = [
        i for i, op in enumerate(input_transform.ops) if isinstance(op, AdditiveNoiseMix)
    ]
    target_noise_indices = [
        i for i, op in enumerate(target_transform.ops) if isinstance(op, AdditiveNoiseMix)
    ]

    assert len(input_noise_indices) == 1
    assert len(target_noise_indices) == 0
    assert input_noise_indices[0] == len(target_transform.ops)


def test_pair_train_transform_without_noise_add_returns_distinct_transforms() -> None:
    input_transform, target_transform = build_pair_train_transform(
        time_len=64,
        augment_cfg={},
        noise_provider_ctx=None,
    )

    assert input_transform is not target_transform
    assert len(input_transform.ops) == len(target_transform.ops)
    assert not any(isinstance(op, AdditiveNoiseMix) for op in input_transform.ops)
    assert not any(isinstance(op, AdditiveNoiseMix) for op in target_transform.ops)


def test_pair_train_transform_requires_ctx_when_noise_add_is_set() -> None:
    with pytest.raises(ValueError, match='noise_provider_ctx is required'):
        build_pair_train_transform(
            time_len=64,
            augment_cfg=_noise_augment_cfg(),
            noise_provider_ctx=None,
        )


def test_noise_provider_forwards_rng_to_dataset_sample() -> None:
    class _FakeNoiseDataset:
        def __init__(self) -> None:
            self.last_rng = None

        def sample(self, *, rng=None) -> dict:
            self.last_rng = rng
            return {'x': torch.zeros((8, 64), dtype=torch.float32)}

        def close(self) -> None:
            return None

    provider = NoiseTraceSubsetProvider(
        segy_files=['dummy_noise.sgy'],
        subset_traces=8,
        primary_keys=('ffid',),
        secondary_key_fixed=False,
        waveform_mode='eager',
        segy_endian='big',
        header_cache_dir=None,
        use_header_cache=True,
        detect_cfg_overrides=None,
        max_redraw=8,
        seed=None,
    )
    fake_ds = _FakeNoiseDataset()
    provider._dataset = fake_ds
    provider._dataset_pid = int(os.getpid())
    provider._dataset_time_len = 64

    rng = np.random.default_rng(123)
    arr = provider.sample((8, 64), rng=rng)

    assert fake_ds.last_rng is rng
    assert tuple(arr.shape) == (8, 64)


def test_noise_provider_rejects_primary_keys_as_plain_string() -> None:
    with pytest.raises(TypeError, match='primary_keys'):
        NoiseTraceSubsetProvider(
            segy_files=['dummy_noise.sgy'],
            subset_traces=8,
            primary_keys='ffid',
            secondary_key_fixed=False,
            waveform_mode='eager',
            segy_endian='big',
            header_cache_dir=None,
            use_header_cache=True,
            detect_cfg_overrides=None,
            max_redraw=8,
            seed=None,
        )


def test_noise_provider_builds_dataset_with_random_crop_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _CaptureDataset:
        def __init__(self, **kwargs) -> None:
            captured['transform'] = kwargs['transform']

        def close(self) -> None:
            return None

    monkeypatch.setattr(noise_add_module, 'NoiseTraceSubsetDataset', _CaptureDataset)

    provider = NoiseTraceSubsetProvider(
        segy_files=['dummy_noise.sgy'],
        subset_traces=8,
        primary_keys=('ffid',),
        secondary_key_fixed=False,
        waveform_mode='eager',
        segy_endian='big',
        header_cache_dir=None,
        use_header_cache=True,
        detect_cfg_overrides=None,
        max_redraw=8,
        seed=None,
    )
    provider._build_dataset(target_len=64)
    transform = captured['transform']
    assert hasattr(transform, 'ops')
    assert any(isinstance(op, RandomCropOrPad) for op in transform.ops)
