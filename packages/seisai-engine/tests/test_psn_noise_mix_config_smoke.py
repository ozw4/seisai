from __future__ import annotations

import numpy as np
import torch

from seisai_engine.pipelines.psn.train import (
    _maybe_wrap_train_dataset_with_noise_mix,
    _parse_noise_segy_files,
)


class _DummyTrainDataset:
    def __init__(self) -> None:
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return 32

    def __getitem__(self, i: int) -> dict:
        _ = int(i)
        h = 8
        w = 16
        mask = torch.ones((h,), dtype=torch.bool)
        return {
            'input': torch.zeros((1, h, w), dtype=torch.float32),
            'target': torch.zeros((3, h, w), dtype=torch.float32),
            'trace_valid': mask,
            'label_valid': mask,
        }


def test_noise_mix_not_enabled_when_noise_file_list_is_empty() -> None:
    base_ds = _DummyTrainDataset()
    cfg = {
        'paths': {
            'noise_segy_files': [],
        },
        'noise': {
            'mix_prob': 0.5,
            'mix_period': 16,
        },
    }

    wrapped = _maybe_wrap_train_dataset_with_noise_mix(
        cfg=cfg,
        ds_train_full=base_ds,
        train_transform=lambda x, rng, return_meta: (x, {}),
        subset_traces=8,
    )
    assert wrapped is base_ds


def test_noise_mix_not_enabled_when_mix_prob_is_zero() -> None:
    base_ds = _DummyTrainDataset()
    cfg = {
        'paths': {
            # 非存在パスを置いても、mix_prob=0 の時点で分岐が止まることを確認
            'noise_segy_files': ['/this/path/does/not/exist.sgy'],
        },
        'noise': {
            'mix_prob': 0.0,
            'mix_period': 16,
        },
    }

    wrapped = _maybe_wrap_train_dataset_with_noise_mix(
        cfg=cfg,
        ds_train_full=base_ds,
        train_transform=lambda x, rng, return_meta: (x, {}),
        subset_traces=8,
    )
    assert wrapped is base_ds


def test_parse_noise_segy_files_expands_by_expand_cfg_listfiles(monkeypatch) -> None:
    seen: dict[str, list[str]] = {}

    def _fake_expand_cfg_listfiles(raw_list: list[str]) -> tuple[list[str], dict]:
        seen['raw_list'] = list(raw_list)
        return ['a.sgy', 'b.sgy'], {}

    monkeypatch.setattr(
        'seisai_engine.pipelines.psn.train.expand_cfg_listfiles',
        _fake_expand_cfg_listfiles,
    )

    parsed = _parse_noise_segy_files({'noise_segy_files': ['@noise_sources.txt']})
    assert seen['raw_list'] == ['@noise_sources.txt']
    assert parsed == ['a.sgy', 'b.sgy']


def test_parse_noise_segy_files_accepts_listfile_str(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_expand_cfg_listfiles(cfg: dict, *, keys: list[str]) -> dict:
        seen['keys'] = list(keys)
        seen['raw'] = cfg['paths']['noise_segy_files']
        cfg['paths']['noise_segy_files'] = ['a.sgy', 'b.sgy']
        return cfg

    monkeypatch.setattr(
        'seisai_engine.pipelines.psn.train.expand_cfg_listfiles',
        _fake_expand_cfg_listfiles,
    )

    parsed = _parse_noise_segy_files({'noise_segy_files': 'noise_sources.txt'})
    assert seen['keys'] == ['paths.noise_segy_files']
    assert seen['raw'] == 'noise_sources.txt'
    assert parsed == ['a.sgy', 'b.sgy']
