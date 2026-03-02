from __future__ import annotations

import numpy as np
import torch

from seisai_engine.pipelines.psn.mix_dataset import MixWithNoiseDataset


class _DummyPsnDataset:
    def __init__(self, *, src: str, h: int = 8, w: int = 16) -> None:
        self.src = str(src)
        self.h = int(h)
        self.w = int(w)
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return 1_000

    def __getitem__(self, i: int) -> dict:
        _ = int(i)
        trace_valid = torch.ones((self.h,), dtype=torch.bool)
        return {
            'input': torch.zeros((1, self.h, self.w), dtype=torch.float32),
            'target': torch.zeros((3, self.h, self.w), dtype=torch.float32),
            'trace_valid': trace_valid,
            'label_valid': trace_valid.clone(),
            'src': self.src,
        }


def test_mix_with_noise_period_ratio_is_exact() -> None:
    base_ds = _DummyPsnDataset(src='base')
    noise_ds = _DummyPsnDataset(src='noise')
    mix = MixWithNoiseDataset(base_ds, noise_ds, p_noise=0.25, period=100)

    n_noise = 0
    for i in range(100):
        sample = mix[i]
        if sample['src'] == 'noise':
            n_noise += 1
    assert n_noise == 25


def test_mix_with_noise_rebuilds_schedule_after_rng_swap() -> None:
    base_ds = _DummyPsnDataset(src='base')
    noise_ds = _DummyPsnDataset(src='noise')
    mix = MixWithNoiseDataset(base_ds, noise_ds, p_noise=0.25, period=100)

    mix._rng = np.random.default_rng(123)
    _ = mix[0]
    schedule_a = mix._schedule.copy()

    mix._rng = np.random.default_rng(456)
    _ = mix[0]
    schedule_b = mix._schedule.copy()

    assert schedule_a.shape == (100,)
    assert schedule_b.shape == (100,)
    match_ratio = float(np.mean(schedule_a == schedule_b))
    assert match_ratio < 1.0
