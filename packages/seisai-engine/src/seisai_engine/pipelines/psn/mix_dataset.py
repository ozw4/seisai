from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

__all__ = ['MixWithNoiseDataset']


class MixWithNoiseDataset(Dataset):
    def __init__(
        self,
        base_ds,
        noise_ds,
        *,
        p_noise: float,
        period: int,
    ) -> None:
        if isinstance(p_noise, bool) or not isinstance(p_noise, (int, float)):
            msg = 'p_noise must be float in [0, 1]'
            raise TypeError(msg)
        if isinstance(period, bool) or not isinstance(period, int):
            msg = 'period must be int >= 1'
            raise TypeError(msg)
        p_noise_f = float(p_noise)
        if p_noise_f < 0.0 or p_noise_f > 1.0:
            msg = 'p_noise must be in [0, 1]'
            raise ValueError(msg)
        if int(period) < 1:
            msg = 'period must be >= 1'
            raise ValueError(msg)

        self.base_ds = base_ds
        self.noise_ds = noise_ds
        self.p_noise = p_noise_f
        self.period = int(period)
        self._rng = np.random.default_rng()
        self._rng_id_last: int | None = None
        self._schedule = np.zeros((self.period,), dtype=np.bool_)

    def __len__(self) -> int:
        return 1_000_000

    def _rebuild_schedule(self) -> None:
        n_noise = int(round(self.p_noise * float(self.period)))
        if n_noise < 0:
            n_noise = 0
        if n_noise > self.period:
            n_noise = self.period
        schedule = np.zeros((self.period,), dtype=np.bool_)
        if n_noise > 0:
            schedule[:n_noise] = True
        self._rng.shuffle(schedule)
        self._schedule = schedule
        self._rng_id_last = id(self._rng)

    def _ensure_schedule(self) -> None:
        rng_id = id(self._rng)
        if self._rng_id_last != rng_id:
            self._rebuild_schedule()

    def __getitem__(self, i):
        if hasattr(self.base_ds, '_rng'):
            self.base_ds._rng = self._rng
        if hasattr(self.noise_ds, '_rng'):
            self.noise_ds._rng = self._rng

        self._ensure_schedule()
        ii = int(i)
        idx = ii % self.period
        if bool(self._schedule[idx]):
            return self.noise_ds[ii]
        return self.base_ds[ii]

    def close(self) -> None:
        close_base = getattr(self.base_ds, 'close', None)
        close_noise = getattr(self.noise_ds, 'close', None)
        if callable(close_base):
            close_base()
        if callable(close_noise):
            close_noise()
