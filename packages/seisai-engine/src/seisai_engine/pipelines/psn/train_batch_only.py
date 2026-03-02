from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

__all__ = ['PsnTrainBatchOnlyDataset']

_REQUIRED_KEYS = ('input', 'target', 'trace_valid', 'label_valid')


class PsnTrainBatchOnlyDataset(Dataset):
    def __init__(self, base_ds) -> None:
        self.base_ds = base_ds
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        if hasattr(self.base_ds, '__len__'):
            return int(len(self.base_ds))
        return 1_000_000

    def __getitem__(self, i):
        if hasattr(self.base_ds, '_rng'):
            self.base_ds._rng = self._rng

        sample = self.base_ds[i]
        if not isinstance(sample, dict):
            msg = (
                'base_ds must return dict, '
                f'got {type(sample).__name__} from {type(self.base_ds).__name__}'
            )
            raise TypeError(msg)

        out: dict = {}
        for key in _REQUIRED_KEYS:
            if key not in sample:
                ds_name = type(self.base_ds).__name__
                msg = f'{ds_name} is missing required key: {key}'
                raise KeyError(msg)
            out[key] = sample[key]
        return out

    def close(self) -> None:
        close_fn = getattr(self.base_ds, 'close', None)
        if callable(close_fn):
            close_fn()
