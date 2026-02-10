from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import LoaderConfig


class TraceSubsetLoader:
    def __init__(self, cfg: LoaderConfig) -> None:
        if cfg.pad_traces_to <= 0:
            msg = 'pad_traces_to must be positive'
            raise ValueError(msg)
        self.cfg = cfg

    # ---- split steps ----
    def load_traces(self, mmap, indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        if isinstance(mmap, np.ndarray):
            return mmap[idx].astype(np.float32)  # (H0, T)

        if not hasattr(mmap, '__getitem__'):
            msg = 'mmap must be np.ndarray or a trace accessor'
            raise TypeError(msg)

        if idx.size == 0:
            return np.zeros((0, 0), dtype=np.float32)

        if _is_contiguous(idx):
            sl = slice(int(idx[0]), int(idx[-1]) + 1)
            data = mmap[sl]
        else:
            data = [mmap[int(i)] for i in idx.tolist()]

        if isinstance(data, np.ndarray):
            arr = data
        else:
            data_list = list(data)
            if len(data_list) == 0:
                return np.zeros((0, 0), dtype=np.float32)
            arr = np.asarray(data_list)
            if arr.dtype == object:
                arr = np.stack(data_list, axis=0)

        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            msg = 'trace accessor must return 2D array'
            raise ValueError(msg)

        return arr.astype(np.float32)  # (H0, T)

    def pad_traces_to_H(self, x: np.ndarray) -> np.ndarray:
        H0, T = x.shape
        if self.cfg.pad_traces_to <= H0:
            return x
        pad = self.cfg.pad_traces_to - H0
        z = np.zeros((pad, T), dtype=x.dtype)
        return np.concatenate([x, z], axis=0)

    # ---- convenience (keeps old call-sites simple) ----
    def load(self, mmap, indices: np.ndarray) -> np.ndarray:
        x = self.load_traces(mmap, indices)
        return self.pad_traces_to_H(x)


def _is_contiguous(idx: np.ndarray) -> bool:
    if idx.size <= 1:
        return True
    return bool(np.all(np.diff(idx) == 1))
