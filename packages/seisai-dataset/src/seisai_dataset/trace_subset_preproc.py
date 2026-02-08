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
        return mmap[idx].astype(np.float32)  # (H0, T)

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
