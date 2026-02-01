from __future__ import annotations

import numpy as np

from .config import LoaderConfig


class TraceSubsetLoader:
	def __init__(self, cfg: LoaderConfig):
		if cfg.pad_traces_to <= 0:
			raise ValueError('pad_traces_to must be positive')
		self.cfg = cfg

	# ---- split steps ----
	def load_traces(self, mmap, indices: np.ndarray) -> np.ndarray:
		idx = np.asarray(indices, dtype=np.int64)
		x = mmap[idx].astype(np.float32)  # (H0, T)
		return x

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
		x = self.pad_traces_to_H(x)
		return x
