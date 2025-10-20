# packages/seisai-transforms/src/seisai_transforms/augment.py
from __future__ import annotations

import numpy as np

from .config import FreqAugConfig, SpaceAugConfig, TimeAugConfig
from .kernels import _apply_freq_augment, _spatial_stretch_sameH, _time_stretch_poly
from .signal_ops import standardize_per_trace


class RandomFreqFilter:
	def __init__(self, cfg: FreqAugConfig = FreqAugConfig()):
		self.cfg = cfg

	def __call__(
		self,
		x_hw: np.ndarray,
		rng: np.random.Generator | None = None,
		return_meta: bool = False,
	):
		p = float(self.cfg.prob or 0.0)
		# ← ここが大事：prob=0 は無条件で No-Op！
		if p <= 0.0:
			return (x_hw, {}) if return_meta else x_hw

		r = rng.random() if rng is not None else np.random.random()
		if r >= p:
			return (x_hw, {}) if return_meta else x_hw

		y = _apply_freq_augment(
			x_hw,
			self.cfg.kinds,
			self.cfg.band,
			self.cfg.width,
			self.cfg.roll,
			self.cfg.restandardize,
			rng=rng,  # ★ rng を下に渡す（再現性◯）
		)
		return (y, {}) if return_meta else y


class RandomTimeStretch:
	def __init__(self, cfg: TimeAugConfig = TimeAugConfig()):
		self.cfg = cfg

	def __call__(
		self,
		x_hw: np.ndarray,
		rng: np.random.Generator | None = None,
		return_meta: bool = False,
	):
		p = max(0.0, min(1.0, float(self.cfg.prob)))
		if p <= 0.0:
			meta = {'factor': 1.0}
			return (x_hw, meta) if return_meta else x_hw

		r = rng or np.random.default_rng()
		if r.random() >= p:
			meta = {'factor': 1.0}
			return (x_hw, meta) if return_meta else x_hw

		f = float(r.uniform(*self.cfg.factor_range))
		L = self.cfg.target_len or x_hw.shape[1]
		y = _time_stretch_poly(x_hw, f, L)
		meta = {'factor': f}
		return (y, meta) if return_meta else y


class RandomSpatialStretchSameH:
	def __init__(self, cfg: SpaceAugConfig = SpaceAugConfig()):
		self.cfg = cfg

	def __call__(
		self,
		x_hw: np.ndarray,
		rng: np.random.Generator | None = None,
		return_meta: bool = False,
	):
		p = float(self.cfg.prob or 0.0)
		if p <= 0.0:
			# No-Op
			return (
				(x_hw, {'did_space': False, 'factor_h': 1.0}) if return_meta else x_hw
			)

		r = rng or np.random.default_rng()
		u = r.random()
		if u >= p:
			# No-Op
			return (
				(x_hw, {'did_space': False, 'factor_h': 1.0}) if return_meta else x_hw
			)

		# ←ここまで来たら適用！
		f = float(r.uniform(*self.cfg.factor_range))
		y = _spatial_stretch_sameH(x_hw, f)
		meta = {'did_space': True, 'factor_h': f}
		return (y, meta) if return_meta else y


class RandomHFlip:
	def __init__(self, prob: float = 0.5):
		self.prob = float(prob)

	def __call__(
		self,
		x_hw: np.ndarray,
		rng: np.random.Generator | None = None,
		return_meta=False,
	):
		r = rng or np.random.default_rng()
		if r.random() < self.prob:
			y = x_hw[::-1, :].copy()
			meta = {'hflip': True}
			return (y, meta) if return_meta else y
		meta = {'hflip': False}
		return (x_hw, meta) if return_meta else x_hw


class RandomCropOrPad:
	def __init__(self, target_len: int):
		self.L = int(target_len)

	def __call__(
		self,
		x_hw: np.ndarray,
		rng: np.random.Generator | None = None,
		return_meta=False,
	):
		r = rng or np.random.default_rng()
		H, W = x_hw.shape
		if self.L == W:
			meta = {'start': 0}
			return (x_hw, meta) if return_meta else x_hw
		if self.L < W:
			start = int(r.integers(0, W - self.L + 1))
			y = x_hw[:, start : start + self.L]
			meta = {'start': start}
			return (y, meta) if return_meta else y
		pad = self.L - W
		y = np.pad(x_hw, ((0, 0), (0, pad)), mode='constant')
		meta = {'start': 0}
		return (y, meta) if return_meta else y


class DeterministicCropOrPad:
	"""常に決定論"""

	def __init__(self, target_len: int):
		if target_len <= 0:
			raise ValueError('target_len must be positive')
		self.L = int(target_len)

	def __call__(
		self,
		x_hw: np.ndarray,
		rng: np.random.Generator | None = None,
		return_meta=False,
	):
		H, W = x_hw.shape
		if self.L == W:
			meta = {'start': 0}
			return (x_hw, meta) if return_meta else x_hw
		if self.L < W:
			start = (W - self.L) // 2
			y = x_hw[:, start : start + self.L]
			meta = {'start': int(start)}
			return (y, meta) if return_meta else y
		# W < L -> symmetric pad
		total = self.L - W
		y = np.pad(x_hw, ((0, 0), (0, total)), mode='constant')
		meta = {'start': 0}
		return (y, meta) if return_meta else y


class PerTraceStandardize:
	def __init__(self, eps: float = 1e-10):
		self.eps = float(eps)

	def __call__(
		self, x_hw: np.ndarray, rng: np.random.Generator | None = None
	) -> np.ndarray:
		return standardize_per_trace(x_hw, eps=self.eps)


class ViewCompose:
	def __init__(self, ops):
		self.ops = list(ops)

	def __call__(
		self,
		x_hw: np.ndarray,
		rng: np.random.Generator | None = None,
		return_meta: bool = False,
	):
		r = rng or np.random.default_rng()
		meta = {}
		for op in self.ops:
			# 各 op が return_meta をサポートしていれば meta を収集
			try:
				y = op(x_hw, r, return_meta=True)
			except TypeError:
				y = op(x_hw, r)
			if isinstance(y, tuple):
				x_hw, mu = y
				if isinstance(mu, dict) and mu:
					meta.update(mu)
			else:
				x_hw = y
		return (x_hw, meta) if return_meta else x_hw
