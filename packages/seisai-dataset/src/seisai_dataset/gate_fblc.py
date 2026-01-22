# gate_fblc.py （抜粋）---------------------------------------------
from typing import Literal

import numpy as np

from .config import FirstBreakGateConfig


class FirstBreakGate:
	def __init__(self, cfg: FirstBreakGateConfig):
		self._validate_config(
			percentile=cfg.percentile,
			thresh_ms=cfg.thresh_ms,
			min_pairs=cfg.min_pairs,
			apply_on=cfg.apply_on,
		)
		self.cfg = cfg

	@staticmethod
	def _validate_config(
		*,
		percentile: float,
		thresh_ms: float,
		min_pairs: int,
		apply_on: Literal['any', 'super_only', 'off'] | None = None,
	) -> None:
		if not (0.0 < float(percentile) < 100.0):
			raise ValueError('percentile must be in (0, 100)')
		if not (float(thresh_ms) > 0.0):
			raise ValueError('thresh_ms must be positive')
		if int(min_pairs) < 0:
			raise ValueError('min_pairs must be non-negative')
		if apply_on is not None and apply_on not in ('any', 'super_only', 'off'):
			raise ValueError("apply_on must be 'any', 'super_only', or 'off'")

	def should_apply(
		self,
		*,
		did_super: bool,
		apply_on: Literal['any', 'super_only', 'off'] | None = None,
	) -> bool:
		ap = self.cfg.apply_on if apply_on is None else apply_on
		if ap == 'off':
			return False
		if ap == 'any':
			return True
		if ap == 'super_only':
			return bool(did_super)
		raise ValueError("apply_on must be 'any', 'super_only', or 'off'")

	def min_pick_accept(self, fb_idx_win: np.ndarray) -> tuple[bool, int, float]:
		r = self.cfg.min_pick_ratio
		if r is None or float(r) == 0.0:
			return True, 0, 0.0
		v = fb_idx_win.astype(np.int64, copy=False)
		H = int(v.size)
		if H == 0:
			return False, 0, 0.0
		valid = v > 0
		n_valid = int(valid.sum())
		ratio = n_valid / H
		return (ratio >= float(r), n_valid, ratio)

	def fblc_accept(
		self,
		fb_idx_win: np.ndarray,
		dt_eff_sec: float,
		*,
		did_super: bool = False,
		percentile: float | None = None,
		thresh_ms: float | None = None,
		min_pairs: int | None = None,
		apply_on: Literal['any', 'super_only', 'off'] | None = None,
	) -> tuple[bool, float | None, int]:
		# 無効化 or 適用不要なら素通し
		if not self.should_apply(did_super=did_super, apply_on=apply_on):
			return True, None, 0

		p = self.cfg.percentile if percentile is None else float(percentile)
		th = self.cfg.thresh_ms if thresh_ms is None else float(thresh_ms)
		mp = self.cfg.min_pairs if min_pairs is None else int(min_pairs)
		self._validate_config(
			percentile=p,
			thresh_ms=th,
			min_pairs=mp,
			apply_on=apply_on,
		)

		v = fb_idx_win.astype(np.float64, copy=False)
		valid = v > 0
		m = valid[1:] & valid[:-1]
		valid_pairs = int(m.sum())
		if valid_pairs < mp:
			return False, None, valid_pairs

		diffs = np.abs(v[1:] - v[:-1])[m]
		q = float(np.percentile(diffs, p))
		p_ms = q * float(dt_eff_sec) * 1000.0
		return (p_ms <= th), p_ms, valid_pairs
