# ============================================================
# 目的: トレンド推定の戦略をインスタンスで差し替える(Strategyパターン)
# 依存: torch, seisai_pick.trend.trend_fit
# ============================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from seisai_pick.trend.trend_fit import (
	robust_linear_trend,
	robust_linear_trend_sections_ransac,
)
from torch import Tensor


class TrendFitStrategy(Protocol):
	name: str

	def __call__(
		self,
		*,
		offsets: Tensor,  # (B,H) [m]
		t_sec: Tensor,  # (B,H) [s]
		valid: Tensor,  # (B,H) bool/int
		w_conf: Tensor,  # (B,H)
	) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
		"""Return: trend_t, trend_s, v_trend, w_used, covered (all (B,H))"""
		...


# ---------------- IRLS 戦略 ----------------
@dataclass(frozen=True)
class IRLSStrategy:
	name: str = 'irls'
	section_len: int = 128
	stride: int = 64
	huber_c: float = 1.345
	iters: int = 3
	vmin: float = 300.0
	vmax: float = 8000.0
	sort_offsets: bool = True
	use_taper: bool = True

	def __call__(
		self, *, offsets: Tensor, t_sec: Tensor, valid: Tensor, w_conf: Tensor
	):
		assert offsets.shape == t_sec.shape == valid.shape == w_conf.shape
		return robust_linear_trend(
			offsets=offsets,
			t_sec=t_sec,
			valid=valid,
			w_conf=w_conf,
			section_len=self.section_len,
			stride=self.stride,
			huber_c=self.huber_c,
			iters=self.iters,
			vmin=self.vmin,
			vmax=self.vmax,
			sort_offsets=self.sort_offsets,
			use_taper=self.use_taper,
		)


# ---------------- RANSAC 戦略 ----------------
@dataclass(frozen=True)
class RANSACStrategy:
	name: str = 'ransac'
	section_len: int = 128
	stride: int = 64
	vmin: float = 300.0
	vmax: float = 8000.0
	ransac_trials: int = 64
	ransac_tau: float = 2.0
	ransac_abs_ms: float = 12.0
	ransac_pack: int = 16
	sample_weighted: bool = True
	refine_irls_iters: int = 1
	use_inlier_blend: bool = True
	sort_offsets: bool = True

	def __call__(
		self, *, offsets: Tensor, t_sec: Tensor, valid: Tensor, w_conf: Tensor
	):
		assert offsets.shape == t_sec.shape == valid.shape == w_conf.shape
		return robust_linear_trend_sections_ransac(
			offsets=offsets,
			t_sec=t_sec,
			valid=valid,
			w_conf=w_conf,
			section_len=self.section_len,
			stride=self.stride,
			vmin=self.vmin,
			vmax=self.vmax,
			ransac_trials=self.ransac_trials,
			ransac_tau=self.ransac_tau,
			ransac_abs_ms=self.ransac_abs_ms,
			ransac_pack=self.ransac_pack,
			sample_weighted=self.sample_weighted,
			refine_irls_iters=self.refine_irls_iters,
			use_inlier_blend=self.use_inlier_blend,
			sort_offsets=self.sort_offsets,
		)
