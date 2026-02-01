"""Mask generation and application utilities for seismic-style (H, T) arrays.

This module provides:
- boolean mask generators (trace-band masks, jittered checkerboard blocks)
- MaskGenerator to generate masks and apply corruption ('replace' or 'add') with Gaussian noise
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
	from collections.abc import Callable

MaskMode = Literal['replace', 'add']


class _InvalidMaskModeError(ValueError):
	def __init__(self, mode: str) -> None:
		super().__init__(f"invalid mode: {mode!r} (expected 'replace' or 'add')")


# =========================================================
# マスク生成: トレース単位 / 2Dチェッカー(ジッター・不規則)
# 返り値は必ず bool (H,T) のマスク(True: マスク/破壊/学習で隠す)
# =========================================================


def mask_traces_bool(
	H: int,
	T: int,
	*,
	ratio: float,
	rng: np.random.Generator,
	width: int = 1,
) -> np.ndarray:
	"""トレース単位のバンド・マスクを (H,T) の bool で返す(True=マスク).

	- ratio: [0,1] 相当の「目安の被覆率」。おおよそ ratio*H 本のトレースを覆うように
		バンド中心数を ceil(ratio*H / width) で決める。
	- width: バンドのトレース幅(>=1)。各中心から高さ方向に連続 'width' 本をマスク。

	例) ratio=0.3, width=5 なら、中心をだいたい ceil(0.3*H/5) 点サンプルし、
		各中心の周囲 'width' 本を連続マスクする(端はクリップ、重なりは許容)。
	"""
	if H <= 0 or T <= 0:
		msg = f'invalid shape H={H}, T={T}'
		raise ValueError(msg)
	r = float(ratio)
	if not (0.0 <= r <= 1.0):
		msg = 'ratio must be in [0,1]'
		raise ValueError(msg)
	w = int(width)
	if w < 1:
		msg = 'width must be >= 1'
		raise ValueError(msg)

	# 目標被覆本数(端クリップや重なりでブレるのは許容)
	target = round(r * H)
	if target == 0:
		return np.zeros((H, T), dtype=bool)
	if target >= H or w >= H:
		return np.ones((H, T), dtype=bool)

	# 必要な中心数を見積もってユニークに選ぶ
	n_centers = int(np.ceil(target / w))
	n_centers = max(1, min(n_centers, H))
	centers = rng.choice(H, size=n_centers, replace=False)

	m = np.zeros((H, T), dtype=bool)
	for c in centers:
		# 幅 w をできるだけ保ったまま [0,H) に収める
		h0 = c - (w // 2)
		h1 = h0 + w
		if h0 < 0:
			h0, h1 = 0, min(w, H)
		if h1 > H:
			h1, h0 = H, max(0, H - w)
		if h0 < h1:
			m[h0:h1, :] = True
	return m


@dataclass(frozen=True)
class CheckerJitterConfig:
	"""ジッター付き2Dブロック・チェッカー生成の設定(不規則・非周期).

	- 全体を cell_h x cell_t の粗い格子に分割
	- 各セルに確率 keep_prob で 1 個のブロックを置く
	- ブロックは block_h x block_t の大きさ、セル内で±jitter_{h,t} だけ平行移動
	- offset_{h,t} はセル基準の位相(推論アンサンブル時の完全被覆に利用可能).
	"""

	block_h: int
	block_t: int
	cell_h: int
	cell_t: int
	jitter_h: int = 0
	jitter_t: int = 0
	keep_prob: float = 1.0
	offset_h: int = 0
	offset_t: int = 0


def mask_checkerboard_jitter_bool(
	H: int, T: int, *, cfg: CheckerJitterConfig, rng: np.random.Generator
) -> np.ndarray:
	"""ジッター付き2Dチェッカーブロック(不規則)を (H,T) bool で返す."""
	if H <= 0 or T <= 0:
		msg = f'invalid shape H={H}, T={T}'
		raise ValueError(msg)
	if cfg.block_h <= 0 or cfg.block_t <= 0:
		msg = 'block_h/block_t must be > 0'
		raise ValueError(msg)
	if cfg.cell_h <= 0 or cfg.cell_t <= 0:
		msg = 'cell_h/cell_t must be > 0'
		raise ValueError(msg)
	if not (0.0 <= cfg.keep_prob <= 1.0):
		msg = 'keep_prob must be in [0,1]'
		raise ValueError(msg)

	mask = np.zeros((H, T), dtype=bool)

	# セル位相(負や大きすぎる位相はモジュロで正規化)
	off_h = int(cfg.offset_h) % cfg.cell_h
	off_t = int(cfg.offset_t) % cfg.cell_t

	# 何セル必要か(端のはみ出しも含めて走査)
	n_cells_h = (H + cfg.cell_h - 1) // cfg.cell_h + 1  # 端埋め用に+1
	n_cells_t = (T + cfg.cell_t - 1) // cfg.cell_t + 1

	for gh in range(n_cells_h):
		base_h = gh * cfg.cell_h + off_h
		for gt in range(n_cells_t):
			base_t = gt * cfg.cell_t + off_t

			# このセルにブロックを置くか(密度調整)
			if rng.random() > cfg.keep_prob:
				continue

			# セル内ジッター
			jh = (
				0
				if cfg.jitter_h <= 0
				else int(rng.integers(-cfg.jitter_h, cfg.jitter_h + 1))
			)
			jt = (
				0
				if cfg.jitter_t <= 0
				else int(rng.integers(-cfg.jitter_t, cfg.jitter_t + 1))
			)

			# ブロック左上(bounds内にクリップ)
			h0 = max(0, min(base_h + jh, H - cfg.block_h))
			t0 = max(0, min(base_t + jt, T - cfg.block_t))
			h1 = h0 + cfg.block_h
			t1 = t0 + cfg.block_t

			if h0 < H and t0 < T and h1 > 0 and t1 > 0:
				mask[h0:h1, t0:t1] = True

	return mask


class MaskGenerator:
	"""* 役割A: ピクセル単位の bool マスク生成 (True = マスク).

	* 役割B: マスク適用時の破壊プロファイル(mode/noise_std)を保持し apply() で適用.

	使い方:
		rng = np.random.default_rng(0)
		gen = MaskGenerator.traces(ratio=0.3, width=5, mode="replace", noise_std=1.0)
		m = gen.generate(H, T, rng)
		xm, m = gen.apply(x, rng=rng, return_mask=True)

		gen2 = MaskGenerator.checker_jitter(cfg, mode="add", noise_std=0.5)
		xm2 = gen2.apply(x, rng=rng)
	"""

	def __init__(
		self,
		fn: Callable[[int, int, np.random.Generator], np.ndarray],
		*,
		mode: Literal['replace', 'add'] = 'replace',
		noise_std: float = 1.0,
	) -> None:
		"""Initialize a mask generator and its corruption behavior."""
		if mode not in ('replace', 'add'):
			raise _InvalidMaskModeError(mode)
		if noise_std < 0:
			msg = 'noise_std must be >= 0'
			raise ValueError(msg)
		self._fn = fn
		if mode not in ('replace', 'add'):
			msg = 'invalid mode'
			raise ValueError(msg)
		if noise_std < 0:
			msg = 'noise_std must be >= 0'
			raise ValueError(msg)
		self._fn = fn
		self.mode = mode
		self.noise_std = float(noise_std)

	# ---- A) マスク生成 ----
	def generate(self, H: int, T: int, rng: np.random.Generator) -> np.ndarray:
		"""Generate a boolean mask of shape (H, T).

		Parameters
		----------
		H : int
			Height (number of traces).
		T : int
			Width (number of time samples).
		rng : np.random.Generator
			Random number generator used by the underlying mask function.

		Returns
		-------
		np.ndarray
			Boolean mask of shape (H, T) where True indicates masked positions.

		Raises
		------
		ValueError
			If the underlying generator returns a non-bool array or an array of a different shape.

		"""
		m = self._fn(H, T, rng)
		if not np.issubdtype(m.dtype, np.bool_) or m.shape != (H, T):
			msg = 'generator must return bool array of shape (H,T)'
			raise ValueError(msg)
		return m

	# ---- B) マスク適用(破壊)----
	def apply(
		self,
		x: np.ndarray,  # (H,T) or (C,H,T) [channel-first]
		*,
		rng: np.random.Generator,
		mask: np.ndarray | None = None,  # 省略時は generate() で作る
		return_mask: bool = False,
	) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
		"""Apply the mask to an input array using this generator's corruption behavior.

		Parameters
		----------
		x : np.ndarray
			Input array of shape (H, T) or (C, H, T) (channel-first).
		rng : np.random.Generator
			Random number generator used for mask generation (if `mask` is None) and noise.
		mask : np.ndarray | None, optional
			Boolean mask of shape (H, T) where True indicates masked/corrupted positions;
			if None, a mask is generated via `generate(H, T, rng)`.
		return_mask : bool, optional
			If True, return a tuple (masked_x, mask); otherwise return only masked_x.

		Returns
		-------
		np.ndarray | tuple[np.ndarray, np.ndarray]
			Masked/corrupted array with the same shape as `x`, and optionally the mask.

		Raises
		------
		ValueError
			If `x` is not shape (H,T) or (C,H,T), or if `mask` is not bool of shape (H,T).

		"""
		if x.ndim == 2:
			H, T = x.shape
			X = x[None, ...].astype(np.float32, copy=False)  # (1,H,T)
			squeeze = True
		elif x.ndim == 3:
			_, H, T = x.shape
			X = x.astype(np.float32, copy=False)
			squeeze = False
		else:
			msg = f'x must be (H,T) or (C,H,T), got {x.shape}'
			raise ValueError(msg)

		m = mask if mask is not None else self.generate(H, T, rng)
		if not np.issubdtype(m.dtype, np.bool_) or m.shape != (H, T):
			msg = 'mask must be bool of shape (H,T)'
			raise ValueError(msg)

		M = m[None, :, :]  # (1,H,T) → (C,H,T) にブロードキャスト
		if self.mode == 'replace':
			if self.noise_std == 0.0:
				Y = np.where(M, 0.0, X)
			else:
				N = rng.normal(0.0, self.noise_std, size=X.shape).astype(np.float32)
				Y = np.where(M, N, X)
		elif self.noise_std == 0.0:
			Y = X.copy()
		else:
			N = rng.normal(0.0, self.noise_std, size=X.shape).astype(np.float32)
			Y = X + (N * M.astype(np.float32))

		Y = Y[0] if squeeze else Y
		return (Y, m) if return_mask else Y

	# ---- ファクトリ ----
	@staticmethod
	def traces(
		*,
		ratio: float,
		width: int = 1,
		mode: Literal['replace', 'add'] = 'replace',
		noise_std: float = 1.0,
	) -> MaskGenerator:
		"""Create a trace-band mask generator.

		Parameters
		----------
		ratio : float
			Approximate fraction of traces (height axis) to be masked, in [0, 1].
		width : int, optional
			Band width (number of consecutive traces) to mask per selected center, by default 1.
		mode : {'replace', 'add'}, optional
			How to corrupt masked positions, by default 'replace'.
		noise_std : float, optional
			Standard deviation of Gaussian noise used for corruption, by default 1.0.

		Returns
		-------
		MaskGenerator
			A configured mask generator.

		Raises
		------
		ValueError
			If `ratio` is not in [0, 1] or `width` is less than 1.

		"""
		if not (0.0 <= float(ratio) <= 1.0):
			msg = 'ratio must be in [0,1]'
			raise ValueError(msg)
		if width < 1:
			msg = 'width must be >= 1'
			raise ValueError(msg)
		return MaskGenerator(
			lambda H, T, rng: mask_traces_bool(
				H, T, ratio=ratio, width=int(width), rng=rng
			),
			mode=mode,
			noise_std=noise_std,
		)

	@staticmethod
	def checker_jitter(
		cfg: CheckerJitterConfig,
		*,
		mode: Literal['replace', 'add'] = 'replace',
		noise_std: float = 1.0,
	) -> MaskGenerator:
		"""Create a jittered checkerboard mask generator.

		Parameters
		----------
		cfg : CheckerJitterConfig
			Configuration for the jittered checkerboard mask.
		mode : {'replace', 'add'}, optional
			How to corrupt masked positions, by default 'replace'.
		noise_std : float, optional
			Standard deviation of Gaussian noise used for corruption, by default 1.0.

		Returns
		-------
		MaskGenerator
			A configured mask generator.

		"""
		return MaskGenerator(
			lambda H, T, rng: mask_checkerboard_jitter_bool(H, T, cfg=cfg, rng=rng),
			mode=mode,
			noise_std=noise_std,
		)
