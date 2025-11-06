from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

__all__ = [
	'_apply_freq_augment',
	'_cosine_ramp',
	'_fit_time_len_np',
	'_make_freq_mask',
	'_spatial_stretch',
	'_time_stretch_poly',
]


def _time_stretch_poly(x_hw: np.ndarray, factor: float) -> np.ndarray:
	"""時間軸のみを factor 倍にリサンプルする（中心固定ではなく t=0 起点）。
	- 入力: x_hw (H,W)
	- 出力: (H,W') で W' は factor に応じて変化
	- 補間: resample_poly（IIR前提の polyphase）、端のパディングは 'line'
	- 例外: factor <= 0.0 は即時失敗
	"""
	if x_hw.ndim != 2:
		raise ValueError('x_hw must be (H,W)')
	if factor <= 0.0:
		raise ValueError('factor must be positive')
	if abs(factor - 1.0) < 1e-4:
		return x_hw

	H, _ = x_hw.shape
	frac = Fraction(float(factor)).limit_denominator(128)
	up, down = frac.numerator, frac.denominator
	y = np.stack(
		[resample_poly(x_hw[h], up, down, padtype='line') for h in range(H)],
		axis=0,
	)
	return y.astype(np.float32, copy=False)


def _spatial_stretch(x_hw: np.ndarray, factor: float) -> np.ndarray:
	"""幾何ストレッチ: H方向(トレース方向)のみ中心固定で座標写像し、出力は (H,W) を保つ。
	- 伸縮は1回の座標変換で実施（ぼかし用の拡大→縮小の2段ズームは廃止）
	- T軸は不変（zoom=(?, 1.0) 相当）
	- 補間: H方向の線形補間（境界はedge-clamp）

	契約:
	x_hw: (H, W) float/np.ndarray
	factor: >0。1.0で恒等。中心 c=(H-1)/2 を固定して y[h,:] = x[src(h),:]
	例外:
	- factor <= 0.0 は ValueError
	- H==0 or W==0 は ValueError
	"""
	if factor <= 0.0:
		raise ValueError('factor must be positive')
	if x_hw.ndim != 2:
		raise ValueError('x_hw must be 2D (H,W)')
	H, W = x_hw.shape
	if H == 0 or W == 0:
		raise ValueError('empty input')

	if abs(factor - 1.0) <= 1e-6:
		return x_hw  # 恒等

	# 中心固定の写像: dst h -> src = c + (h - c)/factor
	c = (H - 1) * 0.5
	dst = np.arange(H, dtype=np.float32)
	src = c + (dst - c) / float(factor)

	# 境界はエッジ複製（reflectではなくclip）
	src = np.clip(src, 0.0, H - 1.0)
	h0 = np.floor(src).astype(np.int64)
	h1 = np.clip(h0 + 1, 0, H - 1)
	w = (src - h0).astype(np.float32)  # (H,)

	# H方向1次補間（ブロードキャストで (H,W) を一括計算）
	y = (1.0 - w)[:, None] * x_hw[h0, :] + w[:, None] * x_hw[h1, :]
	return y.astype(x_hw.dtype, copy=False)


def _cosine_ramp(x: np.ndarray, a: float, b: float, invert: bool = False) -> np.ndarray:
	"""Cosine ramp from 0 to 1 (or inverted) over [a, b]."""
	if b <= a:
		return np.ones_like(x) if not invert else np.zeros_like(x)
	t = np.clip((x - a) / (b - a), 0.0, 1.0)
	ramp = 0.5 - 0.5 * np.cos(np.pi * t)
	return (1.0 - ramp) if invert else ramp


def _make_freq_mask(
	n_rfft: int,
	kind: str,
	f_lo: float | None,
	f_hi: float | None,
	roll: float,
) -> np.ndarray:
	"""Create smooth frequency mask for rFFT."""
	f = np.linspace(0.0, 1.0, n_rfft, dtype=np.float32)
	m = np.ones_like(f, dtype=np.float32)
	if kind == 'bandpass':
		assert f_lo is not None and f_hi is not None and f_hi > f_lo
		bw = max(f_hi - f_lo, 1e-4)
		r = max(roll, 0.05 * bw)
		up = _cosine_ramp(f, max(0.0, f_lo - r), min(1.0, f_lo + r), invert=False)
		dn = _cosine_ramp(f, max(0.0, f_hi - r), min(1.0, f_hi + r), invert=True)
		m = up * dn
	elif kind == 'lowpass':
		assert f_hi is not None
		r = max(roll, 0.05 * f_hi)
		dn = _cosine_ramp(f, max(0.0, f_hi - r), min(1.0, f_hi + r), invert=True)
		m = dn
	elif kind == 'highpass':
		assert f_lo is not None
		r = max(roll, 0.05 * (1.0 - f_lo))
		up = _cosine_ramp(f, max(0.0, f_lo - r), min(1.0, f_lo + r), invert=False)
		m = up
	else:
		raise ValueError(f'unknown freq-augment kind: {kind}')
	return m.astype(np.float32)


def _apply_freq_augment(
	x_hw: np.ndarray,
	augment_freq_kinds: tuple[str, ...],
	augment_freq_band: tuple[float, float],
	augment_freq_width: tuple[float, float],
	augment_freq_roll: float,
	augment_freq_restandardize: bool,
	rng: np.random.Generator | None = None,  # ← 追加
) -> np.ndarray:
	"""Apply same frequency mask to all traces (deterministic if rng given)."""
	r = rng or np.random.default_rng()
	H, W = x_hw.shape
	kind = r.choice(augment_freq_kinds)
	lo_min, hi_max = augment_freq_band

	if kind == 'bandpass':
		min_w, max_w = augment_freq_width
		bw = float(r.uniform(min_w, max_w))
		f_lo = float(r.uniform(lo_min, max(1e-3, hi_max - bw)))
		f_hi = min(hi_max, f_lo + bw)
	elif kind == 'lowpass':
		f_lo, f_hi = None, float(r.uniform(lo_min + 0.02, hi_max))
	elif kind == 'highpass':
		f_lo, f_hi = float(r.uniform(lo_min, hi_max - 0.02)), None
	else:
		raise ValueError(f'unknown freq-augment kind: {kind}')

	n_rfft = W // 2 + 1
	mask = _make_freq_mask(n_rfft, kind, f_lo, f_hi, augment_freq_roll)
	X = np.fft.rfft(x_hw, axis=1)
	Y = X * mask[None, :]
	y = np.fft.irfft(Y, n=W, axis=1).astype(np.float32)
	if augment_freq_restandardize:
		m = y.mean(axis=1, keepdims=True)
		s = y.std(axis=1, keepdims=True) + 1e-10
		y = (y - m) / s
	return y
