from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numba import njit, prange
from seisai_utils.validator import validate_array

# ===== helpers =====


def _percentile(a: np.ndarray, q: float, axis: int, keepdims: bool) -> np.ndarray:
	assert isinstance(a, np.ndarray)
	assert 0.0 <= float(q) <= 100.0
	return np.percentile(a, float(q), axis=axis, keepdims=keepdims, method='linear')


def _robust_loc_scale(seg_nw: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
	"""seg_nw: (N, Wseg)

	Returns:
	  mu_n:   (N,)
	  sigma_n:(N,)

	"""
	assert isinstance(seg_nw, np.ndarray)
	assert seg_nw.ndim == 2
	if method == 'mad':
		mu = np.median(seg_nw, axis=1)
		mad = np.median(np.abs(seg_nw - mu[:, None]), axis=1)
		sigma = 1.4826 * mad
		return mu, sigma
	if method == 'iqr':
		mu = np.median(seg_nw, axis=1)
		q75 = _percentile(seg_nw, 75.0, axis=1, keepdims=False)
		q25 = _percentile(seg_nw, 25.0, axis=1, keepdims=False)
		sigma = (q75 - q25) / 1.349
		return mu, sigma
	raise ValueError("method must be 'mad' or 'iqr'")


def _build_anchors(
	W: int, win: int, hop: int, causal: bool
) -> tuple[np.ndarray, Callable[[int], tuple[int, int]]]:
	assert int(W) > 0 and int(win) > 0 and int(hop) > 0
	W = int(W)
	win = int(win)
	hop = int(hop)

	if causal:
		anchors = np.arange(win // 2, W, hop, dtype=np.int64)

		def _bounds(t: int) -> tuple[int, int]:
			end = int(t)
			start = max(0, end - win)
			return start, end
	else:
		anchors = np.arange(0, W, hop, dtype=np.int64)

		def _bounds(center: int) -> tuple[int, int]:
			half = win // 2
			start = max(0, int(center) - half)
			end = min(W, start + win)
			start = max(0, end - win)
			return start, end

	if anchors.size == 0 or anchors[0] != 0:
		anchors = np.insert(anchors, 0, 0)
	if anchors[-1] != (W - 1):
		anchors = np.append(anchors, W - 1)
	return anchors, _bounds


@njit(parallel=True)
def _interp_rows(
	anchors: np.ndarray,  # (K,)
	values_nk: np.ndarray,  # (N, K)
	tgrid: np.ndarray,  # (T,)
	out_nw: np.ndarray,  # (N, T)
) -> None:
	N = values_nk.shape[0]
	K = values_nk.shape[1]
	T = tgrid.size

	for i in prange(N):
		k = 0
		for ti in range(T):
			t = tgrid[ti]

			# どの区間 [anchors[k], anchors[k+1]] に入るか前に進める
			while (k + 1) < K and anchors[k + 1] <= t:
				k += 1

			if (k + 1) >= K:
				# 最後のアンカーを超えた場合は末尾値をそのまま使う
				out_nw[i, ti] = values_nk[i, K - 1]
			else:
				x0 = anchors[k]
				x1 = anchors[k + 1]
				y0 = values_nk[i, k]
				y1 = values_nk[i, k + 1]

				if x1 == x0:
					out_nw[i, ti] = y0
				else:
					alpha = (t - x0) / (x1 - x0)
					out_nw[i, ti] = y0 + (y1 - y0) * alpha


# ===== main =====
def robust_agc_np(
	x: np.ndarray,
	*,
	win: int = 6000,
	hop: int | None = None,
	method: str = 'mad',  # "mad" or "iqr"
	gamma: float = 0.75,  # 0.5〜1.0
	eps: float = 1e-8,
	clamp_pct: tuple[float, float] = (5.0, 95.0),
	causal: bool = False,
	chunk: int = 1_000_000,  # 補間・適用の時間方向チャンク長（メモリ対策）
	return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
	"""入力形状: (W,) / (H,W) / (C,H,W) / (B,C,H,W)
	  すべて W は最後の軸（time）であること。
	出力は入力と同形状・同dtype。
	"""
	validate_array(x, allowed_ndims=(1, 2, 3, 4), name='x')

	Hdims = x.shape[:-1]
	W = int(x.shape[-1])

	win = int(win)
	if win <= 0:
		raise ValueError('win must be positive')
	hop = int(hop if hop is not None else max(1, win // 4))
	if hop <= 0:
		raise ValueError('hop must be positive')
	if method not in ('mad', 'iqr'):
		raise ValueError("method must be 'mad' or 'iqr'")
	if not (0.0 < float(gamma) <= 1.0):
		raise ValueError('gamma must be in (0, 1]')
	p_lo, p_hi = float(clamp_pct[0]), float(clamp_pct[1])
	if not (0.0 <= p_lo < p_hi <= 100.0):
		raise ValueError('clamp_pct must satisfy 0 <= lo < hi <= 100')
	chunk = int(chunk)
	if chunk <= 0:
		raise ValueError('chunk must be positive')

	# ---- reshape to (N, W) view ----
	N = int(np.prod(Hdims, dtype=np.int64)) if Hdims else 1
	x_nw = x.reshape(N, W)

	anchors, bounds = _build_anchors(W, win, hop, causal)
	K = int(anchors.size)

	# 統計は倍精度で計算
	x_nw64 = x_nw.astype(np.float64, copy=False)

	mu_k = np.empty((N, K), dtype=np.float64)
	sg_k = np.empty((N, K), dtype=np.float64)

	# ---- robust stats over sliding anchors ----
	for j, t in enumerate(anchors):
		s, e = bounds(int(t))
		if e <= s:
			# アンカー窓が空にならないように保証
			e = min(W, s + 1)
		seg = x_nw64[:, s:e]  # (N, Wseg)
		mu, sigma = _robust_loc_scale(seg, method)
		mu_k[:, j] = mu
		sg_k[:, j] = sigma

	# ---- clamp σ per row (over K) and floor by eps ----
	p5 = _percentile(sg_k, p_lo, axis=1, keepdims=True)
	p95 = _percentile(sg_k, p_hi, axis=1, keepdims=True)
	sg_k = np.clip(sg_k, p5, p95)
	sg_k = np.maximum(sg_k, float(eps))

	# ---- interpolate & apply in T-chunks ----
	y_nw = np.empty_like(x_nw, dtype=x.dtype)
	t = 0
	while t < W:
		t0 = t
		t1 = min(W, t0 + chunk)
		tgrid = np.arange(t0, t1, dtype=np.float64)  # (Tchunk,)

		mu_seg = np.empty((N, t1 - t0), dtype=np.float64)
		sg_seg = np.empty((N, t1 - t0), dtype=np.float64)
		_interp_rows(anchors, mu_k, tgrid, mu_seg)
		_interp_rows(anchors, sg_k, tgrid, sg_seg)

		# castは最後に一括、入力dtypeを維持
		y_chunk = (
			(x_nw64[:, t0:t1] - mu_seg) / (sg_seg + float(eps)) ** float(gamma)
		).astype(x.dtype, copy=False)
		y_nw[:, t0:t1] = y_chunk
		t = t1

	# ---- reshape back ----
	y = y_nw.reshape(*Hdims, W)

	if return_meta:
		meta = {
			'anchors': anchors,  # (K,)
			'mu_k': mu_k.reshape(*Hdims, K),  # (*Hdims, K)
			'sigma_k': sg_k.reshape(*Hdims, K),  # (*Hdims, K)
			'gamma': float(gamma),
			'method': method,
			'win': int(win),
			'hop': int(hop),
			'clamp_pct': (p_lo, p_hi),
			'causal': bool(causal),
			'eps': float(eps),
			'chunk': int(chunk),
		}
		return y, meta
	return y
