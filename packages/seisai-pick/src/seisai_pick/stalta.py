import numpy as np
from numba import njit, prange


# === 1D: ユーザーの逐次和アルゴリズム（そのまま採用） =====================
@njit(cache=True, fastmath=True)
def stalta_1d(trc, ns=10, nl=100, eps=1e-12):
	tmax = trc.size
	R = np.zeros(tmax, dtype=np.float64)
	s_sta = 0.0
	s_lta = 0.0
	for i in range(tmax):
		x2 = float(trc[i]) * float(trc[i])
		s_sta += x2
		s_lta += x2
		if i >= ns:
			y = float(trc[i - ns])
			s_sta -= y * y
			sta = s_sta / ns
		else:
			sta = s_sta / (i + 1)
		if i >= nl:
			y = float(trc[i - nl])
			s_lta -= y * y
			lta = s_lta / nl
		else:
			lta = s_lta / (i + 1)
		R[i] = sta / lta if lta > eps else 0.0
	return R


# === (H,T): 2Dを各トレース独立に並列処理（時間=最後の軸） ==================
@njit(cache=True, fastmath=True, parallel=True)
def stalta_2d(x_2d, ns=10, nl=100, eps=1e-12):
	"""x_2d: shape (H, T) の連続配列（C連続推奨）
	戻り: shape (H, T) の float64
	"""
	H, T = x_2d.shape
	out = np.empty((H, T), dtype=np.float64)
	for h in prange(H):
		out[h, :] = stalta_1d(x_2d[h], ns, nl, eps)
	return out


# === 汎用ラッパ: 任意次元 (..., T) に対応 / axis で時間軸を指定 =================


def stalta(x, ns=10, nl=100, eps=1e-12, axis=-1, out_dtype=None):
	"""任意次元の配列 x に対して、指定 axis (=時間軸) に沿って STALTA を計算。
	- x shape 例:
		(T,), (H,T), (B,H,T), (B,C,H,T), ...
	- axis は時間軸（デフォ = -1 = 最後の軸）
	- 出力 shape は入力と同じ。dtype は out_dtype か、なければ入力 dtype を継承（float32/64のみ）。
	"""
	# ガード（サイレント劣化禁止）
	if not (
		isinstance(ns, int) and isinstance(nl, int) and ns > 0 and nl > 0 and nl >= ns
	):
		raise ValueError(
			f'Invalid window sizes: ns={ns}, nl={nl}. Require integers, ns>0, nl>=ns.'
		)

	x = np.asarray(x)
	# 1) 時間軸を末尾へ
	x_last = np.moveaxis(x, axis, -1)
	T = x_last.shape[-1]
	lead_shape = x_last.shape[:-1]

	# 2) 2D (N, T) へまとめる（C連続＆float64でJITに優しい形に）
	x_2d = x_last.reshape(-1, T)
	x_2d = np.ascontiguousarray(x_2d, dtype=np.float64)

	# 3) Numba で各トレース処理
	r_2d = stalta_2d(x_2d, ns=ns, nl=nl, eps=eps)

	# 4) 元の形に戻す→元の axis に戻す
	r_last = r_2d.reshape(lead_shape + (T,))
	r = np.moveaxis(r_last, -1, axis)

	# 5) 出力 dtype
	if out_dtype is None:
		out_dtype = x.dtype if x.dtype in (np.float32, np.float64) else np.float64
	return r.astype(out_dtype, copy=False)
