from __future__ import annotations

import numpy as np


def _resample_idx_nearest(v: np.ndarray, factor_h: float) -> np.ndarray:
	"""(H,) 整数インデックス列を H を保ったまま「最近傍補間」で再サンプルする。
	目的:
	  - fb_idx のような「離散位置」を線形でぼかさず、段差を保ったまま伸縮する。

	契約:
	  - v: (H,), int 系（-1 は無効）
	  - factor_h: >0、1.0 で恒等。中心 (H-1)/2 を固定してスケール。
	  - 端はクリップ（外挿なし）
	無効の扱い:
	  - 最近傍に選ばれた元値が -1 なら出力も -1（無効の“にじみ”は起こさない）
	戻り値:
	  - (H,), int64
	"""
	H = int(v.shape[0])
	if H == 0 or abs(factor_h - 1.0) <= 1e-6:
		return v.copy()

	c = (H - 1) * 0.5
	dst = np.arange(H, dtype=np.float32)
	src = c + (dst - c) / float(factor_h)  # 連続座標
	j = np.floor(src + 0.5).astype(np.int64)  # 最近傍
	j = np.clip(j, 0, H - 1)  # 端クリップ

	out = v[j].astype(np.int64, copy=True)  # -1 はそのまま保持
	return out


def _resample_float_linear(v: np.ndarray, factor_h: float) -> np.ndarray:
	"""トレース方向（H軸）の中心固定・線形補間で、(H,) の float 系列を
	同じ H のままリサンプリングする。

	契約:
	- 入力 v: 形状 (H,), dtype は float を想定（例: offsets）
	- factor_h: >0 の拡大率。1.0 で恒等。中心 (H-1)/2 を固定して拡大/縮小。
	- 端はクリップ（外挿なし）
	戻り値:
	- 形状 (H,), dtype=float32
	例外:
	- H==0 の場合は v.astype(np.float32, copy=True) を返す（恒等）
	"""
	H = int(v.shape[0])
	if H == 0 or abs(factor_h - 1.0) <= 1e-6:
		return v.astype(np.float32, copy=True)
	c = (H - 1) * 0.5
	dst = np.arange(H, dtype=np.float32)
	src = c + (dst - c) / float(factor_h)
	src = np.clip(src, 0.0, H - 1.0)
	h0 = np.floor(src).astype(np.int64)
	h1 = np.clip(h0 + 1, 0, H - 1)
	w = (src - h0).astype(np.float32)
	return ((1.0 - w) * v[h0] + w * v[h1]).astype(np.float32)


def project_fb_idx_view(fb_idx: np.ndarray, H: int, W: int, meta: dict) -> np.ndarray:
	"""生の初動インデックス列 `fb_idx`（0-based, -1 無効）を、
	meta（hflip/factor_h/factor/start）に基づいて View 空間へ投影する。

	処理順:
	1) H方向: hflip → factor_h による再サンプル（線形、無効値伝播ルール適用）
	2) T方向: 0-based のまま round(fb * factor) - start で時間窓へ写像
	3) 範囲外は -1（無効）にする。0..W-1 のみ有効。

	引数:
	- fb_idx: (H,), int。0-based。-1 は無効。
	- H, W : x_view の形状に一致するトレース数・サンプル数
	- meta : dict。{'hflip':bool, 'factor_h':float, 'factor':float, 'start':int}
	戻り値:
	- fb_idx_view: (H,), int64。0..W-1 が有効、-1 が無効。
	例外:
	- H と fb_idx 長さ不一致、factor<=0 などの不正は ValueError。
	"""
	fb = np.asarray(fb_idx, dtype=np.int64).copy()
	if fb.shape[0] != H:
		raise ValueError(f'fb_idx length {fb.shape[0]} != H {H}')
	if meta.get('hflip', False):
		fb = fb[::-1]
	f_h = float(meta.get('factor_h', 1.0))
	if abs(f_h - 1.0) > 1e-6:
		fb = _resample_idx_nearest(fb, f_h)
	factor = float(meta.get('factor', 1.0))
	start = int(meta.get('start', 0))
	fb = np.round(fb * factor).astype(np.int64) - start  # 0-based & round
	fb[(fb <= 0) | (fb >= W)] = -1
	return fb


def project_offsets_view(offsets: np.ndarray, H: int, meta: dict) -> np.ndarray:
	"""生のオフセット列 `offsets`（(H,) float）を、
	meta の H方向変換（hflip/factor_h）に合わせて View 空間へ投影する。

	注意:
	- offsets は時間軸と独立のトレース属性のため、T方向（factor/start）の補正は行わない。

	引数:
	- offsets: (H,), float
	- H: x_view のトレース数
	- meta: {'hflip':bool, 'factor_h':float}
	戻り値:
	- offsets_view: (H,), float32
	例外:
	- H と offsets 長さ不一致は ValueError。
	"""
	off = np.asarray(offsets, dtype=np.float32).copy()
	if off.shape[0] != H:
		raise ValueError(f'offsets length {off.shape[0]} != H {H}')
	if meta.get('hflip', False):
		off = off[::-1]
	f_h = float(meta.get('factor_h', 1.0))
	if abs(f_h - 1.0) > 1e-6:
		off = _resample_float_linear(off, f_h)
	return off


def project_time_view(time_1d: np.ndarray, H: int, W: int, meta: dict) -> np.ndarray:
	"""生の時間軸（1D, 秒）を meta の時間ストレッチ/クロップに追従させ、
	(H,W) の time_view（秒）を生成する。

	前提:
	- time_1d は長さ >=2 の等間隔時刻列（例: np.arange(W0)*dt0 + t0）
	- meta は少なくとも {'factor':>0, 'start':>=0} を含む
	- hflip/factor_h は時間軸に無関係のため無視

	変換:
	- dt0 = mean(diff(time_1d))
	- t_view[w] = t0 + (start + w) * (dt0 / factor)

	引数:
	- time_1d: (W0,), float
	- H, W: x_view の形状に一致するトレース数・サンプル数
	- meta: {'factor':float, 'start':int}
	戻り値:
	- time_view: (H,W), float32（各行同一の時間格子）
	例外:
	- 非1次元、長さ不足、factor<=0 は ValueError。
	"""
	t_raw = np.asarray(time_1d, dtype=np.float64)
	if t_raw.ndim != 1 or t_raw.size < 2:
		raise ValueError('time must be 1D with length >= 2')
	dt0 = float(np.mean(np.diff(t_raw)))
	t0 = float(t_raw[0])
	factor = float(meta.get('factor', 1.0))
	start = int(meta.get('start', 0))
	tv = t0 + (np.arange(W, dtype=np.float64) + start) * (dt0 / factor)  # (W,)
	return np.repeat(tv[None, :].astype(np.float32), H, axis=0)  # (H,W)
