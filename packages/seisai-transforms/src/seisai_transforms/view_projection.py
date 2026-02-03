"""View-space projection utilities for SeisAI transforms.

This module provides helper functions to project 1D per-trace arrays into a
"view" coordinate system based on transformation metadata (e.g., horizontal
flip, H-axis scaling, time scaling, and window start).

Public functions:
- project_fb_idx_view: project first-break indices into view space.
- project_offsets_view: project per-trace offsets into view space.
- project_time_view: project a 1D time grid into view space.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6  # 比較時の許容誤差(ほぼ等しいかの判定に使用)


def _resample_idx_nearest(v: np.ndarray, factor_h: float) -> np.ndarray:
	"""(H,) 整数インデックス列を H を保ったまま「最近傍補間」で再サンプルする.

	目的:
	- fb_idx のような「離散位置」を線形でぼかさず、段差を保ったまま伸縮する。

	契約:
	- v: (H,), int 系(-1 は無効)
	- factor_h: >0、1.0 で恒等。中心 (H-1)/2 を固定してスケール。
	- 端はクリップ(外挿なし)

	無効の扱い:
	- 最近傍に選ばれた元値が -1 なら出力も -1(無効の“にじみ”は起こさない)

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

	return v[j].astype(np.int64, copy=True)


def _resample_float_linear(v: np.ndarray, factor_h: float) -> np.ndarray:
	"""(H,) の float 系列を中心固定・線形補間で同じ H のままリサンプリングする.

	トレース方向(H軸)の中心を固定し、線形補間で拡大/縮小する。

	契約:
	- 入力 v: 形状 (H,), dtype は float を想定(例: offsets)
	- factor_h: >0 の拡大率。1.0 で恒等。中心 (H-1)/2 を固定して拡大/縮小。
	- 端はクリップ(外挿なし)
	戻り値:
	- 形状 (H,), dtype=float32
	例外:
	- H==0 の場合は v.astype(np.float32, copy=True) を返す(恒等)
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
	"""生の初動インデックス列 `fb_idx` を、meta(hflip/factor_h/factor/start)に基づいて View 空間へ投影する.

	前提:
	- インデックスは 0-based だが、0 は無効扱い(有効なのは 1..W-1).
	- -1 は無効値。

	処理順:
	1) H方向: hflip → factor_h による再サンプル(最近傍、無効値伝播ルール適用)
	2) T方向: round(fb * factor) - start で時間窓へ写像(0以下は無効扱い)
	3) 範囲外は -1(無効)にする。有効範囲は 1..W-1。

	引数:
	- fb_idx: (H,), int。0-based。-1 は無効。0 も無効扱い。
	- H, W : x_view の形状に一致するトレース数・サンプル数
	- meta : dict。{'hflip':bool, 'factor_h':float, 'factor':float, 'start':int}

	戻り値:
	- fb_idx_view: (H,), int64。1..W-1 が有効、-1 が無効。

	例外:
	- H と fb_idx 長さ不一致、factor<=0、start<0、factor_h<=0 などの不正は ValueError。
	"""
	fb = np.asarray(fb_idx, dtype=np.int64).copy()
	if fb.shape[0] != H:
		msg = f'fb_idx length {fb.shape[0]} != H {H}'
		raise ValueError(msg)

	if meta.get('hflip', False):
		fb = fb[::-1]

	f_h = float(meta.get('factor_h', 1.0))
	if f_h <= 0.0:
		msg = "meta['factor_h'] must be > 0"
		raise ValueError(msg)
	if abs(f_h - 1.0) > 1e-6:
		fb = _resample_idx_nearest(fb, f_h)

	factor = float(meta.get('factor', 1.0))
	if factor <= 0.0:
		msg = "meta['factor'] must be > 0"
		raise ValueError(msg)

	start_raw = meta.get('start', 0)
	start_f = float(start_raw)
	if start_f < 0.0:
		msg = "meta['start'] must be >= 0"
		raise ValueError(msg)
	if abs(start_f - round(start_f)) > 1e-9:
		msg = "meta['start'] must be an integer >= 0"
		raise ValueError(msg)
	start = round(start_f)

	fb = (
		np.round(fb * factor).astype(np.int64) - start
	)  # 0-based mapping; 0 is treated as invalid
	fb[(fb <= 0) | (fb >= W)] = -1
	return fb


def project_offsets_view(offsets: np.ndarray, H: int, meta: dict) -> np.ndarray:
	"""生のオフセット列 (H,) を meta の H 方向変換(hflip / factor_h)に合わせて View 空間へ投影して返す.

	Parameters
	----------
	offsets : np.ndarray
		形状 (H,) の 1D 配列(float 可)。各トレースの受信点オフセット [m]。
	H : int
		ビューのトレース数(offsets 長と一致している必要がある)。
	meta : dict
		{'hflip': bool, 'factor_h': float} を想定。存在しない場合は既定値 False / 1.0。

	Returns
	-------
	np.ndarray
		形状 (H,) の float32 配列。ビュー空間のオフセット列。

	Raises
	------
	ValueError
		offsets が 1D でない、長さが H と一致しない、または factor_h <= 0 の場合。

	"""
	off = np.asarray(offsets, dtype=np.float32)
	if off.ndim != 1:
		msg = 'offsets must be 1D'
		raise ValueError(msg)
	if off.shape[0] != H:
		msg = f'offsets length {off.shape[0]} != H {H}'
		raise ValueError(msg)

	if bool(meta.get('hflip', False)):
		off = off[::-1].copy()

	factor_h = float(meta.get('factor_h', 1.0))
	if factor_h <= 0.0:
		msg = "meta['factor_h'] must be > 0"
		raise ValueError(msg)

	if abs(factor_h - 1.0) <= _EPS:
		return off

	out = _resample_float_linear(off, factor_h)
	return np.asarray(out, dtype=np.float32)


def project_time_view(time_1d: np.ndarray, H: int, W: int, meta: dict) -> np.ndarray:
	"""Project a 1D time axis (seconds) into view space based on meta.

	生の時間軸(1D, 秒)を meta の時間ストレッチ / クロップに追従させ、
	全トレース共通の 1D 時間グリッド (W,) を返す。
	注: H はインターフェイス整合のための引数で計算には使用しない。

	Parameters
	----------
	time_1d : np.ndarray
		形状 (W0,) の等間隔時刻列(例: np.arange(W0) * dt0 + t0)。
	H : int
		ビューのトレース数(未使用)。
	W : int
		出力時間グリッドのサンプル数。
	meta : dict
		{'factor': float (>0), 'start': int (>=0)} を想定。

	Returns
	-------
	np.ndarray
		形状 (W,) の float32 配列。ビュー空間の時間グリッド [s]。

	Raises
	------
	ValueError
		time_1d が 1D でない、長さ < 2、factor <= 0、または start < 0 の場合。

	"""
	t_raw = np.asarray(time_1d, dtype=np.float64)
	if t_raw.ndim != 1 or t_raw.size < 2:
		msg = 'time_1d must be 1D with length >= 2'
		raise ValueError(msg)

	dt0 = float(np.mean(np.diff(t_raw)))
	t0 = float(t_raw[0])

	factor = float(meta.get('factor', 1.0))
	if factor <= 0.0:
		msg = "meta['factor'] must be > 0"
		raise ValueError(msg)

	start = int(meta.get('start', 0))
	if start < 0:
		msg = "meta['start'] must be >= 0"
		raise ValueError(msg)

	tv = t0 + (np.arange(W, dtype=np.float64) + start) * (dt0 / factor)
	return tv.astype(np.float32)
