# agc_numpy.py
from __future__ import annotations

import numpy as np
import torch
from seisai_utils.validator import validate_array
from torch import Tensor


def _moving_rms_axis_last(xf: np.ndarray, win: int, *, causal: bool) -> np.ndarray:
	"""xf: (*, W) float64
	返り値: (*, W) float64 （Wは最後の軸）
	causal=True のとき、時刻 t のRMSは [max(0, t-win+1) .. t] の平均に対応。
	"""
	if win <= 0:
		raise ValueError('win must be positive')
	W = int(xf.shape[-1])
	if W == 0:
		raise ValueError('W must be > 0')
	sq = xf * xf  # (*, W)
	cs = np.cumsum(sq, axis=-1)  # (*, W)

	if causal:
		# 前半（短窓）：t < win-1
		if win > 1:
			idx = np.arange(1, min(win, W), dtype=np.float64)  # 1..win-1 or 1..W-1
			prefix_mean = cs[..., : idx.size] / idx  # (*, win-1 or W-1)
		else:
			prefix_mean = cs[..., :0]  # 空

		# 後半（固定窓長win）
		if win <= W:
			tail_sum = cs[..., win - 1 :] - np.concatenate(
				(np.zeros_like(cs[..., :1]), cs[..., :-win]), axis=-1
			)
			tail_mean = tail_sum / float(win)  # (*, W-win+1)
			rms = np.concatenate((prefix_mean, tail_mean), axis=-1)
		else:
			rms = prefix_mean  # 全部短窓
	else:
		# 非因果（centered）: same レイアウト。パディングして平均→切り落とし。
		pad = win // 2
		# 先頭と末尾をゼロパディング
		sq_pad = np.pad(sq, (*((0, 0),) * (sq.ndim - 1), (pad, pad)), mode='constant')
		cs2 = np.cumsum(sq_pad, axis=-1)
		# 差分で窓和（長さWの列を得る）
		win_sum = cs2[..., win:] - cs2[..., :-win]
		rms = win_sum / float(win)  # (*, W)

	return np.sqrt(rms, dtype=np.float64)


def agc_np(
	x: np.ndarray,
	*,
	win: int = 1024,
	target_rms: float = 0.2,
	clamp_db: tuple[float, float] = (-20.0, 20.0),
	causal: bool = True,
	eps: float = 1e-8,
	return_gain: bool = False,
):
	"""一般的なRMSベースAGC（自動利得制御） NumPy版（CPU）。
	受け付ける形状: (W,) / (H,W) / (C,H,W) / (B,C,H,W) いずれも W は最後の軸。
	出力は入力と同形状・同dtype。

	- RMSは移動平均（窓長=win）で推定（causal/centered選択可）
	- 目標RMS=target_rms に合わせてゲインをスカラーで時変適用
	- ゲインは dB で [clamp_db[0], clamp_db[1]] にクリップ
	"""
	validate_array(x, allowed_ndims=(1, 2, 3, 4), name='x', backend='numpy')
	if not (float(target_rms) > 0.0):
		raise ValueError('target_rms must be > 0')
	gmin = 10.0 ** (float(clamp_db[0]) / 20.0)
	gmax = 10.0 ** (float(clamp_db[1]) / 20.0)
	if not (gmin <= gmax):
		raise ValueError('clamp_db must satisfy min <= max (in dB)')

	Hdims = x.shape[:-1]
	W = int(x.shape[-1])
	N = int(np.prod(Hdims, dtype=np.int64)) if Hdims else 1

	orig_dtype = x.dtype
	x_nw = x.reshape(N, W)
	xf64 = x_nw.astype(np.float64, copy=False)

	# レベル推定（RMS）
	rms = _moving_rms_axis_last(xf64, int(win), causal=causal)  # (N, W)

	# ゲイン計算（RMS→目標RMS）
	inv = target_rms / (rms + float(eps))
	gain = np.clip(inv, gmin, gmax)  # (N, W)

	# 適用（出力dtypeは入力に合わせる）
	y_nw = (xf64 * gain).astype(orig_dtype, copy=False)
	y = y_nw.reshape(*Hdims, W)

	if return_gain:
		return y, gain.reshape(*Hdims, W).astype(np.float32, copy=False)
	return y


@torch.no_grad()
def _moving_rms_axis_last_torch(xf: Tensor, win: int, *, causal: bool) -> Tensor:
	"""xf: (*, W) float64 (任意デバイス)
	返り値: (*, W) float64（Wは最後の軸）
	causal=True: 時刻tのRMSは [max(0, t-win+1) .. t] の平均
	causal=False: 中心化（左右 pad=win//2）した移動平均
	"""
	if win <= 0:
		raise ValueError('win must be positive')
	if xf.ndim < 1:
		raise ValueError('xf must have at least 1 dimension')
	W = int(xf.shape[-1])
	if W == 0:
		raise ValueError('W must be > 0')

	sq = xf * xf  # (*, W)
	cs = torch.cumsum(sq, dim=-1)  # (*, W)

	if causal:
		# 前半（短窓）：t < win-1
		if win > 1:
			n_prefix = min(win - 1, W)
			idx = torch.arange(
				1, n_prefix + 1, device=xf.device, dtype=xf.dtype
			)  # (1..n_prefix)
			prefix_mean = cs[..., :n_prefix] / idx  # (*, n_prefix)
		else:
			prefix_mean = cs[..., :0]  # 空

		# 後半（固定窓長 win）
		if win <= W:
			# sum[t] = cs[t] - cs[t-win]（t>=win-1）。csの先頭に0を付与して差分を取りやすくする。
			z0 = torch.zeros_like(cs[..., :1])
			cs_shift = torch.cat((z0, cs[..., :-win]), dim=-1)
			tail_sum = cs[..., win - 1 :] - cs_shift
			tail_mean = tail_sum / float(win)  # (*, W-win+1)
			mean = torch.cat((prefix_mean, tail_mean), dim=-1)  # (*, W)
		else:
			mean = prefix_mean  # 全区間が短窓
	else:
		# 非因果（centered same）
		pad = win // 2
		z_pre = torch.zeros((*sq.shape[:-1], pad), dtype=sq.dtype, device=sq.device)
		z_post = torch.zeros((*sq.shape[:-1], pad), dtype=sq.dtype, device=sq.device)
		sq_pad = torch.cat((z_pre, sq, z_post), dim=-1)  # (*, W+2*pad)
		cs2 = torch.cumsum(sq_pad, dim=-1)
		win_sum = cs2[..., win:] - cs2[..., :-win]  # (*, W)
		mean = win_sum / float(win)

	return torch.sqrt(mean)


@torch.no_grad()
def agc_torch(
	x: Tensor,
	*,
	win: int = 1024,
	target_rms: float = 0.2,
	clamp_db: tuple[float, float] = (-20.0, 20.0),
	causal: bool = True,
	eps: float = 1e-8,
	return_gain: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
	"""一般的なRMSベースAGC（PyTorch版）。
	受け付ける形状: (W,) / (H,W) / (C,H,W) / (B,C,H,W) いずれも W は最後の軸。
	出力は入力と同形状・同dtype。
	"""
	validate_array(x, allowed_ndims=(1, 2, 3, 4), name='x', backend='torch')
	if not (float(target_rms) > 0.0):
		raise ValueError('target_rms must be > 0')

	gmin = 10.0 ** (float(clamp_db[0]) / 20.0)
	gmax = 10.0 ** (float(clamp_db[1]) / 20.0)
	if gmin > gmax:
		raise ValueError('clamp_db must satisfy min <= max (in dB)')

	Hdims = tuple(x.shape[:-1])
	W = int(x.shape[-1])
	if W <= 0:
		raise ValueError('W must be > 0')
	N = int(torch.tensor(Hdims).prod().item()) if Hdims else 1  # 情報用

	device = x.device
	orig_dtype = x.dtype

	# (N, W) 化（view可能想定）
	x_nw = x.reshape(-1, W)
	xf64 = x_nw.to(dtype=torch.float64)

	# 移動RMS（float64）
	rms = _moving_rms_axis_last_torch(xf64, int(win), causal=causal)  # (N, W)

	# ゲイン計算
	inv = float(target_rms) / (rms + float(eps))
	gain = torch.clamp(inv, min=gmin, max=gmax)  # (N, W)

	# 適用（出力dtypeは入力に戻す）
	y_nw = (xf64 * gain).to(dtype=orig_dtype)
	y = y_nw.reshape(*Hdims, W)

	if return_gain:
		return y, gain.reshape(*Hdims, W).to(dtype=torch.float32, device=device)
	return y
