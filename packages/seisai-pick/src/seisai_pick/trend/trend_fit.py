from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def robust_linear_trend(
	offsets: Tensor,  # (B,H) [m]
	t_sec: Tensor,  # (B,H) predicted pos_sec [s]
	valid: Tensor,  # (B,H) fb_idx>=0 (bool or int)
	*,
	w_conf: Tensor,  # (B,H) per-trace confidence weights (>=0, typically <=1)
	section_len: int = 128,
	stride: int = 64,
	huber_c: float = 1.345,
	iters: int = 3,
	vmin: float = 300.0,
	vmax: float = 8000.0,
	sort_offsets: bool = True,
	use_taper: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
	"""Windowed IRLS で t(x) ≈ a + s·x を推定。確率マップは扱わず、外部で作成した
	トレース別自信度 w_conf を受け取り、IRLS の基底重みとして使用する。

	Parameters
	----------
	offsets : (B,H) [m]
	t_sec   : (B,H) [s]
	valid   : (B,H) bool/int
	w_conf  : (B,H) 事前重み（非負）。大きいほど信頼が高い
	section_len : int >=4
	stride      : int >=1
	huber_c     : Huber のしきい値（標準化残差 = 1 で切替）
	iters       : 反復回数（>=1）
	vmin, vmax  : 速度制約 [m/s]（s=1/v を [1/vmax, 1/vmin] にクランプ）
	sort_offsets: True の場合、内部で offset 昇順に並べ替えてから元順に戻す
	use_taper   : 窓のブレンドに Hann を使用（False なら一様）

	Returns
	-------
	trend_t : (B,H) [s]   窓ブレンド後のトレンド時刻
	trend_s : (B,H) [s/m] クランプ済みスローネス
	v_trend : (B,H) [m/s] = 1 / trend_s
	w_conf  : (B,H) 使用した事前重み（元順に戻したもの）
	covered : (B,H) bool 少なくとも1つの窓でカバーされた位置

	"""
	assert offsets.ndim == 2 and t_sec.ndim == 2 and valid.ndim == 2, (
		'offsets/t_sec/valid must be (B,H)'
	)
	B, H = offsets.shape
	assert t_sec.shape == (B, H) and valid.shape == (B, H), 'shapes must match (B,H)'
	assert w_conf.ndim == 2 and w_conf.shape == (B, H), 'w_conf must be (B,H)'
	assert section_len >= 4 and stride >= 1 and iters >= 1, 'invalid IRLS/window params'
	assert vmin > 0 and vmax > vmin, '0 < vmin < vmax required'

	x0 = offsets
	y0 = t_sec
	v0 = (valid > 0).to(t_sec)
	pw0 = w_conf.to(t_sec)

	# offset で並べ替え（安定化のため）。復元用インデックスも作成。
	if sort_offsets:
		idx = torch.argsort(x0, dim=1)  # (B,H)
		arangeH = torch.arange(H, device=idx.device).unsqueeze(0).expand_as(idx)
		inv = torch.empty_like(idx)
		inv.scatter_(1, idx, arangeH)

		x = torch.gather(x0, 1, idx)
		y = torch.gather(y0, 1, idx)
		v = torch.gather(v0, 1, idx)
		pw = torch.gather(pw0, 1, idx)
	else:
		x, y, v, pw = x0, y0, v0, pw0
		inv = torch.arange(H, device=x.device).view(1, H).expand(B, H)

	trend_t = torch.zeros_like(y)
	trend_s = torch.zeros_like(y)
	counts = torch.zeros_like(y)

	eps = 1e-12
	for start in range(0, H, stride):
		end = min(H, start + section_len)
		L = end - start
		if L < 4:
			continue

		xs = x[:, start:end]  # (B,L)
		ys = y[:, start:end]
		vs = v[:, start:end]
		pws = pw[:, start:end]  # (B,L)

		# 初期重み：valid × 事前重み
		w = (vs * pws).clone()

		a = torch.zeros(B, 1, dtype=y.dtype, device=y.device)
		b = torch.zeros(B, 1, dtype=y.dtype, device=y.device)  # slope (slowness)

		for _ in range(iters):
			Sw = w.sum(dim=1, keepdim=True).clamp_min(eps)
			Sx = (w * xs).sum(dim=1, keepdim=True)
			Sy = (w * ys).sum(dim=1, keepdim=True)
			Sxx = (w * xs * xs).sum(dim=1, keepdim=True)
			Sxy = (w * xs * ys).sum(dim=1, keepdim=True)

			D = (Sw * Sxx - Sx * Sx).clamp_min(eps)
			b = (Sw * Sxy - Sx * Sy) / D
			a = (Sy - b * Sx) / Sw

			yhat = a + b * xs
			res = (ys - yhat) * vs

			# ロバストスケール（MAD）
			scale = (1.4826 * res.abs().median(dim=1, keepdim=True).values).clamp_min(
				1e-6
			)
			r = res / (huber_c * scale)

			# Huber 風の再重み（影響関数が有界になるように）
			w_huber = torch.where(
				r.abs() <= 1.0, vs, vs * (1.0 / r.abs()).clamp_max(10.0)
			)

			# 事前重みは前段係数として保持（stop-grad 前提）
			w = w_huber * pws

		# 物理範囲にクランプ
		s_sec = b.squeeze(1).clamp(min=1.0 / vmax, max=1.0 / vmin)  # (B,)

		# 窓ブレンド（Hann or 一様）
		if use_taper:
			wwin = torch.hann_window(
				L, periodic=False, device=y.device, dtype=y.dtype
			).view(1, L)
		else:
			wwin = torch.ones(1, L, device=y.device, dtype=y.dtype)
		wtap = wwin * vs * pws  # vs と pws でマスク＋重み

		yhat = a + b * xs  # (B,L)
		trend_t[:, start:end] += yhat * wtap
		trend_s[:, start:end] += s_sec[:, None] * wtap
		counts[:, start:end] += wtap

	trend_t = trend_t / counts.clamp_min(1e-6)
	trend_s = trend_s / counts.clamp_min(1e-6)
	v_trend = 1.0 / trend_s.clamp_min(1e-6)
	covered = (counts > 0).to(torch.bool)

	# 元順に復元
	trend_t = torch.gather(trend_t, 1, inv)
	trend_s = torch.gather(trend_s, 1, inv)
	v_trend = torch.gather(v_trend, 1, inv)
	w_used = torch.gather(pw, 1, inv)
	covered = torch.gather(covered, 1, inv)

	return trend_t, trend_s, v_trend, w_used, covered


import torch


@torch.no_grad()
def robust_linear_trend_sections_ransac(
	offsets: Tensor,  # (B,H) [m]
	t_sec: Tensor,  # (B,H) predicted pos_sec [s]
	valid: Tensor,  # (B,H) fb_idx>=0 (bool or int)
	*,
	w_conf: Tensor,  # (B,H) per-trace confidence weights (>=0, typically <=1)
	# windowing
	section_len: int = 128,
	stride: int = 64,
	# physical bounds
	vmin: float = 300.0,
	vmax: float = 6000.0,
	# RANSAC params
	ransac_trials: int = 32,  # total hypotheses per window
	ransac_tau: float = 2.0,  # multiplier for robust scale
	ransac_abs_ms: float = 15.0,  # absolute inlier threshold [ms]
	ransac_pack: int = 16,  # vectorized hypotheses per block
	sample_weighted: bool = True,  # sample two points ∝ confidence
	dx_min: float = 1e-6,  # minimal |x2-x1| to avoid degenerate fit
	refine_irls_iters: int = 1,  # IRLS refinement on best inliers (0=off)
	use_inlier_blend: bool = True,  # blend only inliers into the section output
	sort_offsets: bool = True,  # sort traces by offset for stability
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
	"""RANSAC ベースの線形トレンド推定。確率マップは扱わず、外部で構築した
	トレース別自信度 w_conf を使用する。

	Returns
	-------
	trend_t : (B,H) [s]
	trend_s : (B,H) [s/m]
	v_trend : (B,H) [m/s]
	w_used  : (B,H) 使用した w_conf（元順に復元）
	covered : (B,H) bool 少なくとも1つの窓でカバー

	"""
	assert offsets.ndim == 2 and t_sec.ndim == 2 and valid.ndim == 2, (
		'offsets/t_sec/valid must be (B,H)'
	)
	B, H = offsets.shape
	assert t_sec.shape == (B, H) and valid.shape == (B, H), 'shapes must match (B,H)'
	assert w_conf.ndim == 2 and w_conf.shape == (B, H), 'w_conf must be (B,H)'
	assert section_len >= 4 and stride >= 1 and ransac_trials >= 1, (
		'invalid window/RANSAC params'
	)
	assert vmin > 0 and vmax > vmin, '0 < vmin < vmax required'
	assert ransac_pack >= 1 and refine_irls_iters >= 0, (
		'invalid vectorization/refine params'
	)

	x0, y0 = offsets, t_sec
	v0 = (valid > 0).to(t_sec)
	pw0 = w_conf.to(t_sec)

	# optional sort by offsets for stability
	if sort_offsets:
		idx = torch.argsort(x0, dim=1)  # (B,H)
		arangeH = torch.arange(H, device=idx.device).unsqueeze(0).expand_as(idx)
		inv = torch.empty_like(idx)
		inv.scatter_(1, idx, arangeH)

		x = torch.gather(x0, 1, idx)
		y = torch.gather(y0, 1, idx)
		v = torch.gather(v0, 1, idx)
		pw = torch.gather(pw0, 1, idx)
	else:
		x, y, v, pw = x0, y0, v0, pw0
		inv = torch.arange(H, device=x.device).view(1, H).expand(B, H)

	trend_t = torch.zeros_like(y)
	trend_s = torch.zeros_like(y)
	counts = torch.zeros_like(y)

	eps = 1e-12
	abs_thr_sec = float(ransac_abs_ms) * 1e-3

	for start in range(0, H, stride):
		end = min(H, start + section_len)
		L = end - start
		if L < 4:
			continue

		xs = x[:, start:end]  # (B,L)
		ys = y[:, start:end]
		vs = v[:, start:end]  # (B,L)
		pws = pw[:, start:end]  # (B,L)

		base_w = (vs * pws).clamp_min(0)  # (B,L)
		ps_sum = base_w.sum(dim=1, keepdim=True)  # (B,1)
		active = ps_sum.squeeze(1) > 0  # (B,)
		if not torch.any(active):
			continue

		# operate only on active batch rows
		xs_a = xs[active]  # (Ba,L)
		ys_a = ys[active]
		vs_a = vs[active]
		pws_a = pws[active]
		B_a = xs_a.shape[0]

		# global robust scale per (batch, window)
		med_y = ys_a.median(dim=1, keepdim=True).values
		scale0 = (
			1.4826 * (ys_a - med_y).abs().median(dim=1, keepdim=True).values
		).clamp_min(1e-6)  # (Ba,1)
		thr = torch.maximum(
			ransac_tau * scale0, torch.full_like(scale0, abs_thr_sec)
		)  # (Ba,1)

		# sampling probabilities over L
		ps = (vs_a * pws_a).clamp_min(0)
		ps = ps / ps.sum(dim=1, keepdim=True)  # (Ba,L) ここは Ba>0 保証済み
		if not sample_weighted:
			ps.fill_(1.0 / L)

		best_score = torch.full((B_a,), -1e9, dtype=ys.dtype, device=ys.device)
		best_a = torch.zeros(B_a, dtype=ys.dtype, device=ys.device)
		best_b = torch.zeros(B_a, dtype=ys.dtype, device=ys.device)

		blocks = (ransac_trials + ransac_pack - 1) // ransac_pack
		for _ in range(blocks):
			K = ransac_pack

			# sample two indices per hypothesis
			i1 = torch.multinomial(ps, num_samples=K, replacement=True)  # (Ba,K)
			i2 = torch.multinomial(ps, num_samples=K, replacement=True)  # (Ba,K)
			same = i1 == i2
			if same.any():
				i2 = torch.where(same, (i2 + 1) % L, i2)  # avoid identical pairs

			# gather points
			x1 = xs_a.gather(1, i1)  # (Ba,K)
			y1 = ys_a.gather(1, i1)
			x2 = xs_a.gather(1, i2)
			y2 = ys_a.gather(1, i2)

			dx = x2 - x1
			good = dx.abs() >= dx_min  # (Ba,K)

			b = (y2 - y1) / dx
			a = y1 - b * x1
			b = torch.where(good, b, torch.nan)
			a = torch.where(good, a, torch.nan)

			# residuals per hypothesis: (Ba,K,L)
			yhat = a.unsqueeze(-1) + b.unsqueeze(-1) * xs_a.unsqueeze(1)
			r = (ys_a.unsqueeze(1) - yhat) * vs_a.unsqueeze(1)

			# inlier mask with global thr per (Ba,1,1)
			inlier = (r.abs() <= thr.unsqueeze(-1)) & good.unsqueeze(-1)  # (Ba,K,L)
			score = (pws_a.unsqueeze(1) * inlier.to(ys_a.dtype)).sum(dim=2)  # (Ba,K)

			# choose best per batch row
			score_max, idx_max = score.max(dim=1)  # (Ba,)
			take = score_max > best_score
			if take.any():
				ar = a[torch.arange(B_a, device=ys.device), idx_max]
				br = b[torch.arange(B_a, device=ys.device), idx_max]
				best_a = torch.where(take, ar, best_a)
				best_b = torch.where(take, br, best_b)
				best_score = torch.where(take, score_max, best_score)

		# ensure finite best hypotheses
		assert torch.isfinite(best_a).all() and torch.isfinite(best_b).all(), (
			'RANSAC failed to find a valid model'
		)

		# optional IRLS refinement on best inliers
		if refine_irls_iters > 0:
			a_ref = best_a.clone()
			b_ref = best_b.clone()
			for _ in range(refine_irls_iters):
				yhat = a_ref.view(B_a, 1) + b_ref.view(B_a, 1) * xs_a  # (Ba,L)
				res = (ys_a - yhat) * vs_a
				inl = (res.abs() <= thr).to(ys_a.dtype)  # (Ba,L)
				w = (vs_a * pws_a * inl).clamp_min(0)  # (Ba,L)

				Sw = w.sum(dim=1, keepdim=True).clamp_min(eps)
				Sx = (w * xs_a).sum(dim=1, keepdim=True)
				Sy = (w * ys_a).sum(dim=1, keepdim=True)
				Sxx = (w * xs_a * xs_a).sum(dim=1, keepdim=True)
				Sxy = (w * xs_a * ys_a).sum(dim=1, keepdim=True)
				D = (Sw * Sxx - Sx * Sx).clamp_min(eps)
				b_ref = ((Sw * Sxy - Sx * Sy) / D).squeeze(1)
				a_ref = ((Sy - b_ref.view(B_a, 1) * Sx) / Sw).squeeze(1)

			best_a, best_b = a_ref, b_ref

		# clamp slowness to physical range
		s_sec = best_b.clamp(min=1.0 / vmax, max=1.0 / vmin)  # (Ba,)

		# blending weights (Hann)
		wwin = torch.hann_window(
			L, periodic=False, device=ys.device, dtype=ys.dtype
		).view(1, L)
		yhat_best = best_a.view(B_a, 1) + best_b.view(B_a, 1) * xs_a  # (Ba,L)
		if use_inlier_blend:
			res = (ys_a - yhat_best) * vs_a
			inl = (res.abs() <= thr).to(ys_a.dtype)
			wtap = wwin * vs_a * pws_a * inl
		else:
			wtap = wwin * vs_a * pws_a

		trend_t[active, start:end] += yhat_best * wtap
		trend_s[active, start:end] += s_sec.view(B_a, 1) * wtap
		counts[active, start:end] += wtap

	trend_t = trend_t / counts.clamp_min(1e-6)
	trend_s = trend_s / counts.clamp_min(1e-6)
	v_trend = 1.0 / trend_s.clamp_min(1e-6)
	covered = (counts > 0).to(torch.bool)

	# restore original order
	trend_t = torch.gather(trend_t, 1, inv)
	trend_s = torch.gather(trend_s, 1, inv)
	v_trend = torch.gather(v_trend, 1, inv)
	w_used = torch.gather(pw, 1, inv)
	covered = torch.gather(covered, 1, inv)

	return trend_t, trend_s, v_trend, w_used, covered
