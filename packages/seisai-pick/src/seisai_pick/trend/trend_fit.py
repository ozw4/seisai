# validators.py の関数を用いた入力検証版
from __future__ import annotations

import numpy as np
import torch

# 追加：validators から必要なものだけ import
from seisai_utils.validator import (
	require_all_finite,
	require_boolint_array,
	require_float_array,
	require_non_negative,
	require_same_shape_and_backend,
	validate_array,
)
from torch import Tensor


def _apply_speed_bounds_on_slowness_np(
	b_slope: np.ndarray,  # shape (B,1) or (Ba,)
	vmin: float | None,
	vmax: float | None,
	symmetric: bool,
) -> np.ndarray:
	# slowness bounds: s = 1/v
	min_s = 0.0 if vmax is None else 1.0 / float(vmax)
	max_s = float('inf') if vmin is None else 1.0 / float(vmin)
	if symmetric:
		sm = np.clip(np.abs(b_slope), a_min=min_s, a_max=max_s)
		return np.sign(b_slope) * sm
	return np.clip(b_slope, a_min=min_s, a_max=max_s)


def _apply_speed_bounds_on_slowness_torch(
	b_slope: Tensor,  # slope (= slowness) [s/m], shape (B,1) or (Ba,)
	vmin: float | None,
	vmax: float | None,
	symmetric: bool,  # True: |v|∈[vmin,vmax] を符号保持で許容（v∈[-vmax,-vmin]∪[vmin,vmax]）
) -> Tensor:
	# slowness bounds: s = 1/v
	min_s = 0.0 if vmax is None else 1.0 / float(vmax)
	max_s = float('inf') if vmin is None else 1.0 / float(vmin)
	if symmetric:
		sm = b_slope.abs().clamp(min=min_s, max=max_s)
		return torch.sign(b_slope) * sm
	return b_slope.clamp(min=min_s, max=max_s)


def _validation_torch(
	offsets, t_sec, valid, w_conf, vmax, vmin, section_len, stride, iters
):
	validate_array(
		offsets, allowed_ndims=(2,), name='offsets', backend='torch', shape_hint='(B,H)'
	)
	require_float_array(t_sec, name='t_sec', backend='torch')
	require_all_finite(offsets, name='offsets', backend='torch')
	require_all_finite(t_sec, name='t_sec', backend='torch')
	require_all_finite(w_conf, name='w_conf', backend='torch')
	require_non_negative(w_conf, name='w_conf', backend='torch')

	require_same_shape_and_backend(
		offsets,
		t_sec,
		w_conf,
		name_a='offsets',
		name_b='t_sec',
		other_names=['w_conf'],
		backend='torch',
		shape_hint='(B,H)',
	)

	if valid is not None:
		validate_array(
			valid, allowed_ndims=(2,), name='valid', backend='torch', shape_hint='(B,H)'
		)
		require_boolint_array(valid, name='valid', backend='torch')
		require_same_shape_and_backend(
			offsets,
			valid,
			name_a='offsets',
			name_b='valid',
			backend='torch',
			shape_hint='(B,H)',
		)

	# ---- パラメータ検証（スカラー系） ----
	assert section_len >= 4 and stride >= 1 and iters >= 1, 'invalid IRLS/window params'
	if vmin is not None:
		assert vmin > 0
	if vmax is not None:
		assert vmax > 0
	if (vmin is not None) and (vmax is not None):
		assert vmax > vmin


def _validation_np(
	offsets, t_sec, valid, w_conf, vmax, vmin, section_len, stride, iters
):
	# 形状・型
	validate_array(
		offsets, allowed_ndims=(2,), name='offsets', backend='numpy', shape_hint='(B,H)'
	)
	require_float_array(t_sec, name='t_sec', backend='numpy')
	require_all_finite(offsets, name='offsets', backend='numpy')
	require_all_finite(t_sec, name='t_sec', backend='numpy')
	require_all_finite(w_conf, name='w_conf', backend='numpy')
	require_non_negative(w_conf, name='w_conf', backend='numpy')

	require_same_shape_and_backend(
		offsets,
		t_sec,
		w_conf,
		name_a='offsets',
		name_b='t_sec',
		other_names=['w_conf'],
		backend='numpy',
		shape_hint='(B,H)',
	)

	if valid is not None:
		validate_array(
			valid, allowed_ndims=(2,), name='valid', backend='numpy', shape_hint='(B,H)'
		)
		require_boolint_array(valid, name='valid', backend='numpy')
		require_same_shape_and_backend(
			offsets,
			valid,
			name_a='offsets',
			name_b='valid',
			backend='numpy',
			shape_hint='(B,H)',
		)

	# スカラー検証
	assert section_len >= 4 and stride >= 1 and iters >= 1, 'invalid IRLS/window params'
	if vmin is not None:
		assert vmin > 0
	if vmax is not None:
		assert vmax > 0
	if (vmin is not None) and (vmax is not None):
		assert vmax > vmin


@torch.no_grad()
def robust_linear_trend(
	offsets: Tensor,  # (B,H) [m]
	t_sec: Tensor,  # (B,H) predicted pos_sec [s]
	valid: Tensor | None = None,  # None → 全点有効
	*,
	w_conf: Tensor,  # (B,H) per-trace confidence weights (>=0, typically <=1)
	section_len: int = 128,
	stride: int = 64,
	huber_c: float = 1.345,
	iters: int = 3,
	vmin: float | None = 300.0,  # None で片側無制限
	vmax: float | None = 8000.0,  # None で片側無制限
	sort_offsets: bool = True,
	use_taper: bool = True,
	abs_velocity: bool = False,  # True のとき v∈[-vmax,-vmin]∪[vmin,vmax] を許容（出力は符号付き）
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
	"""Windowed IRLS で t(x) ≈ a + s·x を推定。validators を用いて (B,H) 形状を検証。"""
	if offsets.ndim == 1:
		offsets = offsets.unsqueeze(0)
	if t_sec.ndim == 1:
		t_sec = t_sec.unsqueeze(0)
	if w_conf.ndim == 1:
		w_conf = w_conf.unsqueeze(0)
	if valid is not None and valid.ndim == 1:
		valid = valid.unsqueeze(0)
	# ---- 入力検証 ----
	_validation_torch(
		offsets, t_sec, valid, w_conf, vmax, vmin, section_len, stride, iters
	)
	offsets = offsets.to(t_sec)

	# ---- 本体処理 ----
	B, H = offsets.shape
	x0, y0 = offsets, t_sec
	v0 = torch.ones_like(t_sec) if valid is None else (valid > 0).to(t_sec)
	pw0 = w_conf.to(t_sec)

	if sort_offsets:
		idx = torch.argsort(x0, dim=1)
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

		xs = x[:, start:end]
		ys = y[:, start:end]
		vs = v[:, start:end]
		pws = pw[:, start:end]

		w = (vs * pws).clone()
		a = torch.zeros(B, 1, dtype=y.dtype, device=y.device)
		b = torch.zeros(B, 1, dtype=y.dtype, device=y.device)  # slope (= slowness)

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

			scale = (1.4826 * res.abs().median(dim=1, keepdim=True).values).clamp_min(
				1e-6
			)
			r = res / (huber_c * scale)

			w_huber = torch.where(
				r.abs() <= 1.0, vs, vs * (1.0 / r.abs()).clamp_max(10.0)
			)
			w = w_huber * pws

		s_sec = _apply_speed_bounds_on_slowness_torch(
			b.squeeze(1), vmin, vmax, symmetric=abs_velocity
		)

		wwin = (
			torch.hann_window(L, periodic=False, device=y.device, dtype=y.dtype).view(
				1, L
			)
			if use_taper
			else torch.ones(1, L, device=y.device, dtype=y.dtype)
		)
		wtap = wwin * vs * pws

		yhat = a + b * xs
		trend_t[:, start:end] += yhat * wtap
		trend_s[:, start:end] += s_sec[:, None] * wtap
		counts[:, start:end] += wtap

	trend_t = trend_t / counts.clamp_min(1e-6)
	trend_s = trend_s / counts.clamp_min(1e-6)

	v_trend = torch.sign(trend_s) / trend_s.abs().clamp_min(1e-6)  # 常に符号付き
	covered = (counts > 0).to(torch.bool)

	trend_t = torch.gather(trend_t, 1, inv)
	trend_s = torch.gather(trend_s, 1, inv)
	v_trend = torch.gather(v_trend, 1, inv)
	w_used = torch.gather(pw, 1, inv)
	covered = torch.gather(covered, 1, inv)

	return trend_t, trend_s, v_trend, w_used, covered


@torch.no_grad()
def robust_linear_trend_sections_ransac(
	offsets: Tensor,  # (B,H) [m]
	t_sec: Tensor,  # (B,H) predicted pos_sec [s]
	valid: Tensor | None = None,  # None → 全点有効
	*,
	w_conf: Tensor,  # (B,H) per-trace confidence weights (>=0, typically <=1)
	section_len: int = 128,
	stride: int = 64,
	vmin: float | None = 300.0,
	vmax: float | None = 6000.0,
	ransac_trials: int = 32,
	ransac_tau: float = 2.0,
	ransac_abs_ms: float = 15.0,
	ransac_pack: int = 16,
	sample_weighted: bool = True,
	dx_min: float = 1e-6,
	refine_irls_iters: int = 1,
	use_inlier_blend: bool = True,
	sort_offsets: bool = True,
	abs_velocity: bool = False,  # True で v を対称許容（出力は符号付き）
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
	"""RANSAC ベースの線形トレンド推定。validators を用いて (B,H) 形状を検証。"""
	if offsets.ndim == 1:
		offsets = offsets.unsqueeze(0)
	if t_sec.ndim == 1:
		t_sec = t_sec.unsqueeze(0)
	if w_conf.ndim == 1:
		w_conf = w_conf.unsqueeze(0)
	if valid is not None and valid.ndim == 1:
		valid = valid.unsqueeze(0)
	# ---- 入力検証 ----
	_validation_torch(
		offsets, t_sec, valid, w_conf, vmax, vmin, section_len, stride, ransac_trials
	)
	assert ransac_pack >= 1 and refine_irls_iters >= 0
	offsets = offsets.to(t_sec)

	# ---- 本体処理 ----
	B, H = offsets.shape
	x0, y0 = offsets, t_sec
	v0 = torch.ones_like(t_sec) if valid is None else (valid > 0).to(t_sec)
	pw0 = w_conf.to(t_sec)

	if sort_offsets:
		idx = torch.argsort(x0, dim=1)
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

		xs = x[:, start:end]
		ys = y[:, start:end]
		vs = v[:, start:end]
		pws = pw[:, start:end]

		base_w = (vs * pws).clamp_min(0)
		if not torch.any(base_w.sum(dim=1) > 0).item():
			continue

		active = base_w.sum(dim=1) > 0
		xs_a, ys_a, vs_a, pws_a = xs[active], ys[active], vs[active], pws[active]
		B_a = xs_a.shape[0]

		med_y = ys_a.median(dim=1, keepdim=True).values
		scale0 = (
			1.4826 * (ys_a - med_y).abs().median(dim=1, keepdim=True).values
		).clamp_min(1e-6)
		thr = torch.maximum(ransac_tau * scale0, torch.full_like(scale0, abs_thr_sec))

		ps = (vs_a * pws_a).clamp_min(0)
		ps = ps / ps.sum(dim=1, keepdim=True)
		if not sample_weighted:
			ps.fill_(1.0 / L)

		best_score = torch.full((B_a,), -1e9, dtype=ys.dtype, device=ys.device)
		best_a = torch.zeros(B_a, dtype=ys.dtype, device=ys.device)
		best_b = torch.zeros(B_a, dtype=ys.dtype, device=ys.device)

		blocks = (ransac_trials + ransac_pack - 1) // ransac_pack
		for _ in range(blocks):
			K = ransac_pack
			i1 = torch.multinomial(ps, num_samples=K, replacement=True)
			i2 = torch.multinomial(ps, num_samples=K, replacement=True)
			same = i1 == i2
			if same.any():
				i2 = torch.where(same, (i2 + 1) % L, i2)

			x1 = xs_a.gather(1, i1)
			y1 = ys_a.gather(1, i1)
			x2 = xs_a.gather(1, i2)
			y2 = ys_a.gather(1, i2)

			dx = x2 - x1
			good = dx.abs() >= dx_min

			b = (y2 - y1) / dx
			a = y1 - b * x1
			b = torch.where(good, b, torch.nan)
			a = torch.where(good, a, torch.nan)

			yhat = a.unsqueeze(-1) + b.unsqueeze(-1) * xs_a.unsqueeze(1)  # (Ba,K,L)
			r = (ys_a.unsqueeze(1) - yhat) * vs_a.unsqueeze(1)
			inlier = (r.abs() <= thr.unsqueeze(-1)) & good.unsqueeze(-1)
			score = (pws_a.unsqueeze(1) * inlier.to(ys_a.dtype)).sum(dim=2)

			score_max, idx_max = score.max(dim=1)
			take = score_max > best_score
			if take.any():
				ar = a[torch.arange(B_a, device=ys.device), idx_max]
				br = b[torch.arange(B_a, device=ys.device), idx_max]
				best_a = torch.where(take, ar, best_a)
				best_b = torch.where(take, br, best_b)
				best_score = torch.where(take, score_max, best_score)

		assert torch.isfinite(best_a).all() and torch.isfinite(best_b).all(), (
			'RANSAC failed to find a valid model'
		)

		if refine_irls_iters > 0:
			a_ref, b_ref = best_a.clone(), best_b.clone()
			for _ in range(refine_irls_iters):
				yhat = a_ref.view(B_a, 1) + b_ref.view(B_a, 1) * xs_a
				res = (ys_a - yhat) * vs_a
				inl = (res.abs() <= thr).to(ys_a.dtype)
				w = (vs_a * pws_a * inl).clamp_min(0)

				Sw = w.sum(dim=1, keepdim=True).clamp_min(eps)
				Sx = (w * xs_a).sum(dim=1, keepdim=True)
				Sy = (w * ys_a).sum(dim=1, keepdim=True)
				Sxx = (w * xs_a * xs_a).sum(dim=1, keepdim=True)
				Sxy = (w * xs_a * ys_a).sum(dim=1, keepdim=True)
				D = (Sw * Sxx - Sx * Sx).clamp_min(eps)
				b_ref = ((Sw * Sxy - Sx * Sy) / D).squeeze(1)
				a_ref = ((Sy - b_ref.view(B_a, 1) * Sx) / Sw).squeeze(1)

			best_a, best_b = a_ref, b_ref

		s_sec = _apply_speed_bounds_on_slowness_torch(
			best_b, vmin, vmax, symmetric=abs_velocity
		)

		wwin = torch.hann_window(
			L, periodic=False, device=ys.device, dtype=ys.dtype
		).view(1, L)
		yhat_best = best_a.view(B_a, 1) + best_b.view(B_a, 1) * xs_a
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
	v_trend = torch.sign(trend_s) / trend_s.abs().clamp_min(1e-6)  # 常に符号付き

	covered = (counts > 0).to(torch.bool)
	trend_t = torch.gather(trend_t, 1, inv)
	trend_s = torch.gather(trend_s, 1, inv)
	v_trend = torch.gather(v_trend, 1, inv)
	w_used = torch.gather(pw, 1, inv)
	covered = torch.gather(covered, 1, inv)

	return trend_t, trend_s, v_trend, w_used, covered


def robust_linear_trend_np(
	offsets: np.ndarray,  # (B,H) [m]  整数可（内部で t_sec.dtype に統一）
	t_sec: np.ndarray,  # (B,H) [s]  float必須
	valid: np.ndarray | None = None,  # (B,H) bool/int（Noneなら全点有効）
	*,
	w_conf: np.ndarray,  # (B,H) >=0（0/1マスクも可）→ floatに演算される
	section_len: int = 128,
	stride: int = 64,
	huber_c: float = 1.345,
	iters: int = 3,
	vmin: float | None = 300.0,
	vmax: float | None = 8000.0,
	sort_offsets: bool = True,
	use_taper: bool = True,
	abs_velocity: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	if offsets.ndim == 1:
		offsets = offsets[np.newaxis, :]
	if t_sec.ndim == 1:
		t_sec = t_sec[np.newaxis, :]
	if w_conf.ndim == 1:
		w_conf = w_conf[np.newaxis, :]
	if valid is not None and valid.ndim == 1:
		valid = valid[np.newaxis, :]

	"""Windowed IRLS で t(x) ≈ a + s·x を推定（NumPy版）"""
	_validation_np(
		offsets, t_sec, valid, w_conf, vmax, vmin, section_len, stride, iters
	)

	# dtype統一（offsetsはint許容 → t_sec.dtypeへ明示キャスト）
	offsets = offsets.astype(t_sec.dtype, copy=False)
	w_conf = w_conf.astype(t_sec.dtype, copy=False)

	B, H = offsets.shape
	x0, y0 = offsets, t_sec
	v0 = (
		np.ones_like(t_sec, dtype=t_sec.dtype)
		if valid is None
		else (valid > 0).astype(t_sec.dtype)
	)
	pw0 = w_conf

	if sort_offsets:
		idx = np.argsort(x0, axis=1)
		arangeH = np.arange(H, dtype=idx.dtype)
		inv = np.empty_like(idx)
		inv[np.arange(B)[:, None], idx] = arangeH[None, :]

		x = np.take_along_axis(x0, idx, axis=1)
		y = np.take_along_axis(y0, idx, axis=1)
		v = np.take_along_axis(v0, idx, axis=1)
		pw = np.take_along_axis(pw0, idx, axis=1)
	else:
		x, y, v, pw = x0, y0, v0, pw0
		inv = np.broadcast_to(np.arange(H)[None, :], (B, H))

	trend_t = np.zeros_like(y)
	trend_s = np.zeros_like(y)
	counts = np.zeros_like(y)

	eps = 1e-12
	for start in range(0, H, stride):
		end = min(H, start + section_len)
		L = end - start
		if L < 4:
			continue

		xs = x[:, start:end]
		ys = y[:, start:end]
		vs = v[:, start:end]
		pws = pw[:, start:end]

		w = (vs * pws).copy()
		a = np.zeros((B, 1), dtype=y.dtype)
		b = np.zeros((B, 1), dtype=y.dtype)  # slope (= slowness)

		for _ in range(iters):
			Sw = np.clip(np.sum(w, axis=1, keepdims=True), eps, None)
			Sx = np.sum(w * xs, axis=1, keepdims=True)
			Sy = np.sum(w * ys, axis=1, keepdims=True)
			Sxx = np.sum(w * xs * xs, axis=1, keepdims=True)
			Sxy = np.sum(w * xs * ys, axis=1, keepdims=True)
			D = np.clip(Sw * Sxx - Sx * Sx, eps, None)
			b = (Sw * Sxy - Sx * Sy) / D
			a = (Sy - b * Sx) / Sw

			yhat = a + b * xs
			res = (ys - yhat) * vs

			scale = np.clip(
				1.4826 * np.median(np.abs(res), axis=1, keepdims=True), 1e-6, None
			)
			r = res / (huber_c * scale)

			w_huber = np.where(
				np.abs(r) <= 1.0, vs, vs * np.minimum(1.0 / np.abs(r), 10.0)
			)
			w = w_huber * pws

		s_sec = _apply_speed_bounds_on_slowness_np(
			b.squeeze(1), vmin, vmax, symmetric=abs_velocity
		)

		if use_taper:
			wwin = np.hanning(L).astype(y.dtype)[None, :]
		else:
			wwin = np.ones((1, L), dtype=y.dtype)
		wtap = wwin * vs * pws

		yhat = a + b * xs
		trend_t[:, start:end] += yhat * wtap
		trend_s[:, start:end] += s_sec[:, None] * wtap
		counts[:, start:end] += wtap

	trend_t = trend_t / np.clip(counts, 1e-6, None)
	trend_s = trend_s / np.clip(counts, 1e-6, None)

	# 出力速度は常に符号付き
	v_trend = np.sign(trend_s) / np.clip(np.abs(trend_s), 1e-6, None)
	covered = (counts > 0).astype(bool)

	trend_t = np.take_along_axis(trend_t, inv, axis=1)
	trend_s = np.take_along_axis(trend_s, inv, axis=1)
	v_trend = np.take_along_axis(v_trend, inv, axis=1)
	w_used = np.take_along_axis(pw, inv, axis=1)
	covered = np.take_along_axis(covered, inv, axis=1)

	return trend_t, trend_s, v_trend, w_used, covered


def robust_linear_trend_sections_ransac_np(
	offsets: np.ndarray,  # (B,H) [m]
	t_sec: np.ndarray,  # (B,H) [s]
	valid: np.ndarray | None = None,
	*,
	w_conf: np.ndarray,  # (B,H)
	section_len: int = 128,
	stride: int = 64,
	vmin: float | None = 300.0,
	vmax: float | None = 6000.0,
	ransac_trials: int = 32,
	ransac_tau: float = 2.0,
	ransac_abs_ms: float = 15.0,
	ransac_pack: int = 16,  # NumPy版では内部的には逐次試行。引数は互換のため残置。
	sample_weighted: bool = True,
	dx_min: float = 1e-6,
	refine_irls_iters: int = 1,
	use_inlier_blend: bool = True,
	sort_offsets: bool = True,
	abs_velocity: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	if offsets.ndim == 1:
		offsets = offsets[np.newaxis, :]
	if t_sec.ndim == 1:
		t_sec = t_sec[np.newaxis, :]
	if w_conf.ndim == 1:
		w_conf = w_conf[np.newaxis, :]
	if valid is not None and valid.ndim == 1:
		valid = valid[np.newaxis, :]

	"""RANSACベースの線形トレンド推定（NumPy版）"""
	_validation_np(
		offsets, t_sec, valid, w_conf, vmax, vmin, section_len, stride, ransac_trials
	)
	assert ransac_pack >= 1 and refine_irls_iters >= 0

	# dtype統一
	offsets = offsets.astype(t_sec.dtype, copy=False)
	w_conf = w_conf.astype(t_sec.dtype, copy=False)

	B, H = offsets.shape
	x0, y0 = offsets, t_sec
	v0 = (
		np.ones_like(t_sec, dtype=t_sec.dtype)
		if valid is None
		else (valid > 0).astype(t_sec.dtype)
	)
	pw0 = w_conf

	if sort_offsets:
		idx = np.argsort(x0, axis=1)
		arangeH = np.arange(H, dtype=idx.dtype)
		inv = np.empty_like(idx)
		inv[np.arange(B)[:, None], idx] = arangeH[None, :]

		x = np.take_along_axis(x0, idx, axis=1)
		y = np.take_along_axis(y0, idx, axis=1)
		v = np.take_along_axis(v0, idx, axis=1)
		pw = np.take_along_axis(pw0, idx, axis=1)
	else:
		x, y, v, pw = x0, y0, v0, pw0
		inv = np.broadcast_to(np.arange(H)[None, :], (B, H))

	trend_t = np.zeros_like(y)
	trend_s = np.zeros_like(y)
	counts = np.zeros_like(y)

	eps = 1e-12
	abs_thr_sec = float(ransac_abs_ms) * 1e-3

	rng = np.random.default_rng()
	for start in range(0, H, stride):
		end = min(H, start + section_len)
		L = end - start
		if L < 4:
			continue

		xs = x[:, start:end]
		ys = y[:, start:end]
		vs = v[:, start:end]
		pws = pw[:, start:end]

		base_w = np.clip(vs * pws, 0, None)
		active_mask = np.sum(base_w, axis=1) > 0
		if not np.any(active_mask):
			continue

		xs_a = xs[active_mask]
		ys_a = ys[active_mask]
		vs_a = vs[active_mask]
		pws_a = pws[active_mask]
		Ba = xs_a.shape[0]

		med_y = np.median(ys_a, axis=1, keepdims=True)
		scale0 = np.clip(
			1.4826 * np.median(np.abs(ys_a - med_y), axis=1, keepdims=True), 1e-6, None
		)
		thr = np.maximum(ransac_tau * scale0, abs_thr_sec)

		ps = np.clip(vs_a * pws_a, 0, None)
		row_sum = np.sum(ps, axis=1, keepdims=True)
		ps = ps / row_sum
		if not sample_weighted:
			ps[:] = 1.0 / L

		best_score = np.full((Ba,), -1e9, dtype=ys.dtype)
		best_a = np.zeros((Ba,), dtype=ys.dtype)
		best_b = np.zeros((Ba,), dtype=ys.dtype)

		for _ in range(ransac_trials):
			# 各バッチ独立に1本サンプル（2点）
			i1 = np.empty((Ba,), dtype=int)
			i2 = np.empty((Ba,), dtype=int)
			for b in range(Ba):
				i1[b] = rng.choice(L, p=ps[b])
				i2[b] = rng.choice(L, p=ps[b])
				if i2[b] == i1[b]:
					i2[b] = (i2[b] + 1) % L

			x1 = xs_a[np.arange(Ba), i1]
			y1 = ys_a[np.arange(Ba), i1]
			x2 = xs_a[np.arange(Ba), i2]
			y2 = ys_a[np.arange(Ba), i2]

			dx = x2 - x1
			good = np.abs(dx) >= dx_min

			# 直線パラメータ
			b = np.empty_like(x1)
			a = np.empty_like(x1)
			b[good] = (y2[good] - y1[good]) / dx[good]
			a[good] = y1[good] - b[good] * x1[good]

			# 不正はスコア極小でスキップ
			if not np.all(good):
				bad = ~good
				b[bad] = np.nan
				a[bad] = np.nan

			# スコア（Huberでなく単純なinlier数の重み付き）
			# yhat: (Ba, L)
			yhat = a[:, None] + b[:, None] * xs_a
			r = (ys_a - yhat) * vs_a
			inlier = (np.abs(r) <= thr) & good[:, None]
			score = np.sum(pws_a * inlier.astype(ys_a.dtype), axis=1)

			take = score > best_score
			if np.any(take):
				best_a[take] = a[take]
				best_b[take] = b[take]
				best_score[take] = score[take]

		# RANSAC失敗チェック
		if not np.all(np.isfinite(best_a)) or not np.all(np.isfinite(best_b)):
			raise RuntimeError('RANSAC failed to find a valid model')

		# IRLSでの軽いリファイン
		if refine_irls_iters > 0:
			a_ref = best_a.copy()
			b_ref = best_b.copy()
			for _ in range(refine_irls_iters):
				yhat = a_ref[:, None] + b_ref[:, None] * xs_a
				res = (ys_a - yhat) * vs_a
				inl = (np.abs(res) <= thr).astype(ys_a.dtype)
				w = np.clip(vs_a * pws_a * inl, 0, None)

				Sw = np.clip(np.sum(w, axis=1, keepdims=True), eps, None)
				Sx = np.sum(w * xs_a, axis=1, keepdims=True)
				Sy = np.sum(w * ys_a, axis=1, keepdims=True)
				Sxx = np.sum(w * xs_a * xs_a, axis=1, keepdims=True)
				Sxy = np.sum(w * xs_a * ys_a, axis=1, keepdims=True)
				D = np.clip(Sw * Sxx - Sx * Sx, eps, None)
				b_ref = ((Sw * Sxy - Sx * Sy) / D).squeeze(1)
				a_ref = ((Sy - b_ref[:, None] * Sx) / Sw).squeeze(1)

			best_a, best_b = a_ref, b_ref

		s_sec = _apply_speed_bounds_on_slowness_np(
			best_b, vmin, vmax, symmetric=abs_velocity
		)

		wwin = np.hanning(L).astype(ys.dtype)[None, :]
		yhat_best = best_a[:, None] + best_b[:, None] * xs_a
		if use_inlier_blend:
			res = (ys_a - yhat_best) * vs_a
			inl = (np.abs(res) <= thr).astype(ys_a.dtype)
			wtap = wwin * vs_a * pws_a * inl
		else:
			wtap = wwin * vs_a * pws_a

		trend_t[active_mask, start:end] += yhat_best * wtap
		trend_s[active_mask, start:end] += s_sec[:, None] * wtap
		counts[active_mask, start:end] += wtap

	trend_t = trend_t / np.clip(counts, 1e-6, None)
	trend_s = trend_s / np.clip(counts, 1e-6, None)
	v_trend = np.sign(trend_s) / np.clip(np.abs(trend_s), 1e-6, None)

	covered = (counts > 0).astype(bool)
	trend_t = np.take_along_axis(trend_t, inv, axis=1)
	trend_s = np.take_along_axis(trend_s, inv, axis=1)
	v_trend = np.take_along_axis(v_trend, inv, axis=1)
	w_used = np.take_along_axis(pw, inv, axis=1)
	covered = np.take_along_axis(covered, inv, axis=1)

	return trend_t, trend_s, v_trend, w_used, covered
