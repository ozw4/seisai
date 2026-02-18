# validators.py の関数を用いた入力検証版
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from seisai_utils.convert import to_numpy, to_torch

# 追加：validators から必要なものだけ import
from seisai_utils.validator import (
    require_all_finite,
    require_all_numpy,
    require_boolint_array,
    require_float_array,
    require_non_negative,
    require_same_shape_and_backend,
    validate_array,
)
from torch import Tensor

if TYPE_CHECKING:
    import numpy as np


def _apply_speed_bounds_on_slowness(
    b_slope: Tensor,  # slope (= slowness) [s/m], shape (B,1) or (Ba,)
    vmin: float | None,
    vmax: float | None,
    symmetric: bool,  # True: |v|∈[vmin,vmax] を符号保持で許容(v∈[-vmax,-vmin]∪[vmin,vmax])
) -> Tensor:
    # slowness bounds: s = 1/v
    min_s = 0.0 if vmax is None else 1.0 / float(vmax)
    max_s = float('inf') if vmin is None else 1.0 / float(vmin)
    if symmetric:
        sm = b_slope.abs().clamp(min=min_s, max=max_s)
        return torch.sign(b_slope) * sm
    return b_slope.clamp(min=min_s, max=max_s)


def _validation(
    offsets, t_sec, valid, w_conf, vmax, vmin, section_len, stride, iters
) -> None:
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

    # ---- パラメータ検証(スカラー系) ----
    assert section_len >= 4 and stride >= 1 and iters >= 1, 'invalid IRLS/window params'
    if vmin is not None:
        assert vmin > 0
    if vmax is not None:
        assert vmax > 0
    if (vmin is not None) and (vmax is not None):
        assert vmax > vmin


@torch.no_grad()
def robust_linear_trend(
    offsets: Tensor | np.ndarray,  # (B,H) or (H,)
    t_sec: Tensor | np.ndarray,  # (B,H) or (H,)
    valid: Tensor | np.ndarray | None = None,  # None → 全点有効
    *,
    w_conf: Tensor | np.ndarray,  # (B,H) or (H,)
    section_len: int = 128,
    stride: int = 64,
    huber_c: float = 1.345,
    iters: int = 3,
    vmin: float | None = 300.0,  # None → 片側無制限
    vmax: float | None = 8000.0,  # None → 片側無制限
    sort_offsets: bool = True,
    use_taper: bool = True,
    abs_velocity: bool = False,  # True: v∈[-vmax,-vmin]∪[vmin,vmax] を許容(出力は符号付き)
) -> tuple[
    Tensor | np.ndarray,
    Tensor | np.ndarray,
    Tensor | np.ndarray,
    Tensor | np.ndarray,
    Tensor | np.ndarray,
]:
    """Windowed IRLS で t(x) ≈ a + s·x を推定。
    - 入力がすべてNumPyなら NumPy を返す。そうでなければ Torch を返す。
    - 内部計算は Torch(CPU)で行う。dtype は t_sec に合わせて統一。.
    """
    # 返却形態の決定(全入力が NumPy なら True)
    if valid is None:
        all_numpy = require_all_numpy(offsets, t_sec, w_conf)
    else:
        all_numpy = require_all_numpy(offsets, t_sec, w_conf, valid)

    # Torchへ正規化(CPU/ dtype は t_sec に合わせる)
    t_tsec = to_torch(t_sec)
    t_offsets = to_torch(offsets, like=t_tsec)
    t_wconf = to_torch(w_conf, like=t_tsec)
    t_valid = None if valid is None else to_torch(valid)

    # (H,) → (1,H)
    if t_offsets.ndim == 1:
        t_offsets = t_offsets.unsqueeze(0)
    if t_tsec.ndim == 1:
        t_tsec = t_tsec.unsqueeze(0)
    if t_wconf.ndim == 1:
        t_wconf = t_wconf.unsqueeze(0)
    if (t_valid is not None) and (t_valid.ndim == 1):
        t_valid = t_valid.unsqueeze(0)

    # 入力検証(Torchバックエンド)
    _validation(
        t_offsets, t_tsec, t_valid, t_wconf, vmax, vmin, section_len, stride, iters
    )

    # dtype/device を完全一致
    t_offsets = t_offsets.to(t_tsec)

    # ---- 本体(従来Torch実装そのまま)----
    B, H = t_offsets.shape
    x0, y0 = t_offsets, t_tsec
    v0 = torch.ones_like(t_tsec) if t_valid is None else (t_valid > 0).to(t_tsec)
    pw0 = t_wconf.to(t_tsec)

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

        s_sec = _apply_speed_bounds_on_slowness(
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

    # すべて NumPy 入力のときだけ NumPy で返却
    if all_numpy:
        return to_numpy(trend_t, trend_s, v_trend, w_used, covered)
    return trend_t, trend_s, v_trend, w_used, covered


@torch.no_grad()
def robust_linear_trend_sections_ransac(
    offsets: Tensor | np.ndarray,  # (B,H) or (H,)
    t_sec: Tensor | np.ndarray,  # (B,H) or (H,)
    valid: Tensor | np.ndarray | None = None,  # None → 全点有効
    *,
    w_conf: Tensor | np.ndarray,  # (B,H) or (H,)
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
    abs_velocity: bool = False,  # True で v を対称許容(出力は符号付き)
) -> tuple[
    Tensor | np.ndarray,
    Tensor | np.ndarray,
    Tensor | np.ndarray,
    Tensor | np.ndarray,
    Tensor | np.ndarray,
]:
    """RANSAC ベースの線形トレンド推定(NumPy/Torch両対応)。
    - 入力がすべて NumPy なら NumPy を返す。そうでなければ Torch を返す。
    - 内部計算は Torch(CPU)で行い、dtype は t_sec に合わせる。.
    """
    # 返却形態の決定
    if valid is None:
        all_numpy = require_all_numpy(offsets, t_sec, w_conf)
    else:
        all_numpy = require_all_numpy(offsets, t_sec, w_conf, valid)

    # Torchへ正規化(CPU, dtype を t_sec に合わせる)
    t_tsec = to_torch(t_sec)
    t_offsets = to_torch(offsets, like=t_tsec)
    t_wconf = to_torch(w_conf, like=t_tsec)
    t_valid = None if valid is None else to_torch(valid)
    # (H,) を (1,H) に昇格
    if t_offsets.ndim == 1:
        t_offsets = t_offsets.unsqueeze(0)
    if t_tsec.ndim == 1:
        t_tsec = t_tsec.unsqueeze(0)
    if t_wconf.ndim == 1:
        t_wconf = t_wconf.unsqueeze(0)
    if (t_valid is not None) and (t_valid.ndim == 1):
        t_valid = t_valid.unsqueeze(0)

    # 入力検証(Torchバックエンド)
    _validation(
        t_offsets,
        t_tsec,
        t_valid,
        t_wconf,
        vmax,
        vmin,
        section_len,
        stride,
        ransac_trials,
    )
    assert ransac_pack >= 1
    assert refine_irls_iters >= 0

    # dtype/device 完全一致
    t_offsets = t_offsets.to(t_tsec)

    # ---- 本体処理(従来Torch実装)----
    B, H = t_offsets.shape
    x0, y0 = t_offsets, t_tsec
    v0 = torch.ones_like(t_tsec) if t_valid is None else (t_valid > 0).to(t_tsec)
    pw0 = t_wconf.to(t_tsec)

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

        s_sec = _apply_speed_bounds_on_slowness(
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

    # すべて NumPy 入力のときだけ NumPy で返却
    if all_numpy:
        return to_numpy(trend_t, trend_s, v_trend, w_used, covered)
    return trend_t, trend_s, v_trend, w_used, covered
