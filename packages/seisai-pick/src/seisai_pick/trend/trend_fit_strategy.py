# ============================================================
# 目的: トレンド推定の戦略をインスタンスで差し替える(Strategyパターン)
# 依存: torch, seisai_pick.trend.trend_fit
# ============================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from seisai_pick.trend.trend_fit import (
    robust_linear_trend,
    robust_linear_trend_sections_ransac,
)

if TYPE_CHECKING:
    from torch import Tensor


class TrendFitStrategy(Protocol):
    name: str

    def __call__(
        self,
        *,
        offsets: Tensor,  # (B,H) [m]
        t_sec: Tensor,  # (B,H) [s]
        valid: Tensor,  # (B,H) bool/int
        w_conf: Tensor,  # (B,H)
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return: trend_t, trend_s, v_trend, w_used, covered (all (B,H))."""
        ...


# ---------------- IRLS 戦略 ----------------
@dataclass(frozen=True)
class IRLSStrategy:
    name: str = 'irls'
    section_len: int = 128
    stride: int = 64
    huber_c: float = 1.345
    iters: int = 3
    vmin: float = 300.0
    vmax: float = 8000.0
    sort_offsets: bool = True
    use_taper: bool = True

    def __call__(
        self, *, offsets: Tensor, t_sec: Tensor, valid: Tensor, w_conf: Tensor
    ):
        assert offsets.shape == t_sec.shape == valid.shape == w_conf.shape
        return robust_linear_trend(
            offsets=offsets,
            t_sec=t_sec,
            valid=valid,
            w_conf=w_conf,
            section_len=self.section_len,
            stride=self.stride,
            huber_c=self.huber_c,
            iters=self.iters,
            vmin=self.vmin,
            vmax=self.vmax,
            sort_offsets=self.sort_offsets,
            use_taper=self.use_taper,
        )


# ---------------- RANSAC 戦略 ----------------
@dataclass(frozen=True)
class RANSACStrategy:
    name: str = 'ransac'
    section_len: int = 128
    stride: int = 64
    vmin: float = 300.0
    vmax: float = 8000.0
    ransac_trials: int = 64
    ransac_tau: float = 2.0
    ransac_abs_ms: float = 12.0
    ransac_pack: int = 16
    sample_weighted: bool = True
    refine_irls_iters: int = 1
    use_inlier_blend: bool = True
    sort_offsets: bool = True

    def __call__(
        self, *, offsets: Tensor, t_sec: Tensor, valid: Tensor, w_conf: Tensor
    ):
        assert offsets.shape == t_sec.shape == valid.shape == w_conf.shape
        return robust_linear_trend_sections_ransac(
            offsets=offsets,
            t_sec=t_sec,
            valid=valid,
            w_conf=w_conf,
            section_len=self.section_len,
            stride=self.stride,
            vmin=self.vmin,
            vmax=self.vmax,
            ransac_trials=self.ransac_trials,
            ransac_tau=self.ransac_tau,
            ransac_abs_ms=self.ransac_abs_ms,
            ransac_pack=self.ransac_pack,
            sample_weighted=self.sample_weighted,
            refine_irls_iters=self.refine_irls_iters,
            use_inlier_blend=self.use_inlier_blend,
            sort_offsets=self.sort_offsets,
        )


# ---------------- 2-piece RANSAC(auto break) 戦略 ----------------
@dataclass(frozen=True)
class PiecewiseLinearTrend:
    edges: torch.Tensor
    coef: torch.Tensor

    def predict(self, x_abs: torch.Tensor) -> torch.Tensor:
        x = x_abs.to(dtype=torch.float32)
        if int(x.ndim) != 1:
            msg = f'x_abs must be 1D, got {tuple(x.shape)}'
            raise ValueError(msg)

        edges = self.edges.to(device=x.device, dtype=torch.float32)
        coef = self.coef.to(device=x.device, dtype=torch.float32)
        if tuple(edges.shape) != (3,):
            msg = f'edges must be (3,), got {tuple(edges.shape)}'
            raise ValueError(msg)
        if tuple(coef.shape) != (2, 2):
            msg = f'coef must be (2,2), got {tuple(coef.shape)}'
            raise ValueError(msg)

        xb = float(edges[1].item())
        a1, b1 = float(coef[0, 0].item()), float(coef[0, 1].item())
        a2, b2 = float(coef[1, 0].item()), float(coef[1, 1].item())

        y = x.new_full(x.shape, float('nan'))
        v = x.isfinite()
        if not bool(v.any()):
            return y

        s1 = v & (x <= xb)
        s2 = v & (x > xb)
        if bool(s1.any()):
            y[s1] = a1 * x[s1] + b1
        if bool(s2.any()):
            y[s2] = a2 * x[s2] + b2
        return y


def _fit_line_ls(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    xx = x.to(dtype=torch.float64)
    yy = y.to(dtype=torch.float64)
    if int(xx.ndim) != 1 or int(yy.ndim) != 1 or int(xx.numel()) != int(yy.numel()):
        msg = (
            f'x/y must be 1D with same length, got {tuple(xx.shape)}, {tuple(yy.shape)}'
        )
        raise ValueError(msg)
    if int(xx.numel()) < 2:
        msg = f'need at least 2 points, got {int(xx.numel())}'
        raise ValueError(msg)

    xm = xx.mean()
    ym = yy.mean()
    dx = xx - xm
    den = (dx * dx).sum()
    if float(den.item()) <= 1e-12:
        return 0.0, float(ym.item())

    a = float(((dx * (yy - ym)).sum() / den).item())
    b = float((ym - a * xm).item())
    return a, b


def _ransac_line(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_iter: int,
    inlier_th: float,
    seed: int,
) -> tuple[float, float, torch.Tensor]:
    xx = x.to(dtype=torch.float64)
    yy = y.to(dtype=torch.float64)
    if int(xx.ndim) != 1 or int(yy.ndim) != 1 or int(xx.numel()) != int(yy.numel()):
        msg = (
            f'x/y must be 1D with same length, got {tuple(xx.shape)}, {tuple(yy.shape)}'
        )
        raise ValueError(msg)

    n = int(xx.numel())
    if n < 2:
        msg = f'need at least 2 points, got {n}'
        raise ValueError(msg)
    if n_iter <= 0:
        msg = f'n_iter must be > 0, got {n_iter}'
        raise ValueError(msg)
    if inlier_th <= 0.0:
        msg = f'inlier_th must be > 0, got {inlier_th}'
        raise ValueError(msg)

    if float((xx.max() - xx.min()).item()) <= 1e-12:
        b0 = float(yy.median().item())
        resid0 = (yy - b0).abs()
        in0 = resid0 <= float(inlier_th)
        if int(in0.count_nonzero()) < 2:
            in0 = torch.ones_like(in0, dtype=torch.bool)
        return 0.0, float(yy[in0].mean().item()), in0

    dev = xx.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))

    i0 = torch.randint(n, (int(n_iter),), generator=gen, device=dev)
    step = torch.randint(1, n, (int(n_iter),), generator=gen, device=dev)
    i1 = (i0 + step) % n
    dx = xx[i1] - xx[i0]
    ok = dx.abs() > 1e-12
    if not bool(ok.any()):
        a_ref, b_ref = _fit_line_ls(xx, yy)
        return a_ref, b_ref, torch.ones(n, device=dev, dtype=torch.bool)

    a = (yy[i1] - yy[i0]) / dx
    b = yy[i0] - a * xx[i0]

    resid = (yy[None, :] - (a[:, None] * xx[None, :] + b[:, None])).abs()
    inlier = resid <= float(inlier_th)
    cnt = inlier.sum(dim=1)

    good = ok & (cnt >= 2)
    if not bool(good.any()):
        a_ref, b_ref = _fit_line_ls(xx, yy)
        return a_ref, b_ref, torch.ones(n, device=dev, dtype=torch.bool)

    cntv = torch.where(good, cnt, cnt.new_full(cnt.shape, -1))
    best_n = int(cntv.max().item())
    cand = cntv == best_n

    resid_in = torch.where(inlier, resid, torch.nan)
    med = torch.nanmedian(resid_in, dim=1).values
    med = torch.where(cand, med, med.new_full(med.shape, float('inf')))
    best_i = int(med.argmin().item())
    best_in = inlier[best_i]

    if int(best_in.count_nonzero()) >= 2:
        a_ref, b_ref = _fit_line_ls(xx[best_in], yy[best_in])
    else:
        a_ref, b_ref = _fit_line_ls(xx, yy)

    return a_ref, b_ref, best_in


def _robust_cost_abs(resid: torch.Tensor) -> float:
    if int(resid.numel()) == 0:
        return float('inf')
    return float(resid.abs().median().item())


def _fit_line_irls_huber(
    x: torch.Tensor,
    y: torch.Tensor,
    w_base: torch.Tensor,
    *,
    huber_c: float,
    iters: int,
) -> tuple[float, float]:
    xx = x.to(dtype=torch.float64)
    yy = y.to(dtype=torch.float64)
    ww0 = w_base.to(dtype=torch.float64)

    if int(xx.ndim) != 1 or int(yy.ndim) != 1 or int(ww0.ndim) != 1:
        msg = f'x/y/w must be 1D, got {tuple(xx.shape)}, {tuple(yy.shape)}, {tuple(ww0.shape)}'
        raise ValueError(msg)
    if int(xx.numel()) != int(yy.numel()) or int(xx.numel()) != int(ww0.numel()):
        msg = f'x/y/w length mismatch: {int(xx.numel())}, {int(yy.numel())}, {int(ww0.numel())}'
        raise ValueError(msg)
    n = int(xx.numel())
    if n < 2:
        msg = f'need at least 2 points, got {n}'
        raise ValueError(msg)
    if iters <= 0:
        msg = f'iters must be > 0, got {iters}'
        raise ValueError(msg)
    if huber_c <= 0.0:
        msg = f'huber_c must be > 0, got {huber_c}'
        raise ValueError(msg)

    v = xx.isfinite() & yy.isfinite() & ww0.isfinite() & (ww0 > 0.0)
    xx = xx[v]
    yy = yy[v]
    ww0 = ww0[v]
    if int(xx.numel()) < 2:
        msg = 'not enough finite/positive-weight points'
        raise ValueError(msg)

    eps = 1e-12
    w = ww0.clone()

    a = float(yy.mean().item())
    b = 0.0

    for _ in range(int(iters)):
        Sw = float(w.sum().item())
        if Sw <= eps:
            a = float(yy.mean().item())
            b = 0.0
            break

        Sx = float((w * xx).sum().item())
        Sy = float((w * yy).sum().item())
        Sxx = float((w * xx * xx).sum().item())
        Sxy = float((w * xx * yy).sum().item())
        D = Sw * Sxx - Sx * Sx
        if eps >= D:
            a = Sy / Sw
            b = 0.0
            break

        b = (Sw * Sxy - Sx * Sy) / D
        a = (Sy - b * Sx) / Sw

        res = yy - (a + b * xx)
        scale = float((1.4826 * res.abs().median().clamp_min(1e-6)).item())
        r = res / (float(huber_c) * scale)

        wh = torch.where(
            r.abs() <= 1.0,
            torch.ones_like(r),
            (1.0 / r.abs()).clamp_max(10.0),
        )
        w = ww0 * wh

    return float(b), float(a)  # slope, intercept


@dataclass(frozen=True)
class TwoPieceIRLSAutoBreakStrategy:
    """2-piece + auto break を IRLS(Huber)で実施する版（RANSACなし・決定論）。"""

    name: str = 'two_piece_irls_autobreak'
    huber_c: float = 1.345
    iters: int = 5
    min_pts: int = 8
    n_break_cand: int = 64
    q_lo: float = 0.15
    q_hi: float = 0.85
    slope_eps: float = 1e-6
    sort_offsets: bool = True

    def fit(
        self,
        x_abs: torch.Tensor,
        y_sec: torch.Tensor,
        w_conf: torch.Tensor,
    ) -> PiecewiseLinearTrend | None:
        x = x_abs.to(dtype=torch.float32)
        y = y_sec.to(dtype=torch.float32)
        w = w_conf.to(dtype=torch.float32)

        if int(x.ndim) != 1 or int(y.ndim) != 1 or int(w.ndim) != 1:
            msg = f'x/y/w must be 1D, got {tuple(x.shape)}, {tuple(y.shape)}, {tuple(w.shape)}'
            raise ValueError(msg)
        if int(x.numel()) != int(y.numel()) or int(x.numel()) != int(w.numel()):
            msg = f'x/y/w length mismatch: {int(x.numel())}, {int(y.numel())}, {int(w.numel())}'
            raise ValueError(msg)

        v = x.isfinite() & y.isfinite() & w.isfinite() & (w > 0.0)
        x = x[v]
        y = y[v]
        w = w[v]
        if int(x.numel()) < 2 * int(self.min_pts):
            return None

        if self.sort_offsets:
            order = torch.argsort(x, stable=True)
            x = x[order]
            y = y[order]
            w = w[order]

        xmin = float(x[0].item())
        xmax = float(x[-1].item())
        if float(xmax - xmin) <= 1e-6:
            a, b = _fit_line_ls(x, y)
            edges = torch.tensor(
                [xmin, xmin, xmax], device=x.device, dtype=torch.float32
            )
            coef = torch.tensor([[a, b], [a, b]], device=x.device, dtype=torch.float32)
            return PiecewiseLinearTrend(edges=edges, coef=coef)

        q = torch.linspace(
            float(self.q_lo),
            float(self.q_hi),
            int(self.n_break_cand),
            device=x.device,
            dtype=torch.float32,
        )
        cand = torch.quantile(x, q)
        cand = torch.unique(cand).sort().values

        best_cost = float('inf')
        best_xb: float | None = None
        best_coef: torch.Tensor | None = None

        for xb_t in cand:
            xb = float(xb_t.item())
            s1 = x <= xb
            s2 = x > xb
            n1 = int(s1.count_nonzero())
            n2 = int(s2.count_nonzero())
            if n1 < int(self.min_pts) or n2 < int(self.min_pts):
                continue

            a1, b1 = _fit_line_irls_huber(
                x[s1], y[s1], w[s1], huber_c=float(self.huber_c), iters=int(self.iters)
            )
            a2, b2 = _fit_line_irls_huber(
                x[s2], y[s2], w[s2], huber_c=float(self.huber_c), iters=int(self.iters)
            )
            if not (a1 > a2 + float(self.slope_eps)):
                continue

            yb = a1 * xb + b1
            b2c = yb - a2 * xb

            r1 = y[s1] - (a1 * x[s1] + b1)
            r2 = y[s2] - (a2 * x[s2] + b2c)
            cost = _robust_cost_abs(r1) + _robust_cost_abs(r2)
            if cost < best_cost:
                best_cost = float(cost)
                best_xb = float(xb)
                best_coef = torch.tensor(
                    [[a1, b1], [a2, b2c]], device=x.device, dtype=torch.float32
                )

        if best_xb is None or best_coef is None:
            return None

        edges = torch.tensor(
            [xmin, best_xb, xmax], device=x.device, dtype=torch.float32
        )
        return PiecewiseLinearTrend(edges=edges, coef=best_coef)

    def __call__(
        self, *, offsets: Tensor, t_sec: Tensor, valid: Tensor, w_conf: Tensor
    ):
        assert offsets.shape == t_sec.shape == valid.shape == w_conf.shape
        assert offsets.ndim == 2

        B, H = offsets.shape
        trend_t = t_sec.new_full((B, H), float('nan'))
        trend_s = t_sec.new_full((B, H), float('nan'))
        covered = (valid > 0).to(torch.bool)

        for b in range(int(B)):
            use = (
                covered[b]
                & offsets[b].isfinite()
                & t_sec[b].isfinite()
                & w_conf[b].isfinite()
                & (w_conf[b] > 0)
            )
            if int(use.count_nonzero()) < 2 * int(self.min_pts):
                covered[b].fill_(False)
                continue

            tr = self.fit(offsets[b][use], t_sec[b][use], w_conf[b][use])
            if tr is None:
                covered[b].fill_(False)
                continue

            yhat = tr.predict(offsets[b])
            trend_t[b] = yhat

            xb = float(tr.edges[1].item())
            a1 = float(tr.coef[0, 0].item())
            a2 = float(tr.coef[1, 0].item())
            xf = offsets[b].isfinite()
            s1 = xf & (offsets[b] <= xb)
            s2 = xf & (offsets[b] > xb)
            if bool(s1.any()):
                trend_s[b, s1] = a1
            if bool(s2.any()):
                trend_s[b, s2] = a2

            covered[b] = covered[b] & trend_t[b].isfinite()

        v_trend = trend_s.sign() / trend_s.abs().clamp_min(1e-6)
        return trend_t, trend_s, v_trend, w_conf, covered


@dataclass(frozen=True)
class TwoPieceRansacAutoBreakStrategy:
    """2-piece line + auto break で t(x) を推定する。

    - break は x の分位点から候補を作り、各候補で左右を RANSAC。
    - a1 > a2 を必須(= slope が減少する)。
    """

    name: str = 'two_piece_ransac_autobreak'
    n_iter: int = 200
    inlier_th_ms: float = 4.0
    min_pts: int = 8
    n_break_cand: int = 64
    q_lo: float = 0.15
    q_hi: float = 0.85
    seed: int = 0
    slope_eps: float = 1e-6  # enforce a1 > a2 + slope_eps
    sort_offsets: bool = True

    def fit(
        self, x_abs: torch.Tensor, y_sec: torch.Tensor
    ) -> PiecewiseLinearTrend | None:
        x = x_abs.to(dtype=torch.float32)
        y = y_sec.to(dtype=torch.float32)
        if int(x.ndim) != 1 or int(y.ndim) != 1 or int(x.numel()) != int(y.numel()):
            msg = f'x_abs/y_sec must be 1D same length, got {tuple(x.shape)}, {tuple(y.shape)}'
            raise ValueError(msg)

        v = x.isfinite() & y.isfinite()
        x = x[v]
        y = y[v]
        if int(x.numel()) < 2 * int(self.min_pts):
            return None

        if self.sort_offsets:
            order = torch.argsort(x, stable=True)
            x = x[order]
            y = y[order]

        xmin = float(x[0].item())
        xmax = float(x[-1].item())
        if float(xmax - xmin) <= 1e-6:
            a, b = _fit_line_ls(x, y)
            edges = torch.tensor(
                [xmin, xmin, xmax], device=x.device, dtype=torch.float32
            )
            coef = torch.tensor([[a, b], [a, b]], device=x.device, dtype=torch.float32)
            return PiecewiseLinearTrend(edges=edges, coef=coef)

        q = torch.linspace(
            float(self.q_lo),
            float(self.q_hi),
            int(self.n_break_cand),
            device=x.device,
            dtype=torch.float32,
        )
        cand = torch.quantile(x, q)
        cand = torch.unique(cand).sort().values

        th_sec = float(self.inlier_th_ms) * 1e-3

        best_cost = float('inf')
        best_xb: float | None = None
        best_coef: torch.Tensor | None = None

        for xb_t in cand:
            xb = float(xb_t.item())
            s1 = x <= xb
            s2 = x > xb
            n1 = int(s1.count_nonzero())
            n2 = int(s2.count_nonzero())
            if n1 < int(self.min_pts) or n2 < int(self.min_pts):
                continue

            a1, b1, _ = _ransac_line(
                x[s1],
                y[s1],
                n_iter=int(self.n_iter),
                inlier_th=th_sec,
                seed=int(self.seed) + 11,
            )
            a2, b2, _ = _ransac_line(
                x[s2],
                y[s2],
                n_iter=int(self.n_iter),
                inlier_th=th_sec,
                seed=int(self.seed) + 29,
            )
            if not (a1 > a2 + float(self.slope_eps)):
                continue

            # ---- enforce continuity at break: y1(xb) == y2(xb) ----
            yb = a1 * xb + b1
            b2c = yb - a2 * xb

            r1 = y[s1] - (a1 * x[s1] + b1)
            r2 = y[s2] - (a2 * x[s2] + b2c)
            cost = _robust_cost_abs(r1) + _robust_cost_abs(r2)
            if cost < best_cost:
                best_cost = float(cost)
                best_xb = float(xb)
                best_coef = torch.tensor(
                    [[a1, b1], [a2, b2c]], device=x.device, dtype=torch.float32
                )

        if best_xb is None or best_coef is None:
            return None

        edges = torch.tensor(
            [xmin, best_xb, xmax], device=x.device, dtype=torch.float32
        )
        return PiecewiseLinearTrend(edges=edges, coef=best_coef)

    def __call__(
        self, *, offsets: Tensor, t_sec: Tensor, valid: Tensor, w_conf: Tensor
    ):
        assert offsets.shape == t_sec.shape == valid.shape == w_conf.shape
        assert offsets.ndim == 2

        B, H = offsets.shape
        trend_t = t_sec.new_zeros((B, H))
        trend_s = t_sec.new_zeros((B, H))
        covered = (valid > 0).to(torch.bool)

        for b in range(int(B)):
            use = covered[b] & offsets[b].isfinite() & t_sec[b].isfinite()
            if int(use.count_nonzero()) < 2 * int(self.min_pts):
                covered[b].fill_(False)
                continue

            tr = self.fit(offsets[b][use], t_sec[b][use])
            if tr is None:
                covered[b].fill_(False)
                continue

            yhat = tr.predict(offsets[b]).nan_to_num(0.0)
            trend_t[b] = yhat

            xb = float(tr.edges[1].item())
            a1 = float(tr.coef[0, 0].item())
            a2 = float(tr.coef[1, 0].item())
            xf = offsets[b].isfinite()
            s1 = xf & (offsets[b] <= xb)
            s2 = xf & (offsets[b] > xb)
            if bool(s1.any()):
                trend_s[b, s1] = a1
            if bool(s2.any()):
                trend_s[b, s2] = a2

        v_trend = trend_s.sign() / trend_s.abs().clamp_min(1e-6)
        return trend_t, trend_s, v_trend, w_conf, covered
