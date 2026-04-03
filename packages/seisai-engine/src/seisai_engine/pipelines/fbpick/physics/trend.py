from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PhysicsLiteConfig
from .feasible import FeasibleBandResult
from .pick_table import CoarsePickTable

__all__ = ['TrendResult', 'build_trend_result']


@dataclass(frozen=True)
class TrendResult:
    seed_mask: np.ndarray
    seed_threshold: np.float32
    local_center_sec: np.ndarray
    local_center_valid: np.ndarray
    local_discard_mask: np.ndarray
    global_center_sec: np.ndarray
    trend_center_sec: np.ndarray
    trend_center_i: np.ndarray
    filled_mask: np.ndarray


def _lower_quantile_threshold(values: np.ndarray, frac: float) -> float:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if int(finite.size) == 0:
        msg = 'cannot compute threshold from empty finite values'
        raise ValueError(msg)
    return float(np.quantile(finite, float(frac), method='linear'))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.shape != w.shape or x.ndim != 1:
        msg = f'values/weights must be 1D and same shape, got {x.shape}, {w.shape}'
        raise ValueError(msg)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    if not bool(np.any(valid)):
        msg = 'weighted median requires at least one positive finite weight'
        raise ValueError(msg)
    xv = x[valid]
    wv = w[valid]
    order = np.argsort(xv, kind='mergesort')
    xv = xv[order]
    wv = wv[order]
    cdf = np.cumsum(wv)
    cutoff = 0.5 * float(cdf[-1])
    idx = int(np.searchsorted(cdf, cutoff, side='left'))
    return float(xv[idx])


def _fit_weighted_line(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float] | None:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    ww = np.asarray(weights, dtype=np.float64)
    if xx.shape != yy.shape or xx.shape != ww.shape or xx.ndim != 1:
        msg = f'x/y/weights must be 1D and same shape, got {xx.shape}, {yy.shape}, {ww.shape}'
        raise ValueError(msg)

    valid = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(ww) & (ww > 0.0)
    if int(np.count_nonzero(valid)) < 2:
        return None

    xv = xx[valid]
    yv = yy[valid]
    wv = ww[valid]
    wsum = float(wv.sum())
    if (not np.isfinite(wsum)) or wsum <= 0.0:
        return None

    x_mean = float(np.sum(wv * xv) / wsum)
    y_mean = float(np.sum(wv * yv) / wsum)
    dx = xv - x_mean
    denom = float(np.sum(wv * dx * dx))
    if (not np.isfinite(denom)) or denom <= 1.0e-12:
        return None

    slope = float(np.sum(wv * dx * (yv - y_mean)) / denom)
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def _predict_local_center_sec(
    *,
    x_window_m: np.ndarray,
    y_window_sec: np.ndarray,
    weight_window: np.ndarray,
    x_target_m: float,
    slope_lo: float,
    slope_hi: float,
    huber_c: float,
    iters: int,
) -> float:
    if slope_hi < slope_lo:
        msg = 'slope_hi must be >= slope_lo'
        raise ValueError(msg)

    fit = _fit_weighted_line(x_window_m, y_window_sec, weight_window)
    if fit is None:
        return _weighted_median(y_window_sec, weight_window)

    slope, intercept = fit
    slope = float(np.clip(slope, slope_lo, slope_hi))
    intercept = _weighted_median(
        np.asarray(y_window_sec, dtype=np.float64) - slope * np.asarray(x_window_m, dtype=np.float64),
        weight_window,
    )

    weights = np.asarray(weight_window, dtype=np.float64)
    for _ in range(int(iters)):
        resid = np.asarray(y_window_sec, dtype=np.float64) - (
            slope * np.asarray(x_window_m, dtype=np.float64) + intercept
        )
        resid_med = np.median(resid)
        mad = np.median(np.abs(resid - resid_med))
        scale = max(float(mad) / 0.67448975, 1.0e-6)
        z = np.abs(resid) / (scale * float(huber_c))
        huber_w = np.ones_like(z, dtype=np.float64)
        huber_w[z > 1.0] = 1.0 / z[z > 1.0]
        fit = _fit_weighted_line(
            x_window_m,
            y_window_sec,
            weights * huber_w,
        )
        if fit is None:
            break
        slope, intercept = fit
        slope = float(np.clip(slope, slope_lo, slope_hi))
        intercept = _weighted_median(
            np.asarray(y_window_sec, dtype=np.float64)
            - slope * np.asarray(x_window_m, dtype=np.float64),
            weights * huber_w,
        )
    return float(slope * float(x_target_m) + intercept)


def _expand_trace_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    m = np.asarray(mask, dtype=np.bool_)
    if m.ndim != 1:
        msg = f'mask must be 1D, got {m.shape}'
        raise ValueError(msg)
    if radius <= 0 or not bool(np.any(m)):
        return m.copy()

    out = m.copy()
    idx = np.flatnonzero(m)
    for center in idx.tolist():
        lo = max(0, int(center) - int(radius))
        hi = min(int(m.shape[0]), int(center) + int(radius) + 1)
        out[lo:hi] = True
    return out


def _local_inversion_mask(
    local_center_i: np.ndarray,
    local_valid: np.ndarray,
    *,
    drop_th_samples: float,
    min_consec_steps: int,
) -> np.ndarray:
    center = np.asarray(local_center_i, dtype=np.float32)
    valid = np.asarray(local_valid, dtype=np.bool_)
    if center.shape != valid.shape or center.ndim != 1:
        msg = f'local_center_i/local_valid must be 1D and same shape, got {center.shape}, {valid.shape}'
        raise ValueError(msg)
    if int(center.shape[0]) < 2:
        return np.zeros(center.shape, dtype=np.bool_)

    bad_step = np.zeros((center.shape[0] - 1,), dtype=np.bool_)
    diff = np.diff(center)
    both_valid = valid[:-1] & valid[1:]
    bad_step[both_valid] = diff[both_valid] <= -float(drop_th_samples)

    out = np.zeros(center.shape, dtype=np.bool_)
    run_start = None
    for idx, is_bad in enumerate(bad_step.tolist()):
        if is_bad and run_start is None:
            run_start = idx
        if (not is_bad) and run_start is not None:
            run_len = idx - run_start
            if run_len >= int(min_consec_steps):
                out[run_start : idx + 1] = True
            run_start = None
    if run_start is not None:
        run_len = int(bad_step.shape[0]) - run_start
        if run_len >= int(min_consec_steps):
            out[run_start : int(bad_step.shape[0]) + 1] = True
    return out


def _build_global_center_sec(
    *,
    offset_abs_m: np.ndarray,
    seed_times_sec: np.ndarray,
    seed_weights: np.ndarray,
    target_offsets_abs_m: np.ndarray,
    cfg: PhysicsLiteConfig,
) -> np.ndarray:
    if int(seed_times_sec.shape[0]) == 0:
        msg = 'global trend requires at least one seed'
        raise ValueError(msg)

    if int(seed_times_sec.shape[0]) < int(cfg.robust_center.global_side_min_pts):
        fill = _weighted_median(seed_times_sec, seed_weights)
        return np.full(target_offsets_abs_m.shape, np.float32(fill), dtype=np.float32)

    slope_lo = 1.0 / float(cfg.robust_center.global_vmax_m_s)
    slope_hi = 1.0 / float(cfg.robust_center.global_vmin_m_s)
    first_offset = float(np.asarray(target_offsets_abs_m, dtype=np.float64)[0])
    intercept_target = _predict_local_center_sec(
        x_window_m=np.asarray(offset_abs_m, dtype=np.float64),
        y_window_sec=np.asarray(seed_times_sec, dtype=np.float64),
        weight_window=np.asarray(seed_weights, dtype=np.float64),
        x_target_m=first_offset,
        slope_lo=slope_lo,
        slope_hi=slope_hi,
        huber_c=float(cfg.trend.trend_local_huber_c),
        iters=int(cfg.trend.trend_local_iters),
    )
    fit = _fit_weighted_line(offset_abs_m, seed_times_sec, seed_weights)
    if fit is None:
        return np.full(
            target_offsets_abs_m.shape,
            np.float32(intercept_target),
            dtype=np.float32,
        )
    slope, intercept = fit
    slope = float(np.clip(slope, slope_lo, slope_hi))
    intercept = _weighted_median(
        np.asarray(seed_times_sec, dtype=np.float64)
        - slope * np.asarray(offset_abs_m, dtype=np.float64),
        np.asarray(seed_weights, dtype=np.float64),
    )
    pred = slope * np.asarray(target_offsets_abs_m, dtype=np.float64) + intercept
    return pred.astype(np.float32, copy=False)


def _fill_trend_centers(
    *,
    local_center_sec: np.ndarray,
    local_valid: np.ndarray,
    global_center_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local = np.asarray(local_center_sec, dtype=np.float32)
    valid = np.asarray(local_valid, dtype=np.bool_)
    global_center = np.asarray(global_center_sec, dtype=np.float32)
    if local.shape != valid.shape or local.shape != global_center.shape or local.ndim != 1:
        msg = f'local/valid/global must be 1D and same shape, got {local.shape}, {valid.shape}, {global_center.shape}'
        raise ValueError(msg)

    filled = (~valid).astype(np.bool_, copy=False)
    if bool(np.any(valid)):
        idx = np.arange(local.shape[0], dtype=np.float32)
        out = local.copy()
        out[~valid] = np.interp(
            idx[~valid],
            idx[valid],
            local[valid].astype(np.float32),
        ).astype(np.float32, copy=False)
        return out, filled

    return global_center.copy(), np.ones(local.shape, dtype=np.bool_)


def build_trend_result(
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    cfg: PhysicsLiteConfig,
) -> TrendResult:
    feasible_mask = np.asarray(feasible.feasible_mask, dtype=np.bool_)
    if feasible_mask.shape != (table.n_traces,):
        msg = f'feasible_mask must have shape {(table.n_traces,)}, got {feasible_mask.shape}'
        raise ValueError(msg)

    pmax = np.clip(np.asarray(table.coarse_pmax, dtype=np.float32), 0.0, 1.0)
    if not bool(np.any(feasible_mask)):
        msg = 'physics-lite requires at least one feasible coarse pick'
        raise ValueError(msg)

    seed_threshold = np.float32(
        _lower_quantile_threshold(
            pmax[feasible_mask],
            frac=float(cfg.keep_reject.drop_low_frac),
        )
    )
    seed_mask = feasible_mask & (pmax >= seed_threshold)
    if not bool(np.any(seed_mask)):
        msg = 'physics-lite requires at least one feasible confident seed'
        raise ValueError(msg)

    offset_abs_m = np.abs(np.asarray(table.offset_m, dtype=np.float32)).astype(
        np.float64,
        copy=False,
    )
    times_sec = np.asarray(table.coarse_pick_t_sec, dtype=np.float32).astype(
        np.float64,
        copy=False,
    )
    weights = np.maximum(pmax.astype(np.float64, copy=False), 1.0e-6)
    local_half_win = int(cfg.trend.trend_local_section_len) * int(
        cfg.trend.trend_local_stride
    )
    slope_lo = 1.0 / float(cfg.trend.trend_local_vmax_mps)
    slope_hi = 1.0 / float(cfg.trend.trend_local_vmin_mps)

    local_center_sec = np.full((table.n_traces,), np.nan, dtype=np.float32)
    local_valid = np.zeros((table.n_traces,), dtype=np.bool_)
    for trace_idx in range(table.n_traces):
        lo = max(0, trace_idx - local_half_win)
        hi = min(table.n_traces, trace_idx + local_half_win + 1)
        window_seed = np.flatnonzero(seed_mask[lo:hi]) + lo
        if int(window_seed.size) < int(cfg.trend.trend_min_pts):
            continue
        center = _predict_local_center_sec(
            x_window_m=offset_abs_m[window_seed],
            y_window_sec=times_sec[window_seed],
            weight_window=weights[window_seed],
            x_target_m=float(offset_abs_m[trace_idx]),
            slope_lo=slope_lo,
            slope_hi=slope_hi,
            huber_c=float(cfg.trend.trend_local_huber_c),
            iters=int(cfg.trend.trend_local_iters),
        )
        local_center_sec[trace_idx] = np.float32(center)
        local_valid[trace_idx] = True

    global_center_sec = _build_global_center_sec(
        offset_abs_m=offset_abs_m[seed_mask],
        seed_times_sec=times_sec[seed_mask],
        seed_weights=weights[seed_mask],
        target_offsets_abs_m=offset_abs_m,
        cfg=cfg,
    )
    local_center_i = local_center_sec / np.float32(table.dt_scalar_sec)
    global_center_i = global_center_sec / np.float32(table.dt_scalar_sec)
    diff_bad = local_valid & (
        np.abs(local_center_i - global_center_i)
        >= float(cfg.robust_center.local_global_diff_th_samples)
    )
    inv_bad = _local_inversion_mask(
        local_center_i=local_center_i,
        local_valid=local_valid,
        drop_th_samples=float(cfg.robust_center.local_inv_drop_th_samples),
        min_consec_steps=int(cfg.robust_center.local_inv_min_consec_steps),
    )
    local_discard_mask = _expand_trace_mask(
        diff_bad | inv_bad,
        radius=int(cfg.robust_center.local_discard_radius_traces),
    )
    local_valid = local_valid & (~local_discard_mask)
    local_center_sec = local_center_sec.copy()
    local_center_sec[~local_valid] = np.nan

    trend_center_sec, filled_mask = _fill_trend_centers(
        local_center_sec=local_center_sec,
        local_valid=local_valid,
        global_center_sec=global_center_sec,
    )
    trend_center_i = np.rint(
        trend_center_sec / np.float32(table.dt_scalar_sec)
    ).astype(np.int32, copy=False)
    trend_center_i = np.clip(
        trend_center_i,
        0,
        int(table.n_samples_orig) - 1,
    ).astype(np.int32, copy=False)
    trend_center_sec = (
        trend_center_i.astype(np.float32) * np.float32(table.dt_scalar_sec)
    ).astype(np.float32, copy=False)

    return TrendResult(
        seed_mask=seed_mask.astype(np.bool_, copy=False),
        seed_threshold=seed_threshold,
        local_center_sec=local_center_sec.astype(np.float32, copy=False),
        local_center_valid=local_valid.astype(np.bool_, copy=False),
        local_discard_mask=local_discard_mask.astype(np.bool_, copy=False),
        global_center_sec=global_center_sec.astype(np.float32, copy=False),
        trend_center_sec=trend_center_sec,
        trend_center_i=trend_center_i,
        filled_mask=filled_mask.astype(np.bool_, copy=False),
    )
