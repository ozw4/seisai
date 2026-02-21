# %%
#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import segyio
import torch
from jogsarar_shared import (
    build_groups_by_key,
    find_segy_files,
    read_trace_field,
    require_npz_key,
    valid_pick_mask,
)
from seisai_pick.score.confidence_from_trend_resid import (
    trace_confidence_from_trend_resid_gaussian,
    trace_confidence_from_trend_resid_var,
)
from seisai_pick.trend.trend_fit import robust_linear_trend
from seisai_pick.trend.trend_fit_strategy import TwoPieceIRLSAutoBreakStrategy

# =========================
# CONFIG（ここだけ触ればOK）
# =========================
IN_SEGY_ROOT = Path('/home/dcuser/data/ActiveSeisField/jogsarar')
IN_INFER_ROOT = Path('/home/dcuser/data/ActiveSeisField/jogsarar_out')
OUT_SEGY_ROOT = Path('/home/dcuser/data/ActiveSeisField/jogsarar_psn512')

SEGY_EXTS = ('.sgy', '.segy')

HALF_WIN = 128  # ± samples
UP_FACTOR = 2
OUT_NS = 2 * HALF_WIN * UP_FACTOR  # 512

# 下位除外（p10）
DROP_LOW_FRAC = 0.05  # 下位除外
SCORE_KEYS_FOR_WEIGHT = (
    'conf_prob1',
    'conf_rs1',
)
SCORE_KEYS_FOR_FILTER = (
    'conf_prob1',
    'conf_trend1',
    'conf_rs1',
)
PICK_KEY = 'pick_final'  # infer npz側の最終pick（p1）

# 閾値モード
#   'per_segy' : SEGYごとにDrop_low_frac
#   'global'   : 全SEGYまとめてDrop_low_frac（全データグローバル）
THRESH_MODE = 'global'  # 'global' or 'per_segy'

# semi-global trendline (ffid neighbors in same SEG-Y)
SEMI_GLOBAL_ENABLE = True
SEMI_NEI_K = 3
SEMI_MIN_NEI = 2
SEMI_IRLS_SECTION_LEN = 48
SEMI_IRLS_STRIDE = 16
SEMI_IRLS_HUBER_C = 1.345
SEMI_IRLS_ITERS = 3
SEMI_MIN_SUPPORT_PER_TRACE = 3

# global trendline (proxy sign split)
GLOBAL_VMIN_M_S = 300.0
GLOBAL_VMAX_M_S = 6000.0
GLOBAL_SLOPE_EPS = 1e-6
GLOBAL_SIDE_MIN_PTS = 16  # 2*min_pts of TwoPieceIRLSAutoBreakStrategy default
# Stage1 local trendline baseline (from stage1 .prob.npz)
USE_STAGE1_LOCAL_TRENDLINE_BASELINE = True
LOCAL_GLOBAL_DIFF_TH_SAMPLES = 128  # >= -> discard local baseline around that trace
LOCAL_DISCARD_RADIUS_TRACES = 32  # +/- traces within ffid
LOCAL_TREND_T_SEC_KEY = 'trend_t_sec'
LOCAL_TREND_COVERED_KEY = 'trend_covered'
LOCAL_TREND_DT_SEC_KEY = 'dt_sec'


# conf_trend1: Stage1互換
CONF_TREND_SIGMA_MS = 6.0
CONF_TREND_VAR_HALF_WIN_TRACES = 8
CONF_TREND_VAR_SIGMA_STD_MS = 6.0
CONF_TREND_VAR_MIN_COUNT = 3


# =========================
# Utility
# =========================
@dataclass(frozen=True)
class _TrendBuildResult:
    trend_center_i_raw: np.ndarray
    trend_center_i_local: np.ndarray
    trend_center_i_semi: np.ndarray
    trend_center_i_final: np.ndarray
    trend_center_i_used: np.ndarray
    trend_center_i_global: np.ndarray
    nn_replaced_mask: np.ndarray
    global_replaced_mask: np.ndarray
    global_missing_filled_mask: np.ndarray
    local_discard_mask: np.ndarray
    global_edges_all: np.ndarray
    global_coef_all: np.ndarray
    global_edges_left: np.ndarray
    global_coef_left: np.ndarray
    global_edges_right: np.ndarray
    global_coef_right: np.ndarray
    trend_filled_mask: np.ndarray
    ffid_values: np.ndarray
    ffid_unique_values: np.ndarray
    shot_x_ffid: np.ndarray
    shot_y_ffid: np.ndarray
    semi_used_ffid_mask: np.ndarray
    semi_fallback_count: int
    semi_low_mask: np.ndarray
    semi_covered: np.ndarray
    semi_support_count: np.ndarray
    semi_v_trend: np.ndarray
    conf_trend1: np.ndarray


def infer_npz_path_for_segy(segy_path: Path) -> Path:
    rel = segy_path.relative_to(IN_SEGY_ROOT)
    return IN_INFER_ROOT / rel.parent / f'{segy_path.stem}.prob.npz'


def out_segy_path_for_in(segy_path: Path) -> Path:
    rel = segy_path.relative_to(IN_SEGY_ROOT)
    out_rel = rel.with_suffix('')
    return OUT_SEGY_ROOT / out_rel.parent / f'{out_rel.name}.win512.sgy'


def out_sidecar_npz_path_for_out(out_segy_path: Path) -> Path:
    return out_segy_path.with_suffix('.sidecar.npz')


def out_pick_csr_npz_path_for_out(out_segy_path: Path) -> Path:
    # PSN dataset (load_phase_pick_csr_npz) が読む用
    return out_segy_path.with_suffix('.phase_pick.csr.npz')


def _load_stage1_local_trend_center_i(
    *,
    z: np.lib.npyio.NpzFile,
    n_traces: int,
    dt_sec_in: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not bool(USE_STAGE1_LOCAL_TRENDLINE_BASELINE):
        center_i = np.full(int(n_traces), np.nan, dtype=np.float32)
        ok = np.zeros(int(n_traces), dtype=bool)
        return center_i, ok

    dt_npz = float(require_npz_key(z, LOCAL_TREND_DT_SEC_KEY))
    dt_in = float(dt_sec_in)
    if (not np.isfinite(dt_npz)) or dt_npz <= 0.0:
        msg = f'invalid {LOCAL_TREND_DT_SEC_KEY} in infer npz: {dt_npz}'
        raise ValueError(msg)
    if (not np.isfinite(dt_in)) or dt_in <= 0.0:
        msg = f'invalid dt_sec_in from segy: {dt_in}'
        raise ValueError(msg)

    tol = 1e-7 * max(abs(dt_in), 1.0)
    if abs(dt_npz - dt_in) > tol:
        msg = (
            f'dt mismatch between segy and infer npz: segy={dt_in:.9g} sec, '
            f'npz={dt_npz:.9g} sec (tol={tol:.3g}). '
            f'Run stage1 with the same input segy.'
        )
        raise ValueError(msg)

    t_sec = require_npz_key(z, LOCAL_TREND_T_SEC_KEY).astype(np.float32, copy=False)
    if t_sec.ndim != 1 or t_sec.shape[0] != int(n_traces):
        msg = (
            f'{LOCAL_TREND_T_SEC_KEY} must be (n_traces,), got {t_sec.shape}, '
            f'n_traces={n_traces}'
        )
        raise ValueError(msg)

    covered = require_npz_key(z, LOCAL_TREND_COVERED_KEY).astype(bool, copy=False)
    if covered.ndim != 1 or covered.shape[0] != int(n_traces):
        msg = (
            f'{LOCAL_TREND_COVERED_KEY} must be (n_traces,), got {covered.shape}, '
            f'n_traces={n_traces}'
        )
        raise ValueError(msg)

    center_i = (t_sec / float(dt_in)).astype(np.float32, copy=False)
    ok = covered & np.isfinite(center_i) & (center_i > 0.0)
    return center_i.astype(np.float32, copy=False), ok.astype(bool, copy=False)


def _expand_mask_within_ffid_groups(
    *,
    mask_by_trace: np.ndarray,
    ffid_groups: list[np.ndarray],
    radius: int,
) -> np.ndarray:
    m = np.asarray(mask_by_trace, dtype=bool)
    if m.ndim != 1:
        msg = f'mask_by_trace must be 1D, got {m.shape}'
        raise ValueError(msg)
    if int(radius) < 0:
        msg = f'radius must be >= 0, got {radius}'
        raise ValueError(msg)

    out = np.zeros_like(m, dtype=bool)
    r = int(radius)

    for idx in ffid_groups:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            continue

        g = m[idx]
        if not bool(np.any(g)):
            continue

        pos = np.flatnonzero(g)
        g_out = np.zeros(idx.size, dtype=bool)
        for p in pos:
            lo = max(0, int(p) - r)
            hi = min(int(idx.size), int(p) + r + 1)
            g_out[lo:hi] = True
        out[idx[g_out]] = True

    return out.astype(bool, copy=False)


def _validate_semi_config() -> None:
    if int(SEMI_NEI_K) < 1:
        msg = f'SEMI_NEI_K must be >= 1, got {SEMI_NEI_K}'
        raise ValueError(msg)
    if int(SEMI_MIN_NEI) < 1:
        msg = f'SEMI_MIN_NEI must be >= 1, got {SEMI_MIN_NEI}'
        raise ValueError(msg)
    if int(SEMI_IRLS_SECTION_LEN) < 4:
        msg = f'SEMI_IRLS_SECTION_LEN must be >= 4, got {SEMI_IRLS_SECTION_LEN}'
        raise ValueError(msg)
    if int(SEMI_IRLS_STRIDE) < 1:
        msg = f'SEMI_IRLS_STRIDE must be >= 1, got {SEMI_IRLS_STRIDE}'
        raise ValueError(msg)
    if float(SEMI_IRLS_HUBER_C) <= 0.0:
        msg = f'SEMI_IRLS_HUBER_C must be > 0, got {SEMI_IRLS_HUBER_C}'
        raise ValueError(msg)
    if int(SEMI_IRLS_ITERS) < 1:
        msg = f'SEMI_IRLS_ITERS must be >= 1, got {SEMI_IRLS_ITERS}'
        raise ValueError(msg)
    if int(SEMI_MIN_SUPPORT_PER_TRACE) < 1:
        msg = (
            f'SEMI_MIN_SUPPORT_PER_TRACE must be >= 1, got {SEMI_MIN_SUPPORT_PER_TRACE}'
        )
        raise ValueError(msg)


def _apply_source_group_scalar(values: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    s = np.asarray(scalar, dtype=np.float64)
    if v.shape != s.shape:
        msg = f'shape mismatch values={v.shape}, scalar={s.shape}'
        raise ValueError(msg)

    scale = np.ones_like(s, dtype=np.float64)
    pos = s > 0.0
    neg = s < 0.0
    scale[pos] = s[pos]
    scale[neg] = 1.0 / np.abs(s[neg])
    return (v * scale).astype(np.float64, copy=False)


def _build_shot_xy_by_ffid(
    *,
    ffid_groups: list[np.ndarray],
    shot_x_by_trace: np.ndarray,
    shot_y_by_trace: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sx_tr = np.asarray(shot_x_by_trace, dtype=np.float64)
    sy_tr = np.asarray(shot_y_by_trace, dtype=np.float64)
    if sx_tr.shape != sy_tr.shape:
        msg = f'shot x/y shape mismatch: {sx_tr.shape} vs {sy_tr.shape}'
        raise ValueError(msg)

    n_ffid = len(ffid_groups)
    shot_x_ffid = np.full(n_ffid, np.nan, dtype=np.float64)
    shot_y_ffid = np.full(n_ffid, np.nan, dtype=np.float64)
    for g, idx in enumerate(ffid_groups):
        xs = sx_tr[idx]
        ys = sy_tr[idx]
        ok_x = np.isfinite(xs)
        ok_y = np.isfinite(ys)
        if np.any(ok_x):
            shot_x_ffid[g] = float(np.nanmedian(xs[ok_x]))
        if np.any(ok_y):
            shot_y_ffid[g] = float(np.nanmedian(ys[ok_y]))
    return shot_x_ffid, shot_y_ffid


def _percentile_threshold(x: np.ndarray, *, frac: float) -> float:
    v = np.asarray(x, dtype=np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan')
    q = float(frac) * 100.0
    return float(np.nanpercentile(v, q))


def _build_neighbor_indices(
    shot_x_ffid: np.ndarray,
    shot_y_ffid: np.ndarray,
    *,
    k: int,
) -> list[np.ndarray]:
    sx = np.asarray(shot_x_ffid, dtype=np.float64)
    sy = np.asarray(shot_y_ffid, dtype=np.float64)
    if sx.shape != sy.shape or sx.ndim != 1:
        msg = f'shot_x_ffid/shot_y_ffid must be 1D and same shape, got {sx.shape}, {sy.shape}'
        raise ValueError(msg)
    if k < 1:
        msg = f'k must be >= 1, got {k}'
        raise ValueError(msg)

    n = int(sx.shape[0])
    valid_xy = np.isfinite(sx) & np.isfinite(sy)
    out: list[np.ndarray] = []
    for i in range(n):
        if not bool(valid_xy[i]):
            out.append(np.zeros(0, dtype=np.int64))
            continue
        cand = np.flatnonzero(valid_xy)
        cand = cand[cand != i]
        if cand.size == 0:
            out.append(np.zeros(0, dtype=np.int64))
            continue
        dx = sx[cand] - float(sx[i])
        dy = sy[cand] - float(sy[i])
        dist2 = dx * dx + dy * dy
        order = np.argsort(dist2, kind='mergesort')
        n_take = min(int(k), int(cand.size))
        out.append(cand[order[:n_take]].astype(np.int64, copy=False))
    return out


def _build_offset_signed_proxy_by_ffid(
    *,
    offset_abs_m: np.ndarray,
    ffid_groups: list[np.ndarray],
) -> np.ndarray:
    off = np.asarray(offset_abs_m, dtype=np.float64)
    if off.ndim != 1:
        msg = f'offset_abs_m must be 1D, got {off.shape}'
        raise ValueError(msg)

    proxy = np.full(off.shape[0], np.nan, dtype=np.float32)
    for idx in ffid_groups:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            continue
        off_g = off[idx]
        finite = np.isfinite(off_g)
        if not bool(np.any(finite)):
            continue
        cand = np.where(finite, off_g, np.inf)
        split = int(np.argmin(cand))
        if not np.isfinite(cand[split]):
            continue

        p_g = off_g.astype(np.float32, copy=True)
        p_g[:split] *= -1.0
        p_g[~finite] = np.nan
        proxy[idx] = p_g
    return proxy.astype(np.float32, copy=False)


def _aggregate_knots_by_offset(
    x: np.ndarray, y: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx = np.asarray(x, dtype=np.float32)
    yy = np.asarray(y, dtype=np.float32)
    vv = np.asarray(v, dtype=np.float32)
    if xx.shape != yy.shape or xx.shape != vv.shape or xx.ndim != 1:
        msg = f'x/y/v must be 1D and same shape, got {xx.shape}, {yy.shape}, {vv.shape}'
        raise ValueError(msg)

    use = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(vv)
    xx = xx[use]
    yy = yy[use]
    vv = vv[use]
    if xx.size == 0:
        e = np.zeros(0, dtype=np.float32)
        return e, e, e

    order = np.argsort(xx, kind='mergesort')
    xx = xx[order]
    yy = yy[order]
    vv = vv[order]

    uniq, inv = np.unique(xx, return_inverse=True)
    y_out = np.full(uniq.shape[0], np.nan, dtype=np.float32)
    v_out = np.full(uniq.shape[0], np.nan, dtype=np.float32)
    for i in range(int(uniq.shape[0])):
        sel = inv == i
        y_out[i] = np.float32(np.median(yy[sel]))
        v_out[i] = np.float32(np.median(vv[sel]))
    return (
        uniq.astype(np.float32, copy=False),
        y_out.astype(np.float32, copy=False),
        v_out.astype(np.float32, copy=False),
    )


def _eval_linear_interp_no_extrap(
    *,
    x_query: np.ndarray,
    x_knots: np.ndarray,
    y_knots: np.ndarray,
    v_knots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xq = np.asarray(x_query, dtype=np.float32)
    xk = np.asarray(x_knots, dtype=np.float32)
    yk = np.asarray(y_knots, dtype=np.float32)
    vk = np.asarray(v_knots, dtype=np.float32)

    if xq.ndim != 1:
        msg = f'x_query must be 1D, got {xq.shape}'
        raise ValueError(msg)
    if xk.ndim != 1 or yk.ndim != 1 or vk.ndim != 1:
        msg = f'knots must be 1D, got {xk.shape}, {yk.shape}, {vk.shape}'
        raise ValueError(msg)
    if xk.shape != yk.shape or xk.shape != vk.shape:
        msg = f'knots shape mismatch: {xk.shape}, {yk.shape}, {vk.shape}'
        raise ValueError(msg)

    y = np.full(xq.shape[0], np.nan, dtype=np.float32)
    v = np.full(xq.shape[0], np.nan, dtype=np.float32)
    covered = np.zeros(xq.shape[0], dtype=bool)
    if xk.size == 0:
        return y, v, covered

    q_ok = np.isfinite(xq)
    if not bool(np.any(q_ok)):
        return y, v, covered

    if xk.size == 1:
        use = q_ok & np.isclose(xq, float(xk[0]), rtol=0.0, atol=1e-6)
        y[use] = yk[0]
        v[use] = vk[0]
        covered[use] = True
        return y, v, covered

    in_range = q_ok & (xq >= float(xk[0])) & (xq <= float(xk[-1]))
    if not bool(np.any(in_range)):
        return y, v, covered

    q = xq[in_range].astype(np.float64, copy=False)
    y[in_range] = np.interp(
        q, xk.astype(np.float64, copy=False), yk.astype(np.float64, copy=False)
    ).astype(np.float32, copy=False)
    v[in_range] = np.interp(
        q, xk.astype(np.float64, copy=False), vk.astype(np.float64, copy=False)
    ).astype(np.float32, copy=False)
    covered[in_range] = True
    return y, v, covered


def _fit_side_trend_knots_from_points(
    *,
    x_side: np.ndarray,
    y_side_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    xs = np.asarray(x_side, dtype=np.float32)
    ys = np.asarray(y_side_sec, dtype=np.float32)
    if xs.shape != ys.shape or xs.ndim != 1:
        msg = f'x_side/y_side_sec must be 1D and same shape, got {xs.shape}, {ys.shape}'
        raise ValueError(msg)

    support = int(xs.shape[0])
    if support <= 0:
        e = np.zeros(0, dtype=np.float32)
        return e, e, e, 0

    use = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[use]
    ys = ys[use]
    support = int(xs.shape[0])
    if support <= 0:
        e = np.zeros(0, dtype=np.float32)
        return e, e, e, 0

    valid = np.ones((1, support), dtype=np.uint8)
    w_conf = np.ones((1, support), dtype=np.float32)
    trend_t, _trend_s, v_trend, _w_used, covered = robust_linear_trend(
        offsets=xs.reshape(1, -1),
        t_sec=ys.reshape(1, -1),
        valid=valid,
        w_conf=w_conf,
        section_len=int(SEMI_IRLS_SECTION_LEN),
        stride=int(SEMI_IRLS_STRIDE),
        huber_c=float(SEMI_IRLS_HUBER_C),
        iters=int(SEMI_IRLS_ITERS),
        vmin=float(GLOBAL_VMIN_M_S),
        vmax=float(GLOBAL_VMAX_M_S),
        sort_offsets=True,
        use_taper=True,
        abs_velocity=True,
    )

    tt = np.asarray(trend_t, dtype=np.float32).reshape(-1)
    vv = np.asarray(v_trend, dtype=np.float32).reshape(-1)
    cc = np.asarray(covered, dtype=bool).reshape(-1)
    fit_ok = cc & np.isfinite(tt) & np.isfinite(vv)
    if not bool(np.any(fit_ok)):
        e = np.zeros(0, dtype=np.float32)
        return e, e, e, support

    return (*_aggregate_knots_by_offset(xs[fit_ok], tt[fit_ok], vv[fit_ok]), support)


def _build_semi_trend_center_i_from_picks(
    *,
    pick_final_i: np.ndarray,
    n_samples_in: int,
    dt_sec_in: float,
    trend_offset_signed_proxy: np.ndarray,
    ffid_groups: list[np.ndarray],
    shot_x_ffid: np.ndarray,
    shot_y_ffid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    p = np.asarray(pick_final_i, dtype=np.int64)
    proxy = np.asarray(trend_offset_signed_proxy, dtype=np.float32)
    if p.ndim != 1 or proxy.ndim != 1 or p.shape != proxy.shape:
        msg = (
            'pick_final_i/trend_offset_signed_proxy must be 1D and same shape, '
            f'got {p.shape}, {proxy.shape}'
        )
        raise ValueError(msg)
    if int(n_samples_in) < 2:
        msg = f'n_samples_in must be >= 2, got {n_samples_in}'
        raise ValueError(msg)
    if float(dt_sec_in) <= 0.0:
        msg = f'dt_sec_in must be > 0, got {dt_sec_in}'
        raise ValueError(msg)

    n_traces = int(p.shape[0])
    pick_ok = valid_pick_mask(p, n_samples=int(n_samples_in))
    proxy_ok = np.isfinite(proxy)

    neighbors = _build_neighbor_indices(
        shot_x_ffid,
        shot_y_ffid,
        k=int(SEMI_NEI_K),
    )

    trend_center_i_semi = np.full(n_traces, np.nan, dtype=np.float32)
    semi_covered = np.zeros(n_traces, dtype=bool)
    semi_support_count = np.zeros(n_traces, dtype=np.int32)
    semi_v_trend = np.full(n_traces, np.nan, dtype=np.float32)
    semi_used_ffid_mask = np.zeros(len(ffid_groups), dtype=bool)

    for g, idx in enumerate(ffid_groups):
        nei = np.asarray(neighbors[g], dtype=np.int64)
        if int(nei.size) < int(SEMI_MIN_NEI):
            continue
        train_parts = [np.asarray(idx, dtype=np.int64)]
        train_parts += [np.asarray(ffid_groups[int(j)], dtype=np.int64) for j in nei]
        if len(train_parts) == 0:
            continue
        train_idx = np.concatenate(train_parts).astype(np.int64, copy=False)
        if train_idx.size == 0:
            continue

        tr_ok = pick_ok[train_idx] & proxy_ok[train_idx]
        if not bool(np.any(tr_ok)):
            continue

        x_train = proxy[train_idx][tr_ok].astype(np.float32, copy=False)
        y_train_sec = p[train_idx][tr_ok].astype(np.float32, copy=False) * float(
            dt_sec_in
        )

        (
            xk_l,
            yk_l,
            vk_l,
            supp_l,
        ) = _fit_side_trend_knots_from_points(
            x_side=x_train[x_train < 0.0],
            y_side_sec=y_train_sec[x_train < 0.0],
        )
        (
            xk_r,
            yk_r,
            vk_r,
            supp_r,
        ) = _fit_side_trend_knots_from_points(
            x_side=x_train[x_train > 0.0],
            y_side_sec=y_train_sec[x_train > 0.0],
        )

        if xk_l.size == 0 and xk_r.size == 0:
            continue
        semi_used_ffid_mask[g] = True

        x_t = proxy[idx].astype(np.float32, copy=False)
        y_l, v_l, c_l = _eval_linear_interp_no_extrap(
            x_query=x_t, x_knots=xk_l, y_knots=yk_l, v_knots=vk_l
        )
        y_r, v_r, c_r = _eval_linear_interp_no_extrap(
            x_query=x_t, x_knots=xk_r, y_knots=yk_r, v_knots=vk_r
        )

        y_out = np.full(idx.shape[0], np.nan, dtype=np.float32)
        v_out = np.full(idx.shape[0], np.nan, dtype=np.float32)
        c_out = np.zeros(idx.shape[0], dtype=bool)
        s_out = np.zeros(idx.shape[0], dtype=np.int32)

        m_l = x_t < 0.0
        if bool(np.any(m_l)):
            y_out[m_l] = y_l[m_l]
            v_out[m_l] = v_l[m_l]
            c_out[m_l] = c_l[m_l]
            s_out[m_l] = np.int32(supp_l)

        m_r = x_t > 0.0
        if bool(np.any(m_r)):
            y_out[m_r] = y_r[m_r]
            v_out[m_r] = v_r[m_r]
            c_out[m_r] = c_r[m_r]
            s_out[m_r] = np.int32(supp_r)

        m0 = np.isfinite(x_t) & (x_t == 0.0)
        if bool(np.any(m0)):
            j0 = np.flatnonzero(m0)
            for j in j0:
                if bool(c_l[j]) and bool(c_r[j]):
                    y_out[j] = np.float32(0.5 * (float(y_l[j]) + float(y_r[j])))
                    v_out[j] = np.float32(0.5 * (float(v_l[j]) + float(v_r[j])))
                    c_out[j] = True
                    s_out[j] = np.int32(max(int(supp_l), int(supp_r)))
                elif bool(c_l[j]):
                    y_out[j] = y_l[j]
                    v_out[j] = v_l[j]
                    c_out[j] = True
                    s_out[j] = np.int32(supp_l)
                elif bool(c_r[j]):
                    y_out[j] = y_r[j]
                    v_out[j] = v_r[j]
                    c_out[j] = True
                    s_out[j] = np.int32(supp_r)

        y_i = (y_out.astype(np.float64) / float(dt_sec_in)).astype(
            np.float32, copy=False
        )
        if int(n_samples_in) > 2:
            y_clip = y_i.copy()
            finite = np.isfinite(y_clip)
            y_clip[finite] = np.clip(y_clip[finite], 1.0, float(int(n_samples_in) - 1))
            y_i = y_clip

        trend_center_i_semi[idx] = y_i
        semi_covered[idx] = c_out
        semi_support_count[idx] = s_out
        semi_v_trend[idx] = v_out

    fallback_count = int(np.count_nonzero(~semi_covered))
    return (
        trend_center_i_semi.astype(np.float32, copy=False),
        semi_covered.astype(bool, copy=False),
        semi_support_count.astype(np.int32, copy=False),
        semi_v_trend.astype(np.float32, copy=False),
        semi_used_ffid_mask.astype(bool, copy=False),
        int(fallback_count),
    )


def _fill_global_missing_by_ffid_nearest_trace(
    *,
    trend_center_i_global: np.ndarray,
    ffid_groups: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    cg = np.asarray(trend_center_i_global, dtype=np.float32)
    if cg.ndim != 1:
        msg = f'trend_center_i_global must be 1D, got {cg.shape}'
        raise ValueError(msg)

    out = cg.copy()
    filled = np.zeros(cg.shape[0], dtype=bool)
    global_ok = np.isfinite(out) & (out > 0.0)

    for idx in ffid_groups:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            continue

        ok = idx[global_ok[idx]]
        if ok.size == 0:
            continue

        miss = idx[~global_ok[idx]]
        if miss.size == 0:
            continue

        for tr in miss:
            j = int(ok[np.argmin(np.abs(ok - int(tr)))])
            out[int(tr)] = out[j]
            filled[int(tr)] = True
    return out.astype(np.float32, copy=False), filled.astype(bool, copy=False)


def _compute_conf_trend1_from_trend(
    *,
    pick_final_i: np.ndarray,
    trend_center_i: np.ndarray,
    n_samples_in: int,
    dt_sec_in: float,
) -> np.ndarray:
    p = np.asarray(pick_final_i, dtype=np.int64)
    c = np.asarray(trend_center_i, dtype=np.float32)
    if p.ndim != 1 or c.ndim != 1 or p.shape != c.shape:
        msg = f'pick_final_i/trend shape mismatch: {p.shape}, {c.shape}'
        raise ValueError(msg)

    t_pick_sec = p.astype(np.float32, copy=False) * float(dt_sec_in)
    t_trend_sec = c.astype(np.float32, copy=False) * float(dt_sec_in)
    trend_ok = np.isfinite(c) & (c > 0.0)
    valid = valid_pick_mask(p, n_samples=int(n_samples_in)) & trend_ok

    conf_g = trace_confidence_from_trend_resid_gaussian(
        t_pick_sec,
        t_trend_sec,
        valid,
        sigma_ms=float(CONF_TREND_SIGMA_MS),
    )
    conf_v = trace_confidence_from_trend_resid_var(
        t_pick_sec,
        t_trend_sec,
        valid,
        half_win_traces=int(CONF_TREND_VAR_HALF_WIN_TRACES),
        sigma_std_ms=float(CONF_TREND_VAR_SIGMA_STD_MS),
        min_count=int(CONF_TREND_VAR_MIN_COUNT),
    )
    out = (
        np.asarray(conf_g, dtype=np.float32) * np.asarray(conf_v, dtype=np.float32)
    ).astype(np.float32, copy=False)
    out[~valid] = 0.0
    return out


def _load_ffid_and_shot_xy_from_segy(
    src: segyio.SegyFile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ffid_values = read_trace_field(
        src,
        segyio.TraceField.FieldRecord,
        dtype=np.int64,
        name='ffid_values',
    )
    src_x_raw = read_trace_field(
        src,
        segyio.TraceField.SourceX,
        dtype=np.float64,
        name='source_x',
    )
    src_y_raw = read_trace_field(
        src,
        segyio.TraceField.SourceY,
        dtype=np.float64,
        name='source_y',
    )
    src_grp_scalar = read_trace_field(
        src,
        segyio.TraceField.SourceGroupScalar,
        dtype=np.float64,
        name='source_group_scalar',
    )
    shot_x = _apply_source_group_scalar(src_x_raw, src_grp_scalar)
    shot_y = _apply_source_group_scalar(src_y_raw, src_grp_scalar)
    return (
        ffid_values.astype(np.int64, copy=False),
        shot_x.astype(np.float64, copy=False),
        shot_y.astype(np.float64, copy=False),
    )


def _load_offset_abs_from_segy(src: segyio.SegyFile) -> np.ndarray:
    off = read_trace_field(
        src,
        segyio.TraceField.offset,
        dtype=np.float64,
        name='offset',
    )
    return np.abs(off).astype(np.float64, copy=False)


def _fit_line_wls(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> tuple[float, float]:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    ww = np.asarray(w, dtype=np.float64)
    if xx.shape != yy.shape or xx.shape != ww.shape or xx.ndim != 1:
        msg = f'x/y/w must be 1D and same shape, got {xx.shape}, {yy.shape}, {ww.shape}'
        raise ValueError(msg)

    v = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(ww) & (ww > 0.0)
    xx = xx[v]
    yy = yy[v]
    ww = ww[v]
    if int(xx.size) < 2:
        if int(yy.size) == 0:
            return 0.0, 0.0
        return 0.0, float(np.nanmedian(yy))

    wsum = float(ww.sum())
    if not np.isfinite(wsum) or wsum <= 0.0:
        ww = np.ones_like(xx, dtype=np.float64)
        wsum = float(ww.sum())

    xm = float((ww * xx).sum() / wsum)
    ym = float((ww * yy).sum() / wsum)
    dx = xx - xm
    den = float((ww * dx * dx).sum())
    if not np.isfinite(den) or den <= 1e-12:
        return 0.0, float(ym)

    a = float((ww * dx * (yy - ym)).sum() / den)
    b = float(ym - a * xm)
    return a, b


def _project_two_piece_coef(
    *,
    xb: float,
    a1: float,
    b1: float,
    a2: float,
    b2: float,
    a_min: float,
    a_max: float,
    slope_eps: float,
) -> tuple[float, float, float, float]:
    aa1 = float(np.clip(a1, a_min, a_max))
    aa2 = float(np.clip(a2, a_min, a_max))

    if not (aa1 > aa2 + float(slope_eps)):
        a2_try = float(min(aa2, aa1 - float(slope_eps)))
        if a2_try >= float(a_min):
            aa2 = a2_try
        else:
            a1_try = float(max(aa1, aa2 + float(slope_eps)))
            if a1_try <= float(a_max):
                aa1 = a1_try
            else:
                aa1 = float(a_max)
                aa2 = float(a_min)

    yb = aa1 * float(xb) + float(b1)
    bb2 = float(yb - aa2 * float(xb))
    return aa1, float(b1), aa2, bb2


def _predict_piecewise_linear(
    x_abs: np.ndarray,
    *,
    edges: np.ndarray,
    coef: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x_abs, dtype=np.float32)
    e = np.asarray(edges, dtype=np.float32)
    c = np.asarray(coef, dtype=np.float32)
    if x.ndim != 1:
        msg = f'x_abs must be 1D, got {x.shape}'
        raise ValueError(msg)
    if e.shape != (3,):
        msg = f'edges must be (3,), got {e.shape}'
        raise ValueError(msg)
    if c.shape != (2, 2):
        msg = f'coef must be (2,2), got {c.shape}'
        raise ValueError(msg)

    xb = float(e[1])
    a1 = float(c[0, 0])
    b1 = float(c[0, 1])
    a2 = float(c[1, 0])
    b2 = float(c[1, 1])

    y = np.full(x.shape, np.nan, dtype=np.float32)
    v = np.isfinite(x)
    if not bool(np.any(v)):
        return y

    s1 = v & (x <= float(xb))
    s2 = v & (x > float(xb))
    if bool(np.any(s1)):
        y[s1] = (a1 * x[s1] + b1).astype(np.float32, copy=False)
    if bool(np.any(s2)):
        y[s2] = (a2 * x[s2] + b2).astype(np.float32, copy=False)
    return y


def _fit_two_piece_projected(
    x_abs_m: np.ndarray,
    y_sec: np.ndarray,
    w_conf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (edges, coef, used_deg_line).

    - Try TwoPieceIRLSAutoBreakStrategy.fit (deterministic).
    - If it returns None or data is too thin, fall back to a single WLS line
      and represent it as 2-piece with a midpoint break.
    - Apply projected slope constraints (vmin/vmax + monotonic a1>a2).
    """
    x = np.asarray(x_abs_m, dtype=np.float64)
    y = np.asarray(y_sec, dtype=np.float64)
    w = np.asarray(w_conf, dtype=np.float64)
    if x.shape != y.shape or x.shape != w.shape or x.ndim != 1:
        msg = f'x/y/w must be 1D and same shape, got {x.shape}, {y.shape}, {w.shape}'
        raise ValueError(msg)

    v = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0.0) & (x > 0.0)
    if int(np.count_nonzero(v)) == 0:
        xmin = 0.0
        xmax = 1.0
        xb = 0.5
        a, b = 0.0, 0.0
        used_deg = True
    else:
        xv = x[v]
        yv = y[v]
        wv = w[v]
        xmin = float(np.nanmin(xv))
        xmax = float(np.nanmax(xv))
        if not np.isfinite(xmin) or not np.isfinite(xmax):
            xmin = 0.0
            xmax = 1.0
        if float(xmax - xmin) <= 1e-6:
            xb = float(xmin)
        else:
            xb = float(0.5 * (xmin + xmax))

        tr = TwoPieceIRLSAutoBreakStrategy().fit(
            torch.from_numpy(xv.astype(np.float32, copy=False)),
            torch.from_numpy(yv.astype(np.float32, copy=False)),
            torch.from_numpy(wv.astype(np.float32, copy=False)),
        )

        if tr is None:
            a, b = _fit_line_wls(xv, yv, wv)
            used_deg = True
        else:
            edges_t = tr.edges.detach().cpu().numpy().astype(np.float32, copy=False)
            coef_t = tr.coef.detach().cpu().numpy().astype(np.float32, copy=False)
            if edges_t.shape != (3,) or coef_t.shape != (2, 2):
                msg = f'unexpected two-piece shape edges={edges_t.shape} coef={coef_t.shape}'
                raise ValueError(msg)
            xmin = float(edges_t[0])
            xb = float(edges_t[1])
            xmax = float(edges_t[2])
            a1, b1 = float(coef_t[0, 0]), float(coef_t[0, 1])
            a2, b2 = float(coef_t[1, 0]), float(coef_t[1, 1])
            a_min = 1.0 / float(GLOBAL_VMAX_M_S)
            a_max = 1.0 / float(GLOBAL_VMIN_M_S)
            a1p, b1p, a2p, b2p = _project_two_piece_coef(
                xb=xb,
                a1=a1,
                b1=b1,
                a2=a2,
                b2=b2,
                a_min=a_min,
                a_max=a_max,
                slope_eps=float(GLOBAL_SLOPE_EPS),
            )
            edges = np.asarray([xmin, xb, xmax], dtype=np.float32)
            coef = np.asarray([[a1p, b1p], [a2p, b2p]], dtype=np.float32)
            return edges, coef, False

    a_min = 1.0 / float(GLOBAL_VMAX_M_S)
    a_max = 1.0 / float(GLOBAL_VMIN_M_S)
    a1p, b1p, a2p, b2p = _project_two_piece_coef(
        xb=xb,
        a1=a,
        b1=b,
        a2=a,
        b2=b,
        a_min=a_min,
        a_max=a_max,
        slope_eps=float(GLOBAL_SLOPE_EPS),
    )
    edges = np.asarray([xmin, xb, xmax], dtype=np.float32)
    coef = np.asarray([[a1p, b1p], [a2p, b2p]], dtype=np.float32)
    return edges, coef, bool(used_deg)


def _build_global_trend_center_i(
    *,
    offset_abs_m: np.ndarray,
    trend_offset_signed_proxy: np.ndarray,
    pick_final_i: np.ndarray,
    scores: dict[str, np.ndarray],
    n_samples_in: int,
    dt_sec_in: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    bool,
]:
    off = np.asarray(offset_abs_m, dtype=np.float64)
    proxy = np.asarray(trend_offset_signed_proxy, dtype=np.float32)
    p = np.asarray(pick_final_i, dtype=np.int64)
    if off.shape != proxy.shape or off.shape != p.shape or off.ndim != 1:
        msg = f'offset/proxy/pick must be 1D and same shape, got {off.shape}, {proxy.shape}, {p.shape}'
        raise ValueError(msg)

    w = np.ones_like(off, dtype=np.float64)
    for k in SCORE_KEYS_FOR_WEIGHT:
        s = np.asarray(scores[k], dtype=np.float64)
        if s.shape != w.shape:
            msg = f'score shape mismatch {k}: {s.shape} vs {w.shape}'
            raise ValueError(msg)
        w *= np.where(np.isfinite(s) & (s > 0.0), s, 0.0)

    pick_ok = valid_pick_mask(p, n_samples=int(n_samples_in))
    v_all = pick_ok & np.isfinite(off) & (off > 0.0) & np.isfinite(w) & (w > 0.0)
    y_sec = p.astype(np.float64) * float(dt_sec_in)

    edges_all, coef_all, deg_all = _fit_two_piece_projected(
        off[v_all], y_sec[v_all], w[v_all]
    )

    left_mask = v_all & np.isfinite(proxy) & (proxy < 0.0)
    right_mask = v_all & np.isfinite(proxy) & (proxy > 0.0)

    if int(np.count_nonzero(left_mask)) >= int(GLOBAL_SIDE_MIN_PTS):
        edges_left, coef_left, deg_left = _fit_two_piece_projected(
            off[left_mask], y_sec[left_mask], w[left_mask]
        )
    else:
        edges_left, coef_left, deg_left = edges_all, coef_all, True

    if int(np.count_nonzero(right_mask)) >= int(GLOBAL_SIDE_MIN_PTS):
        edges_right, coef_right, deg_right = _fit_two_piece_projected(
            off[right_mask], y_sec[right_mask], w[right_mask]
        )
    else:
        edges_right, coef_right, deg_right = edges_all, coef_all, True

    y_hat_all = _predict_piecewise_linear(
        off.astype(np.float32, copy=False), edges=edges_all, coef=coef_all
    )
    y_hat_left = _predict_piecewise_linear(
        off.astype(np.float32, copy=False), edges=edges_left, coef=coef_left
    )
    y_hat_right = _predict_piecewise_linear(
        off.astype(np.float32, copy=False), edges=edges_right, coef=coef_right
    )

    y_hat = y_hat_all
    use_l = np.isfinite(proxy) & (proxy < 0.0)
    use_r = np.isfinite(proxy) & (proxy > 0.0)
    y_hat = np.where(use_l, y_hat_left, y_hat)
    y_hat = np.where(use_r, y_hat_right, y_hat)

    center = (y_hat.astype(np.float64) / float(dt_sec_in)).astype(
        np.float32, copy=False
    )
    if int(n_samples_in) > 2:
        c2 = center.copy()
        v = np.isfinite(c2)
        c2[v] = np.clip(c2[v], 1.0, float(int(n_samples_in) - 1))
        center = c2
    return (
        center,
        edges_all,
        coef_all,
        edges_left,
        coef_left,
        edges_right,
        coef_right,
        bool(deg_left or deg_all),
        bool(deg_right or deg_all),
    )


def _build_trend_result(
    *,
    src: segyio.SegyFile,
    n_traces: int,
    n_samples_in: int,
    dt_sec_in: float,
    pick_final_i: np.ndarray,
    scores: dict[str, np.ndarray],
    trend_center_i_local_in: np.ndarray,
    local_trend_ok_in: np.ndarray,
) -> _TrendBuildResult:
    _validate_semi_config()

    ffid_values, shot_x_trace, shot_y_trace = _load_ffid_and_shot_xy_from_segy(src)
    if ffid_values.shape != (n_traces,):
        msg = f'ffid_values must be (n_traces,), got {ffid_values.shape}, n_traces={n_traces}'
        raise ValueError(msg)

    ffid_unique_values, _ffid_inv, ffid_groups = build_groups_by_key(ffid_values)
    shot_x_ffid, shot_y_ffid = _build_shot_xy_by_ffid(
        ffid_groups=ffid_groups,
        shot_x_by_trace=shot_x_trace,
        shot_y_by_trace=shot_y_trace,
    )

    trend_center_i_raw = np.full(n_traces, np.nan, dtype=np.float32)

    trend_center_i_local = np.asarray(trend_center_i_local_in, dtype=np.float32)
    local_trend_ok = np.asarray(local_trend_ok_in, dtype=bool)
    if trend_center_i_local.shape != (n_traces,) or local_trend_ok.shape != (n_traces,):
        msg = (
            f'local trend shape mismatch: center={trend_center_i_local.shape}, '
            f'ok={local_trend_ok.shape}, n_traces={n_traces}'
        )
        raise ValueError(msg)

    if int(n_samples_in) > 2:
        v = np.isfinite(trend_center_i_local) & (trend_center_i_local > 0.0)
        if bool(np.any(v)):
            trend_center_i_local = trend_center_i_local.copy()
            trend_center_i_local[v] = np.clip(
                trend_center_i_local[v],
                1.0,
                float(int(n_samples_in) - 1),
            )

    offset_abs_m = _load_offset_abs_from_segy(src)
    if offset_abs_m.shape != (n_traces,):
        msg = f'offset_abs_m must be (n_traces,), got {offset_abs_m.shape}, n_traces={n_traces}'
        raise ValueError(msg)

    if not bool(SEMI_GLOBAL_ENABLE):
        msg = 'SEMI_GLOBAL_ENABLE must be True for semi-global -> global trend build'
        raise ValueError(msg)

    trend_offset_signed_proxy = _build_offset_signed_proxy_by_ffid(
        offset_abs_m=offset_abs_m,
        ffid_groups=ffid_groups,
    )

    (
        trend_center_i_semi,
        semi_covered,
        semi_support_count,
        semi_v_trend,
        semi_used_ffid_mask,
        semi_fallback_count,
    ) = _build_semi_trend_center_i_from_picks(
        pick_final_i=pick_final_i,
        n_samples_in=int(n_samples_in),
        dt_sec_in=float(dt_sec_in),
        trend_offset_signed_proxy=trend_offset_signed_proxy,
        ffid_groups=ffid_groups,
        shot_x_ffid=shot_x_ffid,
        shot_y_ffid=shot_y_ffid,
    )

    (
        trend_center_i_global,
        global_edges_all,
        global_coef_all,
        global_edges_left,
        global_coef_left,
        global_edges_right,
        global_coef_right,
        deg_left,
        deg_right,
    ) = _build_global_trend_center_i(
        offset_abs_m=offset_abs_m,
        trend_offset_signed_proxy=trend_offset_signed_proxy,
        pick_final_i=pick_final_i,
        scores=scores,
        n_samples_in=int(n_samples_in),
        dt_sec_in=float(dt_sec_in),
    )

    if bool(deg_left):
        print('[WARN] globaltrend(left) used degenerate 1-line representation')
    if bool(deg_right):
        print('[WARN] globaltrend(right) used degenerate 1-line representation')

    trend_center_i_global_filled, global_missing_filled_mask = (
        _fill_global_missing_by_ffid_nearest_trace(
            trend_center_i_global=trend_center_i_global,
            ffid_groups=ffid_groups,
        )
    )

    vabs = np.abs(np.asarray(semi_v_trend, dtype=np.float32))
    v_sat = np.isfinite(vabs) & (
        (np.abs(vabs - float(GLOBAL_VMIN_M_S)) <= 0.01 * float(GLOBAL_VMIN_M_S))
        | (np.abs(vabs - float(GLOBAL_VMAX_M_S)) <= 0.01 * float(GLOBAL_VMAX_M_S))
    )
    semi_low_mask = (
        (~np.asarray(semi_covered, dtype=bool))
        | (
            np.asarray(semi_support_count, dtype=np.int32)
            < int(SEMI_MIN_SUPPORT_PER_TRACE)
        )
        | v_sat
    )

    trend_center_i_final = np.where(
        semi_low_mask,
        trend_center_i_global_filled,
        trend_center_i_semi,
    ).astype(np.float32, copy=False)
    if int(n_samples_in) > 2:
        f = np.isfinite(trend_center_i_final)
        trend_center_i_final = trend_center_i_final.copy()
        trend_center_i_final[f] = np.clip(
            trend_center_i_final[f],
            1.0,
            float(int(n_samples_in) - 1),
        )

    global_ok = np.isfinite(trend_center_i_global_filled) & (
        trend_center_i_global_filled > 0.0
    )
    global_replaced_mask = semi_low_mask & global_ok

    if int(LOCAL_GLOBAL_DIFF_TH_SAMPLES) < 0:
        msg = f'LOCAL_GLOBAL_DIFF_TH_SAMPLES must be >= 0, got {LOCAL_GLOBAL_DIFF_TH_SAMPLES}'
        raise ValueError(msg)

    local_ok = (
        np.asarray(local_trend_ok, dtype=bool)
        & np.isfinite(trend_center_i_local)
        & (trend_center_i_local > 0.0)
    )

    diff = np.abs(trend_center_i_global_filled - trend_center_i_local).astype(
        np.float32, copy=False
    )
    bad = (
        local_ok
        & global_ok
        & np.isfinite(diff)
        & (diff >= float(int(LOCAL_GLOBAL_DIFF_TH_SAMPLES)))
    )
    local_discard_mask = _expand_mask_within_ffid_groups(
        mask_by_trace=bad,
        ffid_groups=ffid_groups,
        radius=int(LOCAL_DISCARD_RADIUS_TRACES),
    )

    use_integrated = (~local_ok) | local_discard_mask
    trend_center_i_used = np.where(
        use_integrated, trend_center_i_final, trend_center_i_local
    ).astype(np.float32, copy=False)

    if int(n_samples_in) > 2:
        f = np.isfinite(trend_center_i_used)
        if bool(np.any(f)):
            trend_center_i_used = trend_center_i_used.copy()
            trend_center_i_used[f] = np.clip(
                trend_center_i_used[f],
                1.0,
                float(int(n_samples_in) - 1),
            )

    trend_filled_mask = use_integrated.astype(bool, copy=False)

    nn_replaced_mask = np.zeros(n_traces, dtype=bool)
    conf_trend1 = _compute_conf_trend1_from_trend(
        pick_final_i=pick_final_i,
        trend_center_i=trend_center_i_used,
        n_samples_in=int(n_samples_in),
        dt_sec_in=float(dt_sec_in),
    )

    return _TrendBuildResult(
        trend_center_i_raw=trend_center_i_raw.astype(np.float32, copy=False),
        trend_center_i_local=trend_center_i_local.astype(np.float32, copy=False),
        trend_center_i_semi=trend_center_i_semi.astype(np.float32, copy=False),
        trend_center_i_final=trend_center_i_final.astype(np.float32, copy=False),
        trend_center_i_used=trend_center_i_used.astype(np.float32, copy=False),
        trend_center_i_global=trend_center_i_global_filled.astype(
            np.float32, copy=False
        ),
        nn_replaced_mask=nn_replaced_mask.astype(bool, copy=False),
        global_replaced_mask=global_replaced_mask.astype(bool, copy=False),
        global_missing_filled_mask=global_missing_filled_mask.astype(bool, copy=False),
        local_discard_mask=local_discard_mask.astype(bool, copy=False),
        global_edges_all=global_edges_all.astype(np.float32, copy=False),
        global_coef_all=global_coef_all.astype(np.float32, copy=False),
        global_edges_left=global_edges_left.astype(np.float32, copy=False),
        global_coef_left=global_coef_left.astype(np.float32, copy=False),
        global_edges_right=global_edges_right.astype(np.float32, copy=False),
        global_coef_right=global_coef_right.astype(np.float32, copy=False),
        trend_filled_mask=trend_filled_mask.astype(bool, copy=False),
        ffid_values=ffid_values.astype(np.int64, copy=False),
        ffid_unique_values=ffid_unique_values.astype(np.int64, copy=False),
        shot_x_ffid=shot_x_ffid.astype(np.float64, copy=False),
        shot_y_ffid=shot_y_ffid.astype(np.float64, copy=False),
        semi_used_ffid_mask=semi_used_ffid_mask.astype(bool, copy=False),
        semi_fallback_count=int(semi_fallback_count),
        semi_low_mask=semi_low_mask.astype(bool, copy=False),
        semi_covered=semi_covered.astype(bool, copy=False),
        semi_support_count=semi_support_count.astype(np.int32, copy=False),
        semi_v_trend=semi_v_trend.astype(np.float32, copy=False),
        conf_trend1=conf_trend1.astype(np.float32, copy=False),
    )


def _field_key_to_int(key: object) -> int:
    if isinstance(key, (int, np.integer)):
        return int(key)
    v = getattr(key, 'value', None)
    if isinstance(v, (int, np.integer)):
        return int(v)
    try:
        return int(key)
    except (TypeError, ValueError) as e:
        msg = f'cannot convert segy field key to int: {key!r} (type={type(key)})'
        raise TypeError(msg) from e


def _extract_256(trace: np.ndarray, *, center_i: float) -> tuple[np.ndarray, int]:
    x = np.asarray(trace, dtype=np.float32)
    if x.ndim != 1:
        msg = f'trace must be 1D, got {x.shape}'
        raise ValueError(msg)

    out = np.zeros(2 * HALF_WIN, dtype=np.float32)

    if (not np.isfinite(center_i)) or center_i <= 0.0:
        return out, -1

    c = int(np.rint(float(center_i)))
    start = c - HALF_WIN
    end = c + HALF_WIN

    n = x.shape[0]
    ov_l = max(0, start)
    ov_r = min(n, end)
    if ov_l >= ov_r:
        return out, start

    dst_l = ov_l - start
    dst_r = dst_l + (ov_r - ov_l)
    out[dst_l:dst_r] = x[ov_l:ov_r]
    return out, start


def _upsample_256_to_512_linear(win256: np.ndarray) -> np.ndarray:
    y = np.asarray(win256, dtype=np.float32)
    if y.shape != (2 * HALF_WIN,):
        msg = f'win256 must be (256,), got {y.shape}'
        raise ValueError(msg)

    xp = np.arange(2 * HALF_WIN, dtype=np.float32)  # 0..255
    xq = np.arange(OUT_NS, dtype=np.float32) / 2.0  # 0,0.5,..,255.5
    out = np.interp(xq, xp, y, left=0.0, right=0.0).astype(np.float32, copy=False)
    if out.shape != (OUT_NS,):
        msg = f'unexpected upsample output shape: {out.shape}'
        raise ValueError(msg)
    return out


def _base_valid_mask(
    *,
    pick_final_i: np.ndarray,
    trend_center_i: np.ndarray,
    n_samples_in: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (base_valid, reason_mask_partial)
    reason bits:
      bit0: invalid pick
      bit1: trend missing (after fill)
      bit2: pick outside window (strict, to avoid 0/512 labels)
    """
    n_tr = int(pick_final_i.shape[0])
    if trend_center_i.shape != (n_tr,):
        msg = f'trend_center_i mismatch: {trend_center_i.shape} vs {(n_tr,)}'
        raise ValueError(msg)

    p = np.asarray(pick_final_i, dtype=np.int64)
    c = np.asarray(trend_center_i, dtype=np.float32)

    reason = np.zeros(n_tr, dtype=np.uint8)

    pick_ok = valid_pick_mask(p, n_samples=int(n_samples_in))
    reason[~pick_ok] |= 1 << 0

    trend_ok = np.isfinite(c) & (c > 0.0)
    reason[~trend_ok] |= 1 << 1

    c_round = np.full(n_tr, -1, dtype=np.int64)
    if bool(np.any(trend_ok)):
        c_round[trend_ok] = np.rint(c[trend_ok]).astype(np.int64, copy=False)
    # strict "< HALF_WIN" to keep pick_win_512 in (0, OUT_NS)
    win_ok = pick_ok & trend_ok & (np.abs(p - c_round) < int(HALF_WIN))
    reason[~win_ok] |= 1 << 2

    return win_ok, reason


def build_keep_mask(
    *,
    pick_final_i: np.ndarray,
    trend_center_i: np.ndarray,
    n_samples_in: int,
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float] | None,
) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    """keep_mask, thresholds_used, reason_mask, base_valid

    reason bits:
      bit0: invalid pick
      bit1: trend missing
      bit2: pick outside window
      bit3: conf_prob low
      bit4: conf_trend low
      bit5: conf_rs low
    """
    n_tr = int(pick_final_i.shape[0])

    base_valid, reason = _base_valid_mask(
        pick_final_i=pick_final_i,
        trend_center_i=trend_center_i,
        n_samples_in=n_samples_in,
    )

    bit_map = {'conf_prob1': 3, 'conf_trend1': 4, 'conf_rs1': 5}

    thresholds_used: dict[str, float] = {}
    keep = base_valid.copy()

    for k in SCORE_KEYS_FOR_FILTER:
        s = np.asarray(scores[k], dtype=np.float32)
        if s.shape != (n_tr,):
            msg = f'score {k} must be (n_traces,), got {s.shape}, n_traces={n_tr}'
            raise ValueError(msg)

        if thresholds is None:
            th = _percentile_threshold(s[base_valid], frac=DROP_LOW_FRAC)
        else:
            if k not in thresholds:
                msg = f'thresholds missing key: {k} available={sorted(thresholds)}'
                raise KeyError(msg)
            th = float(thresholds[k])

        thresholds_used[k] = th

        keep_k = np.zeros(n_tr, dtype=bool) if np.isnan(th) else (s >= th)

        low = base_valid & (~keep_k)
        reason[low] |= 1 << bit_map[k]
        keep &= keep_k

    return keep, thresholds_used, reason, base_valid


def _load_minimal_inputs_for_thresholds(
    segy_path: Path,
) -> tuple[
    int,
    int,
    float,
    np.ndarray,
    dict[str, np.ndarray],
    _TrendBuildResult,
]:
    infer_npz = infer_npz_path_for_segy(segy_path)
    if not infer_npz.exists():
        msg = f'infer npz not found for segy: {segy_path} expected={infer_npz}'
        raise FileNotFoundError(msg)

    with segyio.open(str(segy_path), 'r', ignore_geometry=True) as src:
        n_traces = int(src.tracecount)
        if n_traces <= 0:
            msg = f'no traces: {segy_path}'
            raise ValueError(msg)

        ns_in = int(src.samples.size)
        if ns_in <= 0:
            msg = f'invalid n_samples: {ns_in}'
            raise ValueError(msg)

        dt_us_in = int(src.bin[segyio.BinField.Interval])
        if dt_us_in <= 0:
            msg = f'invalid dt_us: {dt_us_in}'
            raise ValueError(msg)
        dt_sec_in = float(dt_us_in) * 1e-6
        with np.load(infer_npz, allow_pickle=False) as z:
            pick_final = require_npz_key(z, PICK_KEY).astype(np.int64, copy=False)
            if pick_final.ndim != 1 or pick_final.shape[0] != n_traces:
                msg = f'{PICK_KEY} must be (n_traces,), got {pick_final.shape}, n_traces={n_traces}'
                raise ValueError(msg)

            scores_weight: dict[str, np.ndarray] = {}
            for k in SCORE_KEYS_FOR_WEIGHT:
                scores_weight[k] = require_npz_key(z, k).astype(np.float32, copy=False)

            trend_center_i_local, local_trend_ok = _load_stage1_local_trend_center_i(
                z=z,
                n_traces=n_traces,
                dt_sec_in=dt_sec_in,
            )

            trend_res = _build_trend_result(
                src=src,
                n_traces=n_traces,
                n_samples_in=ns_in,
                dt_sec_in=dt_sec_in,
                pick_final_i=pick_final,
                scores=scores_weight,
                trend_center_i_local_in=trend_center_i_local,
                local_trend_ok_in=local_trend_ok,
            )
            scores_filter: dict[str, np.ndarray] = {
                'conf_prob1': scores_weight['conf_prob1'],
                'conf_rs1': scores_weight['conf_rs1'],
                'conf_trend1': trend_res.conf_trend1,
            }

            return (
                n_traces,
                ns_in,
                dt_sec_in,
                pick_final,
                scores_filter,
                trend_res,
            )


def compute_global_thresholds(segys: list[Path]) -> dict[str, float]:
    vals: dict[str, list[np.ndarray]] = {k: [] for k in SCORE_KEYS_FOR_FILTER}

    n_files_used = 0
    n_base_total = 0

    for p in segys:
        infer_npz = infer_npz_path_for_segy(p)
        if not infer_npz.exists():
            continue

        (
            _n_traces,
            ns_in,
            _dt_sec_in,
            pick_final,
            scores,
            trend_res,
        ) = _load_minimal_inputs_for_thresholds(p)

        base_valid, _reason = _base_valid_mask(
            pick_final_i=pick_final,
            trend_center_i=trend_res.trend_center_i_used,
            n_samples_in=ns_in,
        )

        n_b = int(np.count_nonzero(base_valid))
        if n_b <= 0:
            continue

        for k in SCORE_KEYS_FOR_FILTER:
            s = np.asarray(scores[k], dtype=np.float32)
            vals[k].append(s[base_valid])

        n_files_used += 1
        n_base_total += n_b

    if n_files_used == 0:
        msg = 'no segy files with infer npz found for global thresholds'
        raise RuntimeError(msg)
    if n_base_total == 0:
        msg = 'no base_valid traces across all files (cannot compute global thresholds)'
        raise RuntimeError(msg)

    thresholds: dict[str, float] = {}
    for k in SCORE_KEYS_FOR_FILTER:
        if not vals[k]:
            msg = f'no values accumulated for score={k}'
            raise RuntimeError(msg)
        v = np.concatenate(vals[k]).astype(np.float32, copy=False)
        th = _percentile_threshold(v, frac=DROP_LOW_FRAC)
        if np.isnan(th):
            msg = f'global threshold became NaN for score={k}'
            raise RuntimeError(msg)
        thresholds[k] = th

    print(
        f'[GLOBAL_TH] files_used={n_files_used} base_valid_total={n_base_total} '
        f'p10 prob={thresholds["conf_prob1"]:.6g} trend={thresholds["conf_trend1"]:.6g} rs={thresholds["conf_rs1"]:.6g}'
    )
    return thresholds


def _build_phase_pick_csr_npz(
    *,
    out_path: Path,
    pick_win_512: np.ndarray,
    keep_mask: np.ndarray,
    n_traces: int,
) -> int:
    """Write CSR pick npz required by seisai_dataset.load_phase_pick_csr_npz.

    P picks: 1 per trace at most (int sample index in [1, OUT_NS-1])
    S picks: empty
    """
    pw = np.asarray(pick_win_512, dtype=np.float32)
    km = np.asarray(keep_mask, dtype=bool)
    if pw.shape != (n_traces,) or km.shape != (n_traces,):
        msg = f'pick_win_512/keep_mask must be (n_traces,), got {pw.shape}, {km.shape}, n_traces={n_traces}'
        raise ValueError(msg)

    p_indptr = np.zeros(n_traces + 1, dtype=np.int64)

    pick_i = np.rint(pw).astype(np.int64, copy=False)
    valid = km & np.isfinite(pw) & (pick_i > 0) & (pick_i < int(OUT_NS))

    nnz = int(np.count_nonzero(valid))
    p_data = np.empty(nnz, dtype=np.int64)

    j = 0
    for i in range(n_traces):
        if bool(valid[i]):
            p_data[j] = int(pick_i[i])
            j += 1
        p_indptr[i + 1] = j

    s_indptr = np.zeros(n_traces + 1, dtype=np.int64)
    s_data = np.zeros(0, dtype=np.int64)

    np.savez_compressed(
        out_path,
        n_traces=np.int32(n_traces),
        p_indptr=p_indptr,
        p_data=p_data,
        s_indptr=s_indptr,
        s_data=s_data,
    )
    return nnz


# =========================
# Main per-file processing
# =========================
def process_one_segy(
    segy_path: Path, *, global_thresholds: dict[str, float] | None
) -> None:
    infer_npz = infer_npz_path_for_segy(segy_path)
    if not infer_npz.exists():
        msg = f'infer npz not found for segy: {segy_path}  expected={infer_npz}'
        raise FileNotFoundError(msg)

    out_segy = out_segy_path_for_in(segy_path)
    out_segy.parent.mkdir(parents=True, exist_ok=True)

    side_npz = out_sidecar_npz_path_for_out(out_segy)
    pick_csr_npz = out_pick_csr_npz_path_for_out(out_segy)

    with segyio.open(str(segy_path), 'r', ignore_geometry=True) as src:
        n_traces = int(src.tracecount)
        if n_traces <= 0:
            msg = f'no traces: {segy_path}'
            raise ValueError(msg)

        ns_in = int(src.samples.size)
        if ns_in <= 0:
            msg = f'invalid n_samples: {ns_in}'
            raise ValueError(msg)

        dt_us_in = int(src.bin[segyio.BinField.Interval])
        if dt_us_in <= 0:
            msg = f'invalid dt_us: {dt_us_in}'
            raise ValueError(msg)

        if dt_us_in % UP_FACTOR != 0:
            msg = f'dt_us must be divisible by {UP_FACTOR}. got {dt_us_in}'
            raise ValueError(msg)

        dt_us_out = dt_us_in // UP_FACTOR
        dt_sec_in = float(dt_us_in) * 1e-6
        dt_sec_out = float(dt_us_out) * 1e-6

        with np.load(infer_npz, allow_pickle=False) as z:
            pick_final = require_npz_key(z, PICK_KEY).astype(np.int64, copy=False)
            if pick_final.ndim != 1 or pick_final.shape[0] != n_traces:
                msg = f'{PICK_KEY} must be (n_traces,), got {pick_final.shape}, n_traces={n_traces}'
                raise ValueError(msg)

            scores_weight: dict[str, np.ndarray] = {}
            for k in SCORE_KEYS_FOR_WEIGHT:
                scores_weight[k] = require_npz_key(z, k).astype(np.float32, copy=False)

            trend_center_i_local, local_trend_ok = _load_stage1_local_trend_center_i(
                z=z,
                n_traces=n_traces,
                dt_sec_in=dt_sec_in,
            )

            trend_res = _build_trend_result(
                src=src,
                n_traces=n_traces,
                n_samples_in=ns_in,
                dt_sec_in=dt_sec_in,
                pick_final_i=pick_final,
                scores=scores_weight,
                trend_center_i_local_in=trend_center_i_local,
                local_trend_ok_in=local_trend_ok,
            )
            scores_filter: dict[str, np.ndarray] = {
                'conf_prob1': scores_weight['conf_prob1'],
                'conf_rs1': scores_weight['conf_rs1'],
                'conf_trend1': trend_res.conf_trend1,
            }
            trend_center_i_raw = trend_res.trend_center_i_raw
            trend_center_i_local = trend_res.trend_center_i_local
            trend_center_i_semi = trend_res.trend_center_i_semi
            trend_center_i_final = trend_res.trend_center_i_final
            trend_center_i_used = trend_res.trend_center_i_used
            trend_filled_mask = trend_res.trend_filled_mask
            ffid_values = trend_res.ffid_values
            ffid_unique_values = trend_res.ffid_unique_values
            shot_x_ffid = trend_res.shot_x_ffid
            shot_y_ffid = trend_res.shot_y_ffid
            semi_used_ffid_mask = trend_res.semi_used_ffid_mask
            semi_fallback_count = int(trend_res.semi_fallback_count)
            semi_low_mask = trend_res.semi_low_mask
            semi_covered = trend_res.semi_covered
            semi_support_count = trend_res.semi_support_count
            semi_v_trend = trend_res.semi_v_trend

            nn_replaced_mask = trend_res.nn_replaced_mask
            global_replaced_mask = trend_res.global_replaced_mask
            global_missing_filled_mask = trend_res.global_missing_filled_mask
            local_discard_mask = trend_res.local_discard_mask
            trend_center_i_global = trend_res.trend_center_i_global
            global_edges_all = trend_res.global_edges_all
            global_coef_all = trend_res.global_coef_all
            global_edges_left = trend_res.global_edges_left
            global_coef_left = trend_res.global_coef_left
            global_edges_right = trend_res.global_edges_right
            global_coef_right = trend_res.global_coef_right
            conf_trend1 = trend_res.conf_trend1

        thresholds_arg = None
        if THRESH_MODE == 'global':
            thresholds_arg = global_thresholds
            if thresholds_arg is None:
                msg = 'THRESH_MODE=global but global_thresholds is None'
                raise RuntimeError(msg)
        elif THRESH_MODE == 'per_segy':
            thresholds_arg = None
        else:
            msg = f"THRESH_MODE must be 'global' or 'per_segy', got {THRESH_MODE!r}"
            raise ValueError(msg)

        keep_mask, thresholds_used, reason_mask, _base_valid = build_keep_mask(
            pick_final_i=pick_final,
            trend_center_i=trend_center_i_used,
            n_samples_in=ns_in,
            scores=scores_filter,
            thresholds=thresholds_arg,
        )

        c_round = np.full(n_traces, -1, dtype=np.int64)
        c_ok = np.isfinite(trend_center_i_used) & (trend_center_i_used > 0.0)
        if bool(np.any(c_ok)):
            c_round[c_ok] = np.rint(trend_center_i_used[c_ok]).astype(
                np.int64, copy=False
            )
        win_start_i = c_round - int(HALF_WIN)

        pick_win_512 = (
            pick_final.astype(np.float32) - win_start_i.astype(np.float32)
        ) * float(UP_FACTOR)
        pick_win_512[~keep_mask] = np.nan

        spec = segyio.spec()
        spec.tracecount = n_traces
        spec.samples = np.arange(OUT_NS, dtype=np.int32)
        spec.format = 5  # IEEE float32

        sorting_val = getattr(src, 'sorting', 1)
        spec.sorting = (
            int(sorting_val) if isinstance(sorting_val, (int, np.integer)) else 1
        )

        with segyio.create(str(out_segy), spec) as dst:
            dst.text[0] = src.text[0]

            for k in src.bin:
                dst.bin[_field_key_to_int(k)] = src.bin[k]
            dst.bin[_field_key_to_int(segyio.BinField.Interval)] = dt_us_out
            dst.bin[_field_key_to_int(segyio.BinField.Samples)] = OUT_NS

            for i in range(n_traces):
                h = {_field_key_to_int(k): v for k, v in dict(src.header[i]).items()}
                h[_field_key_to_int(segyio.TraceField.TRACE_SAMPLE_INTERVAL)] = (
                    dt_us_out
                )
                h[_field_key_to_int(segyio.TraceField.TRACE_SAMPLE_COUNT)] = OUT_NS
                dst.header[i] = h

                tr = np.asarray(src.trace[i], dtype=np.float32)
                w256, _start = _extract_256(tr, center_i=float(trend_center_i_used[i]))
                w512 = _upsample_256_to_512_linear(w256)
                dst.trace[i] = w512

            dst.flush()

    nnz_p = _build_phase_pick_csr_npz(
        out_path=pick_csr_npz,
        pick_win_512=pick_win_512,
        keep_mask=keep_mask,
        n_traces=n_traces,
    )

    np.savez_compressed(
        side_npz,
        src_segy=str(segy_path),
        src_infer_npz=str(infer_npz),
        out_segy=str(out_segy),
        out_pick_csr_npz=str(pick_csr_npz),
        thresh_mode=str(THRESH_MODE),
        drop_low_frac=np.float32(DROP_LOW_FRAC),
        dt_sec_in=np.float32(dt_sec_in),
        dt_sec_out=np.float32(dt_sec_out),
        dt_us_in=np.int32(dt_us_in),
        dt_us_out=np.int32(dt_us_out),
        n_traces=np.int32(n_traces),
        n_samples_in=np.int32(ns_in),
        n_samples_out=np.int32(OUT_NS),
        semi_global_enable=np.asarray(bool(SEMI_GLOBAL_ENABLE)),
        semi_nei_k=np.int32(SEMI_NEI_K),
        semi_min_nei=np.int32(SEMI_MIN_NEI),
        semi_irls_section_len=np.int32(SEMI_IRLS_SECTION_LEN),
        semi_irls_stride=np.int32(SEMI_IRLS_STRIDE),
        semi_min_support_per_trace=np.int32(SEMI_MIN_SUPPORT_PER_TRACE),
        local_global_diff_th_samples=np.int32(LOCAL_GLOBAL_DIFF_TH_SAMPLES),
        local_discard_radius_traces=np.int32(LOCAL_DISCARD_RADIUS_TRACES),
        trend_center_i_raw=trend_center_i_raw.astype(np.float32, copy=False),
        trend_center_i_local=trend_center_i_local.astype(np.float32, copy=False),
        trend_center_i_semi=trend_center_i_semi.astype(np.float32, copy=False),
        trend_center_i_final=trend_center_i_final.astype(np.float32, copy=False),
        trend_center_i_used=trend_center_i_used.astype(np.float32, copy=False),
        trend_center_i_global=trend_center_i_global.astype(np.float32, copy=False),
        nn_replaced_mask=nn_replaced_mask.astype(bool, copy=False),
        global_replaced_mask=global_replaced_mask.astype(bool, copy=False),
        global_missing_filled_mask=global_missing_filled_mask.astype(bool, copy=False),
        global_edges_all=global_edges_all.astype(np.float32, copy=False),
        global_coef_all=global_coef_all.astype(np.float32, copy=False),
        global_edges_left=global_edges_left.astype(np.float32, copy=False),
        global_coef_left=global_coef_left.astype(np.float32, copy=False),
        global_edges_right=global_edges_right.astype(np.float32, copy=False),
        global_coef_right=global_coef_right.astype(np.float32, copy=False),
        trend_center_i=trend_center_i_used.astype(np.float32, copy=False),
        trend_filled_mask=trend_filled_mask.astype(bool, copy=False),
        trend_center_i_round=c_round.astype(np.int64, copy=False),
        semi_used_ffid_mask=semi_used_ffid_mask.astype(bool, copy=False),
        semi_fallback_count=np.int32(semi_fallback_count),
        semi_low_mask=semi_low_mask.astype(bool, copy=False),
        semi_covered=semi_covered.astype(bool, copy=False),
        semi_support_count=semi_support_count.astype(np.int32, copy=False),
        semi_v_trend=semi_v_trend.astype(np.float32, copy=False),
        ffid_values=ffid_values.astype(np.int64, copy=False),
        ffid_unique_values=ffid_unique_values.astype(np.int64, copy=False),
        shot_x_ffid=shot_x_ffid.astype(np.float64, copy=False),
        shot_y_ffid=shot_y_ffid.astype(np.float64, copy=False),
        window_start_i=win_start_i.astype(np.int64, copy=False),
        pick_final_i=pick_final.astype(np.int64, copy=False),
        pick_win_512=pick_win_512.astype(np.float32, copy=False),
        keep_mask=keep_mask.astype(bool, copy=False),
        reason_mask=reason_mask.astype(np.uint8, copy=False),
        th_conf_prob1=np.float32(thresholds_used['conf_prob1']),
        th_conf_trend1=np.float32(thresholds_used['conf_trend1']),
        th_conf_rs1=np.float32(thresholds_used['conf_rs1']),
        conf_prob1=scores_filter['conf_prob1'].astype(np.float32, copy=False),
        conf_trend1=conf_trend1.astype(np.float32, copy=False),
        conf_rs1=scores_filter['conf_rs1'].astype(np.float32, copy=False),
    )

    n_keep = int(np.count_nonzero(keep_mask))
    n_fill = int(np.count_nonzero(trend_filled_mask))
    n_ld = int(np.count_nonzero(local_discard_mask))
    n_nn = int(np.count_nonzero(nn_replaced_mask))
    n_gl = int(np.count_nonzero(global_replaced_mask))
    tag = 'global' if THRESH_MODE == 'global' else 'per_segy'
    print(
        f'[OK] {segy_path.name} -> {out_segy.name}  keep={n_keep}/{n_traces} '
        f'filled_trend={n_fill}/{n_traces} discard_local={n_ld} '
        f'fill_nn={n_nn} fill_global={n_gl} '
        f'labels_written(P)={nnz_p} '
        f'th({tag} p10) prob={thresholds_used["conf_prob1"]:.6g} '
        f'trend={thresholds_used["conf_trend1"]:.6g} rs={thresholds_used["conf_rs1"]:.6g}'
    )
    if SEMI_GLOBAL_ENABLE:
        ffids_total = int(ffid_unique_values.shape[0])
        ffids_used_semi = int(np.count_nonzero(semi_used_ffid_mask))
        print(
            f'[SEMI] {segy_path.name} ffids_total={ffids_total} '
            f'ffids_used_semi={ffids_used_semi} '
            f'semi_missing_traces={semi_fallback_count}'
        )
        if semi_fallback_count > 0:
            print(
                f'[WARN] {segy_path.name} semi missing (filled by global) for '
                f'{semi_fallback_count} traces'
            )


def main() -> None:
    _validate_semi_config()
    segys = find_segy_files(IN_SEGY_ROOT, exts=SEGY_EXTS, recursive=True)
    print(f'[RUN] found {len(segys)} segy files under {IN_SEGY_ROOT}')

    segys2: list[Path] = []
    for p in segys:
        infer_npz = infer_npz_path_for_segy(p)
        if not infer_npz.exists():
            print(f'[SKIP] infer npz missing: {p}  expected={infer_npz}')
            continue
        segys2.append(p)

    global_thresholds = None
    if THRESH_MODE == 'global':
        global_thresholds = compute_global_thresholds(segys2)

    for p in segys2:
        process_one_segy(p, global_thresholds=global_thresholds)


if __name__ == '__main__':
    main()
