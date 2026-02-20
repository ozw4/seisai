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
from seisai_pick.trend.trend_fit_strategy import TwoPieceIRLSAutoBreakStrategy

# =========================
# CONFIG（ここだけ触ればOK）
# =========================
IN_SEGY_ROOT = Path('/home/dcuser/data/ActiveSeisField/jogsarar')
IN_INFER_ROOT = Path('/home/dcuser/data/ActiveSeisField/jogsarar_out')
OUT_SEGY_ROOT = Path('/home/dcuser/data/ActiveSeisField/jogsarar_psn512_up1_drop005')

SEGY_EXTS = ('.sgy', '.segy')

HALF_WIN = 256  # ±128 => 256
UP_FACTOR = 1  # 256 -> 512
OUT_NS = 2 * HALF_WIN * UP_FACTOR  # 512

# 下位除外（p10）
DROP_LOW_FRAC = 0.05  # 下位除外
SCORE_KEYS = (
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
SEMI_BIN_W_M = 25.0
SEMI_NEI_K = 8
SEMI_MIN_NEI = 3
SEMI_PHYS_MAX_NONPOS_FRAC = 0.0

# global trendline (proxy sign split)
GLOBAL_VMIN_M_S = 300.0
GLOBAL_VMAX_M_S = 6000.0
GLOBAL_SLOPE_EPS = 1e-6
GLOBAL_SIDE_MIN_PTS = 16  # 2*min_pts of TwoPieceIRLSAutoBreakStrategy default


# =========================
# Utility
# =========================
@dataclass(frozen=True)
class _TrendBuildResult:
    trend_center_i_raw: np.ndarray
    trend_center_i_local: np.ndarray
    trend_center_i_semi: np.ndarray
    trend_center_i_used: np.ndarray
    trend_center_i_global: np.ndarray
    nn_replaced_mask: np.ndarray
    global_replaced_mask: np.ndarray
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


def _validate_semi_config() -> None:
    if float(SEMI_BIN_W_M) <= 0.0:
        msg = f'SEMI_BIN_W_M must be > 0, got {SEMI_BIN_W_M}'
        raise ValueError(msg)
    if int(SEMI_NEI_K) < 1:
        msg = f'SEMI_NEI_K must be >= 1, got {SEMI_NEI_K}'
        raise ValueError(msg)
    if int(SEMI_MIN_NEI) < 1:
        msg = f'SEMI_MIN_NEI must be >= 1, got {SEMI_MIN_NEI}'
        raise ValueError(msg)
    frac = float(SEMI_PHYS_MAX_NONPOS_FRAC)
    if frac < 0.0 or frac > 1.0:
        msg = (
            'SEMI_PHYS_MAX_NONPOS_FRAC must be in [0,1], '
            f'got {SEMI_PHYS_MAX_NONPOS_FRAC}'
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


def _load_trend_center_i(
    z: np.lib.npyio.NpzFile, *, n_traces: int, dt_sec_from_segy: float
) -> np.ndarray:
    keys = set(z.files)

    if 'trend_center_i' in keys:
        c = np.asarray(z['trend_center_i'], dtype=np.float32)
    else:
        if 'trend_t_sec' in keys:
            t = np.asarray(z['trend_t_sec'], dtype=np.float32)
        elif 't_trend_sec' in keys:
            t = np.asarray(z['t_trend_sec'], dtype=np.float32)
        elif 'trend_center_sec' in keys:
            t = np.asarray(z['trend_center_sec'], dtype=np.float32)
        else:
            msg = (
                "npz needs one of: 'trend_center_i', 'trend_t_sec', 't_trend_sec', 'trend_center_sec'. "
                f'available={sorted(keys)}'
            )
            raise KeyError(msg)

        if t.ndim == 2 and t.shape[0] == 1:
            t = t[0]
        if t.ndim != 1:
            msg = f'trend time must be 1D (n_traces,), got {t.shape}'
            raise ValueError(msg)

        if 'dt_sec' in keys:
            dt = float(np.asarray(z['dt_sec']).item())
        else:
            dt = float(dt_sec_from_segy)
        if dt <= 0.0:
            msg = f'dt_sec must be positive, got {dt}'
            raise ValueError(msg)

        c = t / dt

    if c.ndim == 2 and c.shape[0] == 1:
        c = c[0]

    if c.ndim != 1 or c.shape[0] != n_traces:
        msg = f'trend center must be (n_traces,), got {c.shape}, n_traces={n_traces}'
        raise ValueError(msg)

    return c.astype(np.float32, copy=False)


def _fill_trend_center_i(trend_center_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c0 = np.asarray(trend_center_i, dtype=np.float32)
    if c0.ndim != 1:
        msg = f'trend_center_i must be 1D, got {c0.shape}'
        raise ValueError(msg)

    valid = np.isfinite(c0) & (c0 > 0.0)
    idx_valid = np.flatnonzero(valid)
    if idx_valid.size == 0:
        filled = c0.copy()
        filled_mask = np.zeros_like(valid, dtype=bool)
        return filled, filled_mask

    filled = c0.copy()
    inv_idx = np.flatnonzero(~valid)
    m = int(idx_valid.size)

    for i in inv_idx:
        pos = int(np.searchsorted(idx_valid, int(i)))
        if pos <= 0:
            j = int(idx_valid[0])
        elif pos >= m:
            j = int(idx_valid[m - 1])
        else:
            j0 = int(idx_valid[pos - 1])
            j1 = int(idx_valid[pos])
            j = j0 if (i - j0) <= (j1 - i) else j1
        filled[i] = filled[j]

    filled_mask = ~valid
    return filled.astype(np.float32, copy=False), filled_mask.astype(bool, copy=False)


def _fill_trend_center_i_by_ffid(
    trend_center_i_raw: np.ndarray,
    ffid_values: np.ndarray,
    *,
    trend_offset_signed_proxy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(trend_center_i_raw, dtype=np.float32)
    ff = np.asarray(ffid_values, dtype=np.int64)
    if c.ndim != 1 or ff.ndim != 1:
        msg = f'trend_center_i_raw/ffid_values must be 1D, got {c.shape}, {ff.shape}'
        raise ValueError(msg)
    if c.shape[0] != ff.shape[0]:
        msg = f'trend/ffid length mismatch: {c.shape[0]} vs {ff.shape[0]}'
        raise ValueError(msg)

    proxy = None
    if trend_offset_signed_proxy is not None:
        proxy = np.asarray(trend_offset_signed_proxy, dtype=np.float32)
        if proxy.ndim != 1:
            msg = f'trend_offset_signed_proxy must be 1D, got {proxy.shape}'
            raise ValueError(msg)
        if proxy.shape[0] != c.shape[0]:
            msg = f'trend/proxy length mismatch: {c.shape[0]} vs {proxy.shape[0]}'
            raise ValueError(msg)

    _, _, groups = build_groups_by_key(ff)
    out = c.copy()
    filled_mask = ~(np.isfinite(c) & (c > 0.0))
    for idx in groups:
        c_g = c[idx]
        if proxy is None:
            filled_g, _mask_g = _fill_trend_center_i(c_g)
            out[idx] = filled_g
            continue

        p_g = proxy[idx]
        finite_proxy = np.isfinite(p_g)
        side_masks = (
            finite_proxy & (p_g < 0.0),
            finite_proxy & (p_g > 0.0),
            finite_proxy & (p_g == 0.0),
            ~finite_proxy,
        )
        filled_g = c_g.copy()
        for side_mask in side_masks:
            sub_idx = np.flatnonzero(side_mask)
            if sub_idx.size == 0:
                continue
            filled_sub, _mask_sub = _fill_trend_center_i(c_g[sub_idx])
            filled_g[sub_idx] = filled_sub
        out[idx] = filled_g
    return out.astype(np.float32, copy=False), filled_mask.astype(bool, copy=False)


def _build_bin_median_map(
    bin_ids: np.ndarray, center_i: np.ndarray
) -> dict[int, float]:
    b = np.asarray(bin_ids, dtype=np.int64)
    c = np.asarray(center_i, dtype=np.float32)
    if b.shape != c.shape:
        msg = f'bin/center shape mismatch: {b.shape} vs {c.shape}'
        raise ValueError(msg)
    if b.ndim != 1:
        msg = f'bin_ids must be 1D, got {b.shape}'
        raise ValueError(msg)
    out: dict[int, float] = {}
    if b.size == 0:
        return out
    uniq = np.unique(b)
    for u in uniq:
        vals = c[b == u]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            out[int(u)] = float(np.median(vals))
    return out


def _is_side_physically_ok(
    bin_map: dict[int, float], *, side: str, max_nonpos_frac: float
) -> bool:
    items: list[tuple[int, float]] = []
    for b, c in bin_map.items():
        bi = int(b)
        cv = float(c)
        if not np.isfinite(cv):
            continue
        if side == 'left' and bi < 0:
            items.append((abs(bi), cv))
        if side == 'right' and bi > 0:
            items.append((abs(bi), cv))
    if len(items) < 2:
        return True
    items.sort(key=lambda t: t[0])
    centers = np.asarray([x[1] for x in items], dtype=np.float64)
    dc = np.diff(centers)
    if dc.size == 0:
        return True
    frac = float(np.count_nonzero(dc <= 0.0)) / float(dc.size)
    return frac <= float(max_nonpos_frac)


def _is_physically_ok_bin_map(bin_map: dict[int, float]) -> bool:
    frac = float(SEMI_PHYS_MAX_NONPOS_FRAC)
    return _is_side_physically_ok(bin_map, side='left', max_nonpos_frac=frac) and (
        _is_side_physically_ok(bin_map, side='right', max_nonpos_frac=frac)
    )


def _count_finite_bins(bin_map: dict[int, float]) -> int:
    return int(sum(np.isfinite(float(v)) for v in bin_map.values()))


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
    for k in SCORE_KEYS:
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


def _fill_missing_trend_center_i_nn_then_global(
    *,
    trend_center_i_used: np.ndarray,
    trend_center_i_global: np.ndarray,
    pick_final_i: np.ndarray,
    offset_abs_m: np.ndarray,
    trend_offset_signed_proxy: np.ndarray,
    ffid_groups: list[np.ndarray],
    n_samples_in: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.asarray(trend_center_i_used, dtype=np.float32)
    cg = np.asarray(trend_center_i_global, dtype=np.float32)
    p = np.asarray(pick_final_i, dtype=np.int64)
    off = np.asarray(offset_abs_m, dtype=np.float64)
    proxy = np.asarray(trend_offset_signed_proxy, dtype=np.float32)

    if (
        c.shape != cg.shape
        or c.shape != p.shape
        or c.shape != off.shape
        or c.shape != proxy.shape
    ):
        msg = (
            'shape mismatch in fill: '
            f'c={c.shape} cg={cg.shape} p={p.shape} off={off.shape} proxy={proxy.shape}'
        )
        raise ValueError(msg)

    missing = (~np.isfinite(c)) | (c <= 0.0)
    out = c.copy()
    nn_mask = np.zeros(c.shape[0], dtype=bool)

    pick_ok = valid_pick_mask(p, n_samples=int(n_samples_in))

    for idx in ffid_groups:
        idx = np.asarray(idx, dtype=np.int64)
        miss_g = idx[missing[idx]]
        if miss_g.size == 0:
            continue

        cand_g = idx[~missing[idx]]
        if cand_g.size == 0:
            continue

        cand_off = off[cand_g]
        cand_proxy = proxy[cand_g]

        for tr in miss_g:
            if not bool(pick_ok[int(tr)]):
                continue

            pv = float(proxy[int(tr)])
            cand_use = cand_g
            if np.isfinite(pv) and float(pv) != 0.0:
                if pv < 0.0:
                    same = cand_g[np.isfinite(cand_proxy) & (cand_proxy < 0.0)]
                else:
                    same = cand_g[np.isfinite(cand_proxy) & (cand_proxy > 0.0)]
                if same.size > 0:
                    cand_use = same

            d = np.abs(off[cand_use] - float(off[int(tr)]))
            j = int(np.argmin(d))
            tr_nn = int(cand_use[j])
            c_nn = float(out[tr_nn])
            if np.isfinite(c_nn) and float(abs(float(p[int(tr)]) - c_nn)) < float(
                HALF_WIN
            ):
                out[int(tr)] = np.float32(c_nn)
                nn_mask[int(tr)] = True

    missing2 = ((~np.isfinite(out)) | (out <= 0.0)) & (~nn_mask)
    global_ok = np.isfinite(cg) & (cg > 0.0)
    global_mask = missing2 & global_ok
    out[global_mask] = cg[global_mask]

    return (
        out.astype(np.float32, copy=False),
        nn_mask.astype(bool, copy=False),
        global_mask.astype(bool, copy=False),
    )


def _build_trend_with_optional_semi(
    *,
    trend_center_i_local: np.ndarray,
    trend_offset_signed_proxy: np.ndarray,
    ffid_groups: list[np.ndarray],
    shot_x_ffid: np.ndarray,
    shot_y_ffid: np.ndarray,
    trend_filled_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    local = np.asarray(trend_center_i_local, dtype=np.float32)
    off_proxy = np.asarray(trend_offset_signed_proxy, dtype=np.float32)
    if local.shape != off_proxy.shape or local.ndim != 1:
        msg = f'local/proxy must be 1D and same shape, got {local.shape}, {off_proxy.shape}'
        raise ValueError(msg)

    n_traces = int(local.shape[0])
    filled = np.asarray(trend_filled_mask, dtype=bool)
    if filled.shape != (n_traces,):
        msg = f'trend_filled_mask must be (n_traces,), got {filled.shape}, n_traces={n_traces}'
        raise ValueError(msg)

    bin_valid = np.isfinite(off_proxy)
    bin_id = np.zeros(n_traces, dtype=np.int64)
    bin_id[bin_valid] = np.rint(off_proxy[bin_valid] / float(SEMI_BIN_W_M)).astype(
        np.int64, copy=False
    )

    local_bin_maps: list[dict[int, float]] = []
    for idx in ffid_groups:
        # raw(0/NaN) を local で埋めた点は semi 構築に混ぜない
        valid = (
            bin_valid[idx]
            & (~filled[idx])
            & np.isfinite(local[idx])
            & (local[idx] > 0.0)
        )
        idx_use = idx[valid]
        if idx_use.size == 0:
            local_bin_maps.append({})
            continue
        local_bin_maps.append(_build_bin_median_map(bin_id[idx_use], local[idx_use]))

    neighbors = _build_neighbor_indices(
        shot_x_ffid,
        shot_y_ffid,
        k=int(SEMI_NEI_K),
    )
    semi_bin_maps: list[dict[int, float]] = []
    for g, idx in enumerate(ffid_groups):
        nei = neighbors[g]
        if nei.size < int(SEMI_MIN_NEI):
            semi_bin_maps.append({})
            continue
        idx_bin = idx[bin_valid[idx]]
        if idx_bin.size > 0:
            bins = [int(x) for x in np.unique(bin_id[idx_bin]).tolist()]
        else:
            bins = []
        if len(bins) == 0:
            semi_bin_maps.append({})
            continue
        semi_map: dict[int, float] = {}
        for b in bins:
            vals: list[float] = []
            for n in nei:
                v = local_bin_maps[int(n)].get(int(b), np.nan)
                if np.isfinite(v):
                    vals.append(float(v))
            if vals:
                semi_map[int(b)] = float(np.median(np.asarray(vals, dtype=np.float64)))
            else:
                semi_map[int(b)] = float('nan')
        semi_bin_maps.append(semi_map)

    semi_used_ffid_mask = np.zeros(len(ffid_groups), dtype=bool)
    semi_ok_for_missing_ffid_mask = np.zeros(len(ffid_groups), dtype=bool)
    for g in range(len(ffid_groups)):
        local_phys_ng = not _is_physically_ok_bin_map(local_bin_maps[g])
        semi_map = semi_bin_maps[g]
        semi_has_2 = _count_finite_bins(semi_map) >= 2
        semi_phys_ok_2 = semi_has_2 and _is_physically_ok_bin_map(semi_map)
        semi_used_ffid_mask[g] = bool(local_phys_ng and semi_phys_ok_2)

        # raw欠損トレース向け: 1binでも semi が取れれば使う（物理チェックは関数側で安全）
        semi_has_1 = _count_finite_bins(semi_map) >= 1
        semi_ok_for_missing_ffid_mask[g] = bool(
            semi_has_1 and _is_physically_ok_bin_map(semi_map)
        )

    trend_center_i_semi = np.full(n_traces, np.nan, dtype=np.float32)
    trend_center_i_used = local.astype(np.float32, copy=True)
    fallback_count = 0

    for g, idx in enumerate(ffid_groups):
        use_semi_for_group = bool(semi_used_ffid_mask[g])
        use_semi_for_missing = bool(np.any(filled[idx])) and bool(
            semi_ok_for_missing_ffid_mask[g]
        )
        if not (use_semi_for_group or use_semi_for_missing):
            continue

        semi_map = semi_bin_maps[g]
        bins_g = bin_id[idx]
        bins_valid_g = bin_valid[idx]
        for j, tr_idx in enumerate(idx):
            tr = int(tr_idx)
            # グループ置換ではない場合、raw欠損トレースだけ semi を試す
            if (not use_semi_for_group) and (not bool(filled[tr])):
                continue

            v = float('nan')
            if bool(bins_valid_g[j]):
                v = semi_map.get(int(bins_g[j]), float('nan'))
            if np.isfinite(v):
                trend_center_i_semi[tr] = np.float32(v)
                trend_center_i_used[tr] = np.float32(v)
            else:
                fallback_count += 1

    return (
        trend_center_i_semi.astype(np.float32, copy=False),
        trend_center_i_used.astype(np.float32, copy=False),
        semi_used_ffid_mask.astype(bool, copy=False),
        int(fallback_count),
    )


def _build_semi_global_trend_center_i_from_picks(
    *,
    pick_final_i: np.ndarray,
    n_samples_in: int,
    trend_offset_signed_proxy: np.ndarray,
    ffid_groups: list[np.ndarray],
    shot_x_ffid: np.ndarray,
    shot_y_ffid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build semi-global trend center (sample index) from picks.

    - Build per-FFID bin-median maps from valid pick_final_i.
    - For each FFID and bin, take median of neighbor-FFID bin medians.
    - Assign semi-global value to each trace based on its bin_id.
    - Return (trend_center_i_semi, semi_used_ffid_mask, semi_missing_count).

    Notes:
        - This path does NOT use infer-provided trend arrays.
        - Traces that cannot be assigned by semi (missing proxy/bin or missing neighbor bin)
          remain NaN and are expected to be filled by global trend.

    """
    p = np.asarray(pick_final_i, dtype=np.float32)
    off_proxy = np.asarray(trend_offset_signed_proxy, dtype=np.float32)
    if p.ndim != 1 or off_proxy.ndim != 1 or p.shape != off_proxy.shape:
        msg = (
            'pick_final_i/proxy must be 1D and same shape, '
            f'got {p.shape}, {off_proxy.shape}'
        )
        raise ValueError(msg)

    n_traces = int(p.shape[0])
    if n_samples_in < 2:
        msg = f'n_samples_in must be >= 2, got {n_samples_in}'
        raise ValueError(msg)

    pick_valid = np.isfinite(p) & (p > 0.0) & (p < float(n_samples_in))
    bin_valid = np.isfinite(off_proxy)

    bin_id = np.zeros(n_traces, dtype=np.int64)
    bin_id[bin_valid] = np.rint(off_proxy[bin_valid] / float(SEMI_BIN_W_M)).astype(
        np.int64, copy=False
    )

    local_bin_maps: list[dict[int, float]] = []
    for idx in ffid_groups:
        valid = bin_valid[idx] & pick_valid[idx]
        idx_use = idx[valid]
        if idx_use.size == 0:
            local_bin_maps.append({})
            continue
        local_bin_maps.append(_build_bin_median_map(bin_id[idx_use], p[idx_use]))

    neighbors = _build_neighbor_indices(
        shot_x_ffid,
        shot_y_ffid,
        k=int(SEMI_NEI_K),
    )

    semi_bin_maps: list[dict[int, float]] = []
    for g, idx in enumerate(ffid_groups):
        nei = neighbors[g]
        if nei.size < int(SEMI_MIN_NEI):
            semi_bin_maps.append({})
            continue
        idx_bin = idx[bin_valid[idx]]
        if idx_bin.size > 0:
            bins = [int(x) for x in np.unique(bin_id[idx_bin]).tolist()]
        else:
            bins = []
        if len(bins) == 0:
            semi_bin_maps.append({})
            continue

        semi_map: dict[int, float] = {}
        for b in bins:
            vals: list[float] = []
            for n in nei:
                v = local_bin_maps[int(n)].get(int(b), np.nan)
                if np.isfinite(v):
                    vals.append(float(v))
            if vals:
                semi_map[int(b)] = float(np.median(np.asarray(vals, dtype=np.float64)))
            else:
                semi_map[int(b)] = float('nan')
        semi_bin_maps.append(semi_map)

    semi_used_ffid_mask = np.zeros(len(ffid_groups), dtype=bool)
    for g in range(len(ffid_groups)):
        semi_map = semi_bin_maps[g]
        semi_has_1 = _count_finite_bins(semi_map) >= 1
        semi_used_ffid_mask[g] = bool(
            semi_has_1 and _is_physically_ok_bin_map(semi_map)
        )

    trend_center_i_semi = np.full(n_traces, np.nan, dtype=np.float32)
    missing_count = 0
    for g, idx in enumerate(ffid_groups):
        if not bool(semi_used_ffid_mask[g]):
            missing_count += int(idx.size)
            continue

        semi_map = semi_bin_maps[g]
        bins_g = bin_id[idx]
        bins_valid_g = bin_valid[idx]
        for j, tr_idx in enumerate(idx):
            tr = int(tr_idx)
            v = float('nan')
            if bool(bins_valid_g[j]):
                v = semi_map.get(int(bins_g[j]), float('nan'))
            if np.isfinite(v) and v > 0.0:
                trend_center_i_semi[tr] = np.float32(v)
            else:
                missing_count += 1

    return (
        trend_center_i_semi.astype(np.float32, copy=False),
        semi_used_ffid_mask.astype(bool, copy=False),
        int(missing_count),
    )


def _build_trend_result(
    *,
    src: segyio.SegyFile,
    z: np.lib.npyio.NpzFile,
    n_traces: int,
    n_samples_in: int,
    dt_sec_in: float,
    pick_final_i: np.ndarray,
    scores: dict[str, np.ndarray],
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

    # NOTE: This pipeline variant does not use infer-provided per-trace trend arrays.
    # Trend centers used for window extraction are built as:
    #   semi-global (from neighbor FFIDs, based on pick_final) -> global fallback (offset-time fit)
    trend_center_i_raw = np.full(n_traces, np.nan, dtype=np.float32)
    trend_center_i_local = np.full(n_traces, np.nan, dtype=np.float32)

    offset_abs_m = _load_offset_abs_from_segy(src)
    if offset_abs_m.shape != (n_traces,):
        msg = f'offset_abs_m must be (n_traces,), got {offset_abs_m.shape}, n_traces={n_traces}'
        raise ValueError(msg)

    if not bool(SEMI_GLOBAL_ENABLE):
        msg = 'SEMI_GLOBAL_ENABLE must be True for semi-global -> global trend build'
        raise ValueError(msg)

    trend_offset_signed_proxy = require_npz_key(z, 'trend_offset_signed_proxy').astype(
        np.float32, copy=False
    )
    if trend_offset_signed_proxy.ndim == 2 and trend_offset_signed_proxy.shape[0] == 1:
        trend_offset_signed_proxy = trend_offset_signed_proxy[0]
    if (
        trend_offset_signed_proxy.ndim != 1
        or trend_offset_signed_proxy.shape[0] != n_traces
    ):
        msg = (
            'trend_offset_signed_proxy must be (n_traces,), '
            f'got {trend_offset_signed_proxy.shape}, n_traces={n_traces}'
        )
        raise ValueError(msg)

    (
        trend_center_i_semi,
        semi_used_ffid_mask,
        semi_fallback_count,
    ) = _build_semi_global_trend_center_i_from_picks(
        pick_final_i=pick_final_i,
        n_samples_in=int(n_samples_in),
        trend_offset_signed_proxy=trend_offset_signed_proxy,
        ffid_groups=ffid_groups,
        shot_x_ffid=shot_x_ffid,
        shot_y_ffid=shot_y_ffid,
    )

    trend_center_i_used = trend_center_i_semi.astype(np.float32, copy=True)
    trend_filled_mask = (
        (~np.isfinite(trend_center_i_used)) | (trend_center_i_used <= 0.0)
    ).astype(bool, copy=False)

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

    missing = (~np.isfinite(trend_center_i_used)) | (trend_center_i_used <= 0.0)
    global_ok = np.isfinite(trend_center_i_global) & (trend_center_i_global > 0.0)
    global_replaced_mask = missing & global_ok
    trend_center_i_used[global_replaced_mask] = trend_center_i_global[
        global_replaced_mask
    ]

    nn_replaced_mask = np.zeros(n_traces, dtype=bool)

    return _TrendBuildResult(
        trend_center_i_raw=trend_center_i_raw.astype(np.float32, copy=False),
        trend_center_i_local=trend_center_i_local.astype(np.float32, copy=False),
        trend_center_i_semi=trend_center_i_semi.astype(np.float32, copy=False),
        trend_center_i_used=trend_center_i_used.astype(np.float32, copy=False),
        trend_center_i_global=trend_center_i_global.astype(np.float32, copy=False),
        nn_replaced_mask=nn_replaced_mask.astype(bool, copy=False),
        global_replaced_mask=global_replaced_mask.astype(bool, copy=False),
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

    c_round = np.rint(c).astype(np.int64, copy=False)
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

    for k in SCORE_KEYS:
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

            scores: dict[str, np.ndarray] = {}
            for k in SCORE_KEYS:
                scores[k] = require_npz_key(z, k).astype(np.float32, copy=False)

            trend_res = _build_trend_result(
                src=src,
                z=z,
                n_traces=n_traces,
                n_samples_in=ns_in,
                dt_sec_in=dt_sec_in,
                pick_final_i=pick_final,
                scores=scores,
            )

            return (
                n_traces,
                ns_in,
                dt_sec_in,
                pick_final,
                scores,
                trend_res,
            )


def compute_global_thresholds(segys: list[Path]) -> dict[str, float]:
    vals: dict[str, list[np.ndarray]] = {k: [] for k in SCORE_KEYS}

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

        for k in SCORE_KEYS:
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
    for k in SCORE_KEYS:
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

            scores: dict[str, np.ndarray] = {}
            for k in SCORE_KEYS:
                scores[k] = require_npz_key(z, k).astype(np.float32, copy=False)

            trend_res = _build_trend_result(
                src=src,
                z=z,
                n_traces=n_traces,
                n_samples_in=ns_in,
                dt_sec_in=dt_sec_in,
                pick_final_i=pick_final,
                scores=scores,
            )
            trend_center_i_raw = trend_res.trend_center_i_raw
            trend_center_i_local = trend_res.trend_center_i_local
            trend_center_i_semi = trend_res.trend_center_i_semi
            trend_center_i_used = trend_res.trend_center_i_used
            trend_filled_mask = trend_res.trend_filled_mask
            ffid_values = trend_res.ffid_values
            ffid_unique_values = trend_res.ffid_unique_values
            shot_x_ffid = trend_res.shot_x_ffid
            shot_y_ffid = trend_res.shot_y_ffid
            semi_used_ffid_mask = trend_res.semi_used_ffid_mask
            semi_fallback_count = int(trend_res.semi_fallback_count)

            nn_replaced_mask = trend_res.nn_replaced_mask
            global_replaced_mask = trend_res.global_replaced_mask
            trend_center_i_global = trend_res.trend_center_i_global
            global_edges_all = trend_res.global_edges_all
            global_coef_all = trend_res.global_coef_all
            global_edges_left = trend_res.global_edges_left
            global_coef_left = trend_res.global_coef_left
            global_edges_right = trend_res.global_edges_right
            global_coef_right = trend_res.global_coef_right

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
            scores=scores,
            thresholds=thresholds_arg,
        )

        c_round = np.rint(trend_center_i_used).astype(np.int64, copy=False)
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
        semi_bin_w_m=np.float32(SEMI_BIN_W_M),
        semi_nei_k=np.int32(SEMI_NEI_K),
        semi_min_nei=np.int32(SEMI_MIN_NEI),
        trend_center_i_raw=trend_center_i_raw.astype(np.float32, copy=False),
        trend_center_i_local=trend_center_i_local.astype(np.float32, copy=False),
        trend_center_i_semi=trend_center_i_semi.astype(np.float32, copy=False),
        trend_center_i_used=trend_center_i_used.astype(np.float32, copy=False),
        trend_center_i_global=trend_center_i_global.astype(np.float32, copy=False),
        nn_replaced_mask=nn_replaced_mask.astype(bool, copy=False),
        global_replaced_mask=global_replaced_mask.astype(bool, copy=False),
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
        conf_prob1=scores['conf_prob1'].astype(np.float32, copy=False),
        conf_trend1=scores['conf_trend1'].astype(np.float32, copy=False),
        conf_rs1=scores['conf_rs1'].astype(np.float32, copy=False),
    )

    n_keep = int(np.count_nonzero(keep_mask))
    n_fill = int(np.count_nonzero(trend_filled_mask))
    n_nn = int(np.count_nonzero(nn_replaced_mask))
    n_gl = int(np.count_nonzero(global_replaced_mask))
    tag = 'global' if THRESH_MODE == 'global' else 'per_segy'
    print(
        f'[OK] {segy_path.name} -> {out_segy.name}  keep={n_keep}/{n_traces} '
        f'filled_trend={n_fill}/{n_traces} '
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
