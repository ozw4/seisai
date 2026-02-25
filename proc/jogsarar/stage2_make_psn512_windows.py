# %%
#!/usr/bin/env python3

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
from seisai_pick.trend.trend_fit_strategy import TwoPieceIRLSAutoBreakStrategy

# =========================
# CONFIG（ここだけ触ればOK）
# =========================
@dataclass(frozen=True)
class Stage2Cfg:
    in_segy_root: Path = Path('/home/dcuser/data/ActiveSeisField/jogsarar')
    in_infer_root: Path = Path('/home/dcuser/data/ActiveSeisField/jogsarar_out')
    out_segy_root: Path = Path('/home/dcuser/data/ActiveSeisField/jogsarar_psn512')
    segy_exts: tuple[str, ...] = ('.sgy', '.segy')
    half_win: int = 128  # ± samples
    up_factor: int = 2
    drop_low_frac: float = 0.05  # 下位除外
    score_keys_for_weight: tuple[str, ...] = (
        'conf_prob1',
        'conf_rs1',
    )
    score_keys_for_filter: tuple[str, ...] = (
        'conf_prob1',
        'conf_trend1',
        'conf_rs1',
    )
    pick_key: str = 'pick_final'  # infer npz側の最終pick（p1）
    # 閾値モード
    #   'per_segy' : SEGYごとにDrop_low_frac
    #   'global'   : 全SEGYまとめてDrop_low_frac（全データグローバル）
    thresh_mode: str = 'global'  # 'global' or 'per_segy'
    emit_training_artifacts: bool = True
    # global trendline (proxy sign split)
    global_vmin_m_s: float = 300.0
    global_vmax_m_s: float = 6000.0
    global_slope_eps: float = 1e-6
    global_side_min_pts: int = 16
    # Stage1 local trendline baseline (from stage1 .prob.npz)
    use_stage1_local_trendline_baseline: bool = True
    local_global_diff_th_samples: int = 128
    local_discard_radius_traces: int = 32
    local_trend_t_sec_key: str = 'trend_t_sec'
    local_trend_covered_key: str = 'trend_covered'
    local_trend_dt_sec_key: str = 'dt_sec'
    local_inv_drop_th_samples: float = 10.0
    local_inv_min_consec_steps: int = 2
    # conf_trend1: Stage1互換
    conf_trend_sigma_ms: float = 6.0
    conf_trend_var_half_win_traces: int = 8
    conf_trend_var_sigma_std_ms: float = 6.0
    conf_trend_var_min_count: int = 3

    @property
    def out_ns(self) -> int:
        return int(2 * int(self.half_win) * int(self.up_factor))


DEFAULT_STAGE2_CFG = Stage2Cfg()


# =========================
# Utility
# =========================
@dataclass(frozen=True)
class _TrendBuildResult:
    trend_center_i_raw: np.ndarray
    trend_center_i_local: np.ndarray
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
    conf_trend1: np.ndarray


def infer_npz_path_for_segy(
    segy_path: Path, *, cfg: Stage2Cfg = DEFAULT_STAGE2_CFG
) -> Path:
    rel = segy_path.relative_to(cfg.in_segy_root)
    return cfg.in_infer_root / rel.parent / f'{segy_path.stem}.prob.npz'


def out_segy_path_for_in(
    segy_path: Path, *, cfg: Stage2Cfg = DEFAULT_STAGE2_CFG
) -> Path:
    rel = segy_path.relative_to(cfg.in_segy_root)
    out_rel = rel.with_suffix('')
    return cfg.out_segy_root / out_rel.parent / f'{out_rel.name}.win512.sgy'


def out_sidecar_npz_path_for_out(
    out_segy_path: Path, *, cfg: Stage2Cfg = DEFAULT_STAGE2_CFG
) -> Path:
    del cfg
    return out_segy_path.with_suffix('.sidecar.npz')


def out_pick_csr_npz_path_for_out(
    out_segy_path: Path, *, cfg: Stage2Cfg = DEFAULT_STAGE2_CFG
) -> Path:
    del cfg
    # PSN dataset (load_phase_pick_csr_npz) が読む用
    return out_segy_path.with_suffix('.phase_pick.csr.npz')


def _load_stage1_local_trend_center_i(
    *,
    z: np.lib.npyio.NpzFile,
    n_traces: int,
    dt_sec_in: float,
    cfg: Stage2Cfg = DEFAULT_STAGE2_CFG,
) -> tuple[np.ndarray, np.ndarray]:
    if not bool(cfg.use_stage1_local_trendline_baseline):
        center_i = np.full(int(n_traces), np.nan, dtype=np.float32)
        ok = np.zeros(int(n_traces), dtype=bool)
        return center_i, ok

    dt_npz = float(require_npz_key(z, cfg.local_trend_dt_sec_key))
    dt_in = float(dt_sec_in)
    if (not np.isfinite(dt_npz)) or dt_npz <= 0.0:
        msg = f'invalid {cfg.local_trend_dt_sec_key} in infer npz: {dt_npz}'
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

    t_sec = require_npz_key(z, cfg.local_trend_t_sec_key).astype(np.float32, copy=False)
    if t_sec.ndim != 1 or t_sec.shape[0] != int(n_traces):
        msg = (
            f'{cfg.local_trend_t_sec_key} must be (n_traces,), got {t_sec.shape}, '
            f'n_traces={n_traces}'
        )
        raise ValueError(msg)

    covered = require_npz_key(z, cfg.local_trend_covered_key).astype(bool, copy=False)
    if covered.ndim != 1 or covered.shape[0] != int(n_traces):
        msg = (
            f'{cfg.local_trend_covered_key} must be (n_traces,), got {covered.shape}, '
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


def _local_velocity_inversion_mask_by_split(
    *,
    trend_center_i_local: np.ndarray,
    local_ok: np.ndarray,
    trend_offset_signed_proxy: np.ndarray,
    ffid_groups: list[np.ndarray],
    inv_drop_th_samples: float,
    inv_min_consec_steps: int,
) -> np.ndarray:
    """Return mask for traces belonging to split sides that contain any inversion.

    For each FFID group:
      - split traces by the sign of trend_offset_signed_proxy (<0 / >0)
      - within each side, consider only VALID traces:
            local_ok == True AND isfinite(trend_center_i_local) AND trend_center_i_local > 0
      - sort VALID traces by increasing |proxy|
      - inversion step:
            ord_local[k+1] < ord_local[k] - inv_drop_th_samples
      - if there exists a run of inversion steps of length >= inv_min_consec_steps,
        mark ONLY those VALID traces in that side as True

    Returns:
        inversion_mask (bool, shape=(n_traces,))

    """
    local = np.asarray(trend_center_i_local, dtype=np.float32)
    ok = np.asarray(local_ok, dtype=bool)
    proxy = np.asarray(trend_offset_signed_proxy, dtype=np.float32)

    if local.ndim != 1 or ok.ndim != 1 or proxy.ndim != 1:
        msg = f'local/ok/proxy must be 1D, got {local.shape}, {ok.shape}, {proxy.shape}'
        raise ValueError(msg)
    if local.shape != ok.shape or local.shape != proxy.shape:
        msg = f'shape mismatch local={local.shape}, ok={ok.shape}, proxy={proxy.shape}'
        raise ValueError(msg)

    drop_th = float(inv_drop_th_samples)
    min_steps = int(inv_min_consec_steps)
    if min_steps < 1:
        raise ValueError(f'inv_min_consec_steps must be >= 1, got {min_steps}')

    out = np.zeros(local.shape[0], dtype=bool)

    for idx in ffid_groups:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size < 2:
            continue

        g_proxy = proxy[idx]
        g_proxy_finite = np.isfinite(g_proxy)

        for side_sel in (
            (g_proxy < 0.0) & g_proxy_finite,
            (g_proxy > 0.0) & g_proxy_finite,
        ):
            side_idx = idx[side_sel]
            if side_idx.size < 2:
                continue

            valid = (
                ok[side_idx] & np.isfinite(local[side_idx]) & (local[side_idx] > 0.0)
            )

            need_pts = (min_steps + 1) if min_steps > 1 else 2
            if int(np.count_nonzero(valid)) < int(need_pts):
                continue

            valid_idx = side_idx[valid]

            x = np.abs(proxy[valid_idx]).astype(np.float32, copy=False)
            order = np.argsort(x, kind='mergesort')
            ord_local = local[valid_idx[order]]

            inv_step = ord_local[1:] < (ord_local[:-1] - drop_th)

            if min_steps == 1:
                has_inv = bool(np.any(inv_step))
            else:
                run = 0
                has_inv = False
                for b in inv_step:
                    if bool(b):
                        run += 1
                        if run >= min_steps:
                            has_inv = True
                            break
                    else:
                        run = 0

            if has_inv:
                out[valid_idx] = True

    return out.astype(bool, copy=False)


def _fill_trend_center_i_by_linear_interp(
    *,
    trend_center_i_local: np.ndarray,
    fill_mask: np.ndarray,
    trend_center_i_global: np.ndarray,
    trend_offset_signed_proxy: np.ndarray,
    ffid_groups: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Fill selected traces by linear interpolation in |proxy| order, per split side.

    For each FFID group and each split side (proxy<0 / proxy>0), sort by increasing
    |proxy| and replace contiguous fill segments by linear interpolation between
    immediate outer traces (L-1 and R+1 in that sorted order). If either boundary
    trace is missing, or if the physical constraint is violated (pick decreases as
    |offset| increases), fall back to trend_center_i_global for that segment.

    Returns:
        trend_center_i_used, used_global_mask

    """
    local = np.asarray(trend_center_i_local, dtype=np.float32)
    fm = np.asarray(fill_mask, dtype=bool)
    glob = np.asarray(trend_center_i_global, dtype=np.float32)
    proxy = np.asarray(trend_offset_signed_proxy, dtype=np.float32)

    if local.ndim != 1 or fm.ndim != 1 or glob.ndim != 1 or proxy.ndim != 1:
        msg = (
            'local/fill_mask/global/proxy must be 1D, got '
            f'local={local.shape}, fill_mask={fm.shape}, global={glob.shape}, proxy={proxy.shape}'
        )
        raise ValueError(msg)
    if (
        local.shape != fm.shape
        or local.shape != glob.shape
        or local.shape != proxy.shape
    ):
        msg = (
            f'shape mismatch local={local.shape}, fill_mask={fm.shape}, '
            f'global={glob.shape}, proxy={proxy.shape}'
        )
        raise ValueError(msg)

    out = local.copy()
    used_global = np.zeros(local.shape[0], dtype=bool)

    n_traces = int(local.shape[0])
    if n_traces == 0 or not bool(np.any(fm)):
        return out.astype(np.float32, copy=False), used_global.astype(bool, copy=False)

    for idx in ffid_groups:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            continue

        g_fill = fm[idx]
        if not bool(np.any(g_fill)):
            continue

        g_proxy = proxy[idx]
        g_proxy_finite = np.isfinite(g_proxy)

        # If proxy is missing, we cannot order by offset. Fill such traces by global.
        bad_proxy = g_fill & (~g_proxy_finite)
        if bool(np.any(bad_proxy)):
            seg_idx = idx[bad_proxy]
            out[seg_idx] = glob[seg_idx]
            used_global[seg_idx] = True

        for side_sel in (
            (g_proxy < 0.0) & g_proxy_finite,
            (g_proxy > 0.0) & g_proxy_finite,
        ):
            side_idx = idx[side_sel]
            if side_idx.size == 0:
                continue

            x = np.abs(proxy[side_idx]).astype(np.float32, copy=False)
            order = np.argsort(x, kind='mergesort')
            ord_idx = side_idx[order]
            ord_fill = fm[ord_idx]

            if not bool(np.any(ord_fill)):
                continue

            j = 0
            n = int(ord_idx.size)
            while j < n:
                if not bool(ord_fill[j]):
                    j += 1
                    continue

                start = j
                j += 1
                while j < n and bool(ord_fill[j]):
                    j += 1
                end = j - 1

                seg_idx = ord_idx[start : end + 1]
                has_left = start - 1 >= 0
                has_right = end + 1 < n

                if has_left and has_right:
                    left_t = int(ord_idx[start - 1])
                    right_t = int(ord_idx[end + 1])
                    left_val = float(out[left_t])
                    right_val = float(out[right_t])

                    if (not np.isfinite(left_val)) or (not np.isfinite(right_val)):
                        out[seg_idx] = glob[seg_idx]
                        used_global[seg_idx] = True
                        continue

                    # physical constraint: for increasing |offset|, pick should not decrease.
                    if left_val > right_val:
                        out[seg_idx] = glob[seg_idx]
                        used_global[seg_idx] = True
                        continue

                    denom = float((end - start) + 2)
                    for k, t_i in enumerate(seg_idx, start=1):
                        a = float(k) / denom
                        out[int(t_i)] = (1.0 - a) * left_val + a * right_val
                else:
                    out[seg_idx] = glob[seg_idx]
                    used_global[seg_idx] = True

    return out.astype(np.float32, copy=False), used_global.astype(bool, copy=False)


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
    cfg: Stage2Cfg = DEFAULT_STAGE2_CFG,
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
        sigma_ms=float(cfg.conf_trend_sigma_ms),
    )
    conf_v = trace_confidence_from_trend_resid_var(
        t_pick_sec,
        t_trend_sec,
        valid,
        half_win_traces=int(cfg.conf_trend_var_half_win_traces),
        sigma_std_ms=float(cfg.conf_trend_var_sigma_std_ms),
        min_count=int(cfg.conf_trend_var_min_count),
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
    *,
    cfg: Stage2Cfg = DEFAULT_STAGE2_CFG,
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
            a_min = 1.0 / float(cfg.global_vmax_m_s)
            a_max = 1.0 / float(cfg.global_vmin_m_s)
            a1p, b1p, a2p, b2p = _project_two_piece_coef(
                xb=xb,
                a1=a1,
                b1=b1,
                a2=a2,
                b2=b2,
                a_min=a_min,
                a_max=a_max,
                slope_eps=float(cfg.global_slope_eps),
            )
            edges = np.asarray([xmin, xb, xmax], dtype=np.float32)
            coef = np.asarray([[a1p, b1p], [a2p, b2p]], dtype=np.float32)
            return edges, coef, False

    a_min = 1.0 / float(cfg.global_vmax_m_s)
    a_max = 1.0 / float(cfg.global_vmin_m_s)
    a1p, b1p, a2p, b2p = _project_two_piece_coef(
        xb=xb,
        a1=a,
        b1=b,
        a2=a,
        b2=b,
        a_min=a_min,
        a_max=a_max,
        slope_eps=float(cfg.global_slope_eps),
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
    cfg: Stage2Cfg = DEFAULT_STAGE2_CFG,
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
    for k in cfg.score_keys_for_weight:
        s = np.asarray(scores[k], dtype=np.float64)
        if s.shape != w.shape:
            msg = f'score shape mismatch {k}: {s.shape} vs {w.shape}'
            raise ValueError(msg)
        w *= np.where(np.isfinite(s) & (s > 0.0), s, 0.0)

    pick_ok = valid_pick_mask(p, n_samples=int(n_samples_in))
    v_all = pick_ok & np.isfinite(off) & (off > 0.0) & np.isfinite(w) & (w > 0.0)
    y_sec = p.astype(np.float64) * float(dt_sec_in)

    edges_all, coef_all, deg_all = _fit_two_piece_projected(
        off[v_all], y_sec[v_all], w[v_all], cfg=cfg
    )

    left_mask = v_all & np.isfinite(proxy) & (proxy < 0.0)
    right_mask = v_all & np.isfinite(proxy) & (proxy > 0.0)

    if int(np.count_nonzero(left_mask)) >= int(cfg.global_side_min_pts):
        edges_left, coef_left, deg_left = _fit_two_piece_projected(
            off[left_mask], y_sec[left_mask], w[left_mask], cfg=cfg
        )
    else:
        edges_left, coef_left, deg_left = edges_all, coef_all, True

    if int(np.count_nonzero(right_mask)) >= int(cfg.global_side_min_pts):
        edges_right, coef_right, deg_right = _fit_two_piece_projected(
            off[right_mask], y_sec[right_mask], w[right_mask], cfg=cfg
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
    cfg: Stage2Cfg = DEFAULT_STAGE2_CFG,
) -> _TrendBuildResult:
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

    trend_offset_signed_proxy = _build_offset_signed_proxy_by_ffid(
        offset_abs_m=offset_abs_m,
        ffid_groups=ffid_groups,
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
        cfg=cfg,
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

    trend_center_i_final = trend_center_i_global_filled.astype(np.float32, copy=False)
    if int(n_samples_in) > 2:
        f = np.isfinite(trend_center_i_final)
        if bool(np.any(f)):
            trend_center_i_final = trend_center_i_final.copy()
            trend_center_i_final[f] = np.clip(
                trend_center_i_final[f],
                1.0,
                float(int(n_samples_in) - 1),
            )

    global_ok = np.isfinite(trend_center_i_global_filled) & (
        trend_center_i_global_filled > 0.0
    )

    if int(cfg.local_global_diff_th_samples) < 0:
        msg = (
            f'local_global_diff_th_samples must be >= 0, '
            f'got {cfg.local_global_diff_th_samples}'
        )
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
        & (diff >= float(int(cfg.local_global_diff_th_samples)))
    )

    inversion_mask = _local_velocity_inversion_mask_by_split(
        trend_center_i_local=trend_center_i_local,
        local_ok=local_ok,
        trend_offset_signed_proxy=trend_offset_signed_proxy,
        ffid_groups=ffid_groups,
        inv_drop_th_samples=int(cfg.local_inv_drop_th_samples),
        inv_min_consec_steps=int(cfg.local_inv_min_consec_steps),
    )

    seeds = bad & inversion_mask
    local_discard_mask = _expand_mask_within_ffid_groups(
        mask_by_trace=seeds,
        ffid_groups=ffid_groups,
        radius=int(cfg.local_discard_radius_traces),
    )

    fill_mask = (~local_ok) | local_discard_mask
    trend_center_i_used, global_replaced_mask = _fill_trend_center_i_by_linear_interp(
        trend_center_i_local=trend_center_i_local,
        fill_mask=fill_mask,
        trend_center_i_global=trend_center_i_global_filled,
        trend_offset_signed_proxy=trend_offset_signed_proxy,
        ffid_groups=ffid_groups,
    )

    if int(n_samples_in) > 2:
        f = np.isfinite(trend_center_i_used)
        if bool(np.any(f)):
            trend_center_i_used = trend_center_i_used.copy()
            trend_center_i_used[f] = np.clip(
                trend_center_i_used[f],
                1.0,
                float(int(n_samples_in) - 1),
            )

    trend_filled_mask = fill_mask.astype(bool, copy=False)

    nn_replaced_mask = np.zeros(n_traces, dtype=bool)
    conf_trend1 = _compute_conf_trend1_from_trend(
        pick_final_i=pick_final_i,
        trend_center_i=trend_center_i_used,
        n_samples_in=int(n_samples_in),
        dt_sec_in=float(dt_sec_in),
        cfg=cfg,
    )

    return _TrendBuildResult(
        trend_center_i_raw=trend_center_i_raw.astype(np.float32, copy=False),
        trend_center_i_local=trend_center_i_local.astype(np.float32, copy=False),
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


def _extract_256(
    trace: np.ndarray, *, center_i: float, cfg: Stage2Cfg = DEFAULT_STAGE2_CFG
) -> tuple[np.ndarray, int]:
    x = np.asarray(trace, dtype=np.float32)
    if x.ndim != 1:
        msg = f'trace must be 1D, got {x.shape}'
        raise ValueError(msg)

    out = np.zeros(2 * int(cfg.half_win), dtype=np.float32)

    if (not np.isfinite(center_i)) or center_i <= 0.0:
        return out, -1

    c = int(np.rint(float(center_i)))
    start = c - int(cfg.half_win)
    end = c + int(cfg.half_win)

    n = x.shape[0]
    ov_l = max(0, start)
    ov_r = min(n, end)
    if ov_l >= ov_r:
        return out, start

    dst_l = ov_l - start
    dst_r = dst_l + (ov_r - ov_l)
    out[dst_l:dst_r] = x[ov_l:ov_r]
    return out, start


def _upsample_256_to_512_linear(
    win256: np.ndarray, *, cfg: Stage2Cfg = DEFAULT_STAGE2_CFG
) -> np.ndarray:
    y = np.asarray(win256, dtype=np.float32)
    if y.shape != (2 * int(cfg.half_win),):
        msg = f'win256 must be (256,), got {y.shape}'
        raise ValueError(msg)

    xp = np.arange(2 * int(cfg.half_win), dtype=np.float32)  # 0..255
    xq = np.arange(cfg.out_ns, dtype=np.float32) / float(cfg.up_factor)
    out = np.interp(xq, xp, y, left=0.0, right=0.0).astype(np.float32, copy=False)
    if out.shape != (cfg.out_ns,):
        msg = f'unexpected upsample output shape: {out.shape}'
        raise ValueError(msg)
    return out


def _base_valid_mask(
    *,
    pick_final_i: np.ndarray,
    trend_center_i: np.ndarray,
    n_samples_in: int,
    cfg: Stage2Cfg = DEFAULT_STAGE2_CFG,
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
    # strict "< half_win" to keep pick_win_512 in (0, out_ns)
    win_ok = pick_ok & trend_ok & (np.abs(p - c_round) < int(cfg.half_win))
    reason[~win_ok] |= 1 << 2

    return win_ok, reason


def build_keep_mask(
    *,
    pick_final_i: np.ndarray,
    trend_center_i: np.ndarray,
    n_samples_in: int,
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float] | None,
    cfg: Stage2Cfg = DEFAULT_STAGE2_CFG,
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
        cfg=cfg,
    )

    bit_map = {'conf_prob1': 3, 'conf_trend1': 4, 'conf_rs1': 5}

    thresholds_used: dict[str, float] = {}
    keep = base_valid.copy()

    for k in cfg.score_keys_for_filter:
        s = np.asarray(scores[k], dtype=np.float32)
        if s.shape != (n_tr,):
            msg = f'score {k} must be (n_traces,), got {s.shape}, n_traces={n_tr}'
            raise ValueError(msg)

        if thresholds is None:
            th = _percentile_threshold(s[base_valid], frac=cfg.drop_low_frac)
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
    *,
    cfg: Stage2Cfg = DEFAULT_STAGE2_CFG,
) -> tuple[
    int,
    int,
    float,
    np.ndarray,
    dict[str, np.ndarray],
    _TrendBuildResult,
]:
    infer_npz = infer_npz_path_for_segy(segy_path, cfg=cfg)
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
            pick_final = require_npz_key(z, cfg.pick_key).astype(np.int64, copy=False)
            if pick_final.ndim != 1 or pick_final.shape[0] != n_traces:
                msg = (
                    f'{cfg.pick_key} must be (n_traces,), got {pick_final.shape}, '
                    f'n_traces={n_traces}'
                )
                raise ValueError(msg)

            scores_weight: dict[str, np.ndarray] = {}
            for k in cfg.score_keys_for_weight:
                scores_weight[k] = require_npz_key(z, k).astype(np.float32, copy=False)

            trend_center_i_local, local_trend_ok = _load_stage1_local_trend_center_i(
                z=z,
                n_traces=n_traces,
                dt_sec_in=dt_sec_in,
                cfg=cfg,
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
                cfg=cfg,
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


def compute_global_thresholds(
    segys: list[Path], *, cfg: Stage2Cfg = DEFAULT_STAGE2_CFG
) -> dict[str, float]:
    vals: dict[str, list[np.ndarray]] = {k: [] for k in cfg.score_keys_for_filter}

    n_files_used = 0
    n_base_total = 0

    for p in segys:
        infer_npz = infer_npz_path_for_segy(p, cfg=cfg)
        if not infer_npz.exists():
            continue

        (
            _n_traces,
            ns_in,
            _dt_sec_in,
            pick_final,
            scores,
            trend_res,
        ) = _load_minimal_inputs_for_thresholds(p, cfg=cfg)

        base_valid, _reason = _base_valid_mask(
            pick_final_i=pick_final,
            trend_center_i=trend_res.trend_center_i_used,
            n_samples_in=ns_in,
            cfg=cfg,
        )

        n_b = int(np.count_nonzero(base_valid))
        if n_b <= 0:
            continue

        for k in cfg.score_keys_for_filter:
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
    for k in cfg.score_keys_for_filter:
        if not vals[k]:
            msg = f'no values accumulated for score={k}'
            raise RuntimeError(msg)
        v = np.concatenate(vals[k]).astype(np.float32, copy=False)
        th = _percentile_threshold(v, frac=cfg.drop_low_frac)
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
    cfg: Stage2Cfg = DEFAULT_STAGE2_CFG,
) -> int:
    """Write CSR pick npz required by seisai_dataset.load_phase_pick_csr_npz.

    P picks: 1 per trace at most (int sample index in [1, out_ns-1])
    S picks: empty
    """
    pw = np.asarray(pick_win_512, dtype=np.float32)
    km = np.asarray(keep_mask, dtype=bool)
    if pw.shape != (n_traces,) or km.shape != (n_traces,):
        msg = f'pick_win_512/keep_mask must be (n_traces,), got {pw.shape}, {km.shape}, n_traces={n_traces}'
        raise ValueError(msg)

    p_indptr = np.zeros(n_traces + 1, dtype=np.int64)

    pick_i = np.rint(pw).astype(np.int64, copy=False)
    valid = km & np.isfinite(pw) & (pick_i > 0) & (pick_i < int(cfg.out_ns))

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


def _validate_stage2_threshold_cfg(*, cfg: Stage2Cfg = DEFAULT_STAGE2_CFG) -> None:
    if cfg.thresh_mode not in ('global', 'per_segy'):
        msg = (
            "thresh_mode must be 'global' or 'per_segy', "
            f'got {cfg.thresh_mode!r}'
        )
        raise ValueError(msg)
    if (not bool(cfg.emit_training_artifacts)) and cfg.thresh_mode == 'global':
        msg = (
            'emit_training_artifacts=False does not support thresh_mode=global. '
            "Set thresh_mode='per_segy'."
        )
        raise ValueError(msg)


def _resolve_thresholds_arg_for_training(
    *,
    global_thresholds: dict[str, float] | None,
    cfg: Stage2Cfg = DEFAULT_STAGE2_CFG,
) -> dict[str, float] | None:
    _validate_stage2_threshold_cfg(cfg=cfg)
    if cfg.thresh_mode == 'global':
        if global_thresholds is None:
            msg = 'thresh_mode=global but global_thresholds is None'
            raise RuntimeError(msg)
        return global_thresholds
    return None


# =========================
# Main per-file processing
# =========================
def process_one_segy(
    segy_path: Path,
    *,
    global_thresholds: dict[str, float] | None,
    cfg: Stage2Cfg = DEFAULT_STAGE2_CFG,
) -> None:
    _validate_stage2_threshold_cfg(cfg=cfg)

    infer_npz = infer_npz_path_for_segy(segy_path, cfg=cfg)
    if not infer_npz.exists():
        msg = f'infer npz not found for segy: {segy_path}  expected={infer_npz}'
        raise FileNotFoundError(msg)

    out_segy = out_segy_path_for_in(segy_path, cfg=cfg)
    out_segy.parent.mkdir(parents=True, exist_ok=True)

    side_npz = out_sidecar_npz_path_for_out(out_segy, cfg=cfg)
    pick_csr_npz: Path | None = None
    if bool(cfg.emit_training_artifacts):
        pick_csr_npz = out_pick_csr_npz_path_for_out(out_segy, cfg=cfg)

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

        if dt_us_in % int(cfg.up_factor) != 0:
            msg = f'dt_us must be divisible by {cfg.up_factor}. got {dt_us_in}'
            raise ValueError(msg)

        dt_us_out = dt_us_in // int(cfg.up_factor)
        dt_sec_in = float(dt_us_in) * 1e-6
        dt_sec_out = float(dt_us_out) * 1e-6

        with np.load(infer_npz, allow_pickle=False) as z:
            pick_final = require_npz_key(z, cfg.pick_key).astype(np.int64, copy=False)
            if pick_final.ndim != 1 or pick_final.shape[0] != n_traces:
                msg = (
                    f'{cfg.pick_key} must be (n_traces,), got {pick_final.shape}, '
                    f'n_traces={n_traces}'
                )
                raise ValueError(msg)

            scores_weight: dict[str, np.ndarray] = {}
            for k in cfg.score_keys_for_weight:
                scores_weight[k] = require_npz_key(z, k).astype(np.float32, copy=False)

            trend_center_i_local, local_trend_ok = _load_stage1_local_trend_center_i(
                z=z,
                n_traces=n_traces,
                dt_sec_in=dt_sec_in,
                cfg=cfg,
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
                cfg=cfg,
            )
            scores_filter: dict[str, np.ndarray] = {
                'conf_prob1': scores_weight['conf_prob1'],
                'conf_rs1': scores_weight['conf_rs1'],
                'conf_trend1': trend_res.conf_trend1,
            }
            trend_center_i_raw = trend_res.trend_center_i_raw
            trend_center_i_local = trend_res.trend_center_i_local
            trend_center_i_final = trend_res.trend_center_i_final
            trend_center_i_used = trend_res.trend_center_i_used
            trend_filled_mask = trend_res.trend_filled_mask
            ffid_values = trend_res.ffid_values
            ffid_unique_values = trend_res.ffid_unique_values
            shot_x_ffid = trend_res.shot_x_ffid
            shot_y_ffid = trend_res.shot_y_ffid

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

        c_round = np.full(n_traces, -1, dtype=np.int64)
        c_ok = np.isfinite(trend_center_i_used) & (trend_center_i_used > 0.0)
        if bool(np.any(c_ok)):
            c_round[c_ok] = np.rint(trend_center_i_used[c_ok]).astype(
                np.int64, copy=False
            )
        win_start_i = c_round - int(cfg.half_win)
        if not bool(cfg.emit_training_artifacts):
            # In inference-only mode there is no keep_mask gate on stage4 mapping.
            # Force invalid-trend traces to stay outside raw sample range.
            win_start_i[~c_ok] = np.int64(-int(ns_in))

        keep_mask: np.ndarray | None = None
        thresholds_used: dict[str, float] | None = None
        reason_mask: np.ndarray | None = None
        pick_win_512: np.ndarray | None = None

        if bool(cfg.emit_training_artifacts):
            thresholds_arg = _resolve_thresholds_arg_for_training(
                global_thresholds=global_thresholds, cfg=cfg
            )
            keep_mask, thresholds_used, reason_mask, _base_valid = build_keep_mask(
                pick_final_i=pick_final,
                trend_center_i=trend_center_i_used,
                n_samples_in=ns_in,
                scores=scores_filter,
                thresholds=thresholds_arg,
                cfg=cfg,
            )

            pick_win_512 = (
                pick_final.astype(np.float32) - win_start_i.astype(np.float32)
            ) * float(cfg.up_factor)
            pick_win_512[~keep_mask] = np.nan

        spec = segyio.spec()
        spec.tracecount = n_traces
        spec.samples = np.arange(cfg.out_ns, dtype=np.int32)
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
            dst.bin[_field_key_to_int(segyio.BinField.Samples)] = cfg.out_ns

            for i in range(n_traces):
                h = {_field_key_to_int(k): v for k, v in dict(src.header[i]).items()}
                h[_field_key_to_int(segyio.TraceField.TRACE_SAMPLE_INTERVAL)] = (
                    dt_us_out
                )
                h[_field_key_to_int(segyio.TraceField.TRACE_SAMPLE_COUNT)] = cfg.out_ns
                dst.header[i] = h

                tr = np.asarray(src.trace[i], dtype=np.float32)
                w256, _start = _extract_256(
                    tr, center_i=float(trend_center_i_used[i]), cfg=cfg
                )
                w512 = _upsample_256_to_512_linear(w256, cfg=cfg)
                dst.trace[i] = w512

            dst.flush()

    sidecar_payload: dict[str, object] = {
        'src_segy': str(segy_path),
        'src_infer_npz': str(infer_npz),
        'out_segy': str(out_segy),
        'dt_sec_in': np.float32(dt_sec_in),
        'dt_sec_out': np.float32(dt_sec_out),
        'dt_us_in': np.int32(dt_us_in),
        'dt_us_out': np.int32(dt_us_out),
        'n_traces': np.int32(n_traces),
        'n_samples_in': np.int32(ns_in),
        'n_samples_out': np.int32(cfg.out_ns),
        'window_start_i': win_start_i.astype(np.int64, copy=False),
    }

    nnz_p = 0
    if bool(cfg.emit_training_artifacts):
        if pick_csr_npz is None:
            msg = 'internal error: pick_csr_npz is None in training mode'
            raise RuntimeError(msg)
        if keep_mask is None or thresholds_used is None or reason_mask is None:
            msg = 'internal error: keep/threshold/reason missing in training mode'
            raise RuntimeError(msg)
        if pick_win_512 is None:
            msg = 'internal error: pick_win_512 missing in training mode'
            raise RuntimeError(msg)

        nnz_p = _build_phase_pick_csr_npz(
            out_path=pick_csr_npz,
            pick_win_512=pick_win_512,
            keep_mask=keep_mask,
            n_traces=n_traces,
            cfg=cfg,
        )

        sidecar_payload.update(
            out_pick_csr_npz=str(pick_csr_npz),
            thresh_mode=str(cfg.thresh_mode),
            drop_low_frac=np.float32(cfg.drop_low_frac),
            local_global_diff_th_samples=np.int32(cfg.local_global_diff_th_samples),
            local_discard_radius_traces=np.int32(cfg.local_discard_radius_traces),
            trend_center_i_raw=trend_center_i_raw.astype(np.float32, copy=False),
            trend_center_i_local=trend_center_i_local.astype(np.float32, copy=False),
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
            ffid_values=ffid_values.astype(np.int64, copy=False),
            ffid_unique_values=ffid_unique_values.astype(np.int64, copy=False),
            shot_x_ffid=shot_x_ffid.astype(np.float64, copy=False),
            shot_y_ffid=shot_y_ffid.astype(np.float64, copy=False),
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

    np.savez_compressed(side_npz, **sidecar_payload)

    n_fill = int(np.count_nonzero(trend_filled_mask))
    n_ld = int(np.count_nonzero(local_discard_mask))
    n_nn = int(np.count_nonzero(nn_replaced_mask))
    n_gl = int(np.count_nonzero(global_replaced_mask))
    if bool(cfg.emit_training_artifacts):
        if keep_mask is None or thresholds_used is None:
            msg = 'internal error: summary stats missing in training mode'
            raise RuntimeError(msg)
        n_keep = int(np.count_nonzero(keep_mask))
        tag = 'global' if cfg.thresh_mode == 'global' else 'per_segy'
        print(
            f'[OK] {segy_path.name} -> {out_segy.name}  keep={n_keep}/{n_traces} '
            f'filled_trend={n_fill}/{n_traces} discard_local={n_ld} '
            f'fill_nn={n_nn} fill_global={n_gl} '
            f'labels_written(P)={nnz_p} '
            f'th({tag} p10) prob={thresholds_used["conf_prob1"]:.6g} '
            f'trend={thresholds_used["conf_trend1"]:.6g} rs={thresholds_used["conf_rs1"]:.6g}'
        )
    else:
        print(
            f'[OK] {segy_path.name} -> {out_segy.name}  inference_only=1 '
            f'filled_trend={n_fill}/{n_traces} discard_local={n_ld} '
            f'fill_nn={n_nn} fill_global={n_gl}'
        )


def run_stage2(
    *,
    cfg: Stage2Cfg = DEFAULT_STAGE2_CFG,
    segy_paths: list[Path] | None = None,
) -> None:
    _validate_stage2_threshold_cfg(cfg=cfg)

    if segy_paths is None:
        segys = find_segy_files(cfg.in_segy_root, exts=cfg.segy_exts, recursive=True)
    else:
        segys = list(segy_paths)
    print(f'[RUN] found {len(segys)} segy files under {cfg.in_segy_root}')

    segys2: list[Path] = []
    for p in segys:
        infer_npz = infer_npz_path_for_segy(p, cfg=cfg)
        if not infer_npz.exists():
            print(f'[SKIP] infer npz missing: {p}  expected={infer_npz}')
            continue
        segys2.append(p)

    global_thresholds = None
    if bool(cfg.emit_training_artifacts) and cfg.thresh_mode == 'global':
        global_thresholds = compute_global_thresholds(segys2, cfg=cfg)

    for p in segys2:
        process_one_segy(p, global_thresholds=global_thresholds, cfg=cfg)


def main() -> None:
    run_stage2(cfg=DEFAULT_STAGE2_CFG)


if __name__ == '__main__':
    main()
