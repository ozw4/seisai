# %%
#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segyio
from seisai_utils.viz_wiggle import PickOverlay, WiggleConfig, plot_wiggle

# =========================
# CONFIG (edit here)
# =========================
RAW_SEGY_PATH = Path(
    '/home/dcuser/data/ActiveSeisField/jogsarar/0020_geom_set_1401.sgy'
)

# None の場合は stage2.infer_npz_path_for_segy(raw) を使う
INFER_NPZ_PATH: Path | None = Path(
    '/home/dcuser/data/ActiveSeisField/jogsarar_out/0020_geom_set_1401.prob.npz'
)

FFID = 2467

RAW_SAMPLE_START = 0
RAW_SAMPLE_END = 1000  # end is exclusive

X_MODE = 'trace'  # trace | offset | chno | proxy_abs | proxy_signed
ENDIAN = 'big'  # big | little

OUT_PNG = Path('./debug_final_trendline.png')
FIGSIZE = (13, 9)
LOCAL_INV_MIN_CONSEC_STEPS = 2
# stage2 script path (same dir by default)
STAGE2_PATH = Path(__file__).with_name('stage2_make_psn512_windows.py')
LOCAL_INV_DROP_TH_SAMPLES = 10.0
LOCAL_INV_MIN_CONSEC_STEPS = 2


# =========================
# Stage2 module loader
# =========================
@dataclass(frozen=True)
class Stage2Module:
    mod: object


def load_stage2_module(stage2_path: Path) -> Stage2Module:
    if not stage2_path.exists():
        raise FileNotFoundError(f'stage2 script not found: {stage2_path}')

    spec = importlib.util.spec_from_file_location(
        'stage2_make_psn512_windows', stage2_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to create import spec for: {stage2_path}')

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return Stage2Module(mod=mod)


# =========================
# Debug inversion (NO local_ok exclusion)
# =========================
def inversion_mask_by_split_no_exclusion(
    *,
    trend_center_i_local: np.ndarray,
    trend_offset_signed_proxy: np.ndarray,
    ffid_groups: list[np.ndarray],
) -> np.ndarray:
    """Mark split-sides that contain any inversion, using all finite local points.

    - split by proxy sign (<0 / >0)
    - within each side, order by increasing |proxy|
    - if any step has local trend decreasing, mark ALL traces in that side

    Differences vs production:
      * ignores stage1 local_ok entirely ("下位除外"をしない)
    """
    local = np.asarray(trend_center_i_local, dtype=np.float32)
    proxy = np.asarray(trend_offset_signed_proxy, dtype=np.float32)
    if local.ndim != 1 or proxy.ndim != 1:
        raise ValueError(f'local/proxy must be 1D, got {local.shape}, {proxy.shape}')
    if local.shape != proxy.shape:
        raise ValueError(f'shape mismatch local={local.shape}, proxy={proxy.shape}')

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

            valid = np.isfinite(local[side_idx])
            if int(np.count_nonzero(valid)) < 2:
                continue

            valid_idx = side_idx[valid]
            x = np.abs(proxy[valid_idx]).astype(np.float32, copy=False)
            order = np.argsort(x, kind='mergesort')
            ord_local = local[valid_idx[order]]

            inv_step = ord_local[:-1] > ord_local[1:]
            if bool(np.any(inv_step)):
                out[side_idx] = True

    return out.astype(bool, copy=False)


# =========================
# Plot helpers (based on check_psn512_trendline)
# =========================
def _normalize_traces_per_trace(traces: np.ndarray) -> np.ndarray:
    x = np.asarray(traces, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f'traces must be 2D (nt, ns), got {x.shape}')
    x = x - np.mean(x, axis=1, keepdims=True)
    s = np.std(x, axis=1, keepdims=True)
    return x / (s + 1e-10)


def _load_traces_window(
    segy_path: Path,
    trace_indices: np.ndarray,
    sample_start: int,
    sample_end: int,
    *,
    endian: str,
) -> np.ndarray:
    ti = np.asarray(trace_indices, dtype=np.int64)
    if ti.ndim != 1:
        raise ValueError(f'trace_indices must be 1D, got {ti.shape}')

    with segyio.open(str(segy_path), 'r', ignore_geometry=True, endian=endian) as f:
        n_samples = int(f.samples.size)
        if not (0 <= int(sample_start) < int(sample_end) <= n_samples):
            raise ValueError(
                f'invalid sample window: [{sample_start},{sample_end}) for n_samples={n_samples}'
            )

        xs: list[np.ndarray] = []
        for i in ti:
            x = np.asarray(
                f.trace[int(i)][int(sample_start) : int(sample_end)], dtype=np.float32
            )
            xs.append(x)
        if not xs:
            raise ValueError('empty gather traces')
        return np.stack(xs, axis=0)


def _load_trace_field_full(
    segy_path: Path,
    field: segyio.TraceField,
    *,
    dtype,
    endian: str,
) -> np.ndarray:
    with segyio.open(str(segy_path), 'r', ignore_geometry=True, endian=endian) as f:
        full = np.asarray(f.attributes(field)[:], dtype=dtype)
        if full.ndim != 1 or full.shape[0] != int(f.tracecount):
            raise ValueError(f'bad attribute array shape: {full.shape}')
        return full


def _build_x_axis(
    *,
    x_mode: str,
    offsets: np.ndarray,
    chno: np.ndarray,
    proxy: np.ndarray,
) -> tuple[np.ndarray, str]:
    xm = str(x_mode).lower()
    if xm == 'trace':
        return np.arange(
            int(offsets.size), dtype=np.float32
        ), 'trace index (within ffid)'
    if xm == 'offset':
        return np.asarray(offsets, dtype=np.float32), 'offset'
    if xm == 'chno':
        return np.asarray(chno, dtype=np.float32), 'chno'
    if xm == 'proxy_abs':
        return np.abs(np.asarray(proxy, dtype=np.float32)), '|offset proxy|'
    if xm == 'proxy_signed':
        return np.asarray(proxy, dtype=np.float32), 'signed offset proxy'
    raise ValueError(
        "X_MODE must be one of: 'trace' | 'offset' | 'chno' | 'proxy_abs' | 'proxy_signed'"
    )


def _sec_in_raw_window(x_sec: np.ndarray, *, t0: float, t1: float) -> np.ndarray:
    v = np.asarray(x_sec, dtype=np.float32)
    return np.where(np.isfinite(v) & (v >= float(t0)) & (v <= float(t1)), v, np.nan)


def _to_win512_sample(
    center_i: np.ndarray,
    window_start_i: np.ndarray,
    *,
    up_factor: float,
) -> np.ndarray:
    c = np.asarray(center_i, dtype=np.float32)
    w0 = np.asarray(window_start_i, dtype=np.float32)
    if c.shape != w0.shape:
        raise ValueError(
            f'center_i/window_start_i shape mismatch: {c.shape} vs {w0.shape}'
        )
    y = (c - w0) * float(up_factor)
    y = np.where(np.isfinite(y), y, np.nan)
    return y.astype(np.float32, copy=False)


def _compute_win_start_i(*, center_i_used: np.ndarray, half_win: int) -> np.ndarray:
    c = np.asarray(center_i_used, dtype=np.float32)
    c_round = np.full(c.shape[0], -1, dtype=np.int64)
    ok = np.isfinite(c) & (c > 0.0)
    if bool(np.any(ok)):
        c_round[ok] = np.rint(c[ok]).astype(np.int64, copy=False)
    return (c_round - int(half_win)).astype(np.int64, copy=False)


# =========================
# Main (no argparse)
# =========================
def main() -> None:
    raw_segy_path = RAW_SEGY_PATH
    infer_npz_path = INFER_NPZ_PATH
    ffid = int(FFID)
    raw_start = int(RAW_SAMPLE_START)
    raw_end = int(RAW_SAMPLE_END)
    x_mode = str(X_MODE)
    endian = str(ENDIAN)
    out_png = OUT_PNG
    stage2_path = STAGE2_PATH

    if not raw_segy_path.exists():
        raise FileNotFoundError(f'raw segy not found: {raw_segy_path}')

    stage2 = load_stage2_module(stage2_path).mod

    # keep a reference to the production function for comparison prints
    prod_inversion_fn = stage2._local_velocity_inversion_mask_by_split

    # monkey-patch inversion check for this debug run
    def _patched_inversion(
        *,
        trend_center_i_local: np.ndarray,
        local_ok: np.ndarray,
        trend_offset_signed_proxy: np.ndarray,
        ffid_groups: list[np.ndarray],
    ) -> np.ndarray:
        _ = local_ok  # ignored on purpose
        return inversion_mask_by_split_no_exclusion(
            trend_center_i_local=trend_center_i_local,
            trend_offset_signed_proxy=trend_offset_signed_proxy,
            ffid_groups=ffid_groups,
        )

    stage2._local_velocity_inversion_mask_by_split = _patched_inversion

    with segyio.open(
        str(raw_segy_path), 'r', ignore_geometry=True, endian=endian
    ) as src:
        n_traces = int(src.tracecount)
        if n_traces <= 0:
            raise ValueError(f'no traces: {raw_segy_path}')

        ns_in = int(src.samples.size)
        if ns_in <= 0:
            raise ValueError(f'invalid n_samples: {ns_in}')

        dt_us_in = int(src.bin[segyio.BinField.Interval])
        if dt_us_in <= 0:
            raise ValueError(f'invalid dt_us: {dt_us_in}')
        dt_sec_in = float(dt_us_in) * 1e-6

        if infer_npz_path is None:
            infer_npz_path = stage2.infer_npz_path_for_segy(raw_segy_path)
        if not infer_npz_path.exists():
            raise FileNotFoundError(f'infer npz not found: {infer_npz_path}')

        with np.load(infer_npz_path, allow_pickle=False) as z:
            pick_final = stage2.require_npz_key(z, stage2.PICK_KEY).astype(
                np.int64, copy=False
            )
            if pick_final.ndim != 1 or pick_final.shape[0] != n_traces:
                raise ValueError(
                    f'{stage2.PICK_KEY} must be (n_traces,), got {pick_final.shape}, n_traces={n_traces}'
                )

            scores_weight: dict[str, np.ndarray] = {}
            for k in stage2.SCORE_KEYS_FOR_WEIGHT:
                scores_weight[k] = stage2.require_npz_key(z, k).astype(
                    np.float32, copy=False
                )

            trend_center_i_local, local_trend_ok = (
                stage2._load_stage1_local_trend_center_i(
                    z=z,
                    n_traces=n_traces,
                    dt_sec_in=dt_sec_in,
                )
            )

        # Build final trendline (with patched inversion)
        trend_res = stage2._build_trend_result(
            src=src,
            n_traces=n_traces,
            n_samples_in=ns_in,
            dt_sec_in=dt_sec_in,
            pick_final_i=pick_final,
            scores=scores_weight,
            trend_center_i_local_in=trend_center_i_local,
            local_trend_ok_in=local_trend_ok,
        )

        ffid_values = np.asarray(trend_res.ffid_values, dtype=np.int64)
        m = ffid_values == ffid
        if not bool(np.any(m)):
            raise ValueError(f'ffid={ffid} not found in segy: {raw_segy_path}')

        trace_indices = np.flatnonzero(m).astype(np.int64, copy=False)

        # offset proxy (needed for debug prints / x-axis)
        offset_abs_m = stage2._load_offset_abs_from_segy(src)
        _ffid_unique, _ffid_inv, ffid_groups = stage2.build_groups_by_key(ffid_values)
        proxy_all = stage2._build_offset_signed_proxy_by_ffid(
            offset_abs_m=offset_abs_m,
            ffid_groups=ffid_groups,
        )

    # Load headers needed for x-axis
    offsets_full = _load_trace_field_full(
        raw_segy_path,
        segyio.TraceField.offset,
        dtype=np.float32,
        endian=endian,
    )
    chno_full = _load_trace_field_full(
        raw_segy_path,
        segyio.TraceField.TraceNumber,
        dtype=np.int32,
        endian=endian,
    )

    offsets = offsets_full[trace_indices]
    chno = chno_full[trace_indices]
    proxy = proxy_all[trace_indices]

    x, x_label = _build_x_axis(x_mode=x_mode, offsets=offsets, chno=chno, proxy=proxy)

    # Subset arrays
    c_local = np.asarray(trend_res.trend_center_i_local, dtype=np.float32)[m]
    c_global = np.asarray(trend_res.trend_center_i_final, dtype=np.float32)[
        m
    ]  # global_filled baseline
    c_used = np.asarray(trend_res.trend_center_i_used, dtype=np.float32)[m]

    filled = np.asarray(trend_res.trend_filled_mask, dtype=bool)[m]
    discard = np.asarray(trend_res.local_discard_mask, dtype=bool)[m]
    gl_fallback = np.asarray(trend_res.global_replaced_mask, dtype=bool)[m]

    # Recompute "bad deviation" (production definition) and both inversion masks for display
    local_ok_prod = (
        np.asarray(local_trend_ok, dtype=bool)[m]
        & np.isfinite(c_local)
        & (c_local > 0.0)
    )
    global_ok = np.isfinite(c_global) & (c_global > 0.0)
    diff = np.abs(c_global - c_local).astype(np.float32, copy=False)
    bad = (
        local_ok_prod
        & global_ok
        & np.isfinite(diff)
        & (diff >= float(int(stage2.LOCAL_GLOBAL_DIFF_TH_SAMPLES)))
    )

    # per-ffid groups for inversion functions
    ffid_values_g = np.asarray(trend_res.ffid_values, dtype=np.int64)
    _u, _inv, groups = stage2.build_groups_by_key(ffid_values_g)

    inv_prod_all = prod_inversion_fn(
        trend_center_i_local=np.asarray(
            trend_res.trend_center_i_local, dtype=np.float32
        ),
        local_ok=np.asarray(local_trend_ok, dtype=bool)
        & np.isfinite(trend_res.trend_center_i_local),
        trend_offset_signed_proxy=proxy_all,
        ffid_groups=groups,
    )
    inv_dbg_all = inversion_mask_by_split_no_exclusion(
        trend_center_i_local=np.asarray(
            trend_res.trend_center_i_local, dtype=np.float32
        ),
        trend_offset_signed_proxy=proxy_all,
        ffid_groups=groups,
    )

    inv_prod = inv_prod_all[m]
    inv_dbg = inv_dbg_all[m]

    seeds_prod = bad & inv_prod
    seeds_dbg = bad & inv_dbg

    # Raw traces window
    raw_tr = _load_traces_window(
        raw_segy_path, trace_indices, raw_start, raw_end, endian=endian
    )
    raw_tr = _normalize_traces_per_trace(raw_tr)

    ns_raw = int(raw_end - raw_start)
    t0_sec = float(raw_start) * float(dt_sec_in)
    t1_sec = float(raw_end - 1) * float(dt_sec_in)

    pick_g = np.asarray(pick_final[m], dtype=np.float32)
    pick_g[pick_g <= 0] = np.nan
    pick_win_raw = pick_g - float(raw_start)
    pick_win_raw[(pick_win_raw < 0.0) | (pick_win_raw >= float(ns_raw))] = np.nan

    # trend in raw seconds
    local_sec = _sec_in_raw_window(c_local * float(dt_sec_in), t0=t0_sec, t1=t1_sec)
    global_sec = _sec_in_raw_window(c_global * float(dt_sec_in), t0=t0_sec, t1=t1_sec)
    used_sec = _sec_in_raw_window(c_used * float(dt_sec_in), t0=t0_sec, t1=t1_sec)

    # Build synthetic win512 gather (same as stage2 does)
    win_start_i = _compute_win_start_i(
        center_i_used=c_used, half_win=int(stage2.HALF_WIN)
    )
    up_factor = float(stage2.UP_FACTOR)
    dt_sec_out = float(dt_sec_in) / up_factor

    with segyio.open(
        str(raw_segy_path), 'r', ignore_geometry=True, endian=endian
    ) as src:
        win512_traces: list[np.ndarray] = []
        for ti, c in zip(trace_indices, c_used):
            tr = np.asarray(src.trace[int(ti)], dtype=np.float32)
            w256, _ = stage2._extract_256(tr, center_i=float(c))
            w512 = stage2._upsample_256_to_512_linear(w256)
            win512_traces.append(w512)

    win_tr = _normalize_traces_per_trace(np.stack(win512_traces, axis=0))

    y_local_512 = _to_win512_sample(c_local, win_start_i, up_factor=up_factor)
    y_global_512 = _to_win512_sample(c_global, win_start_i, up_factor=up_factor)
    y_used_512 = _to_win512_sample(c_used, win_start_i, up_factor=up_factor)

    local_512_sec = y_local_512 * float(dt_sec_out)
    global_512_sec = y_global_512 * float(dt_sec_out)
    used_512_sec = y_used_512 * float(dt_sec_out)

    pick_win512 = (pick_g - win_start_i.astype(np.float32)) * float(up_factor)
    pick_win512[(pick_win512 <= 0.0) | (pick_win512 >= float(stage2.OUT_NS))] = np.nan

    # ------------------
    # Plot
    # ------------------
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    plot_wiggle(
        raw_tr,
        ax=ax0,
        cfg=WiggleConfig(
            dt=float(dt_sec_in),
            t0=float(t0_sec),
            time_axis=1,
            x=x,
            normalize='trace',
            gain=2.0,
            fill_positive=True,
            picks=(
                PickOverlay(
                    y=pick_win_raw,
                    unit='sample',
                    label='pick_final',
                    marker='x',
                    size=8.0,
                    color='r',
                    alpha=0.8,
                ),
            ),
            show_legend=False,
        ),
    )

    ax0.plot(x, local_sec, lw=1.5, ls='--', alpha=0.9, label='local trend (stage1)')
    ax0.plot(x, global_sec, lw=1.5, ls='-.', alpha=0.9, label='global trend (filled)')
    ax0.plot(x, used_sec, lw=2.2, ls='-', alpha=0.95, label='final trend (used)')

    if bool(np.any(seeds_prod)):
        ax0.scatter(
            x[seeds_prod],
            np.full(int(np.count_nonzero(seeds_prod)), t0_sec, dtype=np.float32),
            s=24.0,
            marker='^',
            alpha=0.75,
            label='seeds (prod inversion)',
            zorder=9,
        )

    if bool(np.any(seeds_dbg)):
        ax0.scatter(
            x[seeds_dbg],
            np.full(int(np.count_nonzero(seeds_dbg)), t0_sec, dtype=np.float32),
            s=24.0,
            marker='v',
            alpha=0.75,
            label='seeds (NO exclusion inversion)',
            zorder=9,
        )

    if bool(np.any(discard)):
        ax0.scatter(
            x[discard],
            np.full(int(np.count_nonzero(discard)), t0_sec, dtype=np.float32),
            s=16.0,
            marker='.',
            alpha=0.6,
            label='local_discard_mask (expanded)',
            zorder=8,
        )

    if bool(np.any(filled)):
        ax0.scatter(
            x[filled],
            np.full(int(np.count_nonzero(filled)), t0_sec, dtype=np.float32),
            s=16.0,
            marker='s',
            alpha=0.6,
            label='filled (final != local)',
            zorder=8,
        )

    ax0.set_title(
        f'RAW gather | {raw_segy_path.name}  ffid={ffid}  window=[{raw_start},{raw_end})'
    )
    ax0.set_ylabel('time (s)')
    ax0.legend(loc='best')

    plot_wiggle(
        win_tr,
        ax=ax1,
        cfg=WiggleConfig(
            dt=float(dt_sec_out),
            t0=0.0,
            time_axis=1,
            x=x,
            normalize='trace',
            gain=2.0,
            fill_positive=True,
            picks=(
                PickOverlay(
                    y=pick_win512,
                    unit='sample',
                    label='pick_win512 (no keep_mask applied)',
                    marker='x',
                    size=8.0,
                    color='r',
                    alpha=0.7,
                ),
            ),
            show_legend=False,
        ),
    )

    ax1.plot(x, local_512_sec, lw=1.4, ls='--', alpha=0.9, label='local (mapped)')
    ax1.plot(x, global_512_sec, lw=1.4, ls='-.', alpha=0.9, label='global (mapped)')
    ax1.plot(x, used_512_sec, lw=2.2, ls='-', alpha=0.95, label='final (mapped)')

    if bool(np.any(gl_fallback)):
        ax1.scatter(
            x[gl_fallback],
            np.zeros(int(np.count_nonzero(gl_fallback)), dtype=np.float32),
            s=14.0,
            marker='o',
            alpha=0.6,
            label='global fallback segments',
            zorder=8,
        )

    ax1.set_title('Synthetic win512 gather (extracted from RAW using final trend)')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('time (s)')
    ax1.legend(loc='best')

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # ------------------
    # Debug prints for inversion (this ffid only)
    # ------------------
    print(f'[OK] wrote: {out_png}')

    # show inversion status per side (ffid only)
    side_l = (np.asarray(proxy, dtype=np.float32) < 0.0) & np.isfinite(proxy)
    side_r = (np.asarray(proxy, dtype=np.float32) > 0.0) & np.isfinite(proxy)

    def _print_side(tag: str, sel: np.ndarray) -> None:
        idx = np.flatnonzero(sel)
        if idx.size < 2:
            print(f'[{tag}] <2 traces')
            return
        x_abs = np.abs(np.asarray(proxy, dtype=np.float32)[idx]).astype(np.float32)
        order = np.argsort(x_abs, kind='mergesort')
        ord_x = x_abs[order]
        ord_local = np.asarray(c_local, dtype=np.float32)[idx[order]]
        inv_step = ord_local[:-1] > ord_local[1:]
        n_inv = int(np.count_nonzero(inv_step))
        print(f'[{tag}] n={idx.size} inv_steps={n_inv}')
        if n_inv > 0:
            k = int(np.flatnonzero(inv_step)[0])
            print(
                f'  first inversion at |proxy| {ord_x[k]:.3f}->{ord_x[k + 1]:.3f}  local {ord_local[k]:.3f}->{ord_local[k + 1]:.3f}'
            )

    _print_side('LEFT(proxy<0)', side_l)
    _print_side('RIGHT(proxy>0)', side_r)


if __name__ == '__main__':
    main()
