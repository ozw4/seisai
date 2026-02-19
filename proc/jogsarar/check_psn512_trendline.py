# %%
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segyio
from seisai_utils.viz_wiggle import PickOverlay, WiggleConfig, plot_wiggle

# =========================
# CONFIG（ここだけ直して使う）
# =========================
RAW_SEGY_PATH = Path(
    '/home/dcuser/data/ActiveSeisField/jogsarar/0020_geom_set_1401.sgy'
)

WIN512_SEGY_PATH = Path(
    '/home/dcuser/data/ActiveSeisField/jogsarar_psn512_drop005/0020_geom_set_1401.win512.sgy'
)

# 省略可: 通常は win512 と同名で .sidecar.npz がある
SIDECAR_NPZ_PATH = WIN512_SEGY_PATH.with_suffix('.sidecar.npz')

FFID = 2100

# raw 側の表示窓（サンプル index）
RAW_SAMPLE_START = 0
RAW_SAMPLE_END = 1000  # endは含まない

X_MODE = 'trace'  # "trace" | "offset" | "chno"
ENDIAN = 'big'  # "big" | "little"

OUT_PNG = Path('./tmp.png')  # 例: Path("/tmp/ffid2013_psn512_trend.png")。Noneなら表示
FIGSIZE = (18, 12)


# =========================
# Helpers
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
        return np.stack(xs, axis=0)  # (nt, ns)


def _load_trace_field_for_indices(
    segy_path: Path,
    trace_indices: np.ndarray,
    field: segyio.TraceField,
    *,
    dtype,
    endian: str,
) -> np.ndarray:
    ti = np.asarray(trace_indices, dtype=np.int64)
    with segyio.open(str(segy_path), 'r', ignore_geometry=True, endian=endian) as f:
        full = np.asarray(f.attributes(field)[:], dtype=dtype)
        if full.ndim != 1 or full.shape[0] != int(f.tracecount):
            raise ValueError(f'bad attribute array shape: {full.shape}')
        return full[ti]


def _build_x_axis(
    *,
    x_mode: str,
    trace_indices: np.ndarray,
    offsets: np.ndarray,
    chno: np.ndarray,
) -> tuple[np.ndarray, str]:
    xm = str(x_mode).lower()
    nt = int(trace_indices.size)
    if xm == 'trace':
        return np.arange(nt, dtype=np.float32), 'trace index (within ffid)'
    if xm == 'offset':
        return np.asarray(offsets, dtype=np.float32), 'offset'
    if xm == 'chno':
        return np.asarray(chno, dtype=np.float32), 'chno'
    raise ValueError("X_MODE must be one of: 'trace' | 'offset' | 'chno'")


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


# =========================
# Main plot
# =========================
def plot_ffid_psn512_trendlines(
    *,
    raw_segy_path: Path,
    win512_segy_path: Path,
    sidecar_npz_path: Path,
    ffid: int,
    raw_sample_start: int,
    raw_sample_end: int,
    x_mode: str,
    endian: str,
    out_png: Path | None,
) -> None:
    if not sidecar_npz_path.exists():
        raise FileNotFoundError(f'sidecar not found: {sidecar_npz_path}')

    with np.load(sidecar_npz_path, allow_pickle=False) as z:
        dt_sec_in = float(np.asarray(z['dt_sec_in']).item())
        dt_sec_out = float(np.asarray(z['dt_sec_out']).item())

        ffid_values = np.asarray(z['ffid_values'], dtype=np.int64)

        # trend centers (sample index in raw time base)
        c_raw = np.asarray(z['trend_center_i_raw'], dtype=np.float32)
        c_semi = np.asarray(
            z['trend_center_i_semi'], dtype=np.float32
        )  # NaN if not replaced
        c_used = np.asarray(z['trend_center_i_used'], dtype=np.float32)
        if 'trend_center_i_global' not in z.files:
            raise ValueError(
                "sidecar missing 'trend_center_i_global' (regenerate sidecar with newer preproc)"
            )
        c_global = np.asarray(z['trend_center_i_global'], dtype=np.float32)

        window_start_i = np.asarray(z['window_start_i'], dtype=np.int64)

        pick_final_i = np.asarray(z['pick_final_i'], dtype=np.int64)
        pick_win_512 = np.asarray(z['pick_win_512'], dtype=np.float32)

        keep_mask = np.asarray(z['keep_mask'], dtype=bool)
        trend_filled_mask = np.asarray(z['trend_filled_mask'], dtype=bool)

        up_factor = (
            float(np.asarray(z.get('up_factor', 2)).item())
            if 'up_factor' in z.files
            else 2.0
        )
        n_samples_out = int(np.asarray(z['n_samples_out']).item())

    if ffid_values.ndim != 1:
        raise ValueError(f'ffid_values must be 1D, got {ffid_values.shape}')

    m = ffid_values == int(ffid)
    if not np.any(m):
        raise ValueError(f'ffid={ffid} not found in sidecar: {sidecar_npz_path}')

    trace_indices = np.flatnonzero(m).astype(np.int64, copy=False)  # file trace index
    nt = int(trace_indices.size)

    # x-axis meta from RAW header (win512 header is copied soどちらでも良い)
    offsets = _load_trace_field_for_indices(
        raw_segy_path,
        trace_indices,
        segyio.TraceField.offset,
        dtype=np.float32,
        endian=endian,
    )
    chno = _load_trace_field_for_indices(
        raw_segy_path,
        trace_indices,
        segyio.TraceField.TraceNumber,
        dtype=np.int32,
        endian=endian,
    )
    x, x_label = _build_x_axis(
        x_mode=x_mode, trace_indices=trace_indices, offsets=offsets, chno=chno
    )

    # subset arrays
    c_raw_g = c_raw[m]
    c_semi_g = c_semi[m]
    c_used_g = c_used[m]
    c_global_g = c_global[m]
    w0_g = window_start_i[m]
    pick_final_g = pick_final_i[m]
    pick_win512_g = pick_win_512[m]
    keep_g = keep_mask[m]
    filled_g = trend_filled_mask[m]

    # -------- raw panel --------
    raw_tr = _load_traces_window(
        raw_segy_path, trace_indices, raw_sample_start, raw_sample_end, endian=endian
    )
    raw_tr = _normalize_traces_per_trace(raw_tr)

    ns_raw = int(raw_sample_end - raw_sample_start)
    t0_sec = float(raw_sample_start) * dt_sec_in
    t1_sec = float(raw_sample_end - 1) * dt_sec_in

    # pick_final (raw sample index) -> window-relative samples for overlay
    pick_win_raw = pick_final_g.astype(np.float32)
    pick_win_raw[pick_win_raw <= 0] = np.nan
    pick_win_raw = pick_win_raw - float(raw_sample_start)
    pick_win_raw[(pick_win_raw < 0.0) | (pick_win_raw >= float(ns_raw))] = np.nan

    # trend centers -> seconds (raw time base)
    raw_sec = _sec_in_raw_window(
        c_raw_g.astype(np.float32) * dt_sec_in, t0=t0_sec, t1=t1_sec
    )
    semi_sec = _sec_in_raw_window(
        c_semi_g.astype(np.float32) * dt_sec_in, t0=t0_sec, t1=t1_sec
    )
    used_sec = _sec_in_raw_window(
        c_used_g.astype(np.float32) * dt_sec_in, t0=t0_sec, t1=t1_sec
    )
    global_sec = _sec_in_raw_window(
        c_global_g.astype(np.float32) * dt_sec_in, t0=t0_sec, t1=t1_sec
    )

    # -------- win512 panel --------
    win_tr = _load_traces_window(
        win512_segy_path, trace_indices, 0, n_samples_out, endian=endian
    )
    win_tr = _normalize_traces_per_trace(win_tr)

    # map trend centers to win512 sample coordinates
    y_raw_512 = _to_win512_sample(c_raw_g, w0_g, up_factor=up_factor)
    y_semi_512 = _to_win512_sample(c_semi_g, w0_g, up_factor=up_factor)
    y_used_512 = _to_win512_sample(c_used_g, w0_g, up_factor=up_factor)
    y_global_512 = _to_win512_sample(c_global_g, w0_g, up_factor=up_factor)

    # convert to seconds (win512 time base)
    raw_512_sec = y_raw_512 * dt_sec_out
    semi_512_sec = y_semi_512 * dt_sec_out
    used_512_sec = y_used_512 * dt_sec_out
    global_512_sec = y_global_512 * dt_sec_out

    # pick overlay already in win512 sample coordinates
    pick_win512_plot = np.asarray(pick_win512_g, dtype=np.float32)
    pick_win512_plot[
        (pick_win512_plot <= 0.0) | (pick_win512_plot >= float(n_samples_out))
    ] = np.nan
    pick_win512_plot[~keep_g] = np.nan

    # figure
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    # raw wiggle
    plot_wiggle(
        raw_tr,
        ax=ax0,
        cfg=WiggleConfig(
            dt=dt_sec_in,
            t0=t0_sec,
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
                    size=18.0,
                    color='r',
                    alpha=0.9,
                ),
            ),
            show_legend=False,
        ),
    )
    # ax0.plot(
    #    x, raw_sec, lw=1.2, ls='--', alpha=0.85, label='trend_raw (from infer npz)'
    # )
    ax0.plot(
        x,
        semi_sec,
        lw=1.6,
        ls='-',
        alpha=0.9,
        label='trend_semi (semi-global, NaN if not used)',
    )
    ax0.plot(x, used_sec, lw=2.2, ls='-', alpha=0.95, label='trend_used (final)')
    ax0.plot(
        x,
        global_sec,
        lw=1.4,
        ls='-.',
        alpha=0.9,
        label='trend_global (offset-time fit fallback)',
    )

    # markers: filled traces / dropped traces (raw panel)
    if np.any(filled_g):
        ax0.scatter(
            x[filled_g],
            np.full(int(np.count_nonzero(filled_g)), t0_sec, dtype=np.float32),
            s=18.0,
            marker='v',
            alpha=0.7,
            label='raw trend missing -> filled',
            zorder=8,
        )
    if np.any(~keep_g):
        ax0.scatter(
            x[~keep_g],
            np.full(int(np.count_nonzero(~keep_g)), t0_sec, dtype=np.float32),
            s=14.0,
            marker='.',
            alpha=0.5,
            label='dropped (keep_mask=False)',
            zorder=7,
        )

    ax0.set_title(
        f'RAW gather  |  {raw_segy_path.name}  ffid={ffid}  window=[{raw_sample_start},{raw_sample_end})'
    )
    ax0.set_ylabel('time (s)')
    ax0.legend(loc='best')

    # win512 wiggle (PSN input)
    plot_wiggle(
        win_tr,
        ax=ax1,
        cfg=WiggleConfig(
            dt=dt_sec_out,
            t0=0.0,
            time_axis=1,
            x=x,
            normalize='trace',
            gain=2.0,
            fill_positive=True,
            picks=(
                PickOverlay(
                    y=pick_win512_plot,
                    unit='sample',
                    label='pick_win_512 (kept only)',
                    marker='x',
                    size=18.0,
                    color='r',
                    alpha=0.9,
                ),
            ),
            show_legend=False,
        ),
    )

    ax1.plot(
        x, raw_512_sec, lw=1.2, ls='--', alpha=0.85, label='trend_raw mapped to win512'
    )
    ax1.plot(
        x, semi_512_sec, lw=1.6, ls='-', alpha=0.9, label='trend_semi mapped to win512'
    )
    ax1.plot(
        x, used_512_sec, lw=2.2, ls='-', alpha=0.95, label='trend_used mapped to win512'
    )
    ax1.plot(
        x,
        global_512_sec,
        lw=1.4,
        ls='-.',
        alpha=0.9,
        label='trend_global mapped to win512',
    )

    # reference line: crop center (should be near 256 samples => 256*dt_out sec)
    center_sec = (0.5 * float(n_samples_out)) * dt_sec_out
    ax1.axhline(
        center_sec, lw=1.0, ls=':', alpha=0.7, label='crop center (≈256 samples)'
    )

    if np.any(~keep_g):
        ax1.scatter(
            x[~keep_g],
            np.full(int(np.count_nonzero(~keep_g)), 0.0, dtype=np.float32),
            s=14.0,
            marker='.',
            alpha=0.5,
            label='dropped (keep_mask=False)',
            zorder=7,
        )

    ax1.set_title(f'PSN input win512 gather  |  {win512_segy_path.name}  ffid={ffid}')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('time (s)')
    ax1.legend(loc='best')

    fig.tight_layout()

    if out_png is None:
        plt.show()
    else:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200)
        plt.close(fig)


if __name__ == '__main__':
    plot_ffid_psn512_trendlines(
        raw_segy_path=RAW_SEGY_PATH,
        win512_segy_path=WIN512_SEGY_PATH,
        sidecar_npz_path=SIDECAR_NPZ_PATH,
        ffid=FFID,
        raw_sample_start=RAW_SAMPLE_START,
        raw_sample_end=RAW_SAMPLE_END,
        x_mode=X_MODE,
        endian=ENDIAN,
        out_png=OUT_PNG,
    )
