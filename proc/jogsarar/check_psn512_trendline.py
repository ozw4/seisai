# %%
#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
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
    '/home/dcuser/data/ActiveSeisField/jogsarar_psn512/0020_geom_set_1401.win512.sgy'
)

# 省略可: 通常は win512 と同名で .sidecar.npz がある
SIDECAR_NPZ_PATH = WIN512_SEGY_PATH.with_suffix('.sidecar.npz')

# raw 側の表示窓（サンプル index）
RAW_SAMPLE_START = 0
RAW_SAMPLE_END = 1000  # endは含まない

X_MODE = 'trace'  # "trace" | "offset" | "chno"
ENDIAN = 'big'  # "big" | "little"

# ★ここを list で指定（同一SEGY内の複数FFIDをまとめて処理）
FFIDS: list[int] = list([2147, 2467])

# ★保存フォルダ（ユーザーが設定）
OUT_DIR = Path('./ffid_plots')

# ★保存名テンプレ（必ず {ffid} を含める）
# 例: "{raw_stem}__ffid{ffid}__psn512_trend.png"
OUT_NAME_TEMPLATE = '{raw_stem}__{win_stem}__ffid{ffid}__psn512_trend.png'

FIGSIZE = (12, 8)


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
# Sidecar loader
# =========================
@dataclass(frozen=True)
class SidecarData:
    dt_sec_in: float
    dt_sec_out: float
    ffid_values: np.ndarray

    trend_center_i_raw: np.ndarray
    trend_center_i_used: np.ndarray
    trend_center_i_global: np.ndarray

    window_start_i: np.ndarray
    pick_final_i: np.ndarray
    pick_win_512: np.ndarray

    keep_mask: np.ndarray
    trend_filled_mask: np.ndarray

    up_factor: float
    n_samples_out: int


def load_sidecar(sidecar_npz_path: Path) -> SidecarData:
    if not sidecar_npz_path.exists():
        raise FileNotFoundError(f'sidecar not found: {sidecar_npz_path}')

    with np.load(sidecar_npz_path, allow_pickle=False) as z:
        if 'trend_center_i_global' not in z.files:
            raise ValueError(
                "sidecar missing 'trend_center_i_global' (regenerate sidecar with newer preproc)"
            )

        up_factor = (
            float(np.asarray(z['up_factor']).item()) if 'up_factor' in z.files else 2.0
        )

        sc = SidecarData(
            dt_sec_in=float(np.asarray(z['dt_sec_in']).item()),
            dt_sec_out=float(np.asarray(z['dt_sec_out']).item()),
            ffid_values=np.asarray(z['ffid_values'], dtype=np.int64),
            trend_center_i_raw=np.asarray(z['trend_center_i_raw'], dtype=np.float32),
            trend_center_i_used=np.asarray(z['trend_center_i_used'], dtype=np.float32),
            trend_center_i_global=np.asarray(
                z['trend_center_i_global'], dtype=np.float32
            ),
            window_start_i=np.asarray(z['window_start_i'], dtype=np.int64),
            pick_final_i=np.asarray(z['pick_final_i'], dtype=np.int64),
            pick_win_512=np.asarray(z['pick_win_512'], dtype=np.float32),
            keep_mask=np.asarray(z['keep_mask'], dtype=bool),
            trend_filled_mask=np.asarray(z['trend_filled_mask'], dtype=bool),
            up_factor=up_factor,
            n_samples_out=int(np.asarray(z['n_samples_out']).item()),
        )

    if sc.ffid_values.ndim != 1:
        raise ValueError(f'ffid_values must be 1D, got {sc.ffid_values.shape}')
    return sc


# =========================
# Single FFID plot
# =========================
def plot_ffid_psn512_trendlines(
    *,
    raw_segy_path: Path,
    win512_segy_path: Path,
    sidecar: SidecarData,
    ffid: int,
    raw_sample_start: int,
    raw_sample_end: int,
    x_mode: str,
    endian: str,
    out_png: Path,
    offsets_full: np.ndarray | None = None,
    chno_full: np.ndarray | None = None,
) -> None:
    ffid_values = sidecar.ffid_values

    m = ffid_values == int(ffid)
    if not np.any(m):
        raise ValueError(f'ffid={ffid} not found in sidecar')

    trace_indices = np.flatnonzero(m).astype(np.int64, copy=False)  # file trace index

    # x-axis meta from RAW header (win512 header is copied soどちらでも良い)
    if offsets_full is None:
        offsets_full = _load_trace_field_full(
            raw_segy_path, segyio.TraceField.offset, dtype=np.float32, endian=endian
        )
    if chno_full is None:
        chno_full = _load_trace_field_full(
            raw_segy_path,
            segyio.TraceField.TraceNumber,
            dtype=np.int32,
            endian=endian,
        )

    offsets = offsets_full[trace_indices]
    chno = chno_full[trace_indices]
    x, x_label = _build_x_axis(
        x_mode=x_mode, trace_indices=trace_indices, offsets=offsets, chno=chno
    )

    # subset arrays
    c_raw_g = sidecar.trend_center_i_raw[m]
    c_used_g = sidecar.trend_center_i_used[m]
    c_global_g = sidecar.trend_center_i_global[m]
    w0_g = sidecar.window_start_i[m]
    pick_final_g = sidecar.pick_final_i[m]
    pick_win512_g = sidecar.pick_win_512[m]
    keep_g = sidecar.keep_mask[m]
    filled_g = sidecar.trend_filled_mask[m]

    dt_sec_in = sidecar.dt_sec_in
    dt_sec_out = sidecar.dt_sec_out
    up_factor = sidecar.up_factor
    n_samples_out = sidecar.n_samples_out

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
    used_sec = _sec_in_raw_window(c_used_g * dt_sec_in, t0=t0_sec, t1=t1_sec)
    global_sec = _sec_in_raw_window(c_global_g * dt_sec_in, t0=t0_sec, t1=t1_sec)

    # -------- win512 panel --------
    win_tr = _load_traces_window(
        win512_segy_path, trace_indices, 0, n_samples_out, endian=endian
    )
    win_tr = _normalize_traces_per_trace(win_tr)

    # map trend centers to win512 sample coordinates -> seconds
    y_raw_512 = _to_win512_sample(c_raw_g, w0_g, up_factor=up_factor)
    y_used_512 = _to_win512_sample(c_used_g, w0_g, up_factor=up_factor)
    y_global_512 = _to_win512_sample(c_global_g, w0_g, up_factor=up_factor)

    raw_512_sec = y_raw_512 * dt_sec_out
    used_512_sec = y_used_512 * dt_sec_out
    global_512_sec = y_global_512 * dt_sec_out

    # pick overlay already in win512 sample coordinates
    pick_win512_plot = np.asarray(pick_win512_g, dtype=np.float32)
    pick_win512_plot[
        (pick_win512_plot <= 0.0) | (pick_win512_plot >= float(n_samples_out))
    ] = np.nan
    pick_win512_plot[~keep_g] = np.nan

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
                    size=8.0,
                    color='r',
                    alpha=0.8,
                ),
            ),
            show_legend=False,
        ),
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
                    size=8.0,
                    color='r',
                    alpha=0.7,
                ),
            ),
            show_legend=False,
        ),
    )

    ax1.plot(
        x, raw_512_sec, lw=1.2, ls='--', alpha=0.85, label='trend_raw mapped to win512'
    )

    ax1.plot(
        x, used_512_sec, lw=2.2, ls='-', alpha=0.95, label='trend_used mapped to win512'
    )

    # ax1.plot(
    #    x,
    #    global_512_sec,
    #    lw=1.4,
    #    ls='-.',
    #    alpha=0.9,
    #    label='trend_global mapped to win512',
    # )

    # center_sec = (0.5 * float(n_samples_out)) * dt_sec_out
    # ax1.axhline(
    #    center_sec, lw=1.0, ls=':', alpha=0.7, label='crop center (≈256 samples)'
    # )

    if np.any(~keep_g):
        ax1.scatter(
            x[~keep_g],
            np.full(int(np.count_nonzero(~keep_g)), 0.0, dtype=np.float32),
            s=7.0,
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

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =========================
# Batch runner (FFID list)
# =========================
def plot_ffids_psn512_trendlines_batch(
    *,
    raw_segy_path: Path,
    win512_segy_path: Path,
    sidecar_npz_path: Path,
    ffids: list[int],
    raw_sample_start: int,
    raw_sample_end: int,
    x_mode: str,
    endian: str,
    out_dir: Path,
    out_name_template: str,
) -> None:
    sidecar = load_sidecar(sidecar_npz_path)

    offsets_full = _load_trace_field_full(
        raw_segy_path, segyio.TraceField.offset, dtype=np.float32, endian=endian
    )
    chno_full = _load_trace_field_full(
        raw_segy_path, segyio.TraceField.TraceNumber, dtype=np.int32, endian=endian
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    for ffid in ffids:
        name = out_name_template.format(
            raw_stem=raw_segy_path.stem,
            win_stem=win512_segy_path.stem,
            ffid=int(ffid),
            x_mode=str(x_mode).lower(),
        )
        out_png = out_dir / name

        plot_ffid_psn512_trendlines(
            raw_segy_path=raw_segy_path,
            win512_segy_path=win512_segy_path,
            sidecar=sidecar,
            ffid=int(ffid),
            raw_sample_start=raw_sample_start,
            raw_sample_end=raw_sample_end,
            x_mode=x_mode,
            endian=endian,
            out_png=out_png,
            offsets_full=offsets_full,
            chno_full=chno_full,
        )


if __name__ == '__main__':
    plot_ffids_psn512_trendlines_batch(
        raw_segy_path=RAW_SEGY_PATH,
        win512_segy_path=WIN512_SEGY_PATH,
        sidecar_npz_path=SIDECAR_NPZ_PATH,
        ffids=FFIDS,
        raw_sample_start=RAW_SAMPLE_START,
        raw_sample_end=RAW_SAMPLE_END,
        x_mode=X_MODE,
        endian=ENDIAN,
        out_dir=OUT_DIR,
        out_name_template=OUT_NAME_TEMPLATE,
    )
