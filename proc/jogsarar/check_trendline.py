# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segyio
from seisai_utils.viz_wiggle import PickOverlay, WiggleConfig, plot_wiggle

# =========================
# ここだけ直して使う
# =========================
SEGY_PATH = Path(
    '/home/dcuser/data/ActiveSeisField/jogsarar/0020_geom_set_1401.sgy'
)  # 元SEGYファイル
PROB_NPZ_PATH = Path(
    '/home/dcuser/data/ActiveSeisField/jogsarar_out/0020_geom_set_1401.prob.npz'
)  # run_fbp_infer の生成物
FFID = 2013

SAMPLE_START = 0
SAMPLE_END = 100  # 表示したいサンプル範囲（endは含まない）
X_MODE = 'trace'  # "trace" | "offset" | "chno"
ENDIAN = 'big'  # "big" | "little"
OUT_PNG = None  # Path("/tmp/ffid2007.png") のようにすると保存。Noneなら表示


def load_gather_traces(
    segy_path: Path,
    trace_indices: np.ndarray,
    sample_start: int,
    sample_end: int,
    *,
    endian: str,
) -> np.ndarray:
    with segyio.open(
        str(segy_path),
        'r',
        ignore_geometry=True,
        endian=endian,
    ) as f:
        n_samples = int(f.samples.size)
        if not (0 <= sample_start < sample_end <= n_samples):
            raise ValueError(
                f'invalid sample window: [{sample_start}, {sample_end}) '
                f'for n_samples={n_samples}'
            )

        xs = []
        for ti in trace_indices.astype(np.int64):
            x = np.asarray(f.trace[int(ti)][sample_start:sample_end], dtype=np.float32)
            xs.append(x)
        return np.stack(xs, axis=0)  # (nt, ns)


def plot_ffid_trend_wiggle(
    *,
    segy_path: Path,
    prob_npz_path: Path,
    ffid: int,
    sample_start: int,
    sample_end: int,
    x_mode: str,
    endian: str,
    out_png: Path | None,
) -> None:
    z = np.load(prob_npz_path)

    dt = float(z['dt_sec'])
    ffid_values = np.asarray(z['ffid_values'], dtype=np.int32)
    trace_indices_all = np.asarray(z['trace_indices'], dtype=np.int64)

    offsets_all = np.asarray(z['offsets'], dtype=np.float32)
    chno_all = np.asarray(z['chno_values'], dtype=np.int32)

    pick_final_all = np.asarray(z['pick_final'], dtype=np.int32)
    trend_t_sec_all = np.asarray(z['trend_t_sec'], dtype=np.float32)
    trend_covered_all = np.asarray(z['trend_covered'], dtype=bool)

    m = ffid_values == int(ffid)
    if not np.any(m):
        raise ValueError(f'ffid={ffid} not found in npz: {prob_npz_path}')

    trace_indices = trace_indices_all[m]
    order = np.argsort(trace_indices)
    trace_indices = trace_indices[order]

    offsets = offsets_all[m][order]
    chno = chno_all[m][order]
    pick_final = pick_final_all[m][order].astype(np.float32)
    trend_t_sec = trend_t_sec_all[m][order].astype(np.float32)
    trend_covered = trend_covered_all[m][order]

    nt = int(trace_indices.size)
    ns = int(sample_end - sample_start)
    t0_sec = float(sample_start) * dt
    t1_sec = float(sample_end - 1) * dt

    traces = load_gather_traces(
        segy_path,
        trace_indices,
        sample_start,
        sample_end,
        endian=endian,
    )

    # wiggleが見やすいようにトレースごとに標準化
    traces = traces - np.mean(traces, axis=1, keepdims=True)
    traces = traces / (np.std(traces, axis=1, keepdims=True) + 1e-10)

    xm = str(x_mode).lower()
    if xm == 'trace':
        x = np.arange(nt, dtype=np.float32)
        x_label = 'trace index (within ffid)'
    elif xm == 'offset':
        x = offsets.astype(np.float32, copy=False)
        x_label = 'offset'
    elif xm == 'chno':
        x = chno.astype(np.float32)
        x_label = 'chno'
    else:
        raise ValueError('x_mode must be one of: trace | offset | chno')

    # pick_final (raw sample index) -> window内 sample index に変換
    pick_win = pick_final.copy()
    pick_win[pick_win <= 0] = np.nan
    pick_win = pick_win - float(sample_start)
    pick_win[(pick_win < 0) | (pick_win >= float(ns))] = np.nan

    # trend_t_sec は秒なので、そのままtime軸に重ねる（window外はNaN）
    trend_sec = trend_t_sec.copy()
    trend_sec = np.where(
        np.isfinite(trend_sec) & (trend_sec >= t0_sec) & (trend_sec <= t1_sec),
        trend_sec,
        np.nan,
    )

    fig, ax = plt.subplots(figsize=(16, 10))
    plot_wiggle(
        traces,
        ax=ax,
        cfg=WiggleConfig(
            dt=dt,
            t0=t0_sec,
            time_axis=1,  # traces shape (nt, ns)
            x=x,
            normalize='trace',
            gain=2.0,
            fill_positive=True,
            picks=(
                PickOverlay(
                    y=pick_win,
                    unit='sample',
                    label='pick_final',
                    marker='x',
                    size=18.0,
                    color='r',
                    alpha=0.9,
                ),
            ),
            show_legend=True,
        ),
    )

    ax.plot(x, trend_sec, lw=1.6, alpha=0.9, color='g', label='trend_t_sec', zorder=7)

    # trend欠損位置が一目で分かるように、covered=False を上端に打つ（任意）
    miss = ~trend_covered
    if np.any(miss):
        ax.scatter(
            x[miss],
            np.full(int(np.count_nonzero(miss)), t0_sec, dtype=np.float32),
            s=18.0,
            marker='v',
            color='g',
            alpha=0.6,
            label='trend missing (covered=False)',
            zorder=8,
        )

    cov = int(np.count_nonzero(trend_covered))
    ax.set_title(
        f'{segy_path.name}  ffid={ffid}  trend_covered={cov}/{nt} ({cov / nt:.1%})  '
        f'window=[{sample_start},{sample_end})'
    )
    ax.set_xlabel(x_label)
    ax.legend(loc='best')
    fig.tight_layout()

    if out_png is None:
        plt.show()
    else:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200)
        plt.close(fig)


plot_ffid_trend_wiggle(
    segy_path=SEGY_PATH,
    prob_npz_path=PROB_NPZ_PATH,
    ffid=FFID,
    sample_start=SAMPLE_START,
    sample_end=SAMPLE_END,
    x_mode=X_MODE,
    endian=ENDIAN,
    out_png=OUT_PNG,
)
