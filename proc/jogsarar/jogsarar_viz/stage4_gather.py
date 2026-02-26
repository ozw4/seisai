from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from seisai_utils.viz_wiggle import PickOverlay, WiggleConfig, plot_wiggle


def _pick_to_window_samples(
    picks: np.ndarray,
    *,
    start: int,
    end: int,
    zero_is_invalid: bool,
    add_samples: float = 0.0,
) -> np.ndarray:
    p = np.asarray(picks, dtype=np.float32)
    if p.ndim != 1:
        msg = f'picks must be 1D, got {p.shape}'
        raise ValueError(msg)
    if end <= start:
        msg = f'end must be > start, got start={start}, end={end}'
        raise ValueError(msg)

    out = p - float(start) + float(add_samples)
    win_len = float(end - start)
    valid = np.isfinite(p) & (out >= 0.0) & (out < win_len)
    if zero_is_invalid:
        valid &= p > 0.0
    out[~valid] = np.nan
    return out.astype(np.float32, copy=False)


def save_stage4_gather_viz(
    *,
    out_png: Path,
    raw_wave_hw: np.ndarray,
    offsets_m: np.ndarray,
    dt_sec: float,
    pick_psn_orig_i: np.ndarray,
    pick_rs_i: np.ndarray,
    pick_final_i: np.ndarray,
    title: str,
    cfg: Any,
) -> None:
    wave = np.asarray(raw_wave_hw, dtype=np.float32)
    if wave.ndim != 2:
        msg = f'raw_wave_hw must be 2D, got {wave.shape}'
        raise ValueError(msg)

    n_traces, n_samples = wave.shape
    if n_samples <= 0:
        msg = f'raw_wave_hw has invalid sample count: {n_samples}'
        raise ValueError(msg)
    if dt_sec <= 0.0:
        msg = f'dt_sec must be positive, got {dt_sec}'
        raise ValueError(msg)

    offs = np.asarray(offsets_m, dtype=np.float32)
    if offs.shape != (n_traces,):
        msg = f'offsets_m must be (H,), got {offs.shape}, H={n_traces}'
        raise ValueError(msg)

    p_psn = np.asarray(pick_psn_orig_i, dtype=np.float32)
    p_rs = np.asarray(pick_rs_i, dtype=np.float32)
    p_final = np.asarray(pick_final_i, dtype=np.float32)
    valid_psn = np.isfinite(p_psn) & (p_psn >= 0.0) & (p_psn < float(n_samples))
    valid_rs = np.isfinite(p_rs) & (p_rs > 0.0) & (p_rs < float(n_samples))
    valid_final = np.isfinite(p_final) & (p_final > 0.0) & (p_final < float(n_samples))

    p_psn_plot = p_psn.copy()
    p_rs_plot = p_rs.copy()
    p_final_plot = p_final.copy()
    p_psn_plot[~valid_psn] = np.nan
    p_rs_plot[~valid_rs] = np.nan
    p_final_plot[~valid_final] = np.nan

    start = int(max(0, cfg.viz_plot_start))
    end_cfg = int(cfg.viz_plot_end)
    if end_cfg <= 0:
        end = n_samples
    else:
        end = min(n_samples, end_cfg)
    if end <= start:
        msg = (
            f'invalid viz sample range: start={start}, end={end}, n_samples={n_samples}'
        )
        raise ValueError(msg)

    wave_win = wave[:, start:end].astype(np.float32, copy=False)
    keep = np.max(np.abs(wave_win), axis=1) > 0.0
    if not np.any(keep):
        return

    x_keep = np.flatnonzero(keep).astype(np.float32, copy=False)
    wave_plot = wave_win[keep]

    pick_psn_win = _pick_to_window_samples(
        p_psn_plot,
        start=start,
        end=end,
        zero_is_invalid=False,
        add_samples=0.0,
    )[keep]
    pick_rs_win = _pick_to_window_samples(
        p_rs_plot,
        start=start,
        end=end,
        zero_is_invalid=False,
        add_samples=0.0,
    )[keep]
    pick_final_win = _pick_to_window_samples(
        p_final_plot,
        start=start,
        end=end,
        zero_is_invalid=False,
        add_samples=0.0,
    )[keep]

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=cfg.viz_figsize)
    plot_wiggle(
        wave_plot,
        ax=ax,
        cfg=WiggleConfig(
            dt=float(dt_sec),
            t0=float(start) * float(dt_sec),
            time_axis=1,
            x=x_keep,
            normalize='trace',
            gain=float(cfg.viz_gain),
            fill_positive=True,
            picks=(
                PickOverlay(
                    pick_psn_win,
                    unit='sample',
                    label='psn_orig',
                    marker='o',
                    size=12.0,
                    color='r',
                    alpha=0.8,
                ),
                PickOverlay(
                    pick_rs_win,
                    unit='sample',
                    label='psn+rs',
                    marker='x',
                    size=16.0,
                    color='b',
                    alpha=0.9,
                ),
                PickOverlay(
                    pick_final_win,
                    unit='sample',
                    label='final_snap',
                    marker='+',
                    size=20.0,
                    color='g',
                    alpha=0.9,
                ),
            ),
            show_legend=True,
        ),
    )
    ax.set_title(f'{title} (no LMO)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=int(cfg.viz_dpi))
    plt.close(fig)
    print(f'[VIZ] saved {out_png}')


__all__ = ['save_stage4_gather_viz']
