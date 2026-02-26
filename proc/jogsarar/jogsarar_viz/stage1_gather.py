from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from seisai_pick.lmo import apply_lmo_linear, lmo_correct_picks
from seisai_utils.viz_wiggle import PickOverlay, WiggleConfig, plot_wiggle


def _plot_score_panel_1d(
    *,
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    ymax: float | None,
    style: str,
) -> None:
    xx = np.asarray(x, dtype=np.float32)
    yy = np.asarray(y, dtype=np.float32)

    if xx.ndim != 1 or yy.ndim != 1 or xx.shape != yy.shape:
        msg = f'x/y must be (N,), got x={xx.shape}, y={yy.shape}'
        raise ValueError(msg)

    st = str(style).lower()
    if st not in ('bar', 'line'):
        msg = f"style must be 'bar' or 'line', got {style!r}"
        raise ValueError(msg)

    if st == 'bar':
        ax.bar(xx, yy, width=1.0, alpha=0.8)
    else:
        ax.plot(xx, yy, lw=1.2)

    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)

    if ymax is not None:
        ax.set_ylim(0.0, float(ymax))
    else:
        ax.set_ylim(bottom=0.0)


def save_stage1_gather_viz(
    *,
    out_png: Path,
    wave_pad: np.ndarray,
    n_samples_orig: int,
    offsets_m: np.ndarray,
    dt_sec: float,
    pick_argmax: np.ndarray,
    nopick_mask: np.ndarray,
    pick_out_i: np.ndarray,
    invalid_trace_mask: np.ndarray,
    rs_label: str,
    plot_start: int,
    plot_end: int,
    lmo_vel_mps: float,
    lmo_bulk_shift_samples: float,
    viz_score_components: bool,
    conf_prob1: np.ndarray,
    conf_trend1: np.ndarray,
    conf_rs1: np.ndarray,
    viz_conf_prob_scale_enable: bool,
    viz_conf_prob_pct_lo: float,
    viz_conf_prob_pct_hi: float,
    viz_conf_prob_pct_eps: float,
    viz_score_style: str,
    viz_ymax_conf_prob: float | None,
    viz_ymax_conf_trend: float | None,
    viz_ymax_conf_rs: float | None,
    viz_trend_line_enable: bool,
    t_trend_sec: np.ndarray | None,
    viz_trend_line_lw: float,
    viz_trend_line_alpha: float,
    viz_trend_line_color: str,
    viz_trend_line_label: str,
    title: str,
    scale01_by_percentile_fn: Callable[..., tuple[np.ndarray, tuple[float, float]]],
) -> None:
    if float(np.max(np.abs(wave_pad))) <= 0.0:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)

    seis_lmo = apply_lmo_linear(
        wave_pad[:, : int(n_samples_orig)],
        offsets_m,
        dt_sec=float(dt_sec),
        vel_mps=float(lmo_vel_mps),
        fill=0.0,
        bulk_shift_samples=float(lmo_bulk_shift_samples),
    )

    p1_viz = np.asarray(pick_argmax, dtype=np.float32).copy()
    p1_viz[np.asarray(nopick_mask, dtype=bool)] = np.nan

    p2_viz = np.asarray(pick_out_i, dtype=np.float32).copy()
    p2_viz[np.asarray(pick_out_i) == 0] = np.nan
    p2_viz[np.asarray(invalid_trace_mask, dtype=bool)] = np.nan

    p1_lmo = lmo_correct_picks(
        p1_viz,
        offsets_m,
        dt_sec=float(dt_sec),
        vel_mps=float(lmo_vel_mps),
    )
    p2_lmo = lmo_correct_picks(
        p2_viz,
        offsets_m,
        dt_sec=float(dt_sec),
        vel_mps=float(lmo_vel_mps),
    )

    seis_win = seis_lmo[:, int(plot_start) : int(plot_end)].astype(np.float32, copy=False)
    pred1_win = (
        p1_lmo - float(plot_start) + float(lmo_bulk_shift_samples)
    ).astype(np.float32, copy=False)
    pred2_win = (
        p2_lmo - float(plot_start) + float(lmo_bulk_shift_samples)
    ).astype(np.float32, copy=False)

    keep = np.max(np.abs(seis_win), axis=1) > 0.0
    if not np.any(keep):
        return

    x_keep = np.flatnonzero(keep).astype(np.float32)
    seis_win = seis_win[keep]
    pred1_win = pred1_win[keep]
    pred2_win = pred2_win[keep]

    seis_win = (seis_win - np.mean(seis_win, axis=1, keepdims=True)) / (
        np.std(seis_win, axis=1, keepdims=True) + 1e-10
    )

    pick_overlays = (
        PickOverlay(
            pred1_win,
            unit='sample',
            label='argmax',
            marker='o',
            size=14.0,
            color='r',
            alpha=0.9,
        ),
        PickOverlay(
            pred2_win,
            unit='sample',
            label=rs_label,
            marker='x',
            size=18.0,
            color='b',
            alpha=0.9,
        ),
    )

    if bool(viz_score_components):
        fig, axes = plt.subplots(
            4,
            1,
            figsize=(15, 13),
            sharex=True,
            gridspec_kw={'height_ratios': [3.2, 1.0, 1.0, 1.0]},
        )
        ax_wiggle, ax_prob, ax_trend, ax_rs = axes
    else:
        fig, ax_wiggle = plt.subplots(figsize=(15, 10))
        ax_prob = ax_trend = ax_rs = None

    plot_wiggle(
        seis_win,
        ax=ax_wiggle,
        cfg=WiggleConfig(
            dt=float(dt_sec),
            t0=float(plot_start) * float(dt_sec),
            time_axis=1,
            x=x_keep,
            normalize='trace',
            gain=2.0,
            fill_positive=True,
            picks=pick_overlays,
            show_legend=True,
        ),
    )

    if bool(viz_trend_line_enable) and t_trend_sec is not None:
        dt = float(dt_sec)
        if dt <= 0.0:
            msg = f'dt_sec must be positive, got {dt_sec}'
            raise ValueError(msg)

        trend_i = (np.asarray(t_trend_sec, dtype=np.float32) / dt).astype(
            np.float32,
            copy=False,
        )
        trend_i = trend_i.copy()
        trend_i[np.asarray(invalid_trace_mask, dtype=bool)] = np.nan

        trend_lmo = lmo_correct_picks(
            trend_i,
            offsets_m,
            dt_sec=dt,
            vel_mps=float(lmo_vel_mps),
        )

        trend_win = (
            trend_lmo - float(plot_start) + float(lmo_bulk_shift_samples)
        ).astype(np.float32, copy=False)
        trend_win = trend_win[keep]
        y_trend_sec = float(plot_start) * dt + trend_win * dt

        ax_wiggle.plot(
            x_keep,
            y_trend_sec,
            lw=float(viz_trend_line_lw),
            alpha=float(viz_trend_line_alpha),
            color=str(viz_trend_line_color),
            label=str(viz_trend_line_label),
            zorder=7,
        )
        ax_wiggle.legend(loc='best')

    if ax_prob is not None:
        s_prob_raw = np.asarray(conf_prob1, dtype=np.float32)[keep]
        if bool(viz_conf_prob_scale_enable):
            s_prob_viz01, (plo, phi) = scale01_by_percentile_fn(
                s_prob_raw,
                pct_lo=float(viz_conf_prob_pct_lo),
                pct_hi=float(viz_conf_prob_pct_hi),
                eps=float(viz_conf_prob_pct_eps),
            )
            prob_title = (
                f'conf_prob (p1) viz01 pct[{viz_conf_prob_pct_lo:.0f},{viz_conf_prob_pct_hi:.0f}] '
                f'raw={plo:.3g}..{phi:.3g}'
            )
            s_prob_plot = s_prob_viz01
        else:
            prob_title = 'conf_prob (p1) raw'
            s_prob_plot = s_prob_raw

        s_trend = np.asarray(conf_trend1, dtype=np.float32)[keep]
        s_rs = np.asarray(conf_rs1, dtype=np.float32)[keep]

        _plot_score_panel_1d(
            ax=ax_prob,
            x=x_keep,
            y=s_prob_plot,
            title=prob_title,
            ymax=viz_ymax_conf_prob,
            style=viz_score_style,
        )
        _plot_score_panel_1d(
            ax=ax_trend,
            x=x_keep,
            y=s_trend,
            title='conf_trend (p1)',
            ymax=viz_ymax_conf_trend,
            style=viz_score_style,
        )
        _plot_score_panel_1d(
            ax=ax_rs,
            x=x_keep,
            y=s_rs,
            title='conf_rs (p1)',
            ymax=viz_ymax_conf_rs,
            style=viz_score_style,
        )
        ax_rs.set_xlabel('trace index')

    ax_wiggle.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f'[VIZ] saved {out_png}')


__all__ = ['save_stage1_gather_viz']
