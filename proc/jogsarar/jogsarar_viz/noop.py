from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def save_stage1_gather_viz_noop(
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
    del (
        out_png,
        wave_pad,
        n_samples_orig,
        offsets_m,
        dt_sec,
        pick_argmax,
        nopick_mask,
        pick_out_i,
        invalid_trace_mask,
        rs_label,
        plot_start,
        plot_end,
        lmo_vel_mps,
        lmo_bulk_shift_samples,
        viz_score_components,
        conf_prob1,
        conf_trend1,
        conf_rs1,
        viz_conf_prob_scale_enable,
        viz_conf_prob_pct_lo,
        viz_conf_prob_pct_hi,
        viz_conf_prob_pct_eps,
        viz_score_style,
        viz_ymax_conf_prob,
        viz_ymax_conf_trend,
        viz_ymax_conf_rs,
        viz_trend_line_enable,
        t_trend_sec,
        viz_trend_line_lw,
        viz_trend_line_alpha,
        viz_trend_line_color,
        viz_trend_line_label,
        title,
        scale01_by_percentile_fn,
    )


def save_conf_scatter_noop(
    *,
    out_png: Path,
    x_abs: np.ndarray,
    picks_i: np.ndarray,
    dt_ms: float,
    conf_prob_viz01: np.ndarray,
    conf_rs: np.ndarray,
    conf_trend: np.ndarray,
    title: str,
    trend_hat_ms: np.ndarray | None = None,
) -> None:
    del (
        out_png,
        x_abs,
        picks_i,
        dt_ms,
        conf_prob_viz01,
        conf_rs,
        conf_trend,
        title,
        trend_hat_ms,
    )


def save_stage4_gather_viz_noop(
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
    del (
        out_png,
        raw_wave_hw,
        offsets_m,
        dt_sec,
        pick_psn_orig_i,
        pick_rs_i,
        pick_final_i,
        title,
        cfg,
    )


__all__ = [
    'save_conf_scatter_noop',
    'save_stage1_gather_viz_noop',
    'save_stage4_gather_viz_noop',
]
