from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import segyio
from stage2.io import Stage2Seed


@dataclass(frozen=True)
class Stage2CoreResult:
    dt_us_out: int
    dt_sec_out: float
    scores_filter: dict[str, np.ndarray]
    c_round: np.ndarray
    win_start_i: np.ndarray
    thresholds_used: dict[str, float] | None
    keep_mask: np.ndarray | None
    reason_mask: np.ndarray | None
    pick_win_512: np.ndarray | None
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


def run_stage2_core(
    *,
    src: segyio.SegyFile,
    n_traces: int,
    ns_in: int,
    dt_us_in: int,
    dt_sec_in: float,
    global_thresholds: dict[str, float] | None,
    seed: Stage2Seed,
    cfg,
    build_trend_result_fn: Callable[..., Any],
    resolve_thresholds_arg_for_training_fn: Callable[..., dict[str, float] | None],
    build_keep_mask_fn: Callable[..., tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]],
) -> Stage2CoreResult:
    if dt_us_in % int(cfg.up_factor) != 0:
        msg = f'dt_us must be divisible by {cfg.up_factor}. got {dt_us_in}'
        raise ValueError(msg)

    dt_us_out = dt_us_in // int(cfg.up_factor)
    dt_sec_out = float(dt_us_out) * 1e-6

    trend_res = build_trend_result_fn(
        src=src,
        n_traces=n_traces,
        n_samples_in=ns_in,
        dt_sec_in=dt_sec_in,
        pick_final_i=seed.pick_final,
        scores=seed.scores_weight,
        trend_center_i_local_in=seed.trend_center_i_local,
        local_trend_ok_in=seed.local_trend_ok,
        cfg=cfg,
    )
    scores_filter: dict[str, np.ndarray] = {
        'conf_prob1': seed.scores_weight['conf_prob1'],
        'conf_rs1': seed.scores_weight['conf_rs1'],
        'conf_trend1': trend_res.conf_trend1,
    }

    c_round = np.full(n_traces, -1, dtype=np.int64)
    c_ok = np.isfinite(trend_res.trend_center_i_used) & (trend_res.trend_center_i_used > 0.0)
    if bool(np.any(c_ok)):
        c_round[c_ok] = np.rint(trend_res.trend_center_i_used[c_ok]).astype(
            np.int64, copy=False
        )
    win_start_i = c_round - int(cfg.half_win)
    if not bool(cfg.emit_training_artifacts):
        # In inference-only mode there is no keep_mask gate on stage4 mapping.
        # Force invalid-trend traces to stay outside raw sample range.
        win_start_i[~c_ok] = np.int64(-int(ns_in))

    thresholds_used: dict[str, float] | None = None
    keep_mask: np.ndarray | None = None
    reason_mask: np.ndarray | None = None
    pick_win_512: np.ndarray | None = None

    if bool(cfg.emit_training_artifacts):
        thresholds_arg = resolve_thresholds_arg_for_training_fn(
            global_thresholds=global_thresholds, cfg=cfg
        )
        keep_mask, thresholds_used, reason_mask, _base_valid = build_keep_mask_fn(
            pick_final_i=seed.pick_final,
            trend_center_i=trend_res.trend_center_i_used,
            n_samples_in=ns_in,
            scores=scores_filter,
            thresholds=thresholds_arg,
            cfg=cfg,
        )

        pick_win_512 = (
            seed.pick_final.astype(np.float32) - win_start_i.astype(np.float32)
        ) * float(cfg.up_factor)
        pick_win_512[~keep_mask] = np.nan

    return Stage2CoreResult(
        dt_us_out=dt_us_out,
        dt_sec_out=dt_sec_out,
        scores_filter=scores_filter,
        c_round=c_round,
        win_start_i=win_start_i,
        thresholds_used=thresholds_used,
        keep_mask=keep_mask,
        reason_mask=reason_mask,
        pick_win_512=pick_win_512,
        trend_center_i_raw=trend_res.trend_center_i_raw,
        trend_center_i_local=trend_res.trend_center_i_local,
        trend_center_i_final=trend_res.trend_center_i_final,
        trend_center_i_used=trend_res.trend_center_i_used,
        trend_center_i_global=trend_res.trend_center_i_global,
        nn_replaced_mask=trend_res.nn_replaced_mask,
        global_replaced_mask=trend_res.global_replaced_mask,
        global_missing_filled_mask=trend_res.global_missing_filled_mask,
        local_discard_mask=trend_res.local_discard_mask,
        global_edges_all=trend_res.global_edges_all,
        global_coef_all=trend_res.global_coef_all,
        global_edges_left=trend_res.global_edges_left,
        global_coef_left=trend_res.global_coef_left,
        global_edges_right=trend_res.global_edges_right,
        global_coef_right=trend_res.global_coef_right,
        trend_filled_mask=trend_res.trend_filled_mask,
        ffid_values=trend_res.ffid_values,
        ffid_unique_values=trend_res.ffid_unique_values,
        shot_x_ffid=trend_res.shot_x_ffid,
        shot_y_ffid=trend_res.shot_y_ffid,
        conf_trend1=trend_res.conf_trend1,
    )


__all__ = ['Stage2CoreResult', 'run_stage2_core']
