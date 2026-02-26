from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import segyio
from common.segy_io import read_basic_segy_info
from stage2.io import load_stage1_seed_from_infer_npz, resolve_stage2_paths
from stage2.outputs import (
    write_phase_pick_csr_npz_if_enabled,
    write_stage2_sidecar_npz,
    write_win512_segy,
)


# Keep this module focused on per-file processing by receiving stage-local helpers
# from the caller (stage2_make_psn512_windows.py).
def process_one_segy(
    segy_path: Path,
    *,
    global_thresholds: dict[str, float] | None,
    cfg,
    validate_stage2_threshold_cfg_fn: Callable[..., None],
    infer_npz_path_for_segy_fn: Callable[..., Path],
    out_segy_path_for_in_fn: Callable[..., Path],
    out_sidecar_npz_path_for_out_fn: Callable[..., Path],
    out_pick_csr_npz_path_for_out_fn: Callable[..., Path],
    load_stage1_local_trend_center_i_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    build_trend_result_fn: Callable[..., Any],
    resolve_thresholds_arg_for_training_fn: Callable[..., dict[str, float] | None],
    build_keep_mask_fn: Callable[..., tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]],
    field_key_to_int_fn: Callable[[Any], int],
    extract_256_fn: Callable[..., tuple[np.ndarray, int]],
    upsample_256_to_512_linear_fn: Callable[..., np.ndarray],
    build_phase_pick_csr_npz_fn: Callable[..., int],
) -> None:
    validate_stage2_threshold_cfg_fn(cfg=cfg)

    paths = resolve_stage2_paths(
        segy_path,
        cfg=cfg,
        infer_npz_path_for_segy_fn=infer_npz_path_for_segy_fn,
        out_segy_path_for_in_fn=out_segy_path_for_in_fn,
        out_sidecar_npz_path_for_out_fn=out_sidecar_npz_path_for_out_fn,
        out_pick_csr_npz_path_for_out_fn=out_pick_csr_npz_path_for_out_fn,
    )

    with segyio.open(str(segy_path), 'r', ignore_geometry=True) as src:
        n_traces, ns_in, dt_us_in, dt_sec_in = read_basic_segy_info(
            src,
            path=segy_path,
            name='',
        )

        if dt_us_in % int(cfg.up_factor) != 0:
            msg = f'dt_us must be divisible by {cfg.up_factor}. got {dt_us_in}'
            raise ValueError(msg)

        dt_us_out = dt_us_in // int(cfg.up_factor)
        dt_sec_out = float(dt_us_out) * 1e-6

        seed = load_stage1_seed_from_infer_npz(
            infer_npz=paths.infer_npz,
            n_traces=n_traces,
            dt_sec_in=dt_sec_in,
            cfg=cfg,
            load_stage1_local_trend_center_i_fn=load_stage1_local_trend_center_i_fn,
        )
        pick_final = seed.pick_final
        scores_weight = seed.scores_weight
        trend_center_i_local, local_trend_ok = (
            seed.trend_center_i_local,
            seed.local_trend_ok,
        )

        trend_res = build_trend_result_fn(
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
            thresholds_arg = resolve_thresholds_arg_for_training_fn(
                global_thresholds=global_thresholds, cfg=cfg
            )
            keep_mask, thresholds_used, reason_mask, _base_valid = build_keep_mask_fn(
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

        write_win512_segy(
            src=src,
            out_segy=paths.out_segy,
            n_traces=n_traces,
            dt_us_out=dt_us_out,
            trend_center_i_used=trend_center_i_used,
            cfg=cfg,
            field_key_to_int_fn=field_key_to_int_fn,
            extract_256_fn=extract_256_fn,
            upsample_256_to_512_linear_fn=upsample_256_to_512_linear_fn,
        )

    nnz_p = write_phase_pick_csr_npz_if_enabled(
        emit_training_artifacts=bool(cfg.emit_training_artifacts),
        pick_csr_npz=paths.pick_csr_npz,
        keep_mask=keep_mask,
        thresholds_used=thresholds_used,
        reason_mask=reason_mask,
        pick_win_512=pick_win_512,
        n_traces=n_traces,
        cfg=cfg,
        build_phase_pick_csr_npz_fn=build_phase_pick_csr_npz_fn,
    )

    write_stage2_sidecar_npz(
        side_npz=paths.sidecar_npz,
        segy_path=segy_path,
        infer_npz=paths.infer_npz,
        out_segy=paths.out_segy,
        dt_sec_in=dt_sec_in,
        dt_sec_out=dt_sec_out,
        dt_us_in=dt_us_in,
        dt_us_out=dt_us_out,
        n_traces=n_traces,
        ns_in=ns_in,
        win_start_i=win_start_i,
        cfg=cfg,
        emit_training_artifacts=bool(cfg.emit_training_artifacts),
        pick_csr_npz=paths.pick_csr_npz,
        thresholds_used=thresholds_used,
        trend_center_i_raw=trend_center_i_raw,
        trend_center_i_local=trend_center_i_local,
        trend_center_i_final=trend_center_i_final,
        trend_center_i_used=trend_center_i_used,
        trend_center_i_global=trend_center_i_global,
        nn_replaced_mask=nn_replaced_mask,
        global_replaced_mask=global_replaced_mask,
        global_missing_filled_mask=global_missing_filled_mask,
        global_edges_all=global_edges_all,
        global_coef_all=global_coef_all,
        global_edges_left=global_edges_left,
        global_coef_left=global_coef_left,
        global_edges_right=global_edges_right,
        global_coef_right=global_coef_right,
        trend_filled_mask=trend_filled_mask,
        c_round=c_round,
        ffid_values=ffid_values,
        ffid_unique_values=ffid_unique_values,
        shot_x_ffid=shot_x_ffid,
        shot_y_ffid=shot_y_ffid,
        pick_final=pick_final,
        pick_win_512=pick_win_512,
        keep_mask=keep_mask,
        reason_mask=reason_mask,
        scores_filter=scores_filter,
        conf_trend1=conf_trend1,
    )

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
            f'[OK] {segy_path.name} -> {paths.out_segy.name}  keep={n_keep}/{n_traces} '
            f'filled_trend={n_fill}/{n_traces} discard_local={n_ld} '
            f'fill_nn={n_nn} fill_global={n_gl} '
            f'labels_written(P)={nnz_p} '
            f'th({tag} p10) prob={thresholds_used["conf_prob1"]:.6g} '
            f'trend={thresholds_used["conf_trend1"]:.6g} rs={thresholds_used["conf_rs1"]:.6g}'
        )
    else:
        print(
            f'[OK] {segy_path.name} -> {paths.out_segy.name}  inference_only=1 '
            f'filled_trend={n_fill}/{n_traces} discard_local={n_ld} '
            f'fill_nn={n_nn} fill_global={n_gl}'
        )


__all__ = ['process_one_segy']
