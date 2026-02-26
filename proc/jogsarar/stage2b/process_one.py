from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import segyio
from common.segy_io import read_basic_segy_info
from stage2.core import run_stage2_core
from stage2.io import resolve_stage2_paths
from stage2.outputs import (
    write_phase_pick_csr_npz_if_enabled,
    write_stage2_sidecar_npz,
    write_win512_segy,
)
from stage2b.io import load_stage4_seed_from_pred_npz


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
    del load_stage1_local_trend_center_i_fn
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

        seed = load_stage4_seed_from_pred_npz(
            pred_npz=paths.infer_npz,
            n_traces=n_traces,
            dt_sec_in=dt_sec_in,
            cfg=cfg,
        )
        core = run_stage2_core(
            src=src,
            n_traces=n_traces,
            ns_in=ns_in,
            dt_us_in=dt_us_in,
            dt_sec_in=dt_sec_in,
            global_thresholds=global_thresholds,
            seed=seed,
            cfg=cfg,
            build_trend_result_fn=build_trend_result_fn,
            resolve_thresholds_arg_for_training_fn=resolve_thresholds_arg_for_training_fn,
            build_keep_mask_fn=build_keep_mask_fn,
        )

        write_win512_segy(
            src=src,
            out_segy=paths.out_segy,
            n_traces=n_traces,
            dt_us_out=core.dt_us_out,
            trend_center_i_used=core.trend_center_i_used,
            cfg=cfg,
            field_key_to_int_fn=field_key_to_int_fn,
            extract_256_fn=extract_256_fn,
            upsample_256_to_512_linear_fn=upsample_256_to_512_linear_fn,
        )

    nnz_p = write_phase_pick_csr_npz_if_enabled(
        emit_training_artifacts=bool(cfg.emit_training_artifacts),
        pick_csr_npz=paths.pick_csr_npz,
        keep_mask=core.keep_mask,
        thresholds_used=core.thresholds_used,
        reason_mask=core.reason_mask,
        pick_win_512=core.pick_win_512,
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
        dt_sec_out=core.dt_sec_out,
        dt_us_in=dt_us_in,
        dt_us_out=core.dt_us_out,
        n_traces=n_traces,
        ns_in=ns_in,
        win_start_i=core.win_start_i,
        cfg=cfg,
        emit_training_artifacts=bool(cfg.emit_training_artifacts),
        pick_csr_npz=paths.pick_csr_npz,
        thresholds_used=core.thresholds_used,
        trend_center_i_raw=core.trend_center_i_raw,
        trend_center_i_local=core.trend_center_i_local,
        trend_center_i_final=core.trend_center_i_final,
        trend_center_i_used=core.trend_center_i_used,
        trend_center_i_global=core.trend_center_i_global,
        nn_replaced_mask=core.nn_replaced_mask,
        global_replaced_mask=core.global_replaced_mask,
        global_missing_filled_mask=core.global_missing_filled_mask,
        global_edges_all=core.global_edges_all,
        global_coef_all=core.global_coef_all,
        global_edges_left=core.global_edges_left,
        global_coef_left=core.global_coef_left,
        global_edges_right=core.global_edges_right,
        global_coef_right=core.global_coef_right,
        trend_filled_mask=core.trend_filled_mask,
        c_round=core.c_round,
        ffid_values=core.ffid_values,
        ffid_unique_values=core.ffid_unique_values,
        shot_x_ffid=core.shot_x_ffid,
        shot_y_ffid=core.shot_y_ffid,
        pick_final=seed.pick_final,
        pick_win_512=core.pick_win_512,
        keep_mask=core.keep_mask,
        reason_mask=core.reason_mask,
        scores_filter=core.scores_filter,
        conf_trend1=core.conf_trend1,
    )

    n_fill = int(np.count_nonzero(core.trend_filled_mask))
    n_ld = int(np.count_nonzero(core.local_discard_mask))
    n_nn = int(np.count_nonzero(core.nn_replaced_mask))
    n_gl = int(np.count_nonzero(core.global_replaced_mask))
    if bool(cfg.emit_training_artifacts):
        if core.keep_mask is None or core.thresholds_used is None:
            msg = 'internal error: summary stats missing in training mode'
            raise RuntimeError(msg)
        n_keep = int(np.count_nonzero(core.keep_mask))
        tag = 'global' if cfg.thresh_mode == 'global' else 'per_segy'
        print(
            f'[OK] {segy_path.name} -> {paths.out_segy.name}  keep={n_keep}/{n_traces} '
            f'filled_trend={n_fill}/{n_traces} discard_local={n_ld} '
            f'fill_nn={n_nn} fill_global={n_gl} '
            f'labels_written(P)={nnz_p} '
            f'th({tag} p10) prob={core.thresholds_used["conf_prob1"]:.6g} '
            f'trend={core.thresholds_used["conf_trend1"]:.6g} rs={core.thresholds_used["conf_rs1"]:.6g}'
        )
    else:
        print(
            f'[OK] {segy_path.name} -> {paths.out_segy.name}  inference_only=1 '
            f'filled_trend={n_fill}/{n_traces} discard_local={n_ld} '
            f'fill_nn={n_nn} fill_global={n_gl}'
        )


__all__ = ['process_one_segy']
