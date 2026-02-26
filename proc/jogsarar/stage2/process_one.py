from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import segyio
from common.npz_io import npz_1d
from common.segy_io import read_basic_segy_info


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

    infer_npz = infer_npz_path_for_segy_fn(segy_path, cfg=cfg)
    if not infer_npz.exists():
        msg = f'infer npz not found for segy: {segy_path}  expected={infer_npz}'
        raise FileNotFoundError(msg)

    out_segy = out_segy_path_for_in_fn(segy_path, cfg=cfg)
    out_segy.parent.mkdir(parents=True, exist_ok=True)

    side_npz = out_sidecar_npz_path_for_out_fn(out_segy, cfg=cfg)
    pick_csr_npz: Path | None = None
    if bool(cfg.emit_training_artifacts):
        pick_csr_npz = out_pick_csr_npz_path_for_out_fn(out_segy, cfg=cfg)

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

        with np.load(infer_npz, allow_pickle=False) as z:
            pick_final = npz_1d(
                z,
                cfg.pick_key,
                context='infer npz',
                n=int(n_traces),
                dtype=np.int64,
            )

            scores_weight: dict[str, np.ndarray] = {}
            for k in cfg.score_keys_for_weight:
                scores_weight[k] = npz_1d(
                    z,
                    k,
                    context='infer npz',
                    n=int(n_traces),
                    dtype=np.float32,
                )

            trend_center_i_local, local_trend_ok = load_stage1_local_trend_center_i_fn(
                z=z,
                n_traces=n_traces,
                dt_sec_in=dt_sec_in,
                cfg=cfg,
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

        spec = segyio.spec()
        spec.tracecount = n_traces
        spec.samples = np.arange(cfg.out_ns, dtype=np.int32)
        spec.format = 5  # IEEE float32

        sorting_val = getattr(src, 'sorting', 1)
        spec.sorting = (
            int(sorting_val) if isinstance(sorting_val, (int, np.integer)) else 1
        )

        with segyio.create(str(out_segy), spec) as dst:
            dst.text[0] = src.text[0]

            for k in src.bin:
                dst.bin[field_key_to_int_fn(k)] = src.bin[k]
            dst.bin[field_key_to_int_fn(segyio.BinField.Interval)] = dt_us_out
            dst.bin[field_key_to_int_fn(segyio.BinField.Samples)] = cfg.out_ns

            for i in range(n_traces):
                h = {field_key_to_int_fn(k): v for k, v in dict(src.header[i]).items()}
                h[field_key_to_int_fn(segyio.TraceField.TRACE_SAMPLE_INTERVAL)] = dt_us_out
                h[field_key_to_int_fn(segyio.TraceField.TRACE_SAMPLE_COUNT)] = cfg.out_ns
                dst.header[i] = h

                tr = np.asarray(src.trace[i], dtype=np.float32)
                w256, _start = extract_256_fn(
                    tr, center_i=float(trend_center_i_used[i]), cfg=cfg
                )
                w512 = upsample_256_to_512_linear_fn(w256, cfg=cfg)
                dst.trace[i] = w512

            dst.flush()

    sidecar_payload: dict[str, object] = {
        'src_segy': str(segy_path),
        'src_infer_npz': str(infer_npz),
        'out_segy': str(out_segy),
        'dt_sec_in': np.float32(dt_sec_in),
        'dt_sec_out': np.float32(dt_sec_out),
        'dt_us_in': np.int32(dt_us_in),
        'dt_us_out': np.int32(dt_us_out),
        'n_traces': np.int32(n_traces),
        'n_samples_in': np.int32(ns_in),
        'n_samples_out': np.int32(cfg.out_ns),
        'window_start_i': win_start_i.astype(np.int64, copy=False),
    }

    nnz_p = 0
    if bool(cfg.emit_training_artifacts):
        if pick_csr_npz is None:
            msg = 'internal error: pick_csr_npz is None in training mode'
            raise RuntimeError(msg)
        if keep_mask is None or thresholds_used is None or reason_mask is None:
            msg = 'internal error: keep/threshold/reason missing in training mode'
            raise RuntimeError(msg)
        if pick_win_512 is None:
            msg = 'internal error: pick_win_512 missing in training mode'
            raise RuntimeError(msg)

        nnz_p = build_phase_pick_csr_npz_fn(
            out_path=pick_csr_npz,
            pick_win_512=pick_win_512,
            keep_mask=keep_mask,
            n_traces=n_traces,
            cfg=cfg,
        )

        sidecar_payload.update(
            out_pick_csr_npz=str(pick_csr_npz),
            thresh_mode=str(cfg.thresh_mode),
            drop_low_frac=np.float32(cfg.drop_low_frac),
            local_global_diff_th_samples=np.int32(cfg.local_global_diff_th_samples),
            local_discard_radius_traces=np.int32(cfg.local_discard_radius_traces),
            trend_center_i_raw=trend_center_i_raw.astype(np.float32, copy=False),
            trend_center_i_local=trend_center_i_local.astype(np.float32, copy=False),
            trend_center_i_final=trend_center_i_final.astype(np.float32, copy=False),
            trend_center_i_used=trend_center_i_used.astype(np.float32, copy=False),
            trend_center_i_global=trend_center_i_global.astype(np.float32, copy=False),
            nn_replaced_mask=nn_replaced_mask.astype(bool, copy=False),
            global_replaced_mask=global_replaced_mask.astype(bool, copy=False),
            global_missing_filled_mask=global_missing_filled_mask.astype(bool, copy=False),
            global_edges_all=global_edges_all.astype(np.float32, copy=False),
            global_coef_all=global_coef_all.astype(np.float32, copy=False),
            global_edges_left=global_edges_left.astype(np.float32, copy=False),
            global_coef_left=global_coef_left.astype(np.float32, copy=False),
            global_edges_right=global_edges_right.astype(np.float32, copy=False),
            global_coef_right=global_coef_right.astype(np.float32, copy=False),
            trend_center_i=trend_center_i_used.astype(np.float32, copy=False),
            trend_filled_mask=trend_filled_mask.astype(bool, copy=False),
            trend_center_i_round=c_round.astype(np.int64, copy=False),
            ffid_values=ffid_values.astype(np.int64, copy=False),
            ffid_unique_values=ffid_unique_values.astype(np.int64, copy=False),
            shot_x_ffid=shot_x_ffid.astype(np.float64, copy=False),
            shot_y_ffid=shot_y_ffid.astype(np.float64, copy=False),
            pick_final_i=pick_final.astype(np.int64, copy=False),
            pick_win_512=pick_win_512.astype(np.float32, copy=False),
            keep_mask=keep_mask.astype(bool, copy=False),
            reason_mask=reason_mask.astype(np.uint8, copy=False),
            th_conf_prob1=np.float32(thresholds_used['conf_prob1']),
            th_conf_trend1=np.float32(thresholds_used['conf_trend1']),
            th_conf_rs1=np.float32(thresholds_used['conf_rs1']),
            conf_prob1=scores_filter['conf_prob1'].astype(np.float32, copy=False),
            conf_trend1=conf_trend1.astype(np.float32, copy=False),
            conf_rs1=scores_filter['conf_rs1'].astype(np.float32, copy=False),
        )

    np.savez_compressed(side_npz, **sidecar_payload)

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
            f'[OK] {segy_path.name} -> {out_segy.name}  keep={n_keep}/{n_traces} '
            f'filled_trend={n_fill}/{n_traces} discard_local={n_ld} '
            f'fill_nn={n_nn} fill_global={n_gl} '
            f'labels_written(P)={nnz_p} '
            f'th({tag} p10) prob={thresholds_used["conf_prob1"]:.6g} '
            f'trend={thresholds_used["conf_trend1"]:.6g} rs={thresholds_used["conf_rs1"]:.6g}'
        )
    else:
        print(
            f'[OK] {segy_path.name} -> {out_segy.name}  inference_only=1 '
            f'filled_trend={n_fill}/{n_traces} discard_local={n_ld} '
            f'fill_nn={n_nn} fill_global={n_gl}'
        )


__all__ = ['process_one_segy']
