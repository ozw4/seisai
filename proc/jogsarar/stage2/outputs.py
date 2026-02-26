from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import segyio
from common.lineage import (
    cfg_hash as _cfg_hash,
    lineage_npz_payload,
    read_git_sha,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GIT_SHA = read_git_sha(_REPO_ROOT)


def write_win512_segy(
    *,
    src: segyio.SegyFile,
    out_segy: Path,
    n_traces: int,
    dt_us_out: int,
    trend_center_i_used: np.ndarray,
    cfg,
    field_key_to_int_fn: Callable[[Any], int],
    extract_256_fn: Callable[..., tuple[np.ndarray, int]],
    upsample_256_to_512_linear_fn: Callable[..., np.ndarray],
) -> None:
    spec = segyio.spec()
    spec.tracecount = n_traces
    spec.samples = np.arange(cfg.out_ns, dtype=np.int32)
    spec.format = 5  # IEEE float32

    sorting_val = getattr(src, 'sorting', 1)
    spec.sorting = int(sorting_val) if isinstance(sorting_val, (int, np.integer)) else 1

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


def write_phase_pick_csr_npz_if_enabled(
    *,
    emit_training_artifacts: bool,
    pick_csr_npz: Path | None,
    keep_mask: np.ndarray | None,
    thresholds_used: dict[str, float] | None,
    reason_mask: np.ndarray | None,
    pick_win_512: np.ndarray | None,
    n_traces: int,
    cfg,
    build_phase_pick_csr_npz_fn: Callable[..., int],
) -> int:
    if not bool(emit_training_artifacts):
        return 0
    if pick_csr_npz is None:
        msg = 'internal error: pick_csr_npz is None in training mode'
        raise RuntimeError(msg)
    if keep_mask is None or thresholds_used is None or reason_mask is None:
        msg = 'internal error: keep/threshold/reason missing in training mode'
        raise RuntimeError(msg)
    if pick_win_512 is None:
        msg = 'internal error: pick_win_512 missing in training mode'
        raise RuntimeError(msg)

    return build_phase_pick_csr_npz_fn(
        out_path=pick_csr_npz,
        pick_win_512=pick_win_512,
        keep_mask=keep_mask,
        n_traces=n_traces,
        cfg=cfg,
    )


def write_stage2_sidecar_npz(
    *,
    side_npz: Path,
    segy_path: Path,
    infer_npz: Path,
    out_segy: Path,
    dt_sec_in: float,
    dt_sec_out: float,
    dt_us_in: int,
    dt_us_out: int,
    n_traces: int,
    ns_in: int,
    win_start_i: np.ndarray,
    cfg,
    emit_training_artifacts: bool,
    pick_csr_npz: Path | None = None,
    thresholds_used: dict[str, float] | None = None,
    trend_center_i_raw: np.ndarray | None = None,
    trend_center_i_local: np.ndarray | None = None,
    trend_center_i_final: np.ndarray | None = None,
    trend_center_i_used: np.ndarray | None = None,
    trend_center_i_global: np.ndarray | None = None,
    nn_replaced_mask: np.ndarray | None = None,
    global_replaced_mask: np.ndarray | None = None,
    global_missing_filled_mask: np.ndarray | None = None,
    global_edges_all: np.ndarray | None = None,
    global_coef_all: np.ndarray | None = None,
    global_edges_left: np.ndarray | None = None,
    global_coef_left: np.ndarray | None = None,
    global_edges_right: np.ndarray | None = None,
    global_coef_right: np.ndarray | None = None,
    trend_filled_mask: np.ndarray | None = None,
    c_round: np.ndarray | None = None,
    ffid_values: np.ndarray | None = None,
    ffid_unique_values: np.ndarray | None = None,
    shot_x_ffid: np.ndarray | None = None,
    shot_y_ffid: np.ndarray | None = None,
    pick_final: np.ndarray | None = None,
    pick_win_512: np.ndarray | None = None,
    keep_mask: np.ndarray | None = None,
    reason_mask: np.ndarray | None = None,
    scores_filter: dict[str, np.ndarray] | None = None,
    conf_trend1: np.ndarray | None = None,
) -> None:
    infer_name = str(infer_npz)
    if infer_name.endswith('.prob.npz'):
        seed_kind = 'stage1'
    elif infer_name.endswith('.psn_pred.npz'):
        seed_kind = 'stage4'
    else:
        seed_kind = 'unknown'

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
    sidecar_payload.update(
        lineage_npz_payload(
            iter_id=getattr(cfg, 'iter_id', None),
            source_model_id=getattr(cfg, 'source_model_id', None),
            cfg_hash=_cfg_hash(cfg),
            git_sha=_GIT_SHA,
            seed_kind=seed_kind,
        )
    )

    if bool(emit_training_artifacts):
        if pick_csr_npz is None:
            msg = 'internal error: pick_csr_npz is None in training mode'
            raise RuntimeError(msg)
        if keep_mask is None or thresholds_used is None or reason_mask is None:
            msg = 'internal error: keep/threshold/reason missing in training mode'
            raise RuntimeError(msg)
        if pick_win_512 is None:
            msg = 'internal error: pick_win_512 missing in training mode'
            raise RuntimeError(msg)

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


__all__ = [
    'write_phase_pick_csr_npz_if_enabled',
    'write_stage2_sidecar_npz',
    'write_win512_segy',
]
