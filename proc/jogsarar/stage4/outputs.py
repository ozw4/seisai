from __future__ import annotations

from pathlib import Path

import numpy as np
from common.paths import (
    stage4_pred_crd_path as _stage4_pred_crd_path,
    stage4_pred_npz_path as _stage4_pred_npz_path,
    stage4_pred_out_dir as _stage4_pred_out_dir,
)
from seisai_pick.pickio.io_grstat import numpy2fbcrd


def resolve_stage4_out_paths(*, raw_path: Path, cfg) -> tuple[Path, Path, Path]:
    out_dir = _stage4_pred_out_dir(
        raw_path,
        in_raw_root=cfg.in_raw_segy_root,
        out_pred_root=cfg.out_pred_root,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = _stage4_pred_npz_path(
        raw_path,
        in_raw_root=cfg.in_raw_segy_root,
        out_pred_root=cfg.out_pred_root,
    )
    out_crd = _stage4_pred_crd_path(
        raw_path,
        in_raw_root=cfg.in_raw_segy_root,
        out_pred_root=cfg.out_pred_root,
    )
    return out_dir, out_npz, out_crd


def write_stage4_pred_npz(
    *,
    out_npz: Path,
    dt_sec_raw: float,
    n_samples_raw: int,
    n_traces: int,
    ffid_values: np.ndarray,
    chno_values: np.ndarray,
    offsets: np.ndarray,
    pick_psn512: np.ndarray,
    pmax_psn: np.ndarray,
    window_start_i: np.ndarray,
    pick_psn_orig_f: np.ndarray,
    pick_psn_orig_i: np.ndarray,
    delta_pick_rs: np.ndarray,
    cmax_rs: np.ndarray,
    rs_valid_mask: np.ndarray,
    pick_rs_i: np.ndarray,
    pick_final: np.ndarray,
) -> None:
    trace_indices = np.arange(n_traces, dtype=np.int64)
    np.savez_compressed(
        out_npz,
        dt_sec=np.float32(dt_sec_raw),
        n_samples_orig=np.int32(n_samples_raw),
        n_traces=np.int32(n_traces),
        ffid_values=ffid_values.astype(np.int32, copy=False),
        chno_values=chno_values.astype(np.int32, copy=False),
        offsets=offsets.astype(np.float32, copy=False),
        trace_indices=trace_indices,
        pick_psn512=pick_psn512.astype(np.int32, copy=False),
        pmax_psn=pmax_psn.astype(np.float32, copy=False),
        window_start_i=window_start_i.astype(np.int64, copy=False),
        pick_psn_orig_f=pick_psn_orig_f.astype(np.float32, copy=False),
        pick_psn_orig_i=pick_psn_orig_i.astype(np.int32, copy=False),
        delta_pick_rs=delta_pick_rs.astype(np.float32, copy=False),
        cmax_rs=cmax_rs.astype(np.float32, copy=False),
        rs_valid_mask=rs_valid_mask.astype(bool, copy=False),
        pick_rs_i=pick_rs_i.astype(np.int32, copy=False),
        pick_final=pick_final.astype(np.int32, copy=False),
    )


def write_stage4_crd(
    *,
    out_crd: Path,
    dt_ms: float,
    fb_mat: np.ndarray,
    ffids_sorted: list[int],
) -> None:
    numpy2fbcrd(
        dt=float(dt_ms),
        fbnum=fb_mat,
        gather_range=ffids_sorted,
        output_name=str(out_crd),
        original=None,
        mode='gather',
        header_comment='machine learning fb pick',
    )


__all__ = [
    'resolve_stage4_out_paths',
    'write_stage4_crd',
    'write_stage4_pred_npz',
]
