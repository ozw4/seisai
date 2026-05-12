from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from seisai_engine.pipelines.fbpick.common import (
    build_lineage_payload,
    load_coarse_npz,
    save_robust_npz,
)

from .confidence import compute_confidence_terms
from .config import load_physics_lite_config, physics_lite_config_to_dict
from .feasible import compute_feasible_band
from .merge import apply_keep_reject_fill
from .physical_center import build_geometry_two_piece_physical_center
from .pick_table import normalize_coarse_pick_table
from .runtime_diagnostics import (
    PhysicalRuntimeDiagnostics,
    derive_physics_runtime_summary_path,
    runtime_summary_from_npz_fields,
    write_physics_runtime_summary,
)
from .trend import build_trend_result

__all__ = [
    'build_robust_payload_from_coarse',
    'derive_physics_runtime_summary_path',
    'derive_robust_npz_path',
    'run_physics_lite',
]


def derive_robust_npz_path(coarse_npz_path: str | Path) -> Path:
    path = Path(coarse_npz_path).expanduser().resolve()
    if not path.name.endswith('.coarse.npz'):
        msg = f'expected *.coarse.npz input, got {path.name}'
        raise ValueError(msg)
    stem = path.name[: -len('.coarse.npz')]
    return path.with_name(f'{stem}.robust.npz')


def build_robust_payload_from_coarse(
    coarse_npz: Mapping[str, np.ndarray],
    *,
    cfg: dict[str, Any] | None,
    source_model_id: str | None = None,
    iter_id: int | str | None = '',
    repo_root: Path | None = None,
) -> dict[str, np.ndarray]:
    typed_cfg = load_physics_lite_config(cfg)
    canonical_cfg = physics_lite_config_to_dict(typed_cfg)
    runtime_diagnostics = (
        PhysicalRuntimeDiagnostics()
        if bool(typed_cfg.physical_runtime.diagnostics_enabled)
        else None
    )

    if runtime_diagnostics is None:
        table = normalize_coarse_pick_table(coarse_npz)
        feasible = compute_feasible_band(table, typed_cfg.feasible_band)
        trend = build_trend_result(table, feasible, typed_cfg)
        confidence = compute_confidence_terms(table, feasible, trend, typed_cfg)
        merged = apply_keep_reject_fill(table, feasible, trend, confidence, typed_cfg)
        physical = build_geometry_two_piece_physical_center(
            coarse_npz=coarse_npz,
            table=table,
            feasible=feasible,
            trend=trend,
            merged=merged,
            cfg=typed_cfg,
        )
    else:
        with runtime_diagnostics.time_physics():
            table = normalize_coarse_pick_table(coarse_npz)
            feasible = compute_feasible_band(table, typed_cfg.feasible_band)
            trend = build_trend_result(table, feasible, typed_cfg)
            confidence = compute_confidence_terms(table, feasible, trend, typed_cfg)
            merged = apply_keep_reject_fill(
                table,
                feasible,
                trend,
                confidence,
                typed_cfg,
            )
            with runtime_diagnostics.time_physical_center():
                physical = build_geometry_two_piece_physical_center(
                    coarse_npz=coarse_npz,
                    table=table,
                    feasible=feasible,
                    trend=trend,
                    merged=merged,
                    cfg=typed_cfg,
                    runtime_diagnostics=runtime_diagnostics,
                )

    payload = {
        'dt_sec': np.asarray(table.dt_scalar_sec, dtype=np.float32),
        'n_samples_orig': np.asarray(table.n_samples_orig, dtype=np.int32),
        'n_traces': np.asarray(table.n_traces, dtype=np.int32),
        'ffid_values': np.asarray(table.ffid, dtype=np.int32),
        'chno_values': np.asarray(table.chno, dtype=np.int32),
        'offsets_m': np.asarray(table.offset_m, dtype=np.float32),
        'trace_indices': np.asarray(table.trace_id, dtype=np.int64),
        'robust_pick_i': np.asarray(merged.robust_pick_i, dtype=np.int32),
        'robust_pick_t_sec': np.asarray(merged.robust_pick_t_sec, dtype=np.float32),
        'robust_conf': np.asarray(merged.robust_conf, dtype=np.float32),
        'robust_source': np.asarray(merged.robust_source, dtype=np.uint8),
        'used_theoretical_mask': np.asarray(
            merged.used_theoretical_mask,
            dtype=np.bool_,
        ),
        'reason_mask': np.asarray(merged.reason_mask, dtype=np.uint8),
        'conf_prob1': np.asarray(confidence.conf_prob1, dtype=np.float32),
        'conf_trend1': np.asarray(confidence.conf_trend1, dtype=np.float32),
        'conf_rs1': np.asarray(confidence.conf_rs1, dtype=np.float32),
        'trend_center_i': np.asarray(trend.trend_center_i, dtype=np.int32),
        'trend_center_t_sec': np.asarray(trend.trend_center_sec, dtype=np.float32),
        'physical_center_i': np.asarray(physical.physical_center_i, dtype=np.int32),
        'physical_center_t_sec': np.asarray(
            physical.physical_center_t_sec,
            dtype=np.float32,
        ),
        'fine_center_i': np.asarray(physical.fine_center_i, dtype=np.int32),
        'fine_center_t_sec': np.asarray(physical.fine_center_t_sec, dtype=np.float32),
        'physical_model_status': np.asarray(
            physical.physical_model_status,
            dtype=np.uint8,
        ),
        'physical_model_failure_reason': np.asarray(
            physical.physical_model_failure_reason,
            dtype=np.uint8,
        ),
        'physical_offset_source': np.asarray(
            physical.physical_offset_source,
            dtype=np.uint8,
        ),
        'physical_model_break_offset_m': np.asarray(
            physical.physical_model_break_offset_m,
            dtype=np.float32,
        ),
        'physical_model_slope_near_s_per_m': np.asarray(
            physical.physical_model_slope_near_s_per_m,
            dtype=np.float32,
        ),
        'physical_model_slope_far_s_per_m': np.asarray(
            physical.physical_model_slope_far_s_per_m,
            dtype=np.float32,
        ),
        'physical_model_velocity_near_m_s': np.asarray(
            physical.physical_model_velocity_near_m_s,
            dtype=np.float32,
        ),
        'physical_model_velocity_far_m_s': np.asarray(
            physical.physical_model_velocity_far_m_s,
            dtype=np.float32,
        ),
        'physical_model_neighbor_count': np.asarray(
            physical.physical_model_neighbor_count,
            dtype=np.int32,
        ),
        'physical_prefilter_valid_count': np.asarray(
            physical.physical_prefilter_valid_count,
            dtype=np.int32,
        ),
        'physical_model_segment_id': np.asarray(
            physical.physical_model_segment_id,
            dtype=np.int32,
        ),
        'physical_model_side': np.asarray(physical.physical_model_side, dtype=np.int8),
        'physical_model_resid_p50_ms': np.asarray(
            physical.physical_model_resid_p50_ms,
            dtype=np.float32,
        ),
        'physical_model_resid_p90_ms': np.asarray(
            physical.physical_model_resid_p90_ms,
            dtype=np.float32,
        ),
        'physical_runtime_t0_shift_ms': np.asarray(
            physical.physical_runtime_t0_shift_ms,
            dtype=np.float32,
        ),
        'physical_runtime_reuse_resid_p50_ms': np.asarray(
            physical.physical_runtime_reuse_resid_p50_ms,
            dtype=np.float32,
        ),
        'physical_runtime_reuse_resid_p90_ms': np.asarray(
            physical.physical_runtime_reuse_resid_p90_ms,
            dtype=np.float32,
        ),
        'physical_runtime_reuse_valid_count': np.asarray(
            physical.physical_runtime_reuse_valid_count,
            dtype=np.int32,
        ),
        'physical_runtime_refit_mask': np.asarray(
            physical.physical_runtime_refit_mask,
            dtype=np.bool_,
        ),
        'physical_runtime_fit_source': np.asarray(
            physical.physical_runtime_fit_source,
            dtype=np.uint8,
        ),
        'lineage': build_lineage_payload(
            canonical_cfg,
            repo_root=repo_root,
            source_model_id=source_model_id,
            iter_id=iter_id,
        ),
    }
    if (
        bool(typed_cfg.physical_runtime.anchor_selection.enabled)
        or typed_cfg.physical_runtime.fit_policy == 'anchor_source_xy'
    ):
        payload.update(
            {
                'physical_anchor_group_id': np.asarray(
                    physical.physical_anchor_group_id,
                    dtype=np.int32,
                ),
                'physical_anchor_is_anchor': np.asarray(
                    physical.physical_anchor_is_anchor,
                    dtype=np.bool_,
                ),
                'physical_anchor_nearest_anchor_group_id': np.asarray(
                    physical.physical_anchor_nearest_anchor_group_id,
                    dtype=np.int32,
                ),
                'physical_anchor_source_distance_m': np.asarray(
                    physical.physical_anchor_source_distance_m,
                    dtype=np.float32,
                ),
            }
        )
    if runtime_diagnostics is not None:
        payload.update(runtime_diagnostics.to_npz_fields())
    return payload


def run_physics_lite(
    coarse_npz_path: str | Path,
    *,
    cfg: dict[str, Any] | None,
    out_path: str | Path | None = None,
    source_model_id: str | None = None,
    iter_id: int | str | None = '',
    repo_root: Path | None = None,
) -> Path:
    coarse_path = Path(coarse_npz_path).expanduser().resolve()
    typed_cfg = load_physics_lite_config(cfg)
    payload = build_robust_payload_from_coarse(
        load_coarse_npz(coarse_path),
        cfg=cfg,
        source_model_id=source_model_id,
        iter_id=iter_id,
        repo_root=repo_root,
    )
    target_path = (
        derive_robust_npz_path(coarse_path) if out_path is None else Path(out_path)
    )
    saved_path = save_robust_npz(target_path, **payload)
    summary = runtime_summary_from_npz_fields(payload)
    if summary is not None and bool(typed_cfg.physical_runtime.write_runtime_summary):
        write_physics_runtime_summary(saved_path, summary)
    return saved_path
