from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
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
from .progress import build_progress_reporter
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


def _status_counts_line(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        return ''
    unique, counts = np.unique(arr, return_counts=True)
    return ','.join(
        f'{int(status)}:{int(count)}'
        for status, count in zip(unique.tolist(), counts.tolist(), strict=True)
    )


def build_robust_payload_from_coarse(
    coarse_npz: Mapping[str, np.ndarray],
    *,
    cfg: dict[str, Any] | None,
    source_model_id: str | None = None,
    iter_id: int | str | None = '',
    repo_root: Path | None = None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
    progress: Any | None = None,
    progress_context: Mapping[str, object] | None = None,
) -> dict[str, np.ndarray]:
    typed_cfg = load_physics_lite_config(cfg)
    canonical_cfg = physics_lite_config_to_dict(typed_cfg)
    reporter = (
        progress
        if progress is not None
        else build_progress_reporter(typed_cfg.physical_runtime.progress)
    )
    progress_fields = dict(progress_context or {})
    if runtime_diagnostics is None and bool(
        typed_cfg.physical_runtime.diagnostics_enabled
    ):
        runtime_diagnostics = PhysicalRuntimeDiagnostics(
            detailed_timing=bool(
                typed_cfg.physical_runtime.diagnostics.detailed_timing
            )
        )

    if runtime_diagnostics is None:
        reporter.emit('physics.stage_start', **progress_fields, stage='normalize_table')
        stage_start = perf_counter()
        table = normalize_coarse_pick_table(coarse_npz)
        reporter.emit(
            'physics.stage_done',
            **progress_fields,
            stage='normalize_table',
            elapsed=perf_counter() - stage_start,
            n_traces=int(table.n_traces),
        )
        reporter.emit('physics.stage_start', **progress_fields, stage='feasible_band')
        stage_start = perf_counter()
        feasible = compute_feasible_band(table, typed_cfg.feasible_band)
        reporter.emit(
            'physics.stage_done',
            **progress_fields,
            stage='feasible_band',
            elapsed=perf_counter() - stage_start,
        )
        reporter.emit('physics.stage_start', **progress_fields, stage='trend_result')
        stage_start = perf_counter()
        trend = build_trend_result(table, feasible, typed_cfg)
        reporter.emit(
            'physics.stage_done',
            **progress_fields,
            stage='trend_result',
            elapsed=perf_counter() - stage_start,
        )
        reporter.emit('physics.stage_start', **progress_fields, stage='confidence')
        stage_start = perf_counter()
        confidence = compute_confidence_terms(table, feasible, trend, typed_cfg)
        reporter.emit(
            'physics.stage_done',
            **progress_fields,
            stage='confidence',
            elapsed=perf_counter() - stage_start,
        )
        reporter.emit('physics.stage_start', **progress_fields, stage='merge')
        stage_start = perf_counter()
        merged = apply_keep_reject_fill(table, feasible, trend, confidence, typed_cfg)
        reporter.emit(
            'physics.stage_done',
            **progress_fields,
            stage='merge',
            elapsed=perf_counter() - stage_start,
        )
        reporter.emit('physics.stage_start', **progress_fields, stage='physical_center')
        stage_start = perf_counter()
        physical = build_geometry_two_piece_physical_center(
            coarse_npz=coarse_npz,
            table=table,
            feasible=feasible,
            trend=trend,
            merged=merged,
            cfg=typed_cfg,
            progress=reporter,
            progress_context=progress_fields,
        )
        reporter.emit(
            'physics.stage_done',
            **progress_fields,
            stage='physical_center',
            elapsed=perf_counter() - stage_start,
        )
    else:
        with runtime_diagnostics.time_physics():
            reporter.emit('physics.stage_start', **progress_fields, stage='normalize_table')
            stage_start = perf_counter()
            with runtime_diagnostics.time_block('normalize_table_sec'):
                table = normalize_coarse_pick_table(coarse_npz)
            runtime_diagnostics.set_traces(int(table.n_traces))
            reporter.emit(
                'physics.stage_done',
                **progress_fields,
                stage='normalize_table',
                elapsed=perf_counter() - stage_start,
                n_traces=int(table.n_traces),
            )
            reporter.emit('physics.stage_start', **progress_fields, stage='feasible_band')
            stage_start = perf_counter()
            with runtime_diagnostics.time_block('feasible_band_sec'):
                feasible = compute_feasible_band(table, typed_cfg.feasible_band)
            reporter.emit(
                'physics.stage_done',
                **progress_fields,
                stage='feasible_band',
                elapsed=perf_counter() - stage_start,
            )
            reporter.emit('physics.stage_start', **progress_fields, stage='trend_result')
            stage_start = perf_counter()
            with runtime_diagnostics.time_block('trend_result_sec'):
                trend = build_trend_result(table, feasible, typed_cfg)
            reporter.emit(
                'physics.stage_done',
                **progress_fields,
                stage='trend_result',
                elapsed=perf_counter() - stage_start,
            )
            reporter.emit('physics.stage_start', **progress_fields, stage='confidence')
            stage_start = perf_counter()
            with runtime_diagnostics.time_block('confidence_sec'):
                confidence = compute_confidence_terms(
                    table,
                    feasible,
                    trend,
                    typed_cfg,
                )
            reporter.emit(
                'physics.stage_done',
                **progress_fields,
                stage='confidence',
                elapsed=perf_counter() - stage_start,
            )
            reporter.emit('physics.stage_start', **progress_fields, stage='merge')
            stage_start = perf_counter()
            with runtime_diagnostics.time_block('merge_sec'):
                merged = apply_keep_reject_fill(
                    table,
                    feasible,
                    trend,
                    confidence,
                    typed_cfg,
                )
            reporter.emit(
                'physics.stage_done',
                **progress_fields,
                stage='merge',
                elapsed=perf_counter() - stage_start,
            )
            reporter.emit('physics.stage_start', **progress_fields, stage='physical_center')
            stage_start = perf_counter()
            with runtime_diagnostics.time_physical_center():
                physical = build_geometry_two_piece_physical_center(
                    coarse_npz=coarse_npz,
                    table=table,
                    feasible=feasible,
                    trend=trend,
                    merged=merged,
                    cfg=typed_cfg,
                    runtime_diagnostics=runtime_diagnostics,
                    progress=reporter,
                    progress_context=progress_fields,
                )
            reporter.emit(
                'physics.stage_done',
                **progress_fields,
                stage='physical_center',
                elapsed=perf_counter() - stage_start,
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
    if (
        runtime_diagnostics is not None
        and bool(typed_cfg.physical_runtime.diagnostics.save_npz_scalars)
    ):
        start = perf_counter()
        runtime_diagnostics.to_npz_fields()
        runtime_diagnostics.add_timing(
            'diagnostics_aggregate_sec',
            perf_counter() - start,
        )
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
    progress: Any | None = None,
    progress_context: Mapping[str, object] | None = None,
) -> Path:
    coarse_path = Path(coarse_npz_path).expanduser().resolve()
    typed_cfg = load_physics_lite_config(cfg)
    reporter = (
        progress
        if progress is not None
        else build_progress_reporter(typed_cfg.physical_runtime.progress)
    )
    progress_fields = dict(progress_context or {})
    target_path = (
        derive_robust_npz_path(coarse_path) if out_path is None else Path(out_path)
    )
    run_start = perf_counter()
    reporter.emit(
        'physics.start',
        **progress_fields,
        coarse=coarse_path,
        out=target_path,
        fit_kind=str(typed_cfg.physical_trend.fit_kind),
        fit_policy=str(typed_cfg.physical_runtime.fit_policy),
    )
    runtime_diagnostics = (
        PhysicalRuntimeDiagnostics(
            detailed_timing=bool(
                typed_cfg.physical_runtime.diagnostics.detailed_timing
            )
        )
        if bool(typed_cfg.physical_runtime.diagnostics_enabled)
        else None
    )
    reporter.emit(
        'physics.stage_start',
        **progress_fields,
        stage='load_coarse_npz',
    )
    stage_start = perf_counter()
    if runtime_diagnostics is None:
        coarse = load_coarse_npz(coarse_path)
    else:
        with runtime_diagnostics.time_block('load_coarse_npz_sec'):
            coarse = load_coarse_npz(coarse_path)
    reporter.emit(
        'physics.stage_done',
        **progress_fields,
        stage='load_coarse_npz',
        elapsed=perf_counter() - stage_start,
    )
    payload = build_robust_payload_from_coarse(
        coarse,
        cfg=cfg,
        source_model_id=source_model_id,
        iter_id=iter_id,
        repo_root=repo_root,
        runtime_diagnostics=runtime_diagnostics,
        progress=reporter,
        progress_context=progress_fields,
    )
    reporter.emit('physics.stage_start', **progress_fields, stage='save_robust_npz')
    stage_start = perf_counter()
    if runtime_diagnostics is None:
        saved_path = save_robust_npz(target_path, **payload)
    else:
        with runtime_diagnostics.time_block('save_robust_npz_sec'):
            saved_path = save_robust_npz(target_path, **payload)
    reporter.emit(
        'physics.stage_done',
        **progress_fields,
        stage='save_robust_npz',
        elapsed=perf_counter() - stage_start,
        out=saved_path,
    )
    summary = (
        runtime_diagnostics.to_summary()
        if runtime_diagnostics is not None
        else runtime_summary_from_npz_fields(payload)
    )
    summary_path = None
    if summary is not None and bool(typed_cfg.physical_runtime.write_runtime_summary):
        reporter.emit('physics.stage_start', **progress_fields, stage='runtime_summary')
        stage_start = perf_counter()
        summary_path = write_physics_runtime_summary(saved_path, summary)
        reporter.emit(
            'physics.stage_done',
            **progress_fields,
            stage='runtime_summary',
            elapsed=perf_counter() - stage_start,
            summary=summary_path,
        )
    summary_fields: dict[str, object] = {}
    if summary is not None:
        for key in (
            'n_traces',
            'n_source_groups',
            'n_unique_fit_contexts',
            'n_fit_calls',
            'cache_hit_rate',
            'physics_total_sec',
            'physical_center_total_sec',
            'ransac_fit_total_sec',
        ):
            if key in summary:
                summary_fields[key] = summary[key]
    if 'physical_model_status' in payload:
        summary_fields['status_counts'] = _status_counts_line(
            payload['physical_model_status']
        )
    reporter.emit(
        'physics.done',
        **progress_fields,
        elapsed=perf_counter() - run_start,
        out=saved_path,
        summary=summary_path,
        **summary_fields,
    )
    return saved_path
