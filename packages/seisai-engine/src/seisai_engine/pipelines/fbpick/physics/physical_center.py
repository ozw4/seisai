from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

from .config import PhysicsLiteConfig
from .feasible import FeasibleBandResult, compute_velocity_t0_band_from_arrays
from .geometry import (
    CoarseGeometry,
    SourceGroup,
    signed_offset_side_from_geometry,
    split_offset_gap_segments,
)
from .merge import MergeResult
from .physical_center_anchor import (
    _anchor_model_key,
    _apply_anchor_selection_diagnostics,
    build_anchor_source_xy_context,
    fit_anchor_models,
)
from .physical_center_anchor_reuse import (
    _assign_nearest_anchor_reuse_predictions,
    _build_anchor_reuse_group_plan,
    _compatible_anchor_contexts_for_reuse,
    _fallback_no_compatible_anchor,
    _nearest_anchor_reuse_context,
    _run_no_compatible_full_fit_fallback,
)
from .physical_center_context import (
    PhysicalCenterBuildContext,
    PhysicalCenterInputs,
    PhysicalCenterWorkspace,
)
from .physical_center_context_fit import (
    _assign_fit_context_fallback,
    _assign_fit_context_work_item_outcome,
    _build_fit_context_work_items,
    _build_observation_plan,
    _cache_entry_from_fit_task_result,
    _fit_and_assign_context_work_item,
    _fit_and_assign_context_work_items,
    _fit_and_assign_context_work_items_parallel,
    _FitContextWorkItem,
    _FitContextWorkResult,
    _prepare_fit_context_assignments_for_trace_indices,
    _prepare_trace_plan_assignment,
    _record_cached_context_hits,
    _record_new_fit_task_diagnostics,
)
from .physical_center_fallback import (
    _allocate_result_arrays,
    _as_bool_vector,
    _as_vector,
    _assign_fallback,
    _assign_robust_fallback,
    _build_disabled_result,
    _emit_fallback_all_and_done,
    _finalize_result_with_pending_trend_fallback,
    _PendingTrendFallback,
)
from .physical_center_fit import (
    _bin_representative_position,
    _evenly_spaced_positions,
    _fit_cache_key,
    _fit_key_for_obs,
    _fit_min_pts,
    _fit_model_for_plan,
    _fit_progress_fields,
    _fit_strategy,
    _fit_strategy_model,
    _fit_task_cfg_values,
    _fit_task_from_work_item,
    _FitCacheEntry,
    _FitTask,
    _FitTaskCfgValues,
    _FitTaskResult,
    _limit_selected_positions,
    _median_time_position,
    _offset_spread_failure_reason,
    _run_fit_task,
    _run_fit_tasks_with_executor,
    _sample_observation_indices_for_fit,
    _set_fit_worker_torch_num_threads,
    _stable_observation_seed,
    _strategy_from_fit_task_cfg,
)
from .physical_center_full_fit import run_full_fit_policy
from .physical_center_geometry import (
    PhysicalCenterGeometryContext,
    _signed_offset_side_labels,
    _SignedOffsetSideLabels,
    build_physical_center_geometry_context,
    build_physical_center_offset_context,
    build_physical_center_source_group_build,
    build_physical_center_source_group_context,
    load_physical_center_geometry,
)
from .physical_center_observation import (
    _append_index,
    _build_gap_segment_context,
    _build_group_observation_contexts,
    _build_side_observation_context,
    _concat_group_traces,
    _filtered_obs_key,
    _gap_context_key,
    _GapSegmentContext,
    _GroupObservationContext,
    _index_key_contains,
    _indices_key,
    _obs_key_or_build,
    _obs_with_target_gap_segment,
    _obs_with_target_labeled_side,
    _obs_with_target_side,
    _obs_with_target_signed_offset_side,
    _ObservationPlan,
    _ObservationPlanCache,
    _select_group_ids,
    _side_slot,
    _SideObservationContext,
    _stable_unique,
    _trace_position_map,
)
from .physical_center_prediction import (
    _assign_model_diagnostics,
    _assign_model_diagnostics_batch,
    _assign_model_prediction,
    _assign_model_prediction_batch,
    _assign_prepared_model_prediction_batch,
    _diagnostics_for_plan,
    _predict_model_array_sec,
    _predict_model_sec,
    _shift_for_trace_indices,
    _status_for_plan_batch,
    _TraceFitResult,
    _TracePlanAssignment,
)
from .physical_center_types import (
    PHYSICAL_MODEL_FAILURE_FIT_FAILED,
    PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
    PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS,
    PHYSICAL_MODEL_FAILURE_LABELS,
    PHYSICAL_MODEL_FAILURE_NONE,
    PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED,
    PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
    PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND,
    PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP,
    PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT,
    PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST,
    PHYSICAL_MODEL_STATUS_FIT_FAILED,
    PHYSICAL_MODEL_STATUS_GEOMETRY_INVALID,
    PHYSICAL_MODEL_STATUS_INSUFFICIENT_OBSERVATIONS,
    PHYSICAL_MODEL_STATUS_LABELS,
    PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED,
    PHYSICAL_MODEL_STATUS_TWO_PIECE_OK,
    PHYSICAL_OFFSET_SOURCE_GEOMETRY,
    PHYSICAL_OFFSET_SOURCE_HEADER,
    PHYSICAL_OFFSET_SOURCE_LABELS,
    PHYSICAL_OFFSET_SOURCE_NONE,
    PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT,
    PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT,
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND,
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_FULL_FIT_NO_COMPATIBLE_ANCHOR,
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST,
    PHYSICAL_RUNTIME_FIT_SOURCE_FULL_FIT,
    PHYSICAL_RUNTIME_FIT_SOURCE_LABELS,
    PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE,
    PhysicalCenterFallbackPreflight,
    PhysicalCenterResult,
)
from .pick_table import CoarsePickTable
from .progress import build_progress_reporter
from .runtime_diagnostics import PhysicalRuntimeDiagnostics
from .trend import TrendResult

_PHYSICAL_CENTER_PRIVATE_COMPAT = (
    _append_index,
    _assign_fit_context_fallback,
    _assign_fit_context_work_item_outcome,
    _assign_model_diagnostics,
    _assign_model_diagnostics_batch,
    _assign_model_prediction,
    _assign_model_prediction_batch,
    _assign_nearest_anchor_reuse_predictions,
    _assign_prepared_model_prediction_batch,
    _anchor_model_key,
    _apply_anchor_selection_diagnostics,
    _bin_representative_position,
    build_anchor_source_xy_context,
    _build_fit_context_work_items,
    _build_anchor_reuse_group_plan,
    _build_gap_segment_context,
    _build_observation_plan,
    _build_side_observation_context,
    _cache_entry_from_fit_task_result,
    _concat_group_traces,
    _compatible_anchor_contexts_for_reuse,
    _diagnostics_for_plan,
    _evenly_spaced_positions,
    _FitContextWorkItem,
    _FitContextWorkResult,
    _filtered_obs_key,
    _fit_cache_key,
    _fit_and_assign_context_work_item,
    _fit_and_assign_context_work_items,
    _fit_and_assign_context_work_items_parallel,
    _fit_key_for_obs,
    _fallback_no_compatible_anchor,
    _fit_min_pts,
    _fit_model_for_plan,
    _fit_progress_fields,
    _fit_strategy_model,
    _fit_task_cfg_values,
    _fit_task_from_work_item,
    _FitCacheEntry,
    _FitTask,
    _FitTaskCfgValues,
    _FitTaskResult,
    fit_anchor_models,
    _GapSegmentContext,
    _gap_context_key,
    _GroupObservationContext,
    _indices_key,
    _index_key_contains,
    _limit_selected_positions,
    _median_time_position,
    _nearest_anchor_reuse_context,
    _obs_key_or_build,
    _obs_with_target_gap_segment,
    _obs_with_target_labeled_side,
    _obs_with_target_side,
    _obs_with_target_signed_offset_side,
    _offset_spread_failure_reason,
    _predict_model_array_sec,
    _predict_model_sec,
    _prepare_fit_context_assignments_for_trace_indices,
    _prepare_trace_plan_assignment,
    _record_cached_context_hits,
    _record_new_fit_task_diagnostics,
    _run_fit_task,
    _run_fit_tasks_with_executor,
    _run_no_compatible_full_fit_fallback,
    _sample_observation_indices_for_fit,
    _select_group_ids,
    _set_fit_worker_torch_num_threads,
    _SideObservationContext,
    _side_slot,
    _signed_offset_side_labels,
    _SignedOffsetSideLabels,
    _stable_unique,
    _stable_observation_seed,
    _strategy_from_fit_task_cfg,
    _shift_for_trace_indices,
    _status_for_plan_batch,
    _trace_position_map,
    _TraceFitResult,
    _TracePlanAssignment,
    CoarseGeometry,
    SourceGroup,
    signed_offset_side_from_geometry,
    split_offset_gap_segments,
)

__all__ = [
    'PHYSICAL_MODEL_FAILURE_FIT_FAILED',
    'PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID',
    'PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS',
    'PHYSICAL_MODEL_FAILURE_LABELS',
    'PHYSICAL_MODEL_FAILURE_NONE',
    'PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED',
    'PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID',
    'PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND',
    'PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP',
    'PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT',
    'PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST',
    'PHYSICAL_MODEL_STATUS_FIT_FAILED',
    'PHYSICAL_MODEL_STATUS_GEOMETRY_INVALID',
    'PHYSICAL_MODEL_STATUS_INSUFFICIENT_OBSERVATIONS',
    'PHYSICAL_MODEL_STATUS_LABELS',
    'PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED',
    'PHYSICAL_MODEL_STATUS_TWO_PIECE_OK',
    'PHYSICAL_OFFSET_SOURCE_GEOMETRY',
    'PHYSICAL_OFFSET_SOURCE_HEADER',
    'PHYSICAL_OFFSET_SOURCE_LABELS',
    'PHYSICAL_OFFSET_SOURCE_NONE',
    'PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT',
    'PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT',
    'PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND',
    'PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_FULL_FIT_NO_COMPATIBLE_ANCHOR',
    'PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST',
    'PHYSICAL_RUNTIME_FIT_SOURCE_FULL_FIT',
    'PHYSICAL_RUNTIME_FIT_SOURCE_LABELS',
    'PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE',
    'PhysicalCenterFallbackPreflight',
    'PhysicalCenterResult',
    'build_geometry_two_piece_physical_center',
    'preflight_geometry_two_piece_fallback',
]


@dataclass(frozen=True)
class _ReuseShiftStats:
    t0_shift_sec: float
    shift_valid: bool
    valid_count: int
    resid_p50_ms: float
    resid_p90_ms: float


def _validate_table(table: CoarsePickTable) -> None:
    if int(table.n_traces) <= 0:
        msg = 'table.n_traces must be positive'
        raise ValueError(msg)
    if int(table.n_samples_orig) <= 0:
        msg = 'table.n_samples_orig must be positive'
        raise ValueError(msg)
    dt = float(table.dt_scalar_sec)
    if (not np.isfinite(dt)) or dt <= 0.0:
        msg = 'table.dt_scalar_sec must be finite and > 0'
        raise ValueError(msg)


def _compute_physical_prefilter_mask(
    *,
    offset_abs_m: np.ndarray,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    cfg: PhysicsLiteConfig,
) -> np.ndarray:
    n = int(table.n_traces)
    offset_abs_arr = _as_vector(
        'offset_abs_m',
        offset_abs_m,
        n_traces=n,
        dtype=np.float32,
    )
    pick_t_sec = _as_vector(
        'table.coarse_pick_t_sec',
        table.coarse_pick_t_sec,
        n_traces=n,
        dtype=np.float32,
    )
    finite_mask = np.isfinite(offset_abs_arr) & np.isfinite(pick_t_sec)

    physical_mask = np.zeros((n,), dtype=np.bool_)
    if bool(cfg.physical_prefilter.enabled):
        finite_indices = np.flatnonzero(finite_mask)
        if finite_indices.size > 0:
            physical_feasible = compute_velocity_t0_band_from_arrays(
                offset_m=offset_abs_arr[finite_indices],
                pick_t_sec=pick_t_sec[finite_indices],
                vmin_m_s=float(cfg.physical_prefilter.vmin_m_s),
                vmax_m_s=float(cfg.physical_prefilter.vmax_m_s),
                t0_lo_ms=float(cfg.physical_prefilter.t0_lo_ms),
                t0_hi_ms=float(cfg.physical_prefilter.t0_hi_ms),
            )
            physical_mask[finite_indices] = np.asarray(
                physical_feasible.feasible_mask,
                dtype=np.bool_,
            )
    else:
        physical_mask[finite_mask] = True

    pmax = _as_vector(
        'table.coarse_pmax',
        table.coarse_pmax,
        n_traces=n,
        dtype=np.float32,
    )
    valid = (
        physical_mask
        & np.isfinite(pick_t_sec)
        & np.isfinite(pmax)
        & (pmax >= np.float32(cfg.physical_prefilter.pmax_min))
    )
    if bool(cfg.physical_prefilter.use_existing_feasible_mask):
        valid &= _as_bool_vector(
            'feasible.feasible_mask',
            feasible.feasible_mask,
            n_traces=n,
        )
    return valid.astype(np.bool_, copy=False)


def _compute_t0_shift_physical_mask(
    *,
    trace_indices: np.ndarray,
    offset_abs_m: np.ndarray,
    pick_t_sec: np.ndarray,
    feasible: FeasibleBandResult,
    cfg: PhysicsLiteConfig,
) -> np.ndarray:
    indices = np.asarray(trace_indices, dtype=np.int64)
    offsets = np.asarray(offset_abs_m, dtype=np.float32)[indices]
    picks = np.asarray(pick_t_sec, dtype=np.float32)[indices]
    finite = np.isfinite(offsets) & np.isfinite(picks)
    physical = np.zeros((indices.size,), dtype=np.bool_)
    if bool(cfg.physical_prefilter.enabled):
        finite_pos = np.flatnonzero(finite)
        if finite_pos.size > 0:
            physical_feasible = compute_velocity_t0_band_from_arrays(
                offset_m=offsets[finite_pos],
                pick_t_sec=picks[finite_pos],
                vmin_m_s=float(cfg.physical_prefilter.vmin_m_s),
                vmax_m_s=float(cfg.physical_prefilter.vmax_m_s),
                t0_lo_ms=float(cfg.physical_prefilter.t0_lo_ms),
                t0_hi_ms=float(cfg.physical_prefilter.t0_hi_ms),
            )
            physical[finite_pos] = np.asarray(
                physical_feasible.feasible_mask,
                dtype=np.bool_,
            )
    else:
        physical[finite] = True

    if bool(cfg.physical_prefilter.use_existing_feasible_mask):
        feasible_mask = _as_bool_vector(
            'feasible.feasible_mask',
            feasible.feasible_mask,
            n_traces=int(np.asarray(offset_abs_m).shape[0]),
        )
        physical &= feasible_mask[indices]
    return physical.astype(np.bool_, copy=False)


def _compute_reuse_t0_shift_stats(
    *,
    trend_model,
    trace_indices: np.ndarray,
    offset_abs_m: np.ndarray,
    pick_t_sec: np.ndarray,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    cfg: PhysicsLiteConfig,
) -> _ReuseShiftStats:
    indices = np.asarray(trace_indices, dtype=np.int64)
    if indices.size == 0:
        return _ReuseShiftStats(
            t0_shift_sec=0.0,
            shift_valid=False,
            valid_count=0,
            resid_p50_ms=np.nan,
            resid_p90_ms=np.nan,
        )

    offsets = np.asarray(offset_abs_m, dtype=np.float32)[indices]
    picks = np.asarray(pick_t_sec, dtype=np.float32)[indices]
    valid = np.isfinite(offsets) & np.isfinite(picks)
    t0_cfg = cfg.physical_runtime.t0_shift
    if bool(t0_cfg.use_physical_prefilter_mask):
        valid &= _compute_t0_shift_physical_mask(
            trace_indices=indices,
            offset_abs_m=offset_abs_m,
            pick_t_sec=pick_t_sec,
            feasible=feasible,
            cfg=cfg,
        )
    if bool(t0_cfg.use_pmax_min):
        pmax = _as_vector(
            'table.coarse_pmax',
            table.coarse_pmax,
            n_traces=table.n_traces,
            dtype=np.float32,
        )[indices]
        valid &= np.isfinite(pmax) & (
            pmax >= np.float32(cfg.physical_prefilter.pmax_min)
        )

    try:
        pred = _predict_model_array_sec(trend_model, offsets)
    except (TypeError, ValueError, RuntimeError):
        return _ReuseShiftStats(
            t0_shift_sec=0.0,
            shift_valid=False,
            valid_count=0,
            resid_p50_ms=np.nan,
            resid_p90_ms=np.nan,
        )
    valid &= np.isfinite(pred)
    residual = picks.astype(np.float64, copy=False) - pred
    residual = residual[valid & np.isfinite(residual)]
    valid_count = int(residual.size)

    shift_valid = (
        bool(t0_cfg.enabled)
        and valid_count >= int(t0_cfg.min_valid_for_t0_shift)
    )
    if shift_valid:
        shift_sec = float(np.median(residual))
        clip_sec = float(t0_cfg.t0_shift_clip_ms) * 1.0e-3
        shift_sec = float(np.clip(shift_sec, -clip_sec, clip_sec))
    else:
        shift_sec = 0.0

    if valid_count == 0:
        resid_p50 = np.nan
        resid_p90 = np.nan
    else:
        residual_ms = np.abs(residual - float(shift_sec)) * 1000.0
        resid_p50 = float(np.percentile(residual_ms, 50.0))
        resid_p90 = float(np.percentile(residual_ms, 90.0))

    return _ReuseShiftStats(
        t0_shift_sec=shift_sec,
        shift_valid=shift_valid,
        valid_count=valid_count,
        resid_p50_ms=resid_p50,
        resid_p90_ms=resid_p90,
    )


def _assign_reuse_runtime_diagnostics(
    arrays: dict[str, np.ndarray],
    trace_indices: np.ndarray,
    stats: _ReuseShiftStats,
) -> None:
    indices = np.asarray(trace_indices, dtype=np.int64)
    if indices.size == 0:
        return
    if bool(stats.shift_valid):
        arrays['physical_runtime_t0_shift_ms'][indices] = np.float32(
            float(stats.t0_shift_sec) * 1000.0
        )
    arrays['physical_runtime_reuse_resid_p50_ms'][indices] = np.float32(
        stats.resid_p50_ms
    )
    arrays['physical_runtime_reuse_resid_p90_ms'][indices] = np.float32(
        stats.resid_p90_ms
    )
    arrays['physical_runtime_reuse_valid_count'][indices] = np.int32(
        stats.valid_count
    )


def _adaptive_refit_triggered(
    *,
    stats: _ReuseShiftStats,
    plan: _ObservationPlan,
    cfg: PhysicsLiteConfig,
    min_fit_obs: int,
) -> bool:
    adaptive = cfg.physical_runtime.adaptive_refit
    if not bool(adaptive.enabled):
        return False
    resid_trigger = (
        np.isfinite(stats.resid_p90_ms)
        and float(stats.resid_p90_ms) > float(adaptive.resid_p90_ms_gt)
    )
    shift_trigger = (
        bool(stats.shift_valid)
        and abs(float(stats.t0_shift_sec) * 1000.0)
        > float(adaptive.median_abs_shift_ms_gt)
    )
    insufficient_trigger = (
        int(stats.valid_count) < int(adaptive.min_valid_for_resid_check)
        and int(plan.obs_indices.size) >= int(min_fit_obs)
    )
    return bool(resid_trigger or shift_trigger or insufficient_trigger)


def preflight_geometry_two_piece_fallback(
    *,
    coarse_npz: Mapping[str, np.ndarray],
    table: CoarsePickTable,
    cfg: PhysicsLiteConfig,
) -> PhysicalCenterFallbackPreflight:
    geometry_context = build_physical_center_geometry_context(
        coarse_npz=coarse_npz,
        table=table,
        cfg=cfg,
        include_offsets=False,
    )
    if geometry_context.geometry_required_missing:
        return PhysicalCenterFallbackPreflight(
            status='geometry_invalid',
            reason='geometry_invalid',
            fallback_mode=str(cfg.physical_runtime.geometry_invalid_fallback),
            geometry_loaded=False,
            groups=None,
        )

    if geometry_context.source_grouping_invalid:
        return PhysicalCenterFallbackPreflight(
            status='geometry_invalid',
            reason='source_xy_degenerate',
            fallback_mode=str(cfg.physical_runtime.geometry_invalid_fallback),
            geometry_loaded=geometry_context.geometry is not None,
            groups=len(geometry_context.groups),
        )
    if len(geometry_context.groups) == 0:
        return PhysicalCenterFallbackPreflight(
            status='geometry_invalid',
            reason='source_group_empty',
            fallback_mode=str(cfg.physical_runtime.group_invalid_fallback),
            geometry_loaded=geometry_context.geometry is not None,
            groups=0,
        )
    return PhysicalCenterFallbackPreflight(
        status=None,
        reason=None,
        fallback_mode=None,
        geometry_loaded=geometry_context.geometry is not None,
        groups=len(geometry_context.groups),
    )


def build_geometry_two_piece_physical_center(
    *,
    coarse_npz: Mapping[str, np.ndarray],
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    merged: MergeResult,
    cfg: PhysicsLiteConfig,
    trend_provider: object | None = None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
    progress: object | None = None,
    progress_context: Mapping[str, object] | None = None,
) -> PhysicalCenterResult:
    inputs = PhysicalCenterInputs(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=cfg,
        trend_provider=trend_provider,
    )
    reporter = (
        progress
        if progress is not None
        else build_progress_reporter(cfg.physical_runtime.progress)
    )
    context = dict(progress_context or {})
    _validate_table(table)
    n = int(table.n_traces)
    n_samples = int(table.n_samples_orig)
    dt = float(table.dt_scalar_sec)

    _as_vector(
        'feasible.feasible_lo_sec',
        feasible.feasible_lo_sec,
        n_traces=n,
        dtype=np.float32,
    )
    _as_vector(
        'feasible.feasible_hi_sec',
        feasible.feasible_hi_sec,
        n_traces=n,
        dtype=np.float32,
    )
    _as_vector(
        'trend.trend_center_i',
        trend.trend_center_i,
        n_traces=n,
        dtype=np.int32,
    )
    _as_vector(
        'trend.trend_center_sec',
        trend.trend_center_sec,
        n_traces=n,
        dtype=np.float32,
    )
    _as_vector(
        'merged.robust_pick_i',
        merged.robust_pick_i,
        n_traces=n,
        dtype=np.int32,
    )

    if runtime_diagnostics is not None:
        runtime_diagnostics.set_traces(n)
        sampling = cfg.physical_runtime.observation_sampling
        runtime_diagnostics.set_observation_sampling(
            enabled=bool(sampling.enabled),
            method=str(sampling.method),
            max_obs_per_fit=int(sampling.max_obs_per_fit),
            n_offset_bins=int(sampling.n_offset_bins),
        )
        fit_executor = cfg.physical_runtime.fit_executor
        runtime_diagnostics.set_fit_executor(
            enabled=bool(fit_executor.enabled),
            backend=str(fit_executor.backend),
            max_workers=fit_executor.max_workers,
        )
    sampling = cfg.physical_runtime.observation_sampling
    fit_executor = cfg.physical_runtime.fit_executor
    reporter.emit(
        'physical-center.start',
        **context,
        fit_kind=str(cfg.physical_trend.fit_kind),
        fit_policy=str(cfg.physical_runtime.fit_policy),
        groups='pending',
        sampling=('on' if bool(sampling.enabled) else 'off'),
        executor=(
            f'{fit_executor.backend}:{fit_executor.max_workers}'
            if bool(fit_executor.enabled)
            else 'serial'
        ),
    )

    if not bool(cfg.physical_trend.enabled):
        result = _build_disabled_result(table, trend)
        reporter.emit(
            'physical-center.done',
            **context,
            status='disabled',
            n_traces=n,
        )
        return result

    reporter.emit('physical-center.stage_start', **context, stage='geometry_load')
    stage_start = time.perf_counter()
    with (
        runtime_diagnostics.time_block('geometry_load_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        geometry = load_physical_center_geometry(coarse_npz, n_traces=n)
    reporter.emit(
        'physical-center.stage_done',
        **context,
        stage='geometry_load',
        elapsed=time.perf_counter() - stage_start,
        geometry_loaded=geometry is not None,
    )

    use_geometry_offset = bool(cfg.physical_trend.use_geometry_offset)
    if use_geometry_offset and geometry is None:
        return _emit_fallback_all_and_done(
            status='geometry_invalid',
            reason='geometry_invalid',
            fallback_mode=str(cfg.physical_runtime.geometry_invalid_fallback),
            failure_reason=PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
            inputs=inputs,
            reporter=reporter,
            context=context,
            runtime_diagnostics=runtime_diagnostics,
        )

    offset_context = build_physical_center_offset_context(
        geometry=geometry,
        table=table,
        cfg=cfg,
    )
    offset_abs_m = offset_context.offset_abs_m
    offset_signed_m = offset_context.offset_signed_m
    offset_signed_labels = offset_context.offset_signed_labels
    offset_source = offset_context.offset_source

    reporter.emit('physical-center.stage_start', **context, stage='source_grouping')
    stage_start = time.perf_counter()
    with (
        runtime_diagnostics.time_block('source_grouping_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        source_group_build = build_physical_center_source_group_build(
            coarse_npz=coarse_npz,
            geometry=geometry,
            table=table,
            cfg=cfg,
        )
        groups = source_group_build.groups
    reporter.emit(
        'physical-center.stage_done',
        **context,
        stage='source_grouping',
        elapsed=time.perf_counter() - stage_start,
        groups=len(groups),
    )

    if source_group_build.source_grouping_invalid:
        return _emit_fallback_all_and_done(
            status='geometry_invalid',
            reason='source_xy_degenerate',
            fallback_mode=str(cfg.physical_runtime.geometry_invalid_fallback),
            failure_reason=PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
            inputs=inputs,
            reporter=reporter,
            context=context,
            runtime_diagnostics=runtime_diagnostics,
        )

    if len(groups) == 0:
        return _emit_fallback_all_and_done(
            status='geometry_invalid',
            reason='source_group_empty',
            fallback_mode=str(cfg.physical_runtime.group_invalid_fallback),
            failure_reason=PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
            inputs=inputs,
            reporter=reporter,
            context=context,
            runtime_diagnostics=runtime_diagnostics,
        )

    if runtime_diagnostics is not None:
        runtime_diagnostics.set_source_groups(len(groups))

    reporter.emit(
        'physical-center.stage_start',
        **context,
        stage='source_group_ordering',
    )
    stage_start = time.perf_counter()
    with (
        runtime_diagnostics.time_block('source_group_ordering_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        source_group_context = build_physical_center_source_group_context(
            source_group_build=source_group_build,
            n_traces=n,
        )
        groups_by_id = source_group_context.groups_by_id
        group_id_by_trace = source_group_context.group_id_by_trace
        source_groups_from_geometry = (
            source_group_context.source_groups_from_geometry
        )
    reporter.emit(
        'physical-center.stage_done',
        **context,
        stage='source_group_ordering',
        elapsed=time.perf_counter() - stage_start,
    )
    geometry_context = PhysicalCenterGeometryContext(
        geometry=geometry,
        offset_abs_m=offset_abs_m,
        offset_signed_m=offset_signed_m,
        offset_signed_labels=offset_signed_labels,
        offset_source=offset_source,
        groups=groups,
        groups_by_id=groups_by_id,
        group_id_by_trace=group_id_by_trace,
        source_grouping_invalid=source_group_build.source_grouping_invalid,
        source_groups_from_geometry=source_groups_from_geometry,
        geometry_required_missing=False,
    )

    pick_t_sec = _as_vector(
        'table.coarse_pick_t_sec',
        table.coarse_pick_t_sec,
        n_traces=n,
        dtype=np.float32,
    )
    reporter.emit(
        'physical-center.stage_start',
        **context,
        stage='valid_mask_velocity_prefilter',
    )
    stage_start = time.perf_counter()
    with (
        runtime_diagnostics.time_block('valid_mask_build_sec')
        if runtime_diagnostics is not None
        else nullcontext(), runtime_diagnostics.time_block('velocity_prefilter_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        valid_for_fit = _compute_physical_prefilter_mask(
            offset_abs_m=offset_abs_m,
            table=table,
            feasible=feasible,
            cfg=cfg,
        )
    reporter.emit(
        'physical-center.stage_done',
        **context,
        stage='valid_mask_velocity_prefilter',
        elapsed=time.perf_counter() - stage_start,
        valid=int(np.count_nonzero(valid_for_fit)),
    )

    observation_plan_cache = _ObservationPlanCache(
        offset_signed_labels=offset_signed_labels,
    )
    reporter.emit('physical-center.stage_start', **context, stage='neighbor_plan')
    stage_start = time.perf_counter()
    with (
        runtime_diagnostics.time_block('neighbor_plan_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        group_context_by_id = _build_group_observation_contexts(
            groups=groups,
            groups_by_id=groups_by_id,
            valid_for_fit=valid_for_fit,
            cfg=cfg,
            use_neighbor_context=source_groups_from_geometry,
            offset_signed_labels=offset_signed_labels,
            plan_cache=observation_plan_cache,
            runtime_diagnostics=runtime_diagnostics,
        )
    reporter.emit(
        'physical-center.stage_done',
        **context,
        stage='neighbor_plan',
        elapsed=time.perf_counter() - stage_start,
        contexts=len(group_context_by_id),
    )

    min_fit_obs = 2 * _fit_min_pts(cfg)
    build_context = PhysicalCenterBuildContext(
        geometry_context=geometry_context,
        pick_t_sec=pick_t_sec,
        valid_for_fit=valid_for_fit,
        group_context_by_id=group_context_by_id,
        observation_plan_cache=observation_plan_cache,
        min_fit_obs=min_fit_obs,
    )
    arrays = _allocate_result_arrays(table)
    pending_trend_fallback = _PendingTrendFallback()
    workspace = PhysicalCenterWorkspace(
        arrays=arrays,
        fit_cache={},
        runtime_diagnostics=runtime_diagnostics,
        pending_trend_fallback=pending_trend_fallback,
    )
    reporter.emit('physical-center.stage_start', **context, stage='anchor_selection')
    stage_start = time.perf_counter()
    with (
        runtime_diagnostics.time_block('anchor_selection_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        anchor_selection = _apply_anchor_selection_diagnostics(
            arrays,
            groups=groups,
            cfg=cfg,
            runtime_diagnostics=runtime_diagnostics,
        )
    reporter.emit(
        'physical-center.stage_done',
        **context,
        stage='anchor_selection',
        elapsed=time.perf_counter() - stage_start,
        enabled=anchor_selection is not None,
    )
    strategy = _fit_strategy(cfg)
    fit_cache = workspace.fit_cache

    if cfg.physical_runtime.fit_policy == 'anchor_source_xy':
        anchor_source_xy = build_anchor_source_xy_context(
            anchor_selection=anchor_selection,
            inputs=inputs,
            build_context=build_context,
            workspace=workspace,
            strategy=strategy,
            progress=reporter,
            progress_context=context,
            fit_model_for_plan=_fit_model_for_plan,
        )
        n_reused_predictions = 0
        fallback_full_group_ids: set[int] = set()

        for group in groups:
            group_id = int(group.group_id)
            if bool(anchor_source_xy.is_anchor_by_id.get(group_id, False)):
                continue
            group_trace_indices = np.asarray(group.trace_indices, dtype=np.int64)
            nearest_context = _nearest_anchor_reuse_context(
                group_id=group_id,
                anchor_source_xy=anchor_source_xy,
                inputs=inputs,
                runtime_diagnostics=runtime_diagnostics,
            )
            compatible_anchor_context_by_plan_key = (
                _compatible_anchor_contexts_for_reuse(
                    anchor_source_xy=anchor_source_xy,
                    nearest_context=nearest_context,
                    inputs=inputs,
                    runtime_diagnostics=runtime_diagnostics,
                )
            )
            reuse_group_plan = _build_anchor_reuse_group_plan(
                group=group,
                nearest_context=nearest_context,
                compatible_anchor_context_by_plan_key=(
                    compatible_anchor_context_by_plan_key
                ),
                inputs=inputs,
                build_context=build_context,
                workspace=workspace,
            )
            reuse_items = reuse_group_plan.reuse_items

            if reuse_group_plan.uses_fallback_full_fit:
                fallback_full_group_ids.add(group_id)
                _run_no_compatible_full_fit_fallback(
                    reuse_group_plan.fallback_full_trace_indices,
                    inputs=inputs,
                    build_context=build_context,
                    workspace=workspace,
                    strategy=strategy,
                    progress=reporter,
                    progress_context=context,
                    fit_model_for_plan=_fit_model_for_plan,
                )

            non_anchor_mode = str(cfg.physical_runtime.anchor_reuse.non_anchor_mode)
            if runtime_diagnostics is not None and reuse_items:
                runtime_diagnostics.record_reuse_contexts(len(reuse_items))
            if non_anchor_mode == 'nearest_anchor':
                n_reused_predictions += _assign_nearest_anchor_reuse_predictions(
                    reuse_items,
                    inputs=inputs,
                    build_context=build_context,
                    workspace=workspace,
                )
                continue

            stats_by_key: dict[tuple[int, int, int, bool], _ReuseShiftStats] = {}
            adaptive_refit = False
            for key, items in reuse_items.items():
                key_trace_indices = np.asarray(
                    [trace_idx for trace_idx, _plan, _context in items],
                    dtype=np.int64,
                )
                anchor_context = items[0][2]
                with (
                    runtime_diagnostics.time_block('t0_shift_sec')
                    if runtime_diagnostics is not None
                    else nullcontext()
                ):
                    stats = _compute_reuse_t0_shift_stats(
                        trend_model=anchor_context.trend_model,
                        trace_indices=key_trace_indices,
                        offset_abs_m=offset_abs_m,
                        pick_t_sec=pick_t_sec,
                        table=table,
                        feasible=feasible,
                        cfg=cfg,
                    )
                stats_by_key[key] = stats
                _assign_reuse_runtime_diagnostics(
                    arrays,
                    key_trace_indices,
                    stats,
                )
                with (
                    runtime_diagnostics.time_block('adaptive_refit_decision_sec')
                    if runtime_diagnostics is not None
                    else nullcontext()
                ):
                    triggered = _adaptive_refit_triggered(
                        stats=stats,
                        plan=items[0][1],
                        cfg=cfg,
                        min_fit_obs=min_fit_obs,
                    )
                if runtime_diagnostics is not None:
                    runtime_diagnostics.record_adaptive_refit_decision(
                        triggered=triggered
                    )
                adaptive_refit = adaptive_refit or triggered

            refit_failed = False
            if adaptive_refit:
                arrays['physical_runtime_refit_mask'][group_trace_indices] = True
                refit_assignments_by_fit, _refit_assignments = (
                    _prepare_fit_context_assignments_for_trace_indices(
                        group_trace_indices,
                        inputs=inputs,
                        build_context=build_context,
                        workspace=workspace,
                    )
                )
                refit_work_items = _build_fit_context_work_items(
                    refit_assignments_by_fit,
                    offset_abs_m=offset_abs_m,
                    pick_t_sec=pick_t_sec,
                    coarse_pmax=table.coarse_pmax,
                    runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT,
                )
                reporter.emit(
                    'physical-center.contexts_built',
                    **context,
                    phase='adaptive_refit',
                    work_items=len(refit_work_items),
                    unique_keys=len(refit_assignments_by_fit),
                )
                refit_results = _fit_and_assign_context_work_items(
                    refit_work_items,
                    inputs=inputs,
                    build_context=build_context,
                    workspace=workspace,
                    strategy=strategy,
                    progress=reporter,
                    progress_context=context,
                    fit_model_for_plan=_fit_model_for_plan,
                )
                assigned_count = sum(
                    len(result.valid_trace_indices)
                    for result in refit_results.values()
                )
                success = assigned_count > 0
                refit_failed = not success
                if runtime_diagnostics is not None:
                    runtime_diagnostics.record_adaptive_refit(success=success)
                if success:
                    continue

            fallback_mode = (
                str(cfg.physical_runtime.adaptive_refit.fallback_if_refit_fails)
                if refit_failed
                else 'nearest_anchor_plus_t0_shift'
            )
            group_shifted_count = 0
            group_shift_ms_values: list[float] = []
            group_reuse_resid_p50_values: list[float] = []
            group_reuse_resid_p90_values: list[float] = []
            for key, items in reuse_items.items():
                stats = stats_by_key[key]
                if fallback_mode == 'robust':
                    for trace_idx, _plan, _context in items:
                        _assign_robust_fallback(
                            arrays,
                            trace_idx,
                            failure_reason=PHYSICAL_MODEL_FAILURE_FIT_FAILED,
                            table=table,
                            merged=merged,
                        )
                    continue
                if fallback_mode == 'existing_trend':
                    for trace_idx, _plan, _context in items:
                        _assign_fallback(
                            arrays,
                            trace_idx,
                            failure_reason=PHYSICAL_MODEL_FAILURE_FIT_FAILED,
                            table=table,
                            feasible=feasible,
                            trend=trend,
                            trend_provider=trend_provider,
                            merged=merged,
                            pending_trend_fallback=pending_trend_fallback,
                        )
                    continue

                use_shift = (
                    fallback_mode == 'nearest_anchor_plus_t0_shift'
                    and bool(stats.shift_valid)
                )
                shift_sec = float(stats.t0_shift_sec) if use_shift else 0.0
                trace_indices = np.asarray(
                    [trace_idx for trace_idx, _plan, _context in items],
                    dtype=np.int64,
                )
                anchor_context = items[0][2]
                plan_by_trace = {
                    int(trace_idx): plan for trace_idx, plan, _context in items
                }
                valid = _assign_model_prediction_batch(
                    arrays,
                    trace_indices,
                    trend_model=anchor_context.trend_model,
                    diagnostics=anchor_context.diagnostics,
                    plan_by_trace=plan_by_trace,
                    offset_abs_m=offset_abs_m,
                    dt=dt,
                    n_samples=n_samples,
                    runtime_fit_source=(
                        PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE
                    ),
                    t0_shift_sec=shift_sec,
                    runtime_diagnostics=runtime_diagnostics,
                )
                n_reused_predictions += int(np.count_nonzero(valid))
                if use_shift:
                    group_shifted_count += int(np.count_nonzero(valid))
                for trace_idx in trace_indices[~valid].tolist():
                    _assign_fallback(
                        arrays,
                        int(trace_idx),
                        failure_reason=PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
                        table=table,
                        feasible=feasible,
                        trend=trend,
                        trend_provider=trend_provider,
                        merged=merged,
                        pending_trend_fallback=pending_trend_fallback,
                    )
                if use_shift:
                    group_shift_ms_values.append(float(stats.t0_shift_sec) * 1000.0)
                    if np.isfinite(stats.resid_p50_ms):
                        group_reuse_resid_p50_values.append(float(stats.resid_p50_ms))
                    if np.isfinite(stats.resid_p90_ms):
                        group_reuse_resid_p90_values.append(float(stats.resid_p90_ms))

            if runtime_diagnostics is not None and group_shifted_count > 0:
                resid_values = np.asarray(
                    group_reuse_resid_p90_values,
                    dtype=np.float64,
                )
                resid_p50_values = np.asarray(
                    group_reuse_resid_p50_values,
                    dtype=np.float64,
                )
                runtime_diagnostics.record_t0_shifted_group(
                    t0_shift_ms=float(np.median(group_shift_ms_values)),
                    prediction_count=group_shifted_count,
                    reuse_resid_p50_ms=(
                        float(np.median(resid_p50_values))
                        if resid_p50_values.size > 0
                        else np.nan
                    ),
                    reuse_resid_p90_ms=(
                        float(np.median(resid_values))
                        if resid_values.size > 0
                        else np.nan
                    ),
                )

        if runtime_diagnostics is not None:
            runtime_diagnostics.record_reused_predictions(n_reused_predictions)
            runtime_diagnostics.record_fallback_full_fit_no_compatible_anchor(
                len(fallback_full_group_ids)
            )
            runtime_diagnostics.set_anchor_reuse_groups(
                n_non_anchor_groups=sum(
                    1
                    for group in groups
                    if not bool(
                        anchor_source_xy.is_anchor_by_id.get(
                            int(group.group_id),
                            False,
                        )
                    )
                )
            )
            runtime_diagnostics.set_fit_call_reduction_rate_vs_full(
                full_fit_call_count_estimate=len(groups)
            )
            runtime_diagnostics.set_unique_fit_contexts(len(fit_cache))
        result = _finalize_result_with_pending_trend_fallback(
            arrays,
            pending_trend_fallback=pending_trend_fallback,
            table=table,
            feasible=feasible,
            trend=trend,
            merged=merged,
            trend_provider=trend_provider,
        )
        reporter.emit(
            'physical-center.done',
            **context,
            status='ok',
            n_traces=n,
            n_source_groups=len(groups),
            n_unique_fit_contexts=len(fit_cache),
        )
        return result

    return run_full_fit_policy(
        inputs=inputs,
        build_context=build_context,
        workspace=workspace,
        strategy=strategy,
        progress=reporter,
        progress_context=context,
        fit_model_for_plan=_fit_model_for_plan,
    )
