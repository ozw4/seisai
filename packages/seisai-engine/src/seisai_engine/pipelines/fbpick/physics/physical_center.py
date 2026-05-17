from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from .config import PhysicsLiteConfig
from .feasible import FeasibleBandResult, compute_velocity_t0_band_from_arrays
from .geometry import (
    CoarseGeometry,
    SourceGroup,
    signed_offset_side_from_geometry,
    split_offset_gap_segments,
)
from .merge import MergeResult
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
    _confidence_weights_for_obs,
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
    _model_diagnostics,
    _offset_spread_failure_reason,
    _PhysicalFitStrategy,
    _run_fit_task,
    _run_fit_tasks_with_executor,
    _sample_observation_indices_for_fit,
    _set_fit_worker_torch_num_threads,
    _stable_observation_seed,
    _strategy_from_fit_task_cfg,
    _tensor_to_numpy,
)
from .physical_center_geometry import (
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
from .physical_center_observation import (
    _build_observation_plan as _build_observation_plan_with_min_obs,
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
from .progress import NullProgressReporter, build_progress_reporter
from .runtime_diagnostics import PhysicalRuntimeDiagnostics
from .runtime_policy import (
    SourceXYAnchorSelectionResult,
    select_source_xy_stride_anchors,
)
from .trend import TrendResult

_PHYSICAL_CENTER_PRIVATE_COMPAT = (
    _append_index,
    _bin_representative_position,
    _build_gap_segment_context,
    _build_side_observation_context,
    _concat_group_traces,
    _evenly_spaced_positions,
    _filtered_obs_key,
    _fit_strategy_model,
    _FitTaskCfgValues,
    _GapSegmentContext,
    _gap_context_key,
    _indices_key,
    _index_key_contains,
    _limit_selected_positions,
    _median_time_position,
    _obs_key_or_build,
    _obs_with_target_gap_segment,
    _obs_with_target_labeled_side,
    _obs_with_target_side,
    _obs_with_target_signed_offset_side,
    _offset_spread_failure_reason,
    _run_fit_task,
    _select_group_ids,
    _set_fit_worker_torch_num_threads,
    _SideObservationContext,
    _side_slot,
    _signed_offset_side_labels,
    _SignedOffsetSideLabels,
    _stable_unique,
    _stable_observation_seed,
    _strategy_from_fit_task_cfg,
    _trace_position_map,
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
class _TraceFitResult:
    plan: _ObservationPlan | None
    trend_model: object | None
    diagnostics: tuple[float, float, float, float, float, float, float] | None
    x_obs: np.ndarray | None = None
    y_obs: np.ndarray | None = None


@dataclass(frozen=True)
class _TracePlanAssignment:
    trace_idx: int
    plan: _ObservationPlan
    fit_key: tuple[int, ...]
    obs_count_before_sampling: int


@dataclass(frozen=True)
class _FitContextWorkItem:
    fit_key: tuple[int, ...]
    fit_plan: _ObservationPlan
    obs_count_before_sampling: int
    trace_indices: np.ndarray
    runtime_fit_source: int
    assignments: tuple[_TracePlanAssignment, ...]
    x_obs: np.ndarray
    y_obs: np.ndarray
    w_obs: np.ndarray


@dataclass(frozen=True)
class _FitContextWorkResult:
    trend_model: object | None
    diagnostics: tuple[float, float, float, float, float, float, float] | None
    valid_trace_indices: frozenset[int]


@dataclass(frozen=True)
class _AnchorModelContext:
    trend_model: object
    diagnostics: tuple[float, float, float, float, float, float, float] | None


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


def _apply_anchor_selection_diagnostics(
    arrays: dict[str, np.ndarray],
    *,
    groups: tuple[SourceGroup, ...],
    cfg: PhysicsLiteConfig,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
) -> SourceXYAnchorSelectionResult | None:
    anchor_cfg = cfg.physical_runtime.anchor_selection
    if not (
        bool(anchor_cfg.enabled)
        or cfg.physical_runtime.fit_policy == 'anchor_source_xy'
    ):
        return None
    result = select_source_xy_stride_anchors(
        groups,
        anchor_stride_source_groups=int(anchor_cfg.anchor_stride_source_groups),
        include_first=bool(anchor_cfg.include_first),
        include_last=bool(anchor_cfg.include_last),
    )
    group_pos_by_id = {
        int(group_id): int(pos)
        for pos, group_id in enumerate(np.asarray(result.group_ids).tolist())
    }
    for group in groups:
        pos = group_pos_by_id[int(group.group_id)]
        trace_indices = np.asarray(group.trace_indices, dtype=np.int64)
        arrays['physical_anchor_group_id'][trace_indices] = np.int32(group.group_id)
        arrays['physical_anchor_is_anchor'][trace_indices] = np.bool_(
            result.is_anchor[pos]
        )
        arrays['physical_anchor_nearest_anchor_group_id'][trace_indices] = np.int32(
            result.nearest_anchor_group_id[pos]
        )
        arrays['physical_anchor_source_distance_m'][trace_indices] = np.float32(
            result.source_distance_m[pos]
        )
    if runtime_diagnostics is not None:
        runtime_diagnostics.set_anchor_selection(
            n_anchor_groups=len(result.anchor_group_ids),
            anchor_stride_source_groups=int(anchor_cfg.anchor_stride_source_groups),
            anchor_selection_mode=str(anchor_cfg.mode),
            source_distance_m=result.source_distance_m,
        )
    return result


def _build_observation_plan(
    *,
    trace_idx: int,
    target_group_id: int,
    group_context_by_id: Mapping[int, _GroupObservationContext],
    geometry: CoarseGeometry | None,
    offset_abs_m: np.ndarray,
    offset_signed_m: np.ndarray | None,
    cfg: PhysicsLiteConfig,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
    plan_cache: _ObservationPlanCache | None = None,
    min_fit_obs: int | None = None,
) -> _ObservationPlan | None:
    return _build_observation_plan_with_min_obs(
        trace_idx=trace_idx,
        target_group_id=target_group_id,
        group_context_by_id=group_context_by_id,
        geometry=geometry,
        offset_abs_m=offset_abs_m,
        offset_signed_m=offset_signed_m,
        cfg=cfg,
        min_fit_obs=(
            2 * _fit_min_pts(cfg) if min_fit_obs is None else int(min_fit_obs)
        ),
        runtime_diagnostics=runtime_diagnostics,
        plan_cache=plan_cache,
    )


def _predict_model_sec(trend_model, offset_m: float) -> float:
    pred = trend_model.predict(torch.tensor([float(offset_m)], dtype=torch.float32))
    pred_np = _tensor_to_numpy(pred).astype(np.float64, copy=False)
    if pred_np.shape != (1,):
        msg = f'trend model prediction must have shape (1,), got {pred_np.shape}'
        raise ValueError(msg)
    return float(pred_np[0])


def _predict_model_array_sec(trend_model, offset_m: np.ndarray) -> np.ndarray:
    offsets = np.asarray(offset_m, dtype=np.float32)
    if offsets.ndim != 1:
        msg = 'offset_m must be 1D'
        raise ValueError(msg)
    if offsets.size == 0:
        return np.zeros((0,), dtype=np.float64)
    pred = trend_model.predict(torch.as_tensor(offsets, dtype=torch.float32))
    pred_np = _tensor_to_numpy(pred).astype(np.float64, copy=False)
    if pred_np.shape != offsets.shape:
        msg = (
            'trend model prediction must have shape '
            f'{offsets.shape}, got {pred_np.shape}'
        )
        raise ValueError(msg)
    return pred_np


def _diagnostics_for_plan(
    *,
    plan: _ObservationPlan,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    cache: dict[tuple[int, ...], _FitCacheEntry],
) -> tuple[float, float, float, float, float, float, float] | None:
    cache_key = _fit_cache_key(plan)
    entry = cache[cache_key]
    if bool(entry.fit_failed) or entry.model is None:
        return None
    if bool(entry.diagnostics_computed):
        return entry.diagnostics

    try:
        diagnostics = _model_diagnostics(
            entry.model,
            obs_offsets_m=x_obs,
            obs_times_sec=y_obs,
        )
    except (TypeError, ValueError, RuntimeError):
        diagnostics = None
    cache[cache_key] = _FitCacheEntry(
        model=entry.model,
        diagnostics=diagnostics,
        fit_failed=False,
        diagnostics_computed=True,
        failure_reason=entry.failure_reason,
    )
    return diagnostics


def _assign_model_diagnostics(
    arrays: dict[str, np.ndarray],
    trace_idx: int,
    diagnostics: tuple[float, float, float, float, float, float, float] | None,
) -> None:
    _assign_model_diagnostics_batch(
        arrays,
        np.asarray([int(trace_idx)], dtype=np.int64),
        diagnostics,
    )


def _assign_model_diagnostics_batch(
    arrays: dict[str, np.ndarray],
    trace_indices: np.ndarray,
    diagnostics: tuple[float, float, float, float, float, float, float] | None,
) -> None:
    if diagnostics is None:
        return
    indices = np.asarray(trace_indices, dtype=np.int64)
    if indices.size == 0:
        return
    (
        break_offset,
        slope_near,
        slope_far,
        velocity_near,
        velocity_far,
        resid_p50,
        resid_p90,
    ) = diagnostics
    arrays['physical_model_break_offset_m'][indices] = np.float32(break_offset)
    arrays['physical_model_slope_near_s_per_m'][indices] = np.float32(slope_near)
    arrays['physical_model_slope_far_s_per_m'][indices] = np.float32(slope_far)
    arrays['physical_model_velocity_near_m_s'][indices] = np.float32(velocity_near)
    arrays['physical_model_velocity_far_m_s'][indices] = np.float32(velocity_far)
    arrays['physical_model_resid_p50_ms'][indices] = np.float32(resid_p50)
    arrays['physical_model_resid_p90_ms'][indices] = np.float32(resid_p90)


def _shift_for_trace_indices(
    t0_shift_sec: float | np.ndarray,
    *,
    trace_indices: np.ndarray,
    n_traces: int,
) -> float | np.ndarray:
    shift = np.asarray(t0_shift_sec, dtype=np.float64)
    if shift.ndim == 0:
        return float(shift)
    if shift.ndim != 1:
        msg = 't0_shift_sec must be scalar or 1D'
        raise ValueError(msg)
    indices = np.asarray(trace_indices, dtype=np.int64)
    if shift.shape == indices.shape:
        return shift
    if int(shift.shape[0]) == int(n_traces):
        return shift[indices]
    msg = (
        't0_shift_sec vector must match trace_indices or full trace count, '
        f'got {shift.shape[0]} for {indices.size} indices and {n_traces} traces'
    )
    raise ValueError(msg)


def _status_for_plan_batch(
    trace_indices: np.ndarray,
    plan_by_trace: Mapping[int, _ObservationPlan] | _ObservationPlan,
) -> np.ndarray:
    indices = np.asarray(trace_indices, dtype=np.int64)
    if isinstance(plan_by_trace, _ObservationPlan):
        status = (
            PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT
            if bool(plan_by_trace.relaxed)
            else PHYSICAL_MODEL_STATUS_TWO_PIECE_OK
        )
        return np.full((indices.size,), np.uint8(status), dtype=np.uint8)

    out = np.empty((indices.size,), dtype=np.uint8)
    for pos, trace_idx in enumerate(indices.tolist()):
        plan = plan_by_trace[int(trace_idx)]
        out[pos] = np.uint8(
            PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT
            if bool(plan.relaxed)
            else PHYSICAL_MODEL_STATUS_TWO_PIECE_OK
        )
    return out


def _assign_model_prediction_batch(
    arrays: dict[str, np.ndarray],
    trace_indices: np.ndarray,
    *,
    trend_model,
    diagnostics: tuple[float, float, float, float, float, float, float] | None,
    plan_by_trace: Mapping[int, _ObservationPlan] | _ObservationPlan,
    offset_abs_m: np.ndarray,
    dt: float,
    n_samples: int,
    runtime_fit_source: int,
    t0_shift_sec: float | np.ndarray = 0.0,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> np.ndarray:
    indices = np.asarray(trace_indices, dtype=np.int64)
    if indices.ndim != 1:
        msg = 'trace_indices must be 1D'
        raise ValueError(msg)
    if indices.size == 0:
        return np.zeros((0,), dtype=np.bool_)

    if runtime_diagnostics is not None:
        runtime_diagnostics.inc('n_prediction_calls', int(indices.size))
        runtime_diagnostics.inc('n_prediction_batches')

    offset_arr = np.asarray(offset_abs_m, dtype=np.float32)
    with (
        runtime_diagnostics.time_block('prediction_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        try:
            physical_t_sec = _predict_model_array_sec(
                trend_model,
                offset_arr[indices],
            )
            shift = _shift_for_trace_indices(
                t0_shift_sec,
                trace_indices=indices,
                n_traces=int(offset_arr.shape[0]),
            )
            physical_t_sec = physical_t_sec + shift
        except (TypeError, ValueError, RuntimeError):
            physical_t_sec = np.full((indices.size,), np.nan, dtype=np.float64)

    valid = np.isfinite(physical_t_sec)
    if not bool(np.any(valid)):
        return valid.astype(np.bool_, copy=False)

    with (
        runtime_diagnostics.time_block('assignment_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        valid_indices = indices[valid]
        center_i = np.rint(physical_t_sec[valid] / float(dt)).astype(np.int64)
        center_i = np.clip(center_i, 0, int(n_samples) - 1).astype(np.int32)
        arrays['physical_center_i'][valid_indices] = center_i
        arrays['physical_center_t_sec'][valid_indices] = (
            center_i.astype(np.float64) * float(dt)
        ).astype(np.float32)
        arrays['physical_model_status'][valid_indices] = _status_for_plan_batch(
            valid_indices,
            plan_by_trace,
        )
        arrays['physical_model_failure_reason'][valid_indices] = np.uint8(
            PHYSICAL_MODEL_FAILURE_NONE
        )
        arrays['physical_runtime_fit_source'][valid_indices] = np.uint8(
            runtime_fit_source
        )
        _assign_model_diagnostics_batch(arrays, valid_indices, diagnostics)
    return valid.astype(np.bool_, copy=False)


def _assign_model_prediction(
    arrays: dict[str, np.ndarray],
    trace_idx: int,
    *,
    trend_model,
    diagnostics: tuple[float, float, float, float, float, float, float] | None,
    plan: _ObservationPlan,
    offset_abs_m: np.ndarray,
    dt: float,
    n_samples: int,
    runtime_fit_source: int,
    t0_shift_sec: float = 0.0,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> bool:
    valid = _assign_model_prediction_batch(
        arrays,
        np.asarray([int(trace_idx)], dtype=np.int64),
        trend_model=trend_model,
        diagnostics=diagnostics,
        plan_by_trace=plan,
        offset_abs_m=offset_abs_m,
        dt=dt,
        n_samples=n_samples,
        runtime_fit_source=runtime_fit_source,
        t0_shift_sec=t0_shift_sec,
        runtime_diagnostics=runtime_diagnostics,
    )
    return bool(valid[0])


def _prepare_trace_plan_assignment(
    *,
    arrays: dict[str, np.ndarray],
    trace_idx: int,
    group_id_by_trace: np.ndarray,
    group_context_by_id: Mapping[int, _GroupObservationContext],
    geometry: CoarseGeometry | None,
    offset_abs_m: np.ndarray,
    offset_signed_m: np.ndarray | None,
    offset_source: int,
    pick_t_sec: np.ndarray,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    trend_provider: object | None = None,
    merged: MergeResult,
    cfg: PhysicsLiteConfig,
    observation_plan_cache: _ObservationPlanCache,
    min_fit_obs: int,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
    pending_trend_fallback: _PendingTrendFallback | None = None,
) -> _TracePlanAssignment | None:
    arrays['physical_offset_source'][trace_idx] = np.uint8(offset_source)
    if (
        not np.isfinite(offset_abs_m[trace_idx])
        or int(group_id_by_trace[trace_idx]) < 0
    ):
        _assign_fallback(
            arrays,
            trace_idx,
            failure_reason=PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            pending_trend_fallback=pending_trend_fallback,
        )
        return None

    plan = _build_observation_plan(
        trace_idx=trace_idx,
        target_group_id=int(group_id_by_trace[trace_idx]),
        group_context_by_id=group_context_by_id,
        geometry=geometry,
        offset_abs_m=offset_abs_m,
        offset_signed_m=offset_signed_m,
        cfg=cfg,
        runtime_diagnostics=runtime_diagnostics,
        plan_cache=observation_plan_cache,
    )
    if plan is None:
        _assign_fallback(
            arrays,
            trace_idx,
            failure_reason=PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            pending_trend_fallback=pending_trend_fallback,
        )
        return None

    arrays['physical_model_neighbor_count'][trace_idx] = np.int32(plan.neighbor_count)
    arrays['physical_prefilter_valid_count'][trace_idx] = np.int32(
        plan.prefilter_valid_count
    )
    arrays['physical_model_segment_id'][trace_idx] = np.int32(plan.segment_id)
    arrays['physical_model_side'][trace_idx] = np.int8(plan.side)

    if int(plan.obs_indices.size) < int(min_fit_obs):
        _assign_fallback(
            arrays,
            trace_idx,
            failure_reason=PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            pending_trend_fallback=pending_trend_fallback,
        )
        return None

    obs_count_before_sampling = int(np.asarray(plan.obs_indices).size)
    with (
        runtime_diagnostics.time_block('observation_sampling_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        obs_indices = _sample_observation_indices_for_fit(
            obs_indices=plan.obs_indices,
            offset_abs_m=offset_abs_m,
            pick_t_sec=pick_t_sec,
            coarse_pmax=table.coarse_pmax,
            cfg=cfg,
            min_required_obs=int(min_fit_obs),
        )
    sampling_changed = obs_indices is not plan.obs_indices
    precomputed_fit_key = None if sampling_changed else plan.obs_key
    fit_key = _fit_key_for_obs(
        obs_indices,
        precomputed_key=precomputed_fit_key,
        runtime_diagnostics=runtime_diagnostics,
        after_sampling=sampling_changed,
        count_missing_precomputed=not sampling_changed,
    )
    fit_plan = _ObservationPlan(
        obs_indices=obs_indices,
        obs_key=fit_key,
        neighbor_count=plan.neighbor_count,
        prefilter_valid_count=plan.prefilter_valid_count,
        segment_id=plan.segment_id,
        side=plan.side,
        relaxed=plan.relaxed,
    )
    return _TracePlanAssignment(
        trace_idx=int(trace_idx),
        plan=fit_plan,
        fit_key=fit_key,
        obs_count_before_sampling=obs_count_before_sampling,
    )


def _assign_prepared_model_prediction_batch(
    *,
    arrays: dict[str, np.ndarray],
    items: list[tuple[int, _TraceFitResult]],
    offset_abs_m: np.ndarray,
    dt: float,
    n_samples: int,
    runtime_fit_source: int,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    trend_provider: object | None = None,
    merged: MergeResult,
    fit_cache: dict[tuple[int, ...], _FitCacheEntry],
    t0_shift_sec: float | np.ndarray = 0.0,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
    pending_trend_fallback: _PendingTrendFallback | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float, float, float, float] | None]:
    if not items:
        return np.zeros((0,), dtype=np.bool_), None

    trace_indices = np.asarray(
        [trace_idx for trace_idx, _result in items], dtype=np.int64
    )
    first_result = items[0][1]
    if first_result.plan is None or first_result.trend_model is None:
        msg = 'prepared model assignment requires plan and trend_model'
        raise ValueError(msg)

    plan_by_trace = {
        int(trace_idx): result.plan
        for trace_idx, result in items
        if result.plan is not None
    }
    diagnostics = first_result.diagnostics
    valid = _assign_model_prediction_batch(
        arrays,
        trace_indices,
        trend_model=first_result.trend_model,
        diagnostics=diagnostics,
        plan_by_trace=plan_by_trace,
        offset_abs_m=offset_abs_m,
        dt=dt,
        n_samples=n_samples,
        runtime_fit_source=runtime_fit_source,
        t0_shift_sec=t0_shift_sec,
        runtime_diagnostics=runtime_diagnostics,
    )

    if bool(np.any(valid)) and diagnostics is None:
        first_valid_pos = int(np.flatnonzero(valid)[0])
        first_valid_result = items[first_valid_pos][1]
        if (
            first_valid_result.plan is not None
            and first_valid_result.x_obs is not None
            and first_valid_result.y_obs is not None
        ):
            diagnostics = _diagnostics_for_plan(
                plan=first_valid_result.plan,
                x_obs=first_valid_result.x_obs,
                y_obs=first_valid_result.y_obs,
                cache=fit_cache,
            )
            _assign_model_diagnostics_batch(
                arrays,
                trace_indices[valid],
                diagnostics,
            )

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
    return valid, diagnostics


def _build_fit_context_work_items(
    grouped_assignments: Mapping[tuple[int, ...], list[_TracePlanAssignment]],
    *,
    offset_abs_m: np.ndarray,
    pick_t_sec: np.ndarray,
    coarse_pmax: np.ndarray,
    runtime_fit_source: int,
) -> list[_FitContextWorkItem]:
    work_items: list[_FitContextWorkItem] = []
    for fit_key, assignments in grouped_assignments.items():
        if not assignments:
            continue
        first = assignments[0]
        trace_indices = np.asarray(
            [item.trace_idx for item in assignments],
            dtype=np.int64,
        )
        obs_indices = np.asarray(first.plan.obs_indices, dtype=np.int64)
        work_items.append(
            _FitContextWorkItem(
                fit_key=fit_key,
                fit_plan=first.plan,
                obs_count_before_sampling=first.obs_count_before_sampling,
                trace_indices=trace_indices,
                runtime_fit_source=int(runtime_fit_source),
                assignments=tuple(assignments),
                x_obs=np.asarray(offset_abs_m[obs_indices], dtype=np.float32),
                y_obs=np.asarray(pick_t_sec[obs_indices], dtype=np.float32),
                w_obs=_confidence_weights_for_obs(
                    np.asarray(coarse_pmax, dtype=np.float32)[obs_indices]
                ),
            )
        )
    return work_items


def _assign_fit_context_fallback(
    *,
    arrays: dict[str, np.ndarray],
    work_item: _FitContextWorkItem,
    failure_reason: int,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    trend_provider: object | None = None,
    merged: MergeResult,
    pending_trend_fallback: _PendingTrendFallback | None = None,
) -> None:
    for trace_idx in np.asarray(work_item.trace_indices, dtype=np.int64).tolist():
        _assign_fallback(
            arrays,
            int(trace_idx),
            failure_reason=int(failure_reason),
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            pending_trend_fallback=pending_trend_fallback,
        )


def _fit_and_assign_context_work_item(
    *,
    arrays: dict[str, np.ndarray],
    work_item: _FitContextWorkItem,
    offset_abs_m: np.ndarray,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    trend_provider: object | None = None,
    merged: MergeResult,
    cfg: PhysicsLiteConfig,
    strategy: _PhysicalFitStrategy,
    fit_cache: dict[tuple[int, ...], _FitCacheEntry],
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
    pending_trend_fallback: _PendingTrendFallback | None = None,
) -> _FitContextWorkResult:
    trend_model, diagnostics, fit_failure_reason = _fit_model_for_plan(
        strategy=strategy,
        plan=work_item.fit_plan,
        x_obs=work_item.x_obs,
        y_obs=work_item.y_obs,
        w_obs=work_item.w_obs,
        min_pts=_fit_min_pts(cfg),
        min_offset_spread_m=float(cfg.physical_trend.min_offset_spread_m),
        cache=fit_cache,
        runtime_diagnostics=runtime_diagnostics,
        obs_count_before_sampling=int(work_item.obs_count_before_sampling),
    )
    if (
        runtime_diagnostics is not None
        and work_item.fit_key in fit_cache
        and int(work_item.trace_indices.size) > 1
    ):
        runtime_diagnostics.record_cache_hit(int(work_item.trace_indices.size) - 1)

    return _assign_fit_context_work_item_outcome(
        arrays=arrays,
        work_item=work_item,
        offset_abs_m=offset_abs_m,
        table=table,
        feasible=feasible,
        trend=trend,
        trend_provider=trend_provider,
        merged=merged,
        fit_cache=fit_cache,
        trend_model=trend_model,
        diagnostics=diagnostics,
        fit_failure_reason=fit_failure_reason,
        runtime_diagnostics=runtime_diagnostics,
        pending_trend_fallback=pending_trend_fallback,
    )


def _assign_fit_context_work_item_outcome(
    *,
    arrays: dict[str, np.ndarray],
    work_item: _FitContextWorkItem,
    offset_abs_m: np.ndarray,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    trend_provider: object | None = None,
    merged: MergeResult,
    fit_cache: dict[tuple[int, ...], _FitCacheEntry],
    trend_model: object | None,
    diagnostics: tuple[float, float, float, float, float, float, float] | None,
    fit_failure_reason: int | None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
    pending_trend_fallback: _PendingTrendFallback | None = None,
) -> _FitContextWorkResult:
    if fit_failure_reason is not None:
        _assign_fit_context_fallback(
            arrays=arrays,
            work_item=work_item,
            failure_reason=fit_failure_reason,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            pending_trend_fallback=pending_trend_fallback,
        )
        return _FitContextWorkResult(
            trend_model=None,
            diagnostics=None,
            valid_trace_indices=frozenset(),
        )

    if trend_model is None:
        _assign_fit_context_fallback(
            arrays=arrays,
            work_item=work_item,
            failure_reason=PHYSICAL_MODEL_FAILURE_FIT_FAILED,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            pending_trend_fallback=pending_trend_fallback,
        )
        return _FitContextWorkResult(
            trend_model=None,
            diagnostics=None,
            valid_trace_indices=frozenset(),
        )

    items = [
        (
            int(item.trace_idx),
            _TraceFitResult(
                plan=item.plan,
                trend_model=trend_model,
                diagnostics=diagnostics,
                x_obs=work_item.x_obs,
                y_obs=work_item.y_obs,
            ),
        )
        for item in work_item.assignments
    ]
    valid, diagnostics = _assign_prepared_model_prediction_batch(
        arrays=arrays,
        items=items,
        offset_abs_m=offset_abs_m,
        dt=float(table.dt_scalar_sec),
        n_samples=int(table.n_samples_orig),
        runtime_fit_source=int(work_item.runtime_fit_source),
        table=table,
        feasible=feasible,
        trend=trend,
        trend_provider=trend_provider,
        merged=merged,
        fit_cache=fit_cache,
        runtime_diagnostics=runtime_diagnostics,
        pending_trend_fallback=pending_trend_fallback,
    )
    return _FitContextWorkResult(
        trend_model=trend_model,
        diagnostics=diagnostics,
        valid_trace_indices=frozenset(
            int(trace_idx)
            for trace_idx in np.asarray(
                work_item.trace_indices,
                dtype=np.int64,
            )[valid].tolist()
        ),
    )


def _prepare_fit_context_assignments_for_trace_indices(
    trace_indices: np.ndarray,
    *,
    arrays: dict[str, np.ndarray],
    group_id_by_trace: np.ndarray,
    group_context_by_id: Mapping[int, _GroupObservationContext],
    geometry: CoarseGeometry | None,
    offset_abs_m: np.ndarray,
    offset_signed_m: np.ndarray | None,
    offset_source: int,
    pick_t_sec: np.ndarray,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    trend_provider: object | None = None,
    merged: MergeResult,
    cfg: PhysicsLiteConfig,
    observation_plan_cache: _ObservationPlanCache,
    min_fit_obs: int,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
    pending_trend_fallback: _PendingTrendFallback | None = None,
) -> tuple[
    dict[tuple[int, ...], list[_TracePlanAssignment]],
    list[_TracePlanAssignment],
]:
    grouped: dict[tuple[int, ...], list[_TracePlanAssignment]] = {}
    ordered: list[_TracePlanAssignment] = []
    for trace_idx in np.asarray(trace_indices, dtype=np.int64).tolist():
        assignment = _prepare_trace_plan_assignment(
            arrays=arrays,
            trace_idx=int(trace_idx),
            group_id_by_trace=group_id_by_trace,
            group_context_by_id=group_context_by_id,
            geometry=geometry,
            offset_abs_m=offset_abs_m,
            offset_signed_m=offset_signed_m,
            offset_source=offset_source,
            pick_t_sec=pick_t_sec,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            cfg=cfg,
            observation_plan_cache=observation_plan_cache,
            min_fit_obs=min_fit_obs,
            runtime_diagnostics=runtime_diagnostics,
            pending_trend_fallback=pending_trend_fallback,
        )
        if assignment is None:
            continue
        grouped.setdefault(assignment.fit_key, []).append(assignment)
        ordered.append(assignment)
    return grouped, ordered


def _fit_and_assign_context_work_items(
    work_items: list[_FitContextWorkItem],
    *,
    arrays: dict[str, np.ndarray],
    offset_abs_m: np.ndarray,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    trend_provider: object | None = None,
    merged: MergeResult,
    cfg: PhysicsLiteConfig,
    strategy: _PhysicalFitStrategy,
    fit_cache: dict[tuple[int, ...], _FitCacheEntry],
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
    progress: object | None = None,
    progress_context: Mapping[str, object] | None = None,
    pending_trend_fallback: _PendingTrendFallback | None = None,
) -> dict[tuple[int, ...], _FitContextWorkResult]:
    reporter = progress if progress is not None else NullProgressReporter()
    context = dict(progress_context or {})
    fit_start = time.perf_counter()
    if bool(cfg.physical_runtime.fit_executor.enabled) and work_items:
        return _fit_and_assign_context_work_items_parallel(
            work_items,
            arrays=arrays,
            offset_abs_m=offset_abs_m,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            cfg=cfg,
            fit_cache=fit_cache,
            runtime_diagnostics=runtime_diagnostics,
            progress=reporter,
            progress_context=context,
            progress_start_sec=fit_start,
            pending_trend_fallback=pending_trend_fallback,
        )

    results: dict[tuple[int, ...], _FitContextWorkResult] = {}
    total = len(work_items)
    reporter.emit(
        'physical-center.fit_start',
        **context,
        work_items=total,
        executor='serial',
    )
    for done, work_item in enumerate(work_items, start=1):
        results[work_item.fit_key] = _fit_and_assign_context_work_item(
            arrays=arrays,
            work_item=work_item,
            offset_abs_m=offset_abs_m,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            cfg=cfg,
            strategy=strategy,
            fit_cache=fit_cache,
            runtime_diagnostics=runtime_diagnostics,
            pending_trend_fallback=pending_trend_fallback,
        )
        reporter.emit(
            'fit.progress',
            **context,
            **_fit_progress_fields(
                done=done,
                total=total,
                start_sec=fit_start,
                runtime_diagnostics=runtime_diagnostics,
                force=done >= total,
            ),
        )
    return results


def _record_cached_context_hits(
    *,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
    work_item: _FitContextWorkItem,
) -> None:
    if runtime_diagnostics is None:
        return
    extra_hits = int(work_item.trace_indices.size) - 1
    if extra_hits > 0:
        runtime_diagnostics.record_cache_hit(extra_hits)


def _record_new_fit_task_diagnostics(
    *,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
    work_item: _FitContextWorkItem,
    task_result: _FitTaskResult,
) -> None:
    if runtime_diagnostics is None:
        return
    if task_result.failure_reason == PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS:
        return
    runtime_diagnostics.record_cache_miss()
    if bool(task_result.fit_attempted):
        runtime_diagnostics.record_ransac_fit(
            elapsed_sec=float(task_result.elapsed_sec),
            obs_count=int(task_result.obs_count),
            obs_count_before=int(task_result.obs_count_before_sampling),
        )
    extra_hits = int(work_item.trace_indices.size) - 1
    if extra_hits > 0:
        runtime_diagnostics.record_cache_hit(extra_hits)


def _cache_entry_from_fit_task_result(
    task_result: _FitTaskResult,
) -> _FitCacheEntry | None:
    if task_result.failure_reason == PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS:
        return None
    if task_result.failure_reason is not None:
        return _FitCacheEntry(
            model=None,
            diagnostics=None,
            fit_failed=bool(task_result.fit_failed),
            failure_reason=task_result.failure_reason,
        )
    if bool(task_result.fit_failed):
        return _FitCacheEntry(
            model=None,
            diagnostics=None,
            fit_failed=True,
            failure_reason=PHYSICAL_MODEL_FAILURE_FIT_FAILED,
        )
    return _FitCacheEntry(
        model=task_result.trend_model,
        diagnostics=task_result.diagnostics,
        fit_failed=False,
        diagnostics_computed=task_result.diagnostics is not None,
    )


def _fit_and_assign_context_work_items_parallel(
    work_items: list[_FitContextWorkItem],
    *,
    arrays: dict[str, np.ndarray],
    offset_abs_m: np.ndarray,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    trend_provider: object | None = None,
    merged: MergeResult,
    cfg: PhysicsLiteConfig,
    fit_cache: dict[tuple[int, ...], _FitCacheEntry],
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
    progress: object | None = None,
    progress_context: Mapping[str, object] | None = None,
    progress_start_sec: float | None = None,
    pending_trend_fallback: _PendingTrendFallback | None = None,
) -> dict[tuple[int, ...], _FitContextWorkResult]:
    reporter = progress if progress is not None else NullProgressReporter()
    context = dict(progress_context or {})
    fit_start = (
        time.perf_counter()
        if progress_start_sec is None
        else progress_start_sec
    )
    total = len(work_items)
    done = 0
    reporter.emit(
        'physical-center.fit_start',
        **context,
        work_items=total,
        executor=str(cfg.physical_runtime.fit_executor.backend),
        max_workers=cfg.physical_runtime.fit_executor.max_workers,
    )
    results: dict[tuple[int, ...], _FitContextWorkResult] = {}
    pending_items: list[_FitContextWorkItem] = []
    tasks: list[_FitTask] = []
    cfg_values = _fit_task_cfg_values(cfg)

    for work_item in work_items:
        entry = fit_cache.get(work_item.fit_key)
        if entry is None:
            pending_items.append(work_item)
            tasks.append(
                _fit_task_from_work_item(work_item, cfg_values=cfg_values)
            )
            continue

        _record_cached_context_hits(
            runtime_diagnostics=runtime_diagnostics,
            work_item=work_item,
        )
        fit_failure_reason = entry.failure_reason
        if fit_failure_reason is None and bool(entry.fit_failed):
            fit_failure_reason = PHYSICAL_MODEL_FAILURE_FIT_FAILED
        results[work_item.fit_key] = _assign_fit_context_work_item_outcome(
            arrays=arrays,
            work_item=work_item,
            offset_abs_m=offset_abs_m,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            fit_cache=fit_cache,
            trend_model=entry.model,
            diagnostics=entry.diagnostics,
            fit_failure_reason=fit_failure_reason,
            runtime_diagnostics=runtime_diagnostics,
            pending_trend_fallback=pending_trend_fallback,
        )
        done += 1
        reporter.emit(
            'fit.progress',
            **context,
            **_fit_progress_fields(
                done=done,
                total=total,
                start_sec=fit_start,
                runtime_diagnostics=runtime_diagnostics,
                force=done >= total,
            ),
        )

    if tasks:
        start = time.perf_counter()
        task_results_by_key = _run_fit_tasks_with_executor(
            tasks,
            cfg=cfg,
            progress=reporter,
            progress_context=context,
            progress_start_done=done,
            progress_total=total,
            progress_start_sec=fit_start,
            runtime_diagnostics=runtime_diagnostics,
            progress_cache_miss_base=(
                int(runtime_diagnostics.n_cache_misses)
                if runtime_diagnostics is not None
                else 0
            ),
            progress_fit_calls_base=(
                int(runtime_diagnostics.n_fit_calls)
                if runtime_diagnostics is not None
                else 0
            ),
            progress_fit_total_sec_base=(
                float(runtime_diagnostics.ransac_fit_total_sec)
                if runtime_diagnostics is not None
                else 0.0
            ),
        )
        wall_sec = time.perf_counter() - start
        if runtime_diagnostics is not None:
            runtime_diagnostics.record_fit_executor_run(
                wall_sec=wall_sec,
                tasks=len(tasks),
            )

        for work_item in pending_items:
            task_result = task_results_by_key[work_item.fit_key]
            entry = _cache_entry_from_fit_task_result(task_result)
            if entry is not None:
                fit_cache[work_item.fit_key] = entry
            _record_new_fit_task_diagnostics(
                runtime_diagnostics=runtime_diagnostics,
                work_item=work_item,
                task_result=task_result,
            )
            done += 1
            results[work_item.fit_key] = _assign_fit_context_work_item_outcome(
                arrays=arrays,
                work_item=work_item,
                offset_abs_m=offset_abs_m,
                table=table,
                feasible=feasible,
                trend=trend,
                trend_provider=trend_provider,
                merged=merged,
                fit_cache=fit_cache,
                trend_model=task_result.trend_model,
                diagnostics=task_result.diagnostics,
                fit_failure_reason=task_result.failure_reason,
                runtime_diagnostics=runtime_diagnostics,
                pending_trend_fallback=pending_trend_fallback,
            )

    return results


def _anchor_model_key(
    group_id: int,
    plan: _ObservationPlan,
) -> tuple[int, int, int, bool]:
    return (int(group_id), int(plan.side), int(plan.segment_id), bool(plan.relaxed))


def _selection_group_maps(
    selection: SourceXYAnchorSelectionResult,
) -> tuple[dict[int, bool], dict[int, int], dict[int, float]]:
    is_anchor_by_id: dict[int, bool] = {}
    nearest_by_id: dict[int, int] = {}
    distance_by_id: dict[int, float] = {}
    group_ids = np.asarray(selection.group_ids, dtype=np.int64)
    for pos, group_id in enumerate(group_ids.tolist()):
        gid = int(group_id)
        is_anchor_by_id[gid] = bool(np.asarray(selection.is_anchor)[pos])
        nearest_by_id[gid] = int(np.asarray(selection.nearest_anchor_group_id)[pos])
        distance_by_id[gid] = float(np.asarray(selection.source_distance_m)[pos])
    return is_anchor_by_id, nearest_by_id, distance_by_id


def _fallback_no_compatible_anchor(
    *,
    arrays: dict[str, np.ndarray],
    trace_idx: int,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    trend_provider: object | None = None,
    merged: MergeResult,
    cfg: PhysicsLiteConfig,
    pending_trend_fallback: _PendingTrendFallback | None = None,
) -> None:
    fallback = str(cfg.physical_runtime.anchor_reuse.fallback_if_no_compatible_segment)
    if fallback == 'robust':
        _assign_robust_fallback(
            arrays,
            trace_idx,
            failure_reason=PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
            table=table,
            merged=merged,
        )
        return
    _assign_fallback(
        arrays,
        trace_idx,
        failure_reason=PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
        runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND,
        table=table,
        feasible=feasible,
        trend=trend,
        trend_provider=trend_provider,
        merged=merged,
        pending_trend_fallback=pending_trend_fallback,
    )


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
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
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
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
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
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
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

    arrays = _allocate_result_arrays(table)
    pending_trend_fallback = _PendingTrendFallback()
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
    fit_cache: dict[tuple[int, ...], _FitCacheEntry] = {}
    min_fit_obs = 2 * _fit_min_pts(cfg)

    if cfg.physical_runtime.fit_policy == 'anchor_source_xy':
        if anchor_selection is None:
            with (
                runtime_diagnostics.time_block('anchor_selection_sec')
                if runtime_diagnostics is not None
                else nullcontext()
            ):
                anchor_selection = select_source_xy_stride_anchors(
                    groups,
                    anchor_stride_source_groups=int(
                        cfg.physical_runtime.anchor_selection.anchor_stride_source_groups
                    ),
                    include_first=bool(
                        cfg.physical_runtime.anchor_selection.include_first
                    ),
                    include_last=bool(
                        cfg.physical_runtime.anchor_selection.include_last
                    ),
                )
        is_anchor_by_id, nearest_by_id, distance_by_id = _selection_group_maps(
            anchor_selection
        )
        anchor_models: dict[tuple[int, int, int, bool], _AnchorModelContext] = {}
        n_reused_predictions = 0
        fallback_full_group_ids: set[int] = set()

        anchor_trace_chunks = [
            np.asarray(group.trace_indices, dtype=np.int64)
            for group in groups
            if bool(is_anchor_by_id.get(int(group.group_id), False))
        ]
        anchor_trace_indices = (
            np.concatenate(anchor_trace_chunks)
            if anchor_trace_chunks
            else np.zeros((0,), dtype=np.int64)
        )
        anchor_assignments_by_fit, anchor_assignments = (
            _prepare_fit_context_assignments_for_trace_indices(
                anchor_trace_indices,
                arrays=arrays,
                group_id_by_trace=group_id_by_trace,
                group_context_by_id=group_context_by_id,
                geometry=geometry,
                offset_abs_m=offset_abs_m,
                offset_signed_m=offset_signed_m,
                offset_source=offset_source,
                pick_t_sec=pick_t_sec,
                table=table,
                feasible=feasible,
                trend=trend,
                trend_provider=trend_provider,
                merged=merged,
                cfg=cfg,
                observation_plan_cache=observation_plan_cache,
                min_fit_obs=min_fit_obs,
                runtime_diagnostics=runtime_diagnostics,
                pending_trend_fallback=pending_trend_fallback,
            )
        )
        anchor_work_items = _build_fit_context_work_items(
            anchor_assignments_by_fit,
            offset_abs_m=offset_abs_m,
            pick_t_sec=pick_t_sec,
            coarse_pmax=table.coarse_pmax,
            runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT,
        )
        reporter.emit(
            'physical-center.contexts_built',
            **context,
            phase='anchor',
            work_items=len(anchor_work_items),
            unique_keys=len(anchor_assignments_by_fit),
        )
        anchor_fit_calls_before = (
            int(runtime_diagnostics.n_fit_calls)
            if runtime_diagnostics is not None
            else 0
        )
        anchor_results = _fit_and_assign_context_work_items(
            anchor_work_items,
            arrays=arrays,
            offset_abs_m=offset_abs_m,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            cfg=cfg,
            strategy=strategy,
            fit_cache=fit_cache,
            runtime_diagnostics=runtime_diagnostics,
            progress=reporter,
            progress_context=context,
            pending_trend_fallback=pending_trend_fallback,
        )
        if runtime_diagnostics is not None:
            anchor_fit_call_delta = max(
                0,
                int(runtime_diagnostics.n_fit_calls) - anchor_fit_calls_before,
            )
            if anchor_fit_call_delta > 0:
                runtime_diagnostics.record_anchor_fit_calls(anchor_fit_call_delta)
        for assignment in anchor_assignments:
            result = anchor_results.get(assignment.fit_key)
            if result is None or result.trend_model is None:
                continue
            trace_idx = int(assignment.trace_idx)
            if trace_idx not in result.valid_trace_indices:
                continue
            key = _anchor_model_key(int(group_id_by_trace[trace_idx]), assignment.plan)
            anchor_models.setdefault(
                key,
                _AnchorModelContext(
                    trend_model=result.trend_model,
                    diagnostics=result.diagnostics,
                ),
            )
        anchor_models_by_group_id: dict[
            int,
            dict[tuple[int, int, bool], _AnchorModelContext],
        ] = {}
        for model_key, anchor_context in anchor_models.items():
            anchor_group_id, side, segment_id, relaxed = model_key
            anchor_models_by_group_id.setdefault(int(anchor_group_id), {}).setdefault(
                (int(side), int(segment_id), bool(relaxed)),
                anchor_context,
            )

        for group in groups:
            group_id = int(group.group_id)
            if bool(is_anchor_by_id.get(group_id, False)):
                continue
            group_trace_indices = np.asarray(group.trace_indices, dtype=np.int64)
            nearest_anchor_id = int(nearest_by_id.get(group_id, -1))
            anchor_distance_m = float(distance_by_id.get(group_id, np.nan))
            if runtime_diagnostics is not None:
                runtime_diagnostics.record_nearest_anchor_distance(anchor_distance_m)
            max_distance_m = cfg.physical_runtime.anchor_reuse.max_anchor_distance_m
            distance_ok = (
                nearest_anchor_id >= 0
                and np.isfinite(anchor_distance_m)
                and (
                    max_distance_m is None
                    or anchor_distance_m <= float(max_distance_m)
                )
            )
            compatible_anchor_context_by_plan_key: dict[
                tuple[int, int, bool],
                _AnchorModelContext,
            ] = {}
            with (
                runtime_diagnostics.time_block('compatible_anchor_search_sec')
                if runtime_diagnostics is not None
                else nullcontext()
            ):
                if (
                    bool(cfg.physical_runtime.anchor_reuse.enabled)
                    and bool(distance_ok)
                ):
                    with (
                        runtime_diagnostics.time_block('anchor_lookup_sec')
                        if runtime_diagnostics is not None
                        else nullcontext()
                    ):
                        compatible_anchor_context_by_plan_key = (
                            anchor_models_by_group_id.get(nearest_anchor_id, {})
                        )
            reuse_items: dict[
                tuple[int, int, int, bool],
                list[tuple[int, _ObservationPlan, _AnchorModelContext]],
            ] = {}
            fallback_full_trace_indices: list[int] = []
            for trace_idx in np.asarray(group.trace_indices, dtype=np.int64).tolist():
                trace_idx = int(trace_idx)
                arrays['physical_offset_source'][trace_idx] = np.uint8(offset_source)
                if (
                    not np.isfinite(offset_abs_m[trace_idx])
                    or int(group_id_by_trace[trace_idx]) < 0
                ):
                    _assign_fallback(
                        arrays,
                        trace_idx,
                        failure_reason=PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
                        table=table,
                        feasible=feasible,
                        trend=trend,
                        trend_provider=trend_provider,
                        merged=merged,
                        pending_trend_fallback=pending_trend_fallback,
                    )
                    continue

                plan = _build_observation_plan(
                    trace_idx=trace_idx,
                    target_group_id=group_id,
                    group_context_by_id=group_context_by_id,
                    geometry=geometry,
                    offset_abs_m=offset_abs_m,
                    offset_signed_m=offset_signed_m,
                    cfg=cfg,
                    runtime_diagnostics=runtime_diagnostics,
                    plan_cache=observation_plan_cache,
                )
                if plan is not None:
                    arrays['physical_model_neighbor_count'][trace_idx] = np.int32(
                        plan.neighbor_count
                    )
                    arrays['physical_prefilter_valid_count'][trace_idx] = np.int32(
                        plan.prefilter_valid_count
                    )
                    arrays['physical_model_segment_id'][trace_idx] = np.int32(
                        plan.segment_id
                    )
                    arrays['physical_model_side'][trace_idx] = np.int8(plan.side)

                anchor_context = None
                key = None
                if runtime_diagnostics is not None:
                    runtime_diagnostics.record_compatible_anchor_search_candidates(
                        1 if bool(distance_ok) else 0
                    )
                if (
                    bool(cfg.physical_runtime.anchor_reuse.enabled)
                    and plan is not None
                    and distance_ok
                ):
                    key = _anchor_model_key(nearest_anchor_id, plan)
                    anchor_context = compatible_anchor_context_by_plan_key.get(
                        (int(plan.side), int(plan.segment_id), bool(plan.relaxed))
                    )

                if anchor_context is not None and plan is not None and key is not None:
                    reuse_items.setdefault(key, []).append(
                        (trace_idx, plan, anchor_context)
                    )
                    continue
                if runtime_diagnostics is not None:
                    runtime_diagnostics.record_no_compatible_anchor_context()

                fallback = str(
                    cfg.physical_runtime.anchor_reuse.fallback_if_no_compatible_segment
                )
                if fallback == 'full_fit':
                    fallback_full_group_ids.add(group_id)
                    fallback_full_trace_indices.append(trace_idx)
                else:
                    _fallback_no_compatible_anchor(
                        arrays=arrays,
                        trace_idx=trace_idx,
                        table=table,
                        feasible=feasible,
                        trend=trend,
                        trend_provider=trend_provider,
                        merged=merged,
                        cfg=cfg,
                        pending_trend_fallback=pending_trend_fallback,
                    )

            if fallback_full_trace_indices:
                fallback_full_assignments_by_fit, _fallback_full_assignments = (
                    _prepare_fit_context_assignments_for_trace_indices(
                        np.asarray(fallback_full_trace_indices, dtype=np.int64),
                        arrays=arrays,
                        group_id_by_trace=group_id_by_trace,
                        group_context_by_id=group_context_by_id,
                        geometry=geometry,
                        offset_abs_m=offset_abs_m,
                        offset_signed_m=offset_signed_m,
                        offset_source=offset_source,
                        pick_t_sec=pick_t_sec,
                        table=table,
                        feasible=feasible,
                        trend=trend,
                        trend_provider=trend_provider,
                        merged=merged,
                        cfg=cfg,
                        observation_plan_cache=observation_plan_cache,
                        min_fit_obs=min_fit_obs,
                        runtime_diagnostics=runtime_diagnostics,
                        pending_trend_fallback=pending_trend_fallback,
                    )
                )
                fallback_full_work_items = _build_fit_context_work_items(
                    fallback_full_assignments_by_fit,
                    offset_abs_m=offset_abs_m,
                    pick_t_sec=pick_t_sec,
                    coarse_pmax=table.coarse_pmax,
                    runtime_fit_source=(
                        PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_FULL_FIT_NO_COMPATIBLE_ANCHOR
                    ),
                )
                reporter.emit(
                    'physical-center.contexts_built',
                    **context,
                    phase='fallback_full',
                    work_items=len(fallback_full_work_items),
                    unique_keys=len(fallback_full_assignments_by_fit),
                )
                _fit_and_assign_context_work_items(
                    fallback_full_work_items,
                    arrays=arrays,
                    offset_abs_m=offset_abs_m,
                    table=table,
                    feasible=feasible,
                    trend=trend,
                    trend_provider=trend_provider,
                    merged=merged,
                    cfg=cfg,
                    strategy=strategy,
                    fit_cache=fit_cache,
                    runtime_diagnostics=runtime_diagnostics,
                    progress=reporter,
                    progress_context=context,
                    pending_trend_fallback=pending_trend_fallback,
                )

            non_anchor_mode = str(cfg.physical_runtime.anchor_reuse.non_anchor_mode)
            if runtime_diagnostics is not None and reuse_items:
                runtime_diagnostics.record_reuse_contexts(len(reuse_items))
            if non_anchor_mode == 'nearest_anchor':
                for items in reuse_items.values():
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
                        runtime_diagnostics=runtime_diagnostics,
                    )
                    n_reused_predictions += int(np.count_nonzero(valid))
                    for trace_idx in trace_indices[~valid].tolist():
                        _assign_fallback(
                            arrays,
                            int(trace_idx),
                            failure_reason=(PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID),
                            table=table,
                            feasible=feasible,
                            trend=trend,
                            trend_provider=trend_provider,
                            merged=merged,
                            pending_trend_fallback=pending_trend_fallback,
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
                        arrays=arrays,
                        group_id_by_trace=group_id_by_trace,
                        group_context_by_id=group_context_by_id,
                        geometry=geometry,
                        offset_abs_m=offset_abs_m,
                        offset_signed_m=offset_signed_m,
                        offset_source=offset_source,
                        pick_t_sec=pick_t_sec,
                        table=table,
                        feasible=feasible,
                        trend=trend,
                        trend_provider=trend_provider,
                        merged=merged,
                        cfg=cfg,
                        observation_plan_cache=observation_plan_cache,
                        min_fit_obs=min_fit_obs,
                        runtime_diagnostics=runtime_diagnostics,
                        pending_trend_fallback=pending_trend_fallback,
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
                    arrays=arrays,
                    offset_abs_m=offset_abs_m,
                    table=table,
                    feasible=feasible,
                    trend=trend,
                    trend_provider=trend_provider,
                    merged=merged,
                    cfg=cfg,
                    strategy=strategy,
                    fit_cache=fit_cache,
                    runtime_diagnostics=runtime_diagnostics,
                    progress=reporter,
                    progress_context=context,
                    pending_trend_fallback=pending_trend_fallback,
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
                    if not bool(is_anchor_by_id.get(int(group.group_id), False))
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

    reporter.emit(
        'physical-center.stage_start',
        **context,
        stage='fit_context_preparation',
    )
    stage_start = time.perf_counter()
    fit_context_assignments: dict[tuple[int, ...], list[_TracePlanAssignment]] = {}
    for trace_idx in range(n):
        assignment = _prepare_trace_plan_assignment(
            arrays=arrays,
            trace_idx=trace_idx,
            group_id_by_trace=group_id_by_trace,
            group_context_by_id=group_context_by_id,
            geometry=geometry,
            offset_abs_m=offset_abs_m,
            offset_signed_m=offset_signed_m,
            offset_source=offset_source,
            pick_t_sec=pick_t_sec,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            cfg=cfg,
            observation_plan_cache=observation_plan_cache,
            min_fit_obs=min_fit_obs,
            runtime_diagnostics=runtime_diagnostics,
            pending_trend_fallback=pending_trend_fallback,
        )
        if assignment is not None:
            fit_context_assignments.setdefault(assignment.fit_key, []).append(
                assignment
            )
    reporter.emit(
        'physical-center.stage_done',
        **context,
        stage='fit_context_preparation',
        elapsed=time.perf_counter() - stage_start,
        unique_keys=len(fit_context_assignments),
    )

    work_items = _build_fit_context_work_items(
        fit_context_assignments,
        offset_abs_m=offset_abs_m,
        pick_t_sec=pick_t_sec,
        coarse_pmax=table.coarse_pmax,
        runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_FULL_FIT,
    )
    reporter.emit(
        'physical-center.contexts_built',
        **context,
        work_items=len(work_items),
        unique_keys=len(fit_context_assignments),
    )
    _fit_and_assign_context_work_items(
        work_items,
        arrays=arrays,
        offset_abs_m=offset_abs_m,
        table=table,
        feasible=feasible,
        trend=trend,
        trend_provider=trend_provider,
        merged=merged,
        cfg=cfg,
        strategy=strategy,
        fit_cache=fit_cache,
        runtime_diagnostics=runtime_diagnostics,
        progress=reporter,
        progress_context=context,
        pending_trend_fallback=pending_trend_fallback,
    )

    if runtime_diagnostics is not None:
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
