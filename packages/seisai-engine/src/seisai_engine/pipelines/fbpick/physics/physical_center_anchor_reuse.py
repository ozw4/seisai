"""Non-anchor anchor-reuse helpers for physical-center fitting."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .feasible import compute_velocity_t0_band_from_arrays
from .physical_center_anchor import (
    _anchor_model_key,
    _AnchorModelContext,
    _AnchorSourceXYContext,
)
from .physical_center_context_fit import (
    _build_fit_context_work_items,
    _build_observation_plan,
    _fit_and_assign_context_work_items,
    _prepare_fit_context_assignments_for_trace_indices,
)
from .physical_center_fallback import (
    _as_bool_vector,
    _as_vector,
    _assign_fallback,
    _assign_robust_fallback,
)
from .physical_center_observation import _ObservationPlan
from .physical_center_prediction import (
    _assign_model_prediction_batch,
    _predict_model_array_sec,
)
from .physical_center_types import (
    PHYSICAL_MODEL_FAILURE_FIT_FAILED,
    PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
    PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
    PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT,
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND,
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_FULL_FIT_NO_COMPATIBLE_ANCHOR,
    PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE,
)
from .progress import NullProgressReporter

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .geometry import SourceGroup
    from .physical_center_context import (
        PhysicalCenterBuildContext,
        PhysicalCenterInputs,
        PhysicalCenterWorkspace,
    )
    from .physical_center_context_fit import _FitModelForPlan
    from .runtime_diagnostics import PhysicalRuntimeDiagnostics

_CompatibleAnchorPlanKey = tuple[int, int, bool]
_AnchorReuseKey = tuple[int, int, int, bool]
_AnchorReuseItem = tuple[int, _ObservationPlan, _AnchorModelContext]


@dataclass(frozen=True)
class _ReuseShiftStats:
    t0_shift_sec: float
    shift_valid: bool
    valid_count: int
    resid_p50_ms: float
    resid_p90_ms: float


@dataclass(frozen=True)
class _NearestAnchorReuseContext:
    nearest_anchor_id: int
    anchor_distance_m: float
    distance_ok: bool


@dataclass(frozen=True)
class _AnchorReuseGroupPlan:
    reuse_items: dict[_AnchorReuseKey, list[_AnchorReuseItem]]
    fallback_full_trace_indices: np.ndarray

    @property
    def uses_fallback_full_fit(self) -> bool:
        return bool(self.fallback_full_trace_indices.size > 0)


def _nearest_anchor_reuse_context(
    *,
    group_id: int,
    anchor_source_xy: _AnchorSourceXYContext,
    inputs: PhysicalCenterInputs,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
) -> _NearestAnchorReuseContext:
    nearest_anchor_id = int(anchor_source_xy.nearest_by_id.get(int(group_id), -1))
    anchor_distance_m = float(
        anchor_source_xy.distance_by_id.get(int(group_id), np.nan)
    )
    if runtime_diagnostics is not None:
        runtime_diagnostics.record_nearest_anchor_distance(anchor_distance_m)
    max_distance_m = inputs.cfg.physical_runtime.anchor_reuse.max_anchor_distance_m
    distance_ok = (
        nearest_anchor_id >= 0
        and np.isfinite(anchor_distance_m)
        and (max_distance_m is None or anchor_distance_m <= float(max_distance_m))
    )
    return _NearestAnchorReuseContext(
        nearest_anchor_id=nearest_anchor_id,
        anchor_distance_m=anchor_distance_m,
        distance_ok=bool(distance_ok),
    )


def _compatible_anchor_contexts_for_reuse(
    *,
    anchor_source_xy: _AnchorSourceXYContext,
    nearest_context: _NearestAnchorReuseContext,
    inputs: PhysicalCenterInputs,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
) -> Mapping[_CompatibleAnchorPlanKey, _AnchorModelContext]:
    with (
        runtime_diagnostics.time_block('compatible_anchor_search_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        if not (
            bool(inputs.cfg.physical_runtime.anchor_reuse.enabled)
            and bool(nearest_context.distance_ok)
        ):
            return {}
        with (
            runtime_diagnostics.time_block('anchor_lookup_sec')
            if runtime_diagnostics is not None
            else nullcontext()
        ):
            return anchor_source_xy.models_by_group_id.get(
                int(nearest_context.nearest_anchor_id),
                {},
            )


def _fallback_no_compatible_anchor(
    *,
    inputs: PhysicalCenterInputs,
    workspace: PhysicalCenterWorkspace,
    trace_idx: int,
) -> None:
    arrays = workspace.arrays
    cfg = inputs.cfg
    fallback = str(cfg.physical_runtime.anchor_reuse.fallback_if_no_compatible_segment)
    if fallback == 'robust':
        _assign_robust_fallback(
            arrays,
            trace_idx,
            failure_reason=PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
            table=inputs.table,
            merged=inputs.merged,
        )
        return
    _assign_fallback(
        arrays,
        trace_idx,
        failure_reason=PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
        runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND,
        table=inputs.table,
        feasible=inputs.feasible,
        trend=inputs.trend,
        trend_provider=inputs.trend_provider,
        merged=inputs.merged,
        pending_trend_fallback=workspace.pending_trend_fallback,
    )


def _assign_reuse_plan_diagnostics(
    arrays: dict[str, np.ndarray],
    trace_idx: int,
    plan: _ObservationPlan,
) -> None:
    arrays['physical_model_neighbor_count'][trace_idx] = np.int32(
        plan.neighbor_count
    )
    arrays['physical_prefilter_valid_count'][trace_idx] = np.int32(
        plan.prefilter_valid_count
    )
    arrays['physical_model_segment_id'][trace_idx] = np.int32(plan.segment_id)
    arrays['physical_model_side'][trace_idx] = np.int8(plan.side)


def _lookup_reuse_item_context(
    *,
    nearest_context: _NearestAnchorReuseContext,
    compatible_anchor_context_by_plan_key: Mapping[
        _CompatibleAnchorPlanKey,
        _AnchorModelContext,
    ],
    plan: _ObservationPlan | None,
    inputs: PhysicalCenterInputs,
) -> tuple[_AnchorReuseKey | None, _AnchorModelContext | None]:
    if not (
        bool(inputs.cfg.physical_runtime.anchor_reuse.enabled)
        and plan is not None
        and bool(nearest_context.distance_ok)
    ):
        return None, None
    reuse_key = _anchor_model_key(int(nearest_context.nearest_anchor_id), plan)
    anchor_context = compatible_anchor_context_by_plan_key.get(
        (int(plan.side), int(plan.segment_id), bool(plan.relaxed))
    )
    return reuse_key, anchor_context


def _build_anchor_reuse_group_plan(  # noqa: PLR0913
    *,
    group: SourceGroup,
    nearest_context: _NearestAnchorReuseContext,
    compatible_anchor_context_by_plan_key: Mapping[
        _CompatibleAnchorPlanKey,
        _AnchorModelContext,
    ],
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
    workspace: PhysicalCenterWorkspace,
) -> _AnchorReuseGroupPlan:
    arrays = workspace.arrays
    runtime_diagnostics = workspace.runtime_diagnostics
    reuse_items: dict[_AnchorReuseKey, list[_AnchorReuseItem]] = {}
    fallback_full_trace_indices: list[int] = []
    group_id = int(group.group_id)
    offset_abs_m = build_context.offset_abs_m
    group_id_by_trace = build_context.group_id_by_trace

    for trace_idx_raw in np.asarray(group.trace_indices, dtype=np.int64).tolist():
        trace_idx = int(trace_idx_raw)
        arrays['physical_offset_source'][trace_idx] = np.uint8(
            build_context.offset_source
        )
        if (
            not np.isfinite(offset_abs_m[trace_idx])
            or int(group_id_by_trace[trace_idx]) < 0
        ):
            _assign_fallback(
                arrays,
                trace_idx,
                failure_reason=PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
                table=inputs.table,
                feasible=inputs.feasible,
                trend=inputs.trend,
                trend_provider=inputs.trend_provider,
                merged=inputs.merged,
                pending_trend_fallback=workspace.pending_trend_fallback,
            )
            continue

        plan = _build_observation_plan(
            trace_idx=trace_idx,
            target_group_id=group_id,
            group_context_by_id=build_context.group_context_by_id,
            geometry=build_context.geometry,
            offset_abs_m=offset_abs_m,
            offset_signed_m=build_context.offset_signed_m,
            cfg=inputs.cfg,
            runtime_diagnostics=runtime_diagnostics,
            plan_cache=build_context.observation_plan_cache,
        )
        if plan is not None:
            _assign_reuse_plan_diagnostics(arrays, trace_idx, plan)

        if runtime_diagnostics is not None:
            runtime_diagnostics.record_compatible_anchor_search_candidates(
                1 if bool(nearest_context.distance_ok) else 0
            )

        reuse_key, anchor_context = _lookup_reuse_item_context(
            nearest_context=nearest_context,
            compatible_anchor_context_by_plan_key=(
                compatible_anchor_context_by_plan_key
            ),
            plan=plan,
            inputs=inputs,
        )
        if anchor_context is not None and plan is not None and reuse_key is not None:
            reuse_items.setdefault(reuse_key, []).append(
                (trace_idx, plan, anchor_context)
            )
            continue

        if runtime_diagnostics is not None:
            runtime_diagnostics.record_no_compatible_anchor_context()

        fallback = str(
            inputs.cfg.physical_runtime.anchor_reuse.fallback_if_no_compatible_segment
        )
        if fallback == 'full_fit':
            fallback_full_trace_indices.append(trace_idx)
        else:
            _fallback_no_compatible_anchor(
                inputs=inputs,
                workspace=workspace,
                trace_idx=trace_idx,
            )

    return _AnchorReuseGroupPlan(
        reuse_items=reuse_items,
        fallback_full_trace_indices=np.asarray(
            fallback_full_trace_indices,
            dtype=np.int64,
        ),
    )


def _run_no_compatible_full_fit_fallback(  # noqa: PLR0913
    trace_indices: np.ndarray,
    *,
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
    workspace: PhysicalCenterWorkspace,
    strategy: object,
    progress: object | None = None,
    progress_context: Mapping[str, object] | None = None,
    fit_model_for_plan: _FitModelForPlan | None = None,
) -> None:
    indices = np.asarray(trace_indices, dtype=np.int64)
    if indices.size == 0:
        return
    reporter = progress if progress is not None else NullProgressReporter()
    context = dict(progress_context or {})
    fallback_full_assignments_by_fit, _fallback_full_assignments = (
        _prepare_fit_context_assignments_for_trace_indices(
            indices,
            inputs=inputs,
            build_context=build_context,
            workspace=workspace,
        )
    )
    fallback_full_work_items = _build_fit_context_work_items(
        fallback_full_assignments_by_fit,
        offset_abs_m=build_context.offset_abs_m,
        pick_t_sec=build_context.pick_t_sec,
        coarse_pmax=inputs.table.coarse_pmax,
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
        inputs=inputs,
        build_context=build_context,
        workspace=workspace,
        strategy=strategy,
        progress=reporter,
        progress_context=context,
        fit_model_for_plan=fit_model_for_plan,
    )


def _assign_nearest_anchor_reuse_predictions(
    reuse_items: Mapping[_AnchorReuseKey, list[_AnchorReuseItem]],
    *,
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
    workspace: PhysicalCenterWorkspace,
) -> int:
    n_reused_predictions = 0
    arrays = workspace.arrays
    runtime_diagnostics = workspace.runtime_diagnostics
    for items in reuse_items.values():
        trace_indices = np.asarray(
            [trace_idx for trace_idx, _plan, _context in items],
            dtype=np.int64,
        )
        anchor_context = items[0][2]
        plan_by_trace = {int(trace_idx): plan for trace_idx, plan, _context in items}
        valid = _assign_model_prediction_batch(
            arrays,
            trace_indices,
            trend_model=anchor_context.trend_model,
            diagnostics=anchor_context.diagnostics,
            plan_by_trace=plan_by_trace,
            offset_abs_m=build_context.offset_abs_m,
            dt=float(inputs.table.dt_scalar_sec),
            n_samples=int(inputs.table.n_samples_orig),
            runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE,
            runtime_diagnostics=runtime_diagnostics,
        )
        n_reused_predictions += int(np.count_nonzero(valid))
        for trace_idx in trace_indices[~valid].tolist():
            _assign_fallback(
                arrays,
                int(trace_idx),
                failure_reason=PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
                table=inputs.table,
                feasible=inputs.feasible,
                trend=inputs.trend,
                trend_provider=inputs.trend_provider,
                merged=inputs.merged,
                pending_trend_fallback=workspace.pending_trend_fallback,
            )
    return n_reused_predictions


def _trace_indices_for_reuse_items(
    items: list[_AnchorReuseItem],
) -> np.ndarray:
    return np.asarray(
        [trace_idx for trace_idx, _plan, _context in items],
        dtype=np.int64,
    )


def _compute_t0_shift_physical_mask(
    *,
    trace_indices: np.ndarray,
    offset_abs_m: np.ndarray,
    pick_t_sec: np.ndarray,
    inputs: PhysicalCenterInputs,
) -> np.ndarray:
    indices = np.asarray(trace_indices, dtype=np.int64)
    offsets = np.asarray(offset_abs_m, dtype=np.float32)[indices]
    picks = np.asarray(pick_t_sec, dtype=np.float32)[indices]
    finite = np.isfinite(offsets) & np.isfinite(picks)
    physical = np.zeros((indices.size,), dtype=np.bool_)
    cfg = inputs.cfg
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
            inputs.feasible.feasible_mask,
            n_traces=int(np.asarray(offset_abs_m).shape[0]),
        )
        physical &= feasible_mask[indices]
    return physical.astype(np.bool_, copy=False)


def _compute_reuse_t0_shift_stats(
    *,
    trend_model: object,
    trace_indices: np.ndarray,
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
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

    offset_abs_m = build_context.offset_abs_m
    pick_t_sec = build_context.pick_t_sec
    offsets = np.asarray(offset_abs_m, dtype=np.float32)[indices]
    picks = np.asarray(pick_t_sec, dtype=np.float32)[indices]
    valid = np.isfinite(offsets) & np.isfinite(picks)
    cfg = inputs.cfg
    t0_cfg = cfg.physical_runtime.t0_shift
    if bool(t0_cfg.use_physical_prefilter_mask):
        valid &= _compute_t0_shift_physical_mask(
            trace_indices=indices,
            offset_abs_m=offset_abs_m,
            pick_t_sec=pick_t_sec,
            inputs=inputs,
        )
    if bool(t0_cfg.use_pmax_min):
        pmax = _as_vector(
            'table.coarse_pmax',
            inputs.table.coarse_pmax,
            n_traces=inputs.table.n_traces,
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
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
) -> bool:
    adaptive = inputs.cfg.physical_runtime.adaptive_refit
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
        and int(plan.obs_indices.size) >= int(build_context.min_fit_obs)
    )
    return bool(resid_trigger or shift_trigger or insufficient_trigger)


def _compute_reuse_stats_by_key(
    reuse_items: Mapping[_AnchorReuseKey, list[_AnchorReuseItem]],
    *,
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
    workspace: PhysicalCenterWorkspace,
) -> tuple[dict[_AnchorReuseKey, _ReuseShiftStats], bool]:
    runtime_diagnostics = workspace.runtime_diagnostics
    stats_by_key: dict[_AnchorReuseKey, _ReuseShiftStats] = {}
    adaptive_refit = False
    for key, items in reuse_items.items():
        key_trace_indices = _trace_indices_for_reuse_items(items)
        anchor_context = items[0][2]
        with (
            runtime_diagnostics.time_block('t0_shift_sec')
            if runtime_diagnostics is not None
            else nullcontext()
        ):
            stats = _compute_reuse_t0_shift_stats(
                trend_model=anchor_context.trend_model,
                trace_indices=key_trace_indices,
                inputs=inputs,
                build_context=build_context,
            )
        stats_by_key[key] = stats
        _assign_reuse_runtime_diagnostics(
            workspace.arrays,
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
                inputs=inputs,
                build_context=build_context,
            )
        if runtime_diagnostics is not None:
            runtime_diagnostics.record_adaptive_refit_decision(
                triggered=triggered
            )
        adaptive_refit = adaptive_refit or triggered
    return stats_by_key, adaptive_refit


def _run_adaptive_refit_for_group(  # noqa: PLR0913
    group_trace_indices: np.ndarray,
    *,
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
    workspace: PhysicalCenterWorkspace,
    strategy: object,
    progress: object | None = None,
    progress_context: Mapping[str, object] | None = None,
    fit_model_for_plan: _FitModelForPlan | None = None,
) -> bool:
    indices = np.asarray(group_trace_indices, dtype=np.int64)
    workspace.arrays['physical_runtime_refit_mask'][indices] = True
    refit_assignments_by_fit, _refit_assignments = (
        _prepare_fit_context_assignments_for_trace_indices(
            indices,
            inputs=inputs,
            build_context=build_context,
            workspace=workspace,
        )
    )
    refit_work_items = _build_fit_context_work_items(
        refit_assignments_by_fit,
        offset_abs_m=build_context.offset_abs_m,
        pick_t_sec=build_context.pick_t_sec,
        coarse_pmax=inputs.table.coarse_pmax,
        runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT,
    )
    reporter = progress if progress is not None else NullProgressReporter()
    reporter.emit(
        'physical-center.contexts_built',
        **dict(progress_context or {}),
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
        progress_context=progress_context,
        fit_model_for_plan=fit_model_for_plan,
    )
    assigned_count = sum(
        len(result.valid_trace_indices) for result in refit_results.values()
    )
    success = assigned_count > 0
    if workspace.runtime_diagnostics is not None:
        workspace.runtime_diagnostics.record_adaptive_refit(success=success)
    return bool(success)


def _assign_anchor_reuse_failure_fallbacks(
    items: list[_AnchorReuseItem],
    *,
    inputs: PhysicalCenterInputs,
    workspace: PhysicalCenterWorkspace,
    fallback_mode: str,
) -> bool:
    arrays = workspace.arrays
    if fallback_mode == 'robust':
        for trace_idx, _plan, _context in items:
            _assign_robust_fallback(
                arrays,
                trace_idx,
                failure_reason=PHYSICAL_MODEL_FAILURE_FIT_FAILED,
                table=inputs.table,
                merged=inputs.merged,
            )
        return True
    if fallback_mode == 'existing_trend':
        for trace_idx, _plan, _context in items:
            _assign_fallback(
                arrays,
                trace_idx,
                failure_reason=PHYSICAL_MODEL_FAILURE_FIT_FAILED,
                table=inputs.table,
                feasible=inputs.feasible,
                trend=inputs.trend,
                trend_provider=inputs.trend_provider,
                merged=inputs.merged,
                pending_trend_fallback=workspace.pending_trend_fallback,
            )
        return True
    return False


def _record_t0_shifted_group_diagnostics(
    *,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
    group_shifted_count: int,
    group_shift_ms_values: list[float],
    group_reuse_resid_p50_values: list[float],
    group_reuse_resid_p90_values: list[float],
) -> None:
    if runtime_diagnostics is None or group_shifted_count <= 0:
        return
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
            float(np.median(resid_values)) if resid_values.size > 0 else np.nan
        ),
    )


def _assign_t0_shift_reuse_predictions(  # noqa: PLR0913
    reuse_items: Mapping[_AnchorReuseKey, list[_AnchorReuseItem]],
    *,
    stats_by_key: Mapping[_AnchorReuseKey, _ReuseShiftStats],
    fallback_mode: str,
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
    workspace: PhysicalCenterWorkspace,
) -> int:
    n_reused_predictions = 0
    group_shifted_count = 0
    group_shift_ms_values: list[float] = []
    group_reuse_resid_p50_values: list[float] = []
    group_reuse_resid_p90_values: list[float] = []
    arrays = workspace.arrays
    runtime_diagnostics = workspace.runtime_diagnostics

    for key, items in reuse_items.items():
        stats = stats_by_key[key]
        if _assign_anchor_reuse_failure_fallbacks(
            items,
            inputs=inputs,
            workspace=workspace,
            fallback_mode=fallback_mode,
        ):
            continue

        use_shift = (
            fallback_mode == 'nearest_anchor_plus_t0_shift'
            and bool(stats.shift_valid)
        )
        shift_sec = float(stats.t0_shift_sec) if use_shift else 0.0
        trace_indices = _trace_indices_for_reuse_items(items)
        anchor_context = items[0][2]
        plan_by_trace = {int(trace_idx): plan for trace_idx, plan, _context in items}
        valid = _assign_model_prediction_batch(
            arrays,
            trace_indices,
            trend_model=anchor_context.trend_model,
            diagnostics=anchor_context.diagnostics,
            plan_by_trace=plan_by_trace,
            offset_abs_m=build_context.offset_abs_m,
            dt=float(inputs.table.dt_scalar_sec),
            n_samples=int(inputs.table.n_samples_orig),
            runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE,
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
                table=inputs.table,
                feasible=inputs.feasible,
                trend=inputs.trend,
                trend_provider=inputs.trend_provider,
                merged=inputs.merged,
                pending_trend_fallback=workspace.pending_trend_fallback,
            )
        if use_shift:
            group_shift_ms_values.append(float(stats.t0_shift_sec) * 1000.0)
            if np.isfinite(stats.resid_p50_ms):
                group_reuse_resid_p50_values.append(float(stats.resid_p50_ms))
            if np.isfinite(stats.resid_p90_ms):
                group_reuse_resid_p90_values.append(float(stats.resid_p90_ms))

    _record_t0_shifted_group_diagnostics(
        runtime_diagnostics=runtime_diagnostics,
        group_shifted_count=group_shifted_count,
        group_shift_ms_values=group_shift_ms_values,
        group_reuse_resid_p50_values=group_reuse_resid_p50_values,
        group_reuse_resid_p90_values=group_reuse_resid_p90_values,
    )
    return n_reused_predictions


def _assign_anchor_reuse_predictions(  # noqa: PLR0913
    reuse_items: Mapping[_AnchorReuseKey, list[_AnchorReuseItem]],
    *,
    group_trace_indices: np.ndarray,
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
    workspace: PhysicalCenterWorkspace,
    strategy: object,
    progress: object | None = None,
    progress_context: Mapping[str, object] | None = None,
    fit_model_for_plan: _FitModelForPlan | None = None,
) -> int:
    runtime_diagnostics = workspace.runtime_diagnostics
    if runtime_diagnostics is not None and reuse_items:
        runtime_diagnostics.record_reuse_contexts(len(reuse_items))

    non_anchor_mode = str(inputs.cfg.physical_runtime.anchor_reuse.non_anchor_mode)
    if non_anchor_mode == 'nearest_anchor':
        return _assign_nearest_anchor_reuse_predictions(
            reuse_items,
            inputs=inputs,
            build_context=build_context,
            workspace=workspace,
        )

    stats_by_key, adaptive_refit = _compute_reuse_stats_by_key(
        reuse_items,
        inputs=inputs,
        build_context=build_context,
        workspace=workspace,
    )
    refit_failed = False
    if adaptive_refit:
        refit_success = _run_adaptive_refit_for_group(
            group_trace_indices,
            inputs=inputs,
            build_context=build_context,
            workspace=workspace,
            strategy=strategy,
            progress=progress,
            progress_context=progress_context,
            fit_model_for_plan=fit_model_for_plan,
        )
        if refit_success:
            return 0
        refit_failed = True

    fallback_mode = (
        str(inputs.cfg.physical_runtime.adaptive_refit.fallback_if_refit_fails)
        if refit_failed
        else 'nearest_anchor_plus_t0_shift'
    )
    return _assign_t0_shift_reuse_predictions(
        reuse_items,
        stats_by_key=stats_by_key,
        fallback_mode=fallback_mode,
        inputs=inputs,
        build_context=build_context,
        workspace=workspace,
    )
