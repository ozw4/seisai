"""Fit-context preparation, execution, and assignment helpers."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .physical_center_context import (
    PhysicalCenterBuildContext,
    PhysicalCenterInputs,
    PhysicalCenterWorkspace,
)
from .physical_center_fallback import _assign_fallback
from .physical_center_fit import (
    _confidence_weights_for_obs,
    _fit_key_for_obs,
    _fit_min_pts,
    _fit_model_for_plan,
    _fit_progress_fields,
    _fit_task_cfg_values,
    _fit_task_from_work_item,
    _FitCacheEntry,
    _run_fit_tasks_with_executor,
    _sample_observation_indices_for_fit,
)
from .physical_center_observation import (
    _build_observation_plan as _build_observation_plan_with_min_obs,
)
from .physical_center_observation import (
    _ObservationPlan,
)
from .physical_center_prediction import (
    _assign_prepared_model_prediction_batch,
    _TraceFitResult,
    _TracePlanAssignment,
)
from .physical_center_types import (
    PHYSICAL_MODEL_FAILURE_FIT_FAILED,
    PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
    PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS,
)
from .progress import NullProgressReporter

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .config import PhysicsLiteConfig
    from .feasible import FeasibleBandResult
    from .geometry import CoarseGeometry
    from .merge import MergeResult
    from .physical_center_fallback import _PendingTrendFallback
    from .physical_center_fit import (
        _FitTask,
        _FitTaskResult,
        _PhysicalFitStrategy,
    )
    from .physical_center_observation import (
        _GroupObservationContext,
        _ObservationPlanCache,
    )
    from .pick_table import CoarsePickTable
    from .runtime_diagnostics import PhysicalRuntimeDiagnostics
    from .trend import TrendResult

_FitDiagnostics = tuple[float, float, float, float, float, float, float]
_FitModelForPlan = Callable[
    ...,
    tuple[object | None, _FitDiagnostics | None, int | None],
]


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
    diagnostics: _FitDiagnostics | None
    valid_trace_indices: frozenset[int]


def _build_observation_plan(  # noqa: PLR0913
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


def _prepare_trace_plan_assignment(  # noqa: PLR0913
    *,
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
    workspace: PhysicalCenterWorkspace,
    trace_idx: int,
) -> _TracePlanAssignment | None:
    arrays = workspace.arrays
    offset_abs_m = build_context.offset_abs_m
    offset_source = build_context.offset_source
    group_id_by_trace = build_context.group_id_by_trace
    runtime_diagnostics = workspace.runtime_diagnostics
    arrays['physical_offset_source'][trace_idx] = np.uint8(offset_source)
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
        return None

    plan = _build_observation_plan(
        trace_idx=trace_idx,
        target_group_id=int(group_id_by_trace[trace_idx]),
        group_context_by_id=build_context.group_context_by_id,
        geometry=build_context.geometry,
        offset_abs_m=offset_abs_m,
        offset_signed_m=build_context.offset_signed_m,
        cfg=inputs.cfg,
        runtime_diagnostics=runtime_diagnostics,
        plan_cache=build_context.observation_plan_cache,
    )
    if plan is None:
        _assign_fallback(
            arrays,
            trace_idx,
            failure_reason=PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS,
            table=inputs.table,
            feasible=inputs.feasible,
            trend=inputs.trend,
            trend_provider=inputs.trend_provider,
            merged=inputs.merged,
            pending_trend_fallback=workspace.pending_trend_fallback,
        )
        return None

    arrays['physical_model_neighbor_count'][trace_idx] = np.int32(plan.neighbor_count)
    arrays['physical_prefilter_valid_count'][trace_idx] = np.int32(
        plan.prefilter_valid_count
    )
    arrays['physical_model_segment_id'][trace_idx] = np.int32(plan.segment_id)
    arrays['physical_model_side'][trace_idx] = np.int8(plan.side)

    if int(plan.obs_indices.size) < int(build_context.min_fit_obs):
        _assign_fallback(
            arrays,
            trace_idx,
            failure_reason=PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS,
            table=inputs.table,
            feasible=inputs.feasible,
            trend=inputs.trend,
            trend_provider=inputs.trend_provider,
            merged=inputs.merged,
            pending_trend_fallback=workspace.pending_trend_fallback,
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
            pick_t_sec=build_context.pick_t_sec,
            coarse_pmax=inputs.table.coarse_pmax,
            cfg=inputs.cfg,
            min_required_obs=int(build_context.min_fit_obs),
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


def _assign_fit_context_fallback(  # noqa: PLR0913
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


def _fit_and_assign_context_work_item(  # noqa: PLR0913
    *,
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
    workspace: PhysicalCenterWorkspace,
    work_item: _FitContextWorkItem,
    strategy: _PhysicalFitStrategy,
    fit_model_for_plan: _FitModelForPlan | None = None,
) -> _FitContextWorkResult:
    cfg = inputs.cfg
    fit_cache = workspace.fit_cache
    runtime_diagnostics = workspace.runtime_diagnostics
    fit_model = (
        _fit_model_for_plan
        if fit_model_for_plan is None
        else fit_model_for_plan
    )
    trend_model, diagnostics, fit_failure_reason = fit_model(
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
        arrays=workspace.arrays,
        work_item=work_item,
        offset_abs_m=build_context.offset_abs_m,
        table=inputs.table,
        feasible=inputs.feasible,
        trend=inputs.trend,
        trend_provider=inputs.trend_provider,
        merged=inputs.merged,
        fit_cache=fit_cache,
        trend_model=trend_model,
        diagnostics=diagnostics,
        fit_failure_reason=fit_failure_reason,
        runtime_diagnostics=runtime_diagnostics,
        pending_trend_fallback=workspace.pending_trend_fallback,
    )


def _assign_fit_context_work_item_outcome(  # noqa: PLR0913
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
    diagnostics: _FitDiagnostics | None,
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


def _prepare_fit_context_assignments_for_trace_indices(  # noqa: PLR0913
    trace_indices: np.ndarray,
    *,
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
    workspace: PhysicalCenterWorkspace,
) -> tuple[
    dict[tuple[int, ...], list[_TracePlanAssignment]],
    list[_TracePlanAssignment],
]:
    grouped: dict[tuple[int, ...], list[_TracePlanAssignment]] = {}
    ordered: list[_TracePlanAssignment] = []
    for trace_idx in np.asarray(trace_indices, dtype=np.int64).tolist():
        assignment = _prepare_trace_plan_assignment(
            inputs=inputs,
            build_context=build_context,
            workspace=workspace,
            trace_idx=int(trace_idx),
        )
        if assignment is None:
            continue
        grouped.setdefault(assignment.fit_key, []).append(assignment)
        ordered.append(assignment)
    return grouped, ordered


def _fit_and_assign_context_work_items(  # noqa: PLR0913
    work_items: list[_FitContextWorkItem],
    *,
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
    workspace: PhysicalCenterWorkspace,
    strategy: _PhysicalFitStrategy,
    progress: object | None = None,
    progress_context: Mapping[str, object] | None = None,
    fit_model_for_plan: _FitModelForPlan | None = None,
) -> dict[tuple[int, ...], _FitContextWorkResult]:
    cfg = inputs.cfg
    runtime_diagnostics = workspace.runtime_diagnostics
    reporter = progress if progress is not None else NullProgressReporter()
    context = dict(progress_context or {})
    fit_start = time.perf_counter()
    if bool(cfg.physical_runtime.fit_executor.enabled) and work_items:
        return _fit_and_assign_context_work_items_parallel(
            work_items,
            inputs=inputs,
            build_context=build_context,
            workspace=workspace,
            progress=reporter,
            progress_context=context,
            progress_start_sec=fit_start,
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
            inputs=inputs,
            build_context=build_context,
            workspace=workspace,
            work_item=work_item,
            strategy=strategy,
            fit_model_for_plan=fit_model_for_plan,
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


def _fit_and_assign_context_work_items_parallel(  # noqa: PLR0913
    work_items: list[_FitContextWorkItem],
    *,
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
    workspace: PhysicalCenterWorkspace,
    progress: object | None = None,
    progress_context: Mapping[str, object] | None = None,
    progress_start_sec: float | None = None,
) -> dict[tuple[int, ...], _FitContextWorkResult]:
    cfg = inputs.cfg
    arrays = workspace.arrays
    fit_cache = workspace.fit_cache
    runtime_diagnostics = workspace.runtime_diagnostics
    offset_abs_m = build_context.offset_abs_m
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
            table=inputs.table,
            feasible=inputs.feasible,
            trend=inputs.trend,
            trend_provider=inputs.trend_provider,
            merged=inputs.merged,
            fit_cache=fit_cache,
            trend_model=entry.model,
            diagnostics=entry.diagnostics,
            fit_failure_reason=fit_failure_reason,
            runtime_diagnostics=runtime_diagnostics,
            pending_trend_fallback=workspace.pending_trend_fallback,
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
                table=inputs.table,
                feasible=inputs.feasible,
                trend=inputs.trend,
                trend_provider=inputs.trend_provider,
                merged=inputs.merged,
                fit_cache=fit_cache,
                trend_model=task_result.trend_model,
                diagnostics=task_result.diagnostics,
                fit_failure_reason=task_result.failure_reason,
                runtime_diagnostics=runtime_diagnostics,
                pending_trend_fallback=workspace.pending_trend_fallback,
            )

    return results
