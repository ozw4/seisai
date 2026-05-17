"""Non-anchor anchor-reuse helpers for physical-center fitting."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

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
from .physical_center_fallback import _assign_fallback, _assign_robust_fallback
from .physical_center_observation import _ObservationPlan
from .physical_center_prediction import _assign_model_prediction_batch
from .physical_center_types import (
    PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
    PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
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
