"""Anchor-source-xy setup helpers for physical-center fitting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .config import PhysicsLiteConfig
from .geometry import SourceGroup
from .physical_center_context import (
    PhysicalCenterBuildContext,
    PhysicalCenterInputs,
    PhysicalCenterWorkspace,
)
from .physical_center_context_fit import (
    _build_fit_context_work_items,
    _fit_and_assign_context_work_items,
    _FitModelForPlan,
    _prepare_fit_context_assignments_for_trace_indices,
)
from .physical_center_observation import _ObservationPlan
from .physical_center_types import PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT
from .progress import NullProgressReporter
from .runtime_diagnostics import PhysicalRuntimeDiagnostics
from .runtime_policy import (
    SourceXYAnchorSelectionResult,
    select_source_xy_stride_anchors,
)

_FitDiagnostics = tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class _AnchorModelContext:
    trend_model: object
    diagnostics: _FitDiagnostics | None


@dataclass(frozen=True)
class _AnchorSelectionMaps:
    is_anchor_by_id: dict[int, bool]
    nearest_by_id: dict[int, int]
    distance_by_id: dict[int, float]


@dataclass(frozen=True)
class _AnchorSourceXYContext:
    selection: SourceXYAnchorSelectionResult
    is_anchor_by_id: dict[int, bool]
    nearest_by_id: dict[int, int]
    distance_by_id: dict[int, float]
    models_by_group_id: dict[int, dict[tuple[int, int, bool], _AnchorModelContext]]


def _select_anchor_groups(
    *,
    groups: tuple[SourceGroup, ...],
    cfg: PhysicsLiteConfig,
) -> SourceXYAnchorSelectionResult:
    anchor_cfg = cfg.physical_runtime.anchor_selection
    return select_source_xy_stride_anchors(
        groups,
        anchor_stride_source_groups=int(anchor_cfg.anchor_stride_source_groups),
        include_first=bool(anchor_cfg.include_first),
        include_last=bool(anchor_cfg.include_last),
    )


def _assign_anchor_selection_diagnostics(
    arrays: dict[str, np.ndarray],
    *,
    groups: tuple[SourceGroup, ...],
    selection: SourceXYAnchorSelectionResult,
) -> None:
    group_pos_by_id = {
        int(group_id): int(pos)
        for pos, group_id in enumerate(np.asarray(selection.group_ids).tolist())
    }
    for group in groups:
        pos = group_pos_by_id[int(group.group_id)]
        trace_indices = np.asarray(group.trace_indices, dtype=np.int64)
        arrays['physical_anchor_group_id'][trace_indices] = np.int32(group.group_id)
        arrays['physical_anchor_is_anchor'][trace_indices] = np.bool_(
            selection.is_anchor[pos]
        )
        arrays['physical_anchor_nearest_anchor_group_id'][trace_indices] = np.int32(
            selection.nearest_anchor_group_id[pos]
        )
        arrays['physical_anchor_source_distance_m'][trace_indices] = np.float32(
            selection.source_distance_m[pos]
        )


def _record_anchor_selection_diagnostics(
    *,
    selection: SourceXYAnchorSelectionResult,
    cfg: PhysicsLiteConfig,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
) -> None:
    if runtime_diagnostics is None:
        return
    anchor_cfg = cfg.physical_runtime.anchor_selection
    runtime_diagnostics.set_anchor_selection(
        n_anchor_groups=len(selection.anchor_group_ids),
        anchor_stride_source_groups=int(anchor_cfg.anchor_stride_source_groups),
        anchor_selection_mode=str(anchor_cfg.mode),
        source_distance_m=selection.source_distance_m,
    )


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
    selection = _select_anchor_groups(groups=groups, cfg=cfg)
    _assign_anchor_selection_diagnostics(
        arrays,
        groups=groups,
        selection=selection,
    )
    _record_anchor_selection_diagnostics(
        selection=selection,
        cfg=cfg,
        runtime_diagnostics=runtime_diagnostics,
    )
    return selection


def _ensure_anchor_selection(
    selection: SourceXYAnchorSelectionResult | None,
    *,
    arrays: dict[str, np.ndarray],
    groups: tuple[SourceGroup, ...],
    cfg: PhysicsLiteConfig,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
) -> SourceXYAnchorSelectionResult:
    if selection is not None:
        return selection
    selection = _select_anchor_groups(groups=groups, cfg=cfg)
    _assign_anchor_selection_diagnostics(
        arrays,
        groups=groups,
        selection=selection,
    )
    _record_anchor_selection_diagnostics(
        selection=selection,
        cfg=cfg,
        runtime_diagnostics=runtime_diagnostics,
    )
    return selection


def _anchor_model_key(
    group_id: int,
    plan: _ObservationPlan,
) -> tuple[int, int, int, bool]:
    return (int(group_id), int(plan.side), int(plan.segment_id), bool(plan.relaxed))


def _selection_group_maps(
    selection: SourceXYAnchorSelectionResult,
) -> _AnchorSelectionMaps:
    is_anchor_by_id: dict[int, bool] = {}
    nearest_by_id: dict[int, int] = {}
    distance_by_id: dict[int, float] = {}
    group_ids = np.asarray(selection.group_ids, dtype=np.int64)
    is_anchor = np.asarray(selection.is_anchor)
    nearest_anchor_group_id = np.asarray(selection.nearest_anchor_group_id)
    source_distance_m = np.asarray(selection.source_distance_m)
    for pos, group_id in enumerate(group_ids.tolist()):
        gid = int(group_id)
        is_anchor_by_id[gid] = bool(is_anchor[pos])
        nearest_by_id[gid] = int(nearest_anchor_group_id[pos])
        distance_by_id[gid] = float(source_distance_m[pos])
    return _AnchorSelectionMaps(
        is_anchor_by_id=is_anchor_by_id,
        nearest_by_id=nearest_by_id,
        distance_by_id=distance_by_id,
    )


def _anchor_trace_indices(
    *,
    groups: tuple[SourceGroup, ...],
    is_anchor_by_id: Mapping[int, bool],
) -> np.ndarray:
    anchor_trace_chunks = [
        np.asarray(group.trace_indices, dtype=np.int64)
        for group in groups
        if bool(is_anchor_by_id.get(int(group.group_id), False))
    ]
    if not anchor_trace_chunks:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(anchor_trace_chunks)


def _anchor_models_by_group_id(
    anchor_models: Mapping[
        tuple[int, int, int, bool],
        _AnchorModelContext,
    ],
) -> dict[int, dict[tuple[int, int, bool], _AnchorModelContext]]:
    models_by_group_id: dict[
        int,
        dict[tuple[int, int, bool], _AnchorModelContext],
    ] = {}
    for model_key, anchor_context in anchor_models.items():
        anchor_group_id, side, segment_id, relaxed = model_key
        models_by_group_id.setdefault(int(anchor_group_id), {}).setdefault(
            (int(side), int(segment_id), bool(relaxed)),
            anchor_context,
        )
    return models_by_group_id


def fit_anchor_models(
    *,
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
    workspace: PhysicalCenterWorkspace,
    is_anchor_by_id: Mapping[int, bool],
    strategy: object,
    progress: object | None = None,
    progress_context: Mapping[str, object] | None = None,
    fit_model_for_plan: _FitModelForPlan | None = None,
) -> dict[int, dict[tuple[int, int, bool], _AnchorModelContext]]:
    reporter = progress if progress is not None else NullProgressReporter()
    context = dict(progress_context or {})
    runtime_diagnostics = workspace.runtime_diagnostics

    anchor_indices = _anchor_trace_indices(
        groups=build_context.groups,
        is_anchor_by_id=is_anchor_by_id,
    )
    anchor_assignments_by_fit, anchor_assignments = (
        _prepare_fit_context_assignments_for_trace_indices(
            anchor_indices,
            inputs=inputs,
            build_context=build_context,
            workspace=workspace,
        )
    )
    anchor_work_items = _build_fit_context_work_items(
        anchor_assignments_by_fit,
        offset_abs_m=build_context.offset_abs_m,
        pick_t_sec=build_context.pick_t_sec,
        coarse_pmax=inputs.table.coarse_pmax,
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
        inputs=inputs,
        build_context=build_context,
        workspace=workspace,
        strategy=strategy,
        progress=reporter,
        progress_context=context,
        fit_model_for_plan=fit_model_for_plan,
    )
    if runtime_diagnostics is not None:
        anchor_fit_call_delta = max(
            0,
            int(runtime_diagnostics.n_fit_calls) - anchor_fit_calls_before,
        )
        if anchor_fit_call_delta > 0:
            runtime_diagnostics.record_anchor_fit_calls(anchor_fit_call_delta)

    group_id_by_trace = build_context.group_id_by_trace
    anchor_models: dict[tuple[int, int, int, bool], _AnchorModelContext] = {}
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
    return _anchor_models_by_group_id(anchor_models)


def build_anchor_source_xy_context(
    *,
    anchor_selection: SourceXYAnchorSelectionResult | None,
    inputs: PhysicalCenterInputs,
    build_context: PhysicalCenterBuildContext,
    workspace: PhysicalCenterWorkspace,
    strategy: object,
    progress: object | None = None,
    progress_context: Mapping[str, object] | None = None,
    fit_model_for_plan: _FitModelForPlan | None = None,
) -> _AnchorSourceXYContext:
    selection = _ensure_anchor_selection(
        anchor_selection,
        arrays=workspace.arrays,
        groups=build_context.groups,
        cfg=inputs.cfg,
        runtime_diagnostics=workspace.runtime_diagnostics,
    )
    selection_maps = _selection_group_maps(selection)
    models_by_group_id = fit_anchor_models(
        inputs=inputs,
        build_context=build_context,
        workspace=workspace,
        is_anchor_by_id=selection_maps.is_anchor_by_id,
        strategy=strategy,
        progress=progress,
        progress_context=progress_context,
        fit_model_for_plan=fit_model_for_plan,
    )
    return _AnchorSourceXYContext(
        selection=selection,
        is_anchor_by_id=selection_maps.is_anchor_by_id,
        nearest_by_id=selection_maps.nearest_by_id,
        distance_by_id=selection_maps.distance_by_id,
        models_by_group_id=models_by_group_id,
    )
