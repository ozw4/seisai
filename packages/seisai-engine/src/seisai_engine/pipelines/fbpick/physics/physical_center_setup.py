"""Common setup for physical-center policy runners."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .feasible import compute_velocity_t0_band_from_arrays
from .physical_center_anchor import _apply_anchor_selection_diagnostics
from .physical_center_context import (
    PhysicalCenterBuildContext,
    PhysicalCenterWorkspace,
)
from .physical_center_fallback import (
    _allocate_result_arrays,
    _as_bool_vector,
    _as_vector,
    _build_disabled_result,
    _emit_fallback_all_and_done,
    _PendingTrendFallback,
)
from .physical_center_fit import _fit_min_pts, _fit_strategy
from .physical_center_geometry import (
    PhysicalCenterGeometryContext,
    build_physical_center_geometry_context,
    build_physical_center_offset_context,
    build_physical_center_source_group_build,
    build_physical_center_source_group_context,
    load_physical_center_geometry,
)
from .physical_center_observation import (
    _build_group_observation_contexts,
    _ObservationPlanCache,
)
from .physical_center_types import (
    PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
    PhysicalCenterFallbackPreflight,
)
from .progress import build_progress_reporter

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .config import PhysicsLiteConfig
    from .feasible import FeasibleBandResult
    from .physical_center_context import PhysicalCenterInputs
    from .physical_center_types import PhysicalCenterResult
    from .pick_table import CoarsePickTable
    from .runtime_diagnostics import PhysicalRuntimeDiagnostics
    from .runtime_policy import SourceXYAnchorSelectionResult


@dataclass(frozen=True)
class PhysicalCenterPolicySetup:
    """Validated state shared by physical-center policy runners."""

    inputs: PhysicalCenterInputs
    build_context: PhysicalCenterBuildContext
    workspace: PhysicalCenterWorkspace
    anchor_selection: SourceXYAnchorSelectionResult | None
    strategy: object
    progress: object
    progress_context: dict[str, object]


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


def _preflight_geometry_two_piece_fallback(
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


def build_physical_center_policy_setup(  # noqa: PLR0915
    *,
    inputs: PhysicalCenterInputs,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
    progress: object | None = None,
    progress_context: Mapping[str, object] | None = None,
) -> PhysicalCenterPolicySetup | PhysicalCenterResult:
    """Build validated shared state for physical-center policy dispatch."""
    table = inputs.table
    feasible = inputs.feasible
    trend = inputs.trend
    merged = inputs.merged
    cfg = inputs.cfg
    reporter = (
        progress
        if progress is not None
        else build_progress_reporter(cfg.physical_runtime.progress)
    )
    context = dict(progress_context or {})
    _validate_table(table)
    n = int(table.n_traces)

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
        geometry = load_physical_center_geometry(inputs.coarse_npz, n_traces=n)
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
            coarse_npz=inputs.coarse_npz,
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

    return PhysicalCenterPolicySetup(
        inputs=inputs,
        build_context=build_context,
        workspace=workspace,
        anchor_selection=anchor_selection,
        strategy=_fit_strategy(cfg),
        progress=reporter,
        progress_context=context,
    )
