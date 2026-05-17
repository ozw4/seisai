"""Public facade for physical-center fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .geometry import (
    CoarseGeometry,
    SourceGroup,
    signed_offset_side_from_geometry,
    split_offset_gap_segments,
)
from .physical_center_anchor_policy import run_anchor_source_xy_policy
from .physical_center_context import PhysicalCenterInputs
from .physical_center_context_fit import (
    _cache_entry_from_fit_task_result,
    _FitContextWorkItem,
    _record_cached_context_hits,
    _record_new_fit_task_diagnostics,
)
from .physical_center_fallback import _allocate_result_arrays
from .physical_center_fit import (
    _fit_cache_key,
    _fit_key_for_obs,
    _fit_min_pts,
    _fit_model_for_plan,
    _FitTaskResult,
    _sample_observation_indices_for_fit,
)
from .physical_center_full_fit import run_full_fit_policy
from .physical_center_geometry import _signed_offset_side_labels
from .physical_center_observation import (
    _build_group_observation_contexts,
    _build_side_observation_context,
    _concat_group_traces,
    _GroupObservationContext,
    _indices_key,
    _obs_with_target_gap_segment,
    _obs_with_target_signed_offset_side,
    _ObservationPlan,
    _ObservationPlanCache,
    _select_group_ids,
    _SideObservationContext,
    _stable_unique,
)
from .physical_center_observation import (
    _build_observation_plan as _build_observation_plan_impl,
)
from .physical_center_prediction import (
    _assign_model_prediction,
    _assign_model_prediction_batch,
)
from .physical_center_setup import (
    _preflight_geometry_two_piece_fallback,
    build_physical_center_policy_setup,
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

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

    from .config import PhysicsLiteConfig
    from .feasible import FeasibleBandResult
    from .merge import MergeResult
    from .pick_table import CoarsePickTable
    from .runtime_diagnostics import PhysicalRuntimeDiagnostics
    from .trend import TrendResult

# TODO(codex): remove after refactor tests migrate; see #158.  # noqa: FIX002
_PHYSICAL_CENTER_PRIVATE_COMPAT = (
    _allocate_result_arrays,
    _assign_model_prediction,
    _assign_model_prediction_batch,
    _build_group_observation_contexts,
    _build_observation_plan_impl,
    _build_side_observation_context,
    _cache_entry_from_fit_task_result,
    CoarseGeometry,
    _concat_group_traces,
    _fit_cache_key,
    _fit_key_for_obs,
    _fit_min_pts,
    _fit_model_for_plan,
    _FitContextWorkItem,
    _FitTaskResult,
    _GroupObservationContext,
    _indices_key,
    _obs_with_target_gap_segment,
    _obs_with_target_signed_offset_side,
    _ObservationPlan,
    _ObservationPlanCache,
    _record_cached_context_hits,
    _record_new_fit_task_diagnostics,
    _sample_observation_indices_for_fit,
    _select_group_ids,
    _SideObservationContext,
    _signed_offset_side_labels,
    signed_offset_side_from_geometry,
    SourceGroup,
    split_offset_gap_segments,
    _stable_unique,
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


def preflight_geometry_two_piece_fallback(
    *,
    coarse_npz: Mapping[str, np.ndarray],
    table: CoarsePickTable,
    cfg: PhysicsLiteConfig,
) -> PhysicalCenterFallbackPreflight:
    """Return the configured geometry fallback decision without fitting."""
    return _preflight_geometry_two_piece_fallback(
        coarse_npz=coarse_npz,
        table=table,
        cfg=cfg,
    )


def _build_observation_plan(  # noqa: PLR0913
    *,
    trace_idx: int,
    target_group_id: int,
    group_context_by_id: Mapping[int, _GroupObservationContext],
    geometry: CoarseGeometry | None,
    offset_abs_m: np.ndarray,
    offset_signed_m: np.ndarray | None,
    cfg: PhysicsLiteConfig,
    min_fit_obs: int | None = None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
    plan_cache: _ObservationPlanCache | None = None,
) -> _ObservationPlan | None:
    if min_fit_obs is None:
        min_fit_obs = 2 * _fit_min_pts(cfg)
    return _build_observation_plan_impl(
        trace_idx=trace_idx,
        target_group_id=target_group_id,
        group_context_by_id=group_context_by_id,
        geometry=geometry,
        offset_abs_m=offset_abs_m,
        offset_signed_m=offset_signed_m,
        cfg=cfg,
        min_fit_obs=min_fit_obs,
        runtime_diagnostics=runtime_diagnostics,
        plan_cache=plan_cache,
    )


def build_geometry_two_piece_physical_center(  # noqa: PLR0913
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
    """Build physical-center picks with the configured fit policy."""
    inputs = PhysicalCenterInputs(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=cfg,
        trend_provider=trend_provider,
    )
    setup = build_physical_center_policy_setup(
        inputs=inputs,
        runtime_diagnostics=runtime_diagnostics,
        progress=progress,
        progress_context=progress_context,
    )
    if isinstance(setup, PhysicalCenterResult):
        return setup

    fit_policy = str(cfg.physical_runtime.fit_policy)
    if fit_policy == 'anchor_source_xy':
        return run_anchor_source_xy_policy(
            inputs=setup.inputs,
            build_context=setup.build_context,
            workspace=setup.workspace,
            anchor_selection=setup.anchor_selection,
            strategy=setup.strategy,
            progress=setup.progress,
            progress_context=setup.progress_context,
            fit_model_for_plan=_fit_model_for_plan,
        )
    if fit_policy == 'full':
        return run_full_fit_policy(
            inputs=setup.inputs,
            build_context=setup.build_context,
            workspace=setup.workspace,
            strategy=setup.strategy,
            progress=setup.progress,
            progress_context=setup.progress_context,
            fit_model_for_plan=_fit_model_for_plan,
        )

    msg = "physical_runtime.fit_policy must be 'full' or 'anchor_source_xy'"
    raise ValueError(msg)
