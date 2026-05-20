"""Public facade for physical-center fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .physical_center_anchor_policy import run_anchor_source_xy_policy
from .physical_center_context import PhysicalCenterInputs
from .physical_center_full_fit import run_full_fit_policy
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
    PHYSICAL_MODEL_STATUS_SINGLE_LINE_OK,
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
    'PHYSICAL_MODEL_STATUS_SINGLE_LINE_OK',
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
        )
    if fit_policy == 'full':
        return run_full_fit_policy(
            inputs=setup.inputs,
            build_context=setup.build_context,
            workspace=setup.workspace,
            strategy=setup.strategy,
            progress=setup.progress,
            progress_context=setup.progress_context,
        )

    msg = "physical_runtime.fit_policy must be 'full' or 'anchor_source_xy'"
    raise ValueError(msg)
