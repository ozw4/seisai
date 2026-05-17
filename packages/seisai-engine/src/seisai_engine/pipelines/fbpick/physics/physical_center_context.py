"""Orchestration context objects for physical-center fitting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import PhysicsLiteConfig
    from .feasible import FeasibleBandResult
    from .geometry import CoarseGeometry, SourceGroup
    from .merge import MergeResult
    from .physical_center_fallback import _PendingTrendFallback
    from .physical_center_fit import _FitCacheEntry
    from .physical_center_geometry import (
        PhysicalCenterGeometryContext,
        _SignedOffsetSideLabels,
    )
    from .physical_center_observation import (
        _GroupObservationContext,
        _ObservationPlanCache,
    )
    from .pick_table import CoarsePickTable
    from .runtime_diagnostics import PhysicalRuntimeDiagnostics
    from .trend import TrendResult

__all__ = [
    'PhysicalCenterBuildContext',
    'PhysicalCenterInputs',
    'PhysicalCenterWorkspace',
]


@dataclass(frozen=True)
class PhysicalCenterInputs:
    """Immutable inputs for one physical-center build.

    Array-valued inputs are retained by reference to preserve existing behavior.
    """

    coarse_npz: Mapping[str, np.ndarray]
    table: CoarsePickTable
    feasible: FeasibleBandResult
    trend: TrendResult
    merged: MergeResult
    cfg: PhysicsLiteConfig
    trend_provider: object | None = None


@dataclass(frozen=True)
class PhysicalCenterBuildContext:
    """Geometry, source grouping, and fit-preparation state.

    Array-valued fields are retained by reference. Mutable execution state belongs
    in ``PhysicalCenterWorkspace`` instead.
    """

    geometry_context: PhysicalCenterGeometryContext
    pick_t_sec: np.ndarray
    valid_for_fit: np.ndarray
    group_context_by_id: Mapping[int, _GroupObservationContext]
    observation_plan_cache: _ObservationPlanCache
    min_fit_obs: int

    @property
    def geometry(self) -> CoarseGeometry | None:
        return self.geometry_context.geometry

    @property
    def offset_abs_m(self) -> np.ndarray:
        offset_abs_m = self.geometry_context.offset_abs_m
        if offset_abs_m is None:
            msg = 'physical-center build context requires offset_abs_m'
            raise RuntimeError(msg)
        return offset_abs_m

    @property
    def offset_signed_m(self) -> np.ndarray | None:
        return self.geometry_context.offset_signed_m

    @property
    def offset_signed_labels(self) -> _SignedOffsetSideLabels | None:
        return self.geometry_context.offset_signed_labels

    @property
    def offset_source(self) -> int:
        return int(self.geometry_context.offset_source)

    @property
    def groups(self) -> tuple[SourceGroup, ...]:
        return self.geometry_context.groups

    @property
    def groups_by_id(self) -> Mapping[int, SourceGroup]:
        return self.geometry_context.groups_by_id

    @property
    def group_id_by_trace(self) -> np.ndarray:
        return self.geometry_context.group_id_by_trace

    @property
    def source_groups_from_geometry(self) -> bool:
        return bool(self.geometry_context.source_groups_from_geometry)


@dataclass
class PhysicalCenterWorkspace:
    """Mutable execution state for one physical-center build."""

    arrays: dict[str, np.ndarray]
    fit_cache: dict[tuple[int, ...], _FitCacheEntry]
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None
    pending_trend_fallback: _PendingTrendFallback | None = None
