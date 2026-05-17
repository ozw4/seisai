"""Geometry, offset, and source-group setup for physical-center fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .geometry import (
    CoarseGeometry,
    SourceGroup,
    build_source_groups,
    is_source_xy_degenerate,
    load_coarse_geometry_from_npz,
)
from .physical_center_fallback import _as_bool_vector, _as_vector
from .physical_center_types import (
    PHYSICAL_OFFSET_SOURCE_GEOMETRY,
    PHYSICAL_OFFSET_SOURCE_HEADER,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .config import PhysicsLiteConfig
    from .pick_table import CoarsePickTable


@dataclass(frozen=True)
class _SignedOffsetSideLabels:
    side: np.ndarray
    finite: np.ndarray


@dataclass(frozen=True)
class PhysicalCenterOffsetContext:
    """Resolved offset arrays and provenance for physical-center fitting."""

    offset_abs_m: np.ndarray
    offset_signed_m: np.ndarray | None
    offset_signed_labels: _SignedOffsetSideLabels | None
    offset_source: int


@dataclass(frozen=True)
class PhysicalCenterSourceGroupBuild:
    """Raw source-group build result before trace-index lookup maps."""

    groups: tuple[SourceGroup, ...]
    source_grouping_invalid: bool
    source_groups_from_geometry: bool


@dataclass(frozen=True)
class PhysicalCenterSourceGroupContext:
    """Source groups plus trace and group lookup maps."""

    groups: tuple[SourceGroup, ...]
    groups_by_id: dict[int, SourceGroup]
    group_id_by_trace: np.ndarray
    source_grouping_invalid: bool
    source_groups_from_geometry: bool


@dataclass(frozen=True)
class PhysicalCenterGeometryContext:
    """Loaded geometry, resolved offsets, and resolved source groups."""

    geometry: CoarseGeometry | None
    offset_abs_m: np.ndarray | None
    offset_signed_m: np.ndarray | None
    offset_signed_labels: _SignedOffsetSideLabels | None
    offset_source: int
    groups: tuple[SourceGroup, ...]
    groups_by_id: dict[int, SourceGroup]
    group_id_by_trace: np.ndarray
    source_grouping_invalid: bool
    source_groups_from_geometry: bool
    geometry_required_missing: bool


def _signed_offset_side_labels(
    signed_offset_m: np.ndarray,
    *,
    zero_tol_m: float = 1.0e-6,
    finite_mask: np.ndarray | None = None,
) -> _SignedOffsetSideLabels:
    signed = np.asarray(signed_offset_m, dtype=np.float32)
    if signed.ndim != 1:
        msg = 'signed_offset_m must be 1D'
        raise ValueError(msg)

    zero_tol = float(zero_tol_m)
    if zero_tol < 0.0 or not np.isfinite(zero_tol):
        msg = 'zero_tol_m must be finite and >= 0'
        raise ValueError(msg)

    finite = np.isfinite(signed)
    if finite_mask is not None:
        mask = np.asarray(finite_mask, dtype=np.bool_)
        if mask.shape != signed.shape:
            msg = 'finite_mask must have the same shape as signed_offset_m'
            raise ValueError(msg)
        finite = finite & mask

    side = np.zeros(signed.shape, dtype=np.int8)
    signed_valid = np.asarray(signed[finite], dtype=np.float64)
    side[finite] = np.where(
        np.abs(signed_valid) <= zero_tol,
        0,
        np.where(signed_valid < 0.0, -1, 1),
    ).astype(np.int8)
    return _SignedOffsetSideLabels(side=side, finite=finite)


def _table_offset_abs_m(table: CoarsePickTable, *, n_traces: int) -> np.ndarray:
    offset_m = _as_vector(
        'table.offset_m',
        table.offset_m,
        n_traces=n_traces,
        dtype=np.float32,
    )
    return np.abs(offset_m).astype(np.float32, copy=False)


def _build_table_source_groups(
    table: CoarsePickTable,
    *,
    n_traces: int,
) -> tuple[SourceGroup, ...]:
    shot_id = _as_vector(
        'table.shot_id',
        table.shot_id,
        n_traces=n_traces,
        dtype=np.int32,
    )
    groups: list[SourceGroup] = []
    seen: set[int] = set()
    for shot in shot_id.tolist():
        shot_int = int(shot)
        if shot_int in seen:
            continue
        seen.add(shot_int)
        trace_indices = np.flatnonzero(shot_id == np.int32(shot_int)).astype(
            np.int64,
            copy=False,
        )
        group_id = len(groups)
        groups.append(
            SourceGroup(
                group_id=group_id,
                source_key_x=shot_int,
                source_key_y=0,
                source_x_m=float(group_id),
                source_y_m=0.0,
                trace_indices=trace_indices,
            )
        )
    return tuple(groups)


def _load_source_group_geometry_from_npz(
    coarse_npz: Mapping[str, np.ndarray],
    *,
    n_traces: int,
) -> CoarseGeometry | None:
    if 'source_x_m' not in coarse_npz or 'source_y_m' not in coarse_npz:
        return None
    try:
        source_x_m = _as_vector(
            'source_x_m',
            coarse_npz['source_x_m'],
            n_traces=n_traces,
            dtype=np.float32,
        )
        source_y_m = _as_vector(
            'source_y_m',
            coarse_npz['source_y_m'],
            n_traces=n_traces,
            dtype=np.float32,
        )
        if 'geometry_valid_mask' in coarse_npz:
            geometry_valid_mask = _as_bool_vector(
                'geometry_valid_mask',
                coarse_npz['geometry_valid_mask'],
                n_traces=n_traces,
            )
        else:
            geometry_valid_mask = np.ones((n_traces,), dtype=np.bool_)
    except (TypeError, ValueError):
        return None

    geometry_valid_mask = (
        geometry_valid_mask
        & np.isfinite(source_x_m)
        & np.isfinite(source_y_m)
    ).astype(np.bool_, copy=False)
    zeros = np.zeros((n_traces,), dtype=np.float32)
    return CoarseGeometry(
        source_x_m=source_x_m,
        source_y_m=source_y_m,
        receiver_x_m=zeros,
        receiver_y_m=zeros.copy(),
        offset_abs_geom_m=zeros.copy(),
        geometry_valid_mask=geometry_valid_mask,
        offset_signed_geom_m=None,
    )


def load_physical_center_geometry(
    coarse_npz: Mapping[str, np.ndarray],
    *,
    n_traces: int,
) -> CoarseGeometry | None:
    """Load optional coarse geometry, returning None for invalid inputs."""
    try:
        return load_coarse_geometry_from_npz(coarse_npz, n_traces=n_traces)
    except (KeyError, TypeError, ValueError):
        return None


def build_physical_center_offset_context(
    *,
    geometry: CoarseGeometry | None,
    table: CoarsePickTable,
    cfg: PhysicsLiteConfig,
) -> PhysicalCenterOffsetContext:
    """Resolve offset arrays from geometry or table headers."""
    n_traces = int(table.n_traces)
    use_geometry_offset = bool(cfg.physical_trend.use_geometry_offset)
    segment_by_offset_sign = bool(cfg.physical_trend.segment_by_offset_sign)
    offset_signed_labels = None

    if use_geometry_offset:
        if geometry is None:
            msg = 'geometry is required when use_geometry_offset is True'
            raise ValueError(msg)
        offset_abs_m = _as_vector(
            'geometry.offset_abs_geom_m',
            geometry.offset_abs_geom_m,
            n_traces=n_traces,
            dtype=np.float32,
        )
        if segment_by_offset_sign and geometry.offset_signed_geom_m is not None:
            offset_signed_m = _as_vector(
                'geometry.offset_signed_geom_m',
                geometry.offset_signed_geom_m,
                n_traces=n_traces,
                dtype=np.float32,
            )
            offset_signed_labels = _signed_offset_side_labels(
                offset_signed_m,
                finite_mask=geometry.geometry_valid_mask,
            )
        else:
            offset_signed_m = None
        offset_source = PHYSICAL_OFFSET_SOURCE_GEOMETRY
    else:
        offset_abs_m = _table_offset_abs_m(table, n_traces=n_traces)
        if segment_by_offset_sign:
            offset_signed_m = _as_vector(
                'table.offset_m',
                table.offset_m,
                n_traces=n_traces,
                dtype=np.float32,
            )
            offset_signed_labels = _signed_offset_side_labels(offset_signed_m)
        else:
            offset_signed_m = None
        offset_source = PHYSICAL_OFFSET_SOURCE_HEADER

    return PhysicalCenterOffsetContext(
        offset_abs_m=offset_abs_m,
        offset_signed_m=offset_signed_m,
        offset_signed_labels=offset_signed_labels,
        offset_source=offset_source,
    )


def build_physical_center_source_group_build(
    *,
    coarse_npz: Mapping[str, np.ndarray],
    geometry: CoarseGeometry | None,
    table: CoarsePickTable,
    cfg: PhysicsLiteConfig,
) -> PhysicalCenterSourceGroupBuild:
    """Build source groups from geometry, source xy, or table shot ids."""
    n_traces = int(table.n_traces)
    use_geometry_offset = bool(cfg.physical_trend.use_geometry_offset)
    source_grouping_invalid = False
    source_groups_from_geometry = False
    groups: tuple[SourceGroup, ...] = ()
    source_group_geometry = geometry
    if source_group_geometry is None and not use_geometry_offset:
        source_group_geometry = _load_source_group_geometry_from_npz(
            coarse_npz,
            n_traces=n_traces,
        )
    if source_group_geometry is not None:
        coord_group_tol_m = float(cfg.physical_trend.coord_group_tol_m)
        source_xy_degenerate = is_source_xy_degenerate(
            source_group_geometry,
            table=table,
            coord_group_tol_m=coord_group_tol_m,
        )
        if source_xy_degenerate and use_geometry_offset:
            source_grouping_invalid = True
        if not source_xy_degenerate:
            groups = build_source_groups(
                source_group_geometry,
                coord_group_tol_m=coord_group_tol_m,
            )
            source_groups_from_geometry = len(groups) > 0
    if len(groups) == 0 and not use_geometry_offset:
        groups = _build_table_source_groups(table, n_traces=n_traces)

    return PhysicalCenterSourceGroupBuild(
        groups=groups,
        source_grouping_invalid=source_grouping_invalid,
        source_groups_from_geometry=source_groups_from_geometry,
    )


def build_physical_center_source_group_context(
    *,
    source_group_build: PhysicalCenterSourceGroupBuild,
    n_traces: int,
) -> PhysicalCenterSourceGroupContext:
    """Build lookup maps for a source-group build result."""
    groups = source_group_build.groups
    groups_by_id = {int(group.group_id): group for group in groups}
    group_id_by_trace = np.full((int(n_traces),), -1, dtype=np.int32)
    for group in groups:
        group_id_by_trace[np.asarray(group.trace_indices, dtype=np.int64)] = np.int32(
            group.group_id
        )
    return PhysicalCenterSourceGroupContext(
        groups=groups,
        groups_by_id=groups_by_id,
        group_id_by_trace=group_id_by_trace,
        source_grouping_invalid=source_group_build.source_grouping_invalid,
        source_groups_from_geometry=source_group_build.source_groups_from_geometry,
    )


def build_physical_center_geometry_context(
    *,
    coarse_npz: Mapping[str, np.ndarray],
    table: CoarsePickTable,
    cfg: PhysicsLiteConfig,
    include_offsets: bool = True,
) -> PhysicalCenterGeometryContext:
    """Build the complete geometry context used by preflight and main setup."""
    n_traces = int(table.n_traces)
    geometry = load_physical_center_geometry(coarse_npz, n_traces=n_traces)
    use_geometry_offset = bool(cfg.physical_trend.use_geometry_offset)
    offset_source = (
        PHYSICAL_OFFSET_SOURCE_GEOMETRY
        if use_geometry_offset
        else PHYSICAL_OFFSET_SOURCE_HEADER
    )
    if use_geometry_offset and geometry is None:
        empty_source_groups = build_physical_center_source_group_context(
            source_group_build=PhysicalCenterSourceGroupBuild(
                groups=(),
                source_grouping_invalid=False,
                source_groups_from_geometry=False,
            ),
            n_traces=n_traces,
        )
        return PhysicalCenterGeometryContext(
            geometry=None,
            offset_abs_m=None,
            offset_signed_m=None,
            offset_signed_labels=None,
            offset_source=offset_source,
            groups=empty_source_groups.groups,
            groups_by_id=empty_source_groups.groups_by_id,
            group_id_by_trace=empty_source_groups.group_id_by_trace,
            source_grouping_invalid=empty_source_groups.source_grouping_invalid,
            source_groups_from_geometry=(
                empty_source_groups.source_groups_from_geometry
            ),
            geometry_required_missing=True,
        )

    offset_context = (
        build_physical_center_offset_context(
            geometry=geometry,
            table=table,
            cfg=cfg,
        )
        if include_offsets
        else None
    )
    source_group_build = build_physical_center_source_group_build(
        coarse_npz=coarse_npz,
        geometry=geometry,
        table=table,
        cfg=cfg,
    )
    source_group_context = build_physical_center_source_group_context(
        source_group_build=source_group_build,
        n_traces=n_traces,
    )
    return PhysicalCenterGeometryContext(
        geometry=geometry,
        offset_abs_m=(
            offset_context.offset_abs_m if offset_context is not None else None
        ),
        offset_signed_m=(
            offset_context.offset_signed_m if offset_context is not None else None
        ),
        offset_signed_labels=(
            offset_context.offset_signed_labels if offset_context is not None else None
        ),
        offset_source=(
            offset_context.offset_source
            if offset_context is not None
            else offset_source
        ),
        groups=source_group_context.groups,
        groups_by_id=source_group_context.groups_by_id,
        group_id_by_trace=source_group_context.group_id_by_trace,
        source_grouping_invalid=source_group_context.source_grouping_invalid,
        source_groups_from_geometry=source_group_context.source_groups_from_geometry,
        geometry_required_missing=False,
    )
