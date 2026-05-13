from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from seisai_pick.trend.trend_fit_strategy import TwoPieceRansacAutoBreakStrategy

from .config import PhysicsLiteConfig
from .feasible import FeasibleBandResult, compute_velocity_t0_band_from_arrays
from .geometry import (
    CoarseGeometry,
    SourceGroup,
    build_source_groups,
    is_source_xy_degenerate,
    load_coarse_geometry_from_npz,
    select_nearest_source_groups,
    signed_offset_side_from_geometry,
    split_offset_gap_segments,
)
from .merge import MergeResult
from .pick_table import CoarsePickTable
from .runtime_diagnostics import PhysicalRuntimeDiagnostics
from .runtime_policy import (
    SourceXYAnchorSelectionResult,
    select_source_xy_stride_anchors,
)
from .trend import TrendResult

PHYSICAL_MODEL_STATUS_TWO_PIECE_OK = 0
PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT = 1
PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND = 2
PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP = 3
PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST = 4
PHYSICAL_MODEL_STATUS_GEOMETRY_INVALID = 5
PHYSICAL_MODEL_STATUS_INSUFFICIENT_OBSERVATIONS = 6
PHYSICAL_MODEL_STATUS_FIT_FAILED = 7
PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED = 8

PHYSICAL_MODEL_STATUS_LABELS = {
    PHYSICAL_MODEL_STATUS_TWO_PIECE_OK: 'two_piece_ok',
    PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT: 'relaxed_segment_ok',
    PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND: 'fallback_existing_trend',
    PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP: 'fallback_feasible_clip',
    PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST: 'fallback_robust',
    PHYSICAL_MODEL_STATUS_GEOMETRY_INVALID: 'geometry_invalid',
    PHYSICAL_MODEL_STATUS_INSUFFICIENT_OBSERVATIONS: 'insufficient_observations',
    PHYSICAL_MODEL_STATUS_FIT_FAILED: 'fit_failed',
    PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED: 'physical_disabled',
}

PHYSICAL_MODEL_FAILURE_NONE = 0
PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED = 1
PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID = 2
PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS = 3
PHYSICAL_MODEL_FAILURE_FIT_FAILED = 4
PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID = 5

PHYSICAL_MODEL_FAILURE_LABELS = {
    PHYSICAL_MODEL_FAILURE_NONE: 'none',
    PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED: 'physical_disabled',
    PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID: 'geometry_invalid',
    PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS: 'insufficient_observations',
    PHYSICAL_MODEL_FAILURE_FIT_FAILED: 'fit_failed',
    PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID: 'prediction_invalid',
}

PHYSICAL_OFFSET_SOURCE_NONE = 0
PHYSICAL_OFFSET_SOURCE_GEOMETRY = 1
PHYSICAL_OFFSET_SOURCE_HEADER = 2

PHYSICAL_OFFSET_SOURCE_LABELS = {
    PHYSICAL_OFFSET_SOURCE_NONE: 'none',
    PHYSICAL_OFFSET_SOURCE_GEOMETRY: 'geometry_offset',
    PHYSICAL_OFFSET_SOURCE_HEADER: 'header_offset',
}

PHYSICAL_RUNTIME_FIT_SOURCE_FULL_FIT = 0
PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT = 1
PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE = 2
PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_FULL_FIT_NO_COMPATIBLE_ANCHOR = 3
PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND = 4
PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST = 5
PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT = 6

PHYSICAL_RUNTIME_FIT_SOURCE_LABELS = {
    PHYSICAL_RUNTIME_FIT_SOURCE_FULL_FIT: 'full_fit',
    PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT: 'anchor_fit',
    PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE: 'nearest_anchor_reuse',
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_FULL_FIT_NO_COMPATIBLE_ANCHOR: (
        'fallback_full_fit_no_compatible_anchor'
    ),
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND: 'fallback_existing_trend',
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST: 'fallback_robust',
    PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT: 'adaptive_refit',
}

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
    'PhysicalCenterResult',
    'build_geometry_two_piece_physical_center',
]


@dataclass(frozen=True)
class PhysicalCenterResult:
    physical_center_i: np.ndarray
    physical_center_t_sec: np.ndarray
    fine_center_i: np.ndarray
    fine_center_t_sec: np.ndarray
    physical_model_status: np.ndarray
    physical_model_failure_reason: np.ndarray
    physical_offset_source: np.ndarray
    physical_model_break_offset_m: np.ndarray
    physical_model_slope_near_s_per_m: np.ndarray
    physical_model_slope_far_s_per_m: np.ndarray
    physical_model_velocity_near_m_s: np.ndarray
    physical_model_velocity_far_m_s: np.ndarray
    physical_model_neighbor_count: np.ndarray
    physical_prefilter_valid_count: np.ndarray
    physical_model_segment_id: np.ndarray
    physical_model_side: np.ndarray
    physical_model_resid_p50_ms: np.ndarray
    physical_model_resid_p90_ms: np.ndarray
    physical_anchor_group_id: np.ndarray
    physical_anchor_is_anchor: np.ndarray
    physical_anchor_nearest_anchor_group_id: np.ndarray
    physical_anchor_source_distance_m: np.ndarray
    physical_runtime_t0_shift_ms: np.ndarray
    physical_runtime_reuse_resid_p50_ms: np.ndarray
    physical_runtime_reuse_resid_p90_ms: np.ndarray
    physical_runtime_reuse_valid_count: np.ndarray
    physical_runtime_refit_mask: np.ndarray
    physical_runtime_fit_source: np.ndarray


@dataclass(frozen=True)
class _GroupObservationContext:
    group_id: int
    neighbor_group_ids: np.ndarray
    neighbor_indices: np.ndarray
    valid_obs_indices: np.ndarray
    neighbor_count: int
    prefilter_valid_count: int


@dataclass(frozen=True)
class _ObservationPlan:
    obs_indices: np.ndarray
    neighbor_count: int
    prefilter_valid_count: int
    segment_id: int
    side: int
    relaxed: bool


@dataclass(frozen=True)
class _FitCacheEntry:
    model: object | None
    diagnostics: tuple[float, float, float, float, float, float, float] | None
    fit_failed: bool
    diagnostics_computed: bool = False


@dataclass(frozen=True)
class _TraceFitResult:
    plan: _ObservationPlan | None
    trend_model: object | None
    diagnostics: tuple[float, float, float, float, float, float, float] | None
    fit_call_delta: int
    assigned_from_model: bool


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


def _as_vector(name: str, value: np.ndarray, *, n_traces: int, dtype) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 1 or int(arr.shape[0]) != int(n_traces):
        msg = f'{name} must be 1D with length n_traces'
        raise ValueError(msg)
    return arr


def _as_bool_vector(name: str, value: np.ndarray, *, n_traces: int) -> np.ndarray:
    return _as_vector(name, value, n_traces=n_traces, dtype=np.bool_).astype(
        np.bool_,
        copy=False,
    )


def _is_valid_pick_i(value: int, *, n_samples_orig: int) -> bool:
    return 0 <= int(value) < int(n_samples_orig)


def _fallback_center_for_trace(
    trace_idx: int,
    *,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    merged: MergeResult,
) -> tuple[int, np.float32, int]:
    n_samples = int(table.n_samples_orig)
    dt = float(table.dt_scalar_sec)
    idx = int(trace_idx)

    trend_i = int(np.asarray(trend.trend_center_i, dtype=np.int64)[idx])
    trend_t = float(np.asarray(trend.trend_center_sec, dtype=np.float32)[idx])
    if _is_valid_pick_i(trend_i, n_samples_orig=n_samples) and np.isfinite(trend_t):
        return (
            trend_i,
            np.float32(trend_i * dt),
            PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND,
        )

    robust_i = int(np.asarray(merged.robust_pick_i, dtype=np.int64)[idx])
    lo_sec = float(np.asarray(feasible.feasible_lo_sec, dtype=np.float32)[idx])
    hi_sec = float(np.asarray(feasible.feasible_hi_sec, dtype=np.float32)[idx])
    if np.isfinite(lo_sec) and np.isfinite(hi_sec) and lo_sec <= hi_sec:
        lo_i = int(np.ceil(lo_sec / dt))
        hi_i = int(np.floor(hi_sec / dt))
        lo_i = int(np.clip(lo_i, 0, n_samples - 1))
        hi_i = int(np.clip(hi_i, 0, n_samples - 1))
        if lo_i <= hi_i:
            clipped_i = int(np.clip(robust_i, lo_i, hi_i))
            return (
                clipped_i,
                np.float32(clipped_i * dt),
                PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP,
            )

    robust_i = int(np.clip(robust_i, 0, n_samples - 1))
    return (
        robust_i,
        np.float32(robust_i * dt),
        PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST,
    )


def _allocate_result_arrays(table: CoarsePickTable) -> dict[str, np.ndarray]:
    n = int(table.n_traces)
    return {
        'physical_center_i': np.zeros((n,), dtype=np.int32),
        'physical_center_t_sec': np.zeros((n,), dtype=np.float32),
        'fine_center_i': np.zeros((n,), dtype=np.int32),
        'fine_center_t_sec': np.zeros((n,), dtype=np.float32),
        'physical_model_status': np.zeros((n,), dtype=np.uint8),
        'physical_model_failure_reason': np.zeros((n,), dtype=np.uint8),
        'physical_offset_source': np.zeros((n,), dtype=np.uint8),
        'physical_model_break_offset_m': np.full((n,), np.nan, dtype=np.float32),
        'physical_model_slope_near_s_per_m': np.full((n,), np.nan, dtype=np.float32),
        'physical_model_slope_far_s_per_m': np.full((n,), np.nan, dtype=np.float32),
        'physical_model_velocity_near_m_s': np.full((n,), np.nan, dtype=np.float32),
        'physical_model_velocity_far_m_s': np.full((n,), np.nan, dtype=np.float32),
        'physical_model_neighbor_count': np.zeros((n,), dtype=np.int32),
        'physical_prefilter_valid_count': np.zeros((n,), dtype=np.int32),
        'physical_model_segment_id': np.full((n,), -1, dtype=np.int32),
        'physical_model_side': np.zeros((n,), dtype=np.int8),
        'physical_model_resid_p50_ms': np.full((n,), np.nan, dtype=np.float32),
        'physical_model_resid_p90_ms': np.full((n,), np.nan, dtype=np.float32),
        'physical_anchor_group_id': np.full((n,), -1, dtype=np.int32),
        'physical_anchor_is_anchor': np.zeros((n,), dtype=np.bool_),
        'physical_anchor_nearest_anchor_group_id': np.full((n,), -1, dtype=np.int32),
        'physical_anchor_source_distance_m': np.full((n,), np.nan, dtype=np.float32),
        'physical_runtime_t0_shift_ms': np.full((n,), np.nan, dtype=np.float32),
        'physical_runtime_reuse_resid_p50_ms': np.full(
            (n,),
            np.nan,
            dtype=np.float32,
        ),
        'physical_runtime_reuse_resid_p90_ms': np.full(
            (n,),
            np.nan,
            dtype=np.float32,
        ),
        'physical_runtime_reuse_valid_count': np.zeros((n,), dtype=np.int32),
        'physical_runtime_refit_mask': np.zeros((n,), dtype=np.bool_),
        'physical_runtime_fit_source': np.zeros((n,), dtype=np.uint8),
    }


def _finalize_result(arrays: dict[str, np.ndarray]) -> PhysicalCenterResult:
    arrays['fine_center_i'][:] = arrays['physical_center_i']
    arrays['fine_center_t_sec'][:] = arrays['physical_center_t_sec']
    return PhysicalCenterResult(**arrays)


def _assign_fallback(
    arrays: dict[str, np.ndarray],
    trace_idx: int,
    *,
    failure_reason: int,
    runtime_fit_source: int = PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    merged: MergeResult,
) -> None:
    center_i, center_t, fallback_status = _fallback_center_for_trace(
        trace_idx,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
    )
    arrays['physical_center_i'][trace_idx] = np.int32(center_i)
    arrays['physical_center_t_sec'][trace_idx] = np.float32(center_t)
    arrays['physical_model_status'][trace_idx] = np.uint8(fallback_status)
    arrays['physical_model_failure_reason'][trace_idx] = np.uint8(failure_reason)
    source = int(runtime_fit_source)
    if fallback_status == PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST:
        source = PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST
    arrays['physical_runtime_fit_source'][trace_idx] = np.uint8(source)


def _assign_robust_fallback(
    arrays: dict[str, np.ndarray],
    trace_idx: int,
    *,
    failure_reason: int,
    table: CoarsePickTable,
    merged: MergeResult,
) -> None:
    n_samples = int(table.n_samples_orig)
    dt = float(table.dt_scalar_sec)
    robust_i = int(np.asarray(merged.robust_pick_i, dtype=np.int64)[int(trace_idx)])
    robust_i = int(np.clip(robust_i, 0, n_samples - 1))
    arrays['physical_center_i'][trace_idx] = np.int32(robust_i)
    arrays['physical_center_t_sec'][trace_idx] = np.float32(robust_i * dt)
    arrays['physical_model_status'][trace_idx] = np.uint8(
        PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST
    )
    arrays['physical_model_failure_reason'][trace_idx] = np.uint8(failure_reason)
    arrays['physical_runtime_fit_source'][trace_idx] = np.uint8(
        PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST
    )


def _assign_fallback_all(
    *,
    failure_reason: int,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    merged: MergeResult,
) -> PhysicalCenterResult:
    arrays = _allocate_result_arrays(table)
    for trace_idx in range(int(table.n_traces)):
        _assign_fallback(
            arrays,
            trace_idx,
            failure_reason=failure_reason,
            runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND,
            table=table,
            feasible=feasible,
            trend=trend,
            merged=merged,
        )
    return _finalize_result(arrays)


def _build_disabled_result(
    table: CoarsePickTable,
    trend: TrendResult,
) -> PhysicalCenterResult:
    arrays = _allocate_result_arrays(table)
    center_i = _as_vector(
        'trend.trend_center_i',
        trend.trend_center_i,
        n_traces=table.n_traces,
        dtype=np.int32,
    )
    center_t = _as_vector(
        'trend.trend_center_sec',
        trend.trend_center_sec,
        n_traces=table.n_traces,
        dtype=np.float32,
    )
    arrays['physical_center_i'][:] = center_i
    arrays['physical_center_t_sec'][:] = center_t
    arrays['physical_model_status'][:] = np.uint8(
        PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED
    )
    arrays['physical_model_failure_reason'][:] = np.uint8(
        PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED
    )
    arrays['physical_runtime_fit_source'][:] = np.uint8(
        PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND
    )
    return _finalize_result(arrays)


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


def _stable_unique(indices: np.ndarray) -> np.ndarray:
    arr = np.asarray(indices, dtype=np.int64)
    if arr.ndim != 1:
        msg = 'indices must be 1D'
        raise ValueError(msg)
    seen: set[int] = set()
    out: list[int] = []
    for value in arr.tolist():
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return np.asarray(out, dtype=np.int64)


def _append_index(indices: np.ndarray, trace_idx: int) -> np.ndarray:
    return _stable_unique(
        np.concatenate(
            [
                np.asarray(indices, dtype=np.int64),
                np.asarray([int(trace_idx)], dtype=np.int64),
            ]
        )
    )


def _concat_group_traces(
    group_ids: np.ndarray,
    *,
    groups_by_id: Mapping[int, SourceGroup],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for group_id in np.asarray(group_ids, dtype=np.int64).tolist():
        group = groups_by_id.get(int(group_id))
        if group is None:
            continue
        chunks.append(np.asarray(group.trace_indices, dtype=np.int64))
    if not chunks:
        return np.zeros((0,), dtype=np.int64)
    return _stable_unique(np.concatenate(chunks))


def _trace_position_map(indices: np.ndarray) -> dict[int, int]:
    return {int(trace_idx): int(pos) for pos, trace_idx in enumerate(indices.tolist())}


def _select_group_ids(
    *,
    groups: tuple[SourceGroup, ...],
    target_group_id: int,
    cfg: PhysicsLiteConfig,
    use_neighbor_context: bool,
) -> np.ndarray:
    if not bool(use_neighbor_context) or not bool(cfg.neighbor_context.enabled):
        return np.asarray([int(target_group_id)], dtype=np.int64)
    return select_nearest_source_groups(
        groups,
        target_group_id=int(target_group_id),
        k_neighbors=int(cfg.neighbor_context.k_neighbors),
        max_source_distance_m=cfg.neighbor_context.max_source_distance_m,
        include_self=bool(cfg.neighbor_context.include_self),
    )


def _build_group_observation_contexts(
    *,
    groups: tuple[SourceGroup, ...],
    groups_by_id: Mapping[int, SourceGroup],
    valid_for_fit: np.ndarray,
    cfg: PhysicsLiteConfig,
    use_neighbor_context: bool,
) -> dict[int, _GroupObservationContext]:
    valid_mask = np.asarray(valid_for_fit, dtype=np.bool_)
    contexts: dict[int, _GroupObservationContext] = {}
    for group in groups:
        group_id = int(group.group_id)
        neighbor_group_ids = _select_group_ids(
            groups=groups,
            target_group_id=group_id,
            cfg=cfg,
            use_neighbor_context=use_neighbor_context,
        )
        neighbor_indices = _concat_group_traces(
            neighbor_group_ids,
            groups_by_id=groups_by_id,
        )
        valid_obs_indices = neighbor_indices[valid_mask[neighbor_indices]]
        contexts[group_id] = _GroupObservationContext(
            group_id=group_id,
            neighbor_group_ids=neighbor_group_ids,
            neighbor_indices=neighbor_indices,
            valid_obs_indices=valid_obs_indices,
            neighbor_count=int(neighbor_group_ids.size),
            prefilter_valid_count=int(valid_obs_indices.size),
        )
    return contexts


def _obs_with_target_side(
    *,
    trace_idx: int,
    obs_indices: np.ndarray,
    geometry: CoarseGeometry,
) -> tuple[np.ndarray, int, bool]:
    context_indices = _append_index(obs_indices, trace_idx)
    signed = signed_offset_side_from_geometry(geometry, context_indices)
    if not bool(signed.reliable):
        return np.asarray(obs_indices, dtype=np.int64), 0, False

    pos = _trace_position_map(context_indices)
    target_side = int(signed.side[pos[int(trace_idx)]])
    obs_side = np.asarray(
        [int(signed.side[pos[int(obs_idx)]]) for obs_idx in obs_indices.tolist()],
        dtype=np.int8,
    )
    return (
        np.asarray(obs_indices, dtype=np.int64)[obs_side == target_side],
        target_side,
        True,
    )


def _obs_with_target_signed_offset_side(
    *,
    trace_idx: int,
    obs_indices: np.ndarray,
    signed_offset_m: np.ndarray,
    zero_tol_m: float = 1.0e-6,
) -> tuple[np.ndarray, int, bool]:
    context_indices = _append_index(obs_indices, trace_idx)
    signed = np.asarray(signed_offset_m, dtype=np.float32)[context_indices]
    finite = np.isfinite(signed)
    if int(np.count_nonzero(finite)) < 2:
        return np.asarray(obs_indices, dtype=np.int64), 0, False

    zero_tol = float(zero_tol_m)
    if zero_tol < 0.0 or not np.isfinite(zero_tol):
        msg = 'zero_tol_m must be finite and >= 0'
        raise ValueError(msg)

    side = np.zeros((context_indices.size,), dtype=np.int8)
    signed_valid = np.asarray(signed[finite], dtype=np.float64)
    if not np.any(np.abs(signed_valid) > zero_tol):
        return np.asarray(obs_indices, dtype=np.int64), 0, False

    side[finite] = np.where(
        np.abs(signed_valid) <= zero_tol,
        0,
        np.where(signed_valid < 0.0, -1, 1),
    ).astype(np.int8)
    pos = _trace_position_map(context_indices)
    target_side = int(side[pos[int(trace_idx)]])
    obs_side = np.asarray(
        [int(side[pos[int(obs_idx)]]) for obs_idx in obs_indices.tolist()],
        dtype=np.int8,
    )
    return (
        np.asarray(obs_indices, dtype=np.int64)[obs_side == target_side],
        target_side,
        True,
    )


def _obs_with_target_gap_segment(
    *,
    trace_idx: int,
    obs_indices: np.ndarray,
    offset_abs_m: np.ndarray,
    cfg: PhysicsLiteConfig,
) -> tuple[np.ndarray, int]:
    context_indices = _append_index(obs_indices, trace_idx)
    segment_id = split_offset_gap_segments(
        np.asarray(offset_abs_m, dtype=np.float32)[context_indices],
        split_by_offset_gap=bool(cfg.physical_trend.split_by_offset_gap),
        gap_ratio=float(cfg.physical_trend.gap_ratio),
        min_gap_m=cfg.physical_trend.min_gap_m,
    )
    pos = _trace_position_map(context_indices)
    target_segment_id = int(segment_id[pos[int(trace_idx)]])
    obs_segment_id = np.asarray(
        [int(segment_id[pos[int(obs_idx)]]) for obs_idx in obs_indices.tolist()],
        dtype=np.int64,
    )
    return (
        np.asarray(obs_indices, dtype=np.int64)[obs_segment_id == target_segment_id],
        target_segment_id,
    )


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
) -> _ObservationPlan | None:
    with (
        runtime_diagnostics.time_block('neighbor_plan_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        group_context = group_context_by_id.get(int(target_group_id))
    if group_context is None:
        msg = f'observation context not found for group_id={int(target_group_id)}'
        raise ValueError(msg)

    valid_obs = group_context.valid_obs_indices
    neighbor_count = int(group_context.neighbor_count)
    prefilter_valid_count = int(group_context.prefilter_valid_count)
    min_fit_obs = 2 * int(cfg.two_piece_ransac.min_pts)

    if prefilter_valid_count < min_fit_obs:
        return _ObservationPlan(
            obs_indices=valid_obs,
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=-1,
            side=0,
            relaxed=False,
        )

    side_obs = valid_obs
    side = 0
    side_reliable = False
    with (
        runtime_diagnostics.time_block('side_segment_build_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        if bool(cfg.physical_trend.segment_by_offset_sign):
            if offset_signed_m is not None:
                side_obs, side, side_reliable = _obs_with_target_signed_offset_side(
                    trace_idx=trace_idx,
                    obs_indices=valid_obs,
                    signed_offset_m=offset_signed_m,
                )
            elif geometry is not None:
                side_obs, side, side_reliable = _obs_with_target_side(
                    trace_idx=trace_idx,
                    obs_indices=valid_obs,
                    geometry=geometry,
                )

        segment_obs = side_obs
        segment_id = 0
        if bool(cfg.physical_trend.split_by_offset_gap):
            segment_obs, segment_id = _obs_with_target_gap_segment(
                trace_idx=trace_idx,
                obs_indices=side_obs,
                offset_abs_m=offset_abs_m,
                cfg=cfg,
            )

    if int(segment_obs.size) >= min_fit_obs:
        return _ObservationPlan(
            obs_indices=segment_obs,
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=segment_id,
            side=side,
            relaxed=False,
        )

    if (
        bool(cfg.physical_trend.split_by_offset_gap)
        and int(side_obs.size) >= min_fit_obs
    ):
        return _ObservationPlan(
            obs_indices=side_obs,
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=0,
            side=side,
            relaxed=True,
        )

    if side_reliable and int(valid_obs.size) >= min_fit_obs:
        return _ObservationPlan(
            obs_indices=valid_obs,
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=0,
            side=0,
            relaxed=True,
        )

    return _ObservationPlan(
        obs_indices=segment_obs,
        neighbor_count=neighbor_count,
        prefilter_valid_count=prefilter_valid_count,
        segment_id=segment_id,
        side=side,
        relaxed=False,
    )


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _fit_strategy(cfg: PhysicsLiteConfig) -> TwoPieceRansacAutoBreakStrategy:
    return TwoPieceRansacAutoBreakStrategy(
        n_iter=int(cfg.two_piece_ransac.n_iter),
        inlier_th_ms=float(cfg.two_piece_ransac.inlier_th_ms),
        min_pts=int(cfg.two_piece_ransac.min_pts),
        n_break_cand=int(cfg.two_piece_ransac.n_break_cand),
        q_lo=float(cfg.two_piece_ransac.q_lo),
        q_hi=float(cfg.two_piece_ransac.q_hi),
        seed=int(cfg.two_piece_ransac.seed),
        slope_eps=float(cfg.two_piece_ransac.slope_eps),
        sort_offsets=bool(cfg.two_piece_ransac.sort_offsets),
    )


def _median_time_position(local_positions: np.ndarray, y_obs: np.ndarray) -> int:
    positions = np.asarray(local_positions, dtype=np.int64)
    if positions.size == 0:
        msg = 'local_positions must be non-empty'
        raise ValueError(msg)
    y_values = np.asarray(y_obs, dtype=np.float32)[positions]
    finite = np.isfinite(y_values)
    if not np.any(finite):
        return int(positions[0])
    finite_positions = positions[finite]
    finite_y = y_values[finite]
    median_y = float(np.median(finite_y.astype(np.float64, copy=False)))
    return int(finite_positions[int(np.argmin(np.abs(finite_y - median_y)))])


def _stable_observation_seed(seed: int, values: np.ndarray, *, bin_id: int) -> int:
    acc = int(seed) & 0xFFFFFFFF
    for value in np.asarray(values, dtype=np.int64).tolist():
        acc = (acc * 1664525 + int(value) + 1013904223) & 0xFFFFFFFF
    return int((acc + int(bin_id) * 374761393) & 0xFFFFFFFF)


def _bin_representative_position(
    *,
    local_positions: np.ndarray,
    obs_indices: np.ndarray,
    y_obs: np.ndarray,
    p_obs: np.ndarray | None,
    bin_pick: str,
    random_seed: int,
    bin_id: int,
) -> int:
    positions = np.asarray(local_positions, dtype=np.int64)
    if positions.size == 0:
        msg = 'local_positions must be non-empty'
        raise ValueError(msg)
    if bin_pick == 'median_time':
        return _median_time_position(positions, y_obs)
    if bin_pick == 'random':
        seed = _stable_observation_seed(
            random_seed,
            np.asarray(obs_indices, dtype=np.int64)[positions],
            bin_id=int(bin_id),
        )
        rng = np.random.default_rng(seed)
        return int(positions[int(rng.integers(0, int(positions.size)))])

    if p_obs is not None:
        p_values = np.asarray(p_obs, dtype=np.float32)[positions]
        finite = np.isfinite(p_values)
        if np.any(finite):
            finite_positions = positions[finite]
            finite_p = p_values[finite]
            return int(finite_positions[int(np.argmax(finite_p))])
    return _median_time_position(positions, y_obs)


def _evenly_spaced_positions(length: int, count: int) -> np.ndarray:
    n = int(length)
    k = int(count)
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    raw = np.linspace(0.0, float(n - 1), num=k)
    used: set[int] = set()
    out: list[int] = []
    for value in raw.tolist():
        pos = int(np.rint(float(value)))
        if pos in used:
            for delta in range(1, n):
                left = pos - delta
                right = pos + delta
                if left >= 0 and left not in used:
                    pos = left
                    break
                if right < n and right not in used:
                    pos = right
                    break
        used.add(pos)
        out.append(pos)
    return np.asarray(sorted(out), dtype=np.int64)


def _limit_selected_positions(
    selected_count: int,
    *,
    max_count: int,
    preserve_edge_bins: bool,
) -> np.ndarray:
    n = int(selected_count)
    max_n = int(max_count)
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    if bool(preserve_edge_bins) and max_n >= 2 and n >= 2:
        interior_count = max_n - 2
        interior = _evenly_spaced_positions(n - 2, interior_count) + 1
        return np.concatenate(
            [
                np.asarray([0], dtype=np.int64),
                interior,
                np.asarray([n - 1], dtype=np.int64),
            ]
        )
    return _evenly_spaced_positions(n, max_n)


def _sample_observation_indices_for_fit(
    *,
    obs_indices: np.ndarray,
    offset_abs_m: np.ndarray,
    pick_t_sec: np.ndarray,
    coarse_pmax: np.ndarray | None,
    cfg: PhysicsLiteConfig,
    min_required_obs: int = 0,
) -> np.ndarray:
    sampling = cfg.physical_runtime.observation_sampling
    obs = np.asarray(obs_indices, dtype=np.int64)
    insufficient = np.zeros((0,), dtype=np.int64)
    if not bool(sampling.enabled):
        return obs
    max_obs = int(sampling.max_obs_per_fit)
    if int(obs.size) <= max_obs:
        return obs

    x_obs = np.asarray(offset_abs_m, dtype=np.float32)[obs]
    y_obs = np.asarray(pick_t_sec, dtype=np.float32)[obs]
    finite = np.isfinite(x_obs) & np.isfinite(y_obs)
    finite_positions = np.flatnonzero(finite).astype(np.int64, copy=False)
    if int(finite_positions.size) == 0:
        return obs

    finite_x = x_obs[finite_positions]
    x_min = float(np.min(finite_x))
    x_max = float(np.max(finite_x))
    if (not np.isfinite(x_min)) or (not np.isfinite(x_max)) or x_max <= x_min:
        return obs

    n_bins = min(int(sampling.n_offset_bins), int(finite_positions.size))
    edges = np.linspace(x_min, x_max, num=n_bins + 1, dtype=np.float64)
    bin_ids = np.searchsorted(edges, finite_x.astype(np.float64), side='right') - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1).astype(np.int64, copy=False)

    p_obs = (
        None
        if coarse_pmax is None
        else np.asarray(coarse_pmax, dtype=np.float32)[obs]
    )
    selected_positions: list[int] = []
    for bin_id in range(n_bins):
        in_bin = finite_positions[bin_ids == int(bin_id)]
        if int(in_bin.size) == 0:
            continue
        selected_positions.append(
            _bin_representative_position(
                local_positions=in_bin,
                obs_indices=obs,
                y_obs=y_obs,
                p_obs=p_obs,
                bin_pick=str(sampling.bin_pick),
                random_seed=int(cfg.two_piece_ransac.seed),
                bin_id=int(bin_id),
            )
        )

    selected = obs[np.asarray(selected_positions, dtype=np.int64)]
    min_after = max(
        int(sampling.min_obs_per_fit_after_sampling),
        int(min_required_obs),
    )
    if int(selected.size) < min_after:
        return insufficient
    if int(selected.size) > max_obs:
        keep = _limit_selected_positions(
            int(selected.size),
            max_count=max_obs,
            preserve_edge_bins=bool(sampling.preserve_edge_bins),
        )
        selected = selected[keep]
    if int(selected.size) < min_after:
        return insufficient
    return np.asarray(selected, dtype=np.int64)


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


def _model_diagnostics(
    trend_model,
    *,
    obs_offsets_m: np.ndarray,
    obs_times_sec: np.ndarray,
) -> tuple[float, float, float, float, float, float, float]:
    edges = _tensor_to_numpy(trend_model.edges).astype(np.float32, copy=False)
    coef = _tensor_to_numpy(trend_model.coef).astype(np.float32, copy=False)
    if edges.shape != (3,) or coef.shape != (2, 2):
        msg = 'trend model must expose edges (3,) and coef (2,2)'
        raise ValueError(msg)

    slope_near = float(coef[0, 0])
    slope_far = float(coef[1, 0])
    velocity_near = (
        1.0 / slope_near
        if np.isfinite(slope_near) and slope_near > 0.0
        else np.nan
    )
    velocity_far = (
        1.0 / slope_far
        if np.isfinite(slope_far) and slope_far > 0.0
        else np.nan
    )

    x_obs = torch.as_tensor(obs_offsets_m, dtype=torch.float32)
    pred = _tensor_to_numpy(trend_model.predict(x_obs)).astype(np.float64, copy=False)
    residual_ms = np.abs(np.asarray(obs_times_sec, dtype=np.float64) - pred) * 1000.0
    residual_ms = residual_ms[np.isfinite(residual_ms)]
    if residual_ms.size == 0:
        resid_p50 = np.nan
        resid_p90 = np.nan
    else:
        resid_p50 = float(np.percentile(residual_ms, 50.0))
        resid_p90 = float(np.percentile(residual_ms, 90.0))

    return (
        float(edges[1]),
        slope_near,
        slope_far,
        velocity_near,
        velocity_far,
        resid_p50,
        resid_p90,
    )


def _fit_cache_key(plan: _ObservationPlan) -> tuple[int, ...]:
    return tuple(np.asarray(plan.obs_indices, dtype=np.int64).tolist())


def _offset_spread_failure_reason(
    x_obs: np.ndarray,
    *,
    min_pts: int,
    min_offset_spread_m: float,
) -> int | None:
    finite_x = np.asarray(x_obs, dtype=np.float64)
    finite_x = finite_x[np.isfinite(finite_x)]
    if int(finite_x.size) < int(min_pts):
        return PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS
    if float(np.ptp(finite_x)) < float(min_offset_spread_m):
        return PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS
    return None


def _fit_model_for_plan(
    *,
    strategy: TwoPieceRansacAutoBreakStrategy,
    plan: _ObservationPlan,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    min_pts: int,
    min_offset_spread_m: float,
    cache: dict[tuple[int, ...], _FitCacheEntry],
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
    obs_count_before_sampling: int | None = None,
) -> tuple[
    object | None,
    tuple[float, float, float, float, float, float, float] | None,
    int | None,
]:
    spread_failure_reason = _offset_spread_failure_reason(
        x_obs,
        min_pts=int(min_pts),
        min_offset_spread_m=float(min_offset_spread_m),
    )
    if spread_failure_reason is not None:
        return None, None, spread_failure_reason

    cache_key = _fit_cache_key(plan)
    entry = cache.get(cache_key)
    if entry is None:
        if runtime_diagnostics is not None:
            runtime_diagnostics.record_cache_miss()
        try:
            x_tensor = torch.as_tensor(x_obs, dtype=torch.float32)
            y_tensor = torch.as_tensor(y_obs, dtype=torch.float32)
            if runtime_diagnostics is None:
                trend_model = strategy.fit(x_tensor, y_tensor)
            else:
                with runtime_diagnostics.time_ransac_fit(
                    obs_count=int(np.asarray(x_obs).size),
                    obs_count_before=obs_count_before_sampling,
                ):
                    trend_model = strategy.fit(x_tensor, y_tensor)
        except (TypeError, ValueError, RuntimeError):
            trend_model = None

        if trend_model is None:
            entry = _FitCacheEntry(model=None, diagnostics=None, fit_failed=True)
        else:
            entry = _FitCacheEntry(
                model=trend_model,
                diagnostics=None,
                fit_failed=False,
            )
        cache[cache_key] = entry
    elif runtime_diagnostics is not None:
        runtime_diagnostics.record_cache_hit()

    if bool(entry.fit_failed):
        return None, None, PHYSICAL_MODEL_FAILURE_FIT_FAILED
    return entry.model, entry.diagnostics, None


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
    )
    return diagnostics


def _assign_model_diagnostics(
    arrays: dict[str, np.ndarray],
    trace_idx: int,
    diagnostics: tuple[float, float, float, float, float, float, float] | None,
) -> None:
    if diagnostics is None:
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
    arrays['physical_model_break_offset_m'][trace_idx] = np.float32(break_offset)
    arrays['physical_model_slope_near_s_per_m'][trace_idx] = np.float32(slope_near)
    arrays['physical_model_slope_far_s_per_m'][trace_idx] = np.float32(slope_far)
    arrays['physical_model_velocity_near_m_s'][trace_idx] = np.float32(velocity_near)
    arrays['physical_model_velocity_far_m_s'][trace_idx] = np.float32(velocity_far)
    arrays['physical_model_resid_p50_ms'][trace_idx] = np.float32(resid_p50)
    arrays['physical_model_resid_p90_ms'][trace_idx] = np.float32(resid_p90)


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
    if runtime_diagnostics is not None:
        runtime_diagnostics.inc('n_prediction_calls')
    with (
        runtime_diagnostics.time_block('prediction_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        try:
            physical_t_sec = _predict_model_sec(
                trend_model,
                float(offset_abs_m[int(trace_idx)]),
            )
            physical_t_sec += float(t0_shift_sec)
        except (TypeError, ValueError, RuntimeError):
            physical_t_sec = np.nan

    if not np.isfinite(physical_t_sec):
        return False

    with (
        runtime_diagnostics.time_block('assignment_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        center_i = int(np.rint(physical_t_sec / float(dt)))
        center_i = int(np.clip(center_i, 0, int(n_samples) - 1))
        arrays['physical_center_i'][trace_idx] = np.int32(center_i)
        arrays['physical_center_t_sec'][trace_idx] = np.float32(center_i * float(dt))
        arrays['physical_model_status'][trace_idx] = np.uint8(
            PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT
            if bool(plan.relaxed)
            else PHYSICAL_MODEL_STATUS_TWO_PIECE_OK
        )
        arrays['physical_model_failure_reason'][trace_idx] = np.uint8(
            PHYSICAL_MODEL_FAILURE_NONE
        )
        arrays['physical_runtime_fit_source'][trace_idx] = np.uint8(runtime_fit_source)
        _assign_model_diagnostics(arrays, trace_idx, diagnostics)
    return True


def _fit_and_assign_trace(
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
    merged: MergeResult,
    cfg: PhysicsLiteConfig,
    strategy: TwoPieceRansacAutoBreakStrategy,
    fit_cache: dict[tuple[int, ...], _FitCacheEntry],
    min_fit_obs: int,
    runtime_fit_source: int,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
) -> _TraceFitResult:
    arrays['physical_offset_source'][trace_idx] = np.uint8(offset_source)
    fit_calls_before = (
        int(runtime_diagnostics.n_fit_calls)
        if runtime_diagnostics is not None
        else 0
    )
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
            merged=merged,
        )
        return _TraceFitResult(None, None, None, 0, False)

    plan = _build_observation_plan(
        trace_idx=trace_idx,
        target_group_id=int(group_id_by_trace[trace_idx]),
        group_context_by_id=group_context_by_id,
        geometry=geometry,
        offset_abs_m=offset_abs_m,
        offset_signed_m=offset_signed_m,
        cfg=cfg,
        runtime_diagnostics=runtime_diagnostics,
    )
    if plan is None:
        _assign_fallback(
            arrays,
            trace_idx,
            failure_reason=PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS,
            table=table,
            feasible=feasible,
            trend=trend,
            merged=merged,
        )
        return _TraceFitResult(None, None, None, 0, False)

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
            merged=merged,
        )
        return _TraceFitResult(plan, None, None, 0, False)

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
    fit_plan = _ObservationPlan(
        obs_indices=obs_indices,
        neighbor_count=plan.neighbor_count,
        prefilter_valid_count=plan.prefilter_valid_count,
        segment_id=plan.segment_id,
        side=plan.side,
        relaxed=plan.relaxed,
    )
    x_obs = np.asarray(offset_abs_m[obs_indices], dtype=np.float32)
    y_obs = np.asarray(pick_t_sec[obs_indices], dtype=np.float32)
    trend_model, diagnostics, fit_failure_reason = _fit_model_for_plan(
        strategy=strategy,
        plan=fit_plan,
        x_obs=x_obs,
        y_obs=y_obs,
        min_pts=int(cfg.two_piece_ransac.min_pts),
        min_offset_spread_m=float(cfg.physical_trend.min_offset_spread_m),
        cache=fit_cache,
        runtime_diagnostics=runtime_diagnostics,
        obs_count_before_sampling=obs_count_before_sampling,
    )
    fit_calls_after = (
        int(runtime_diagnostics.n_fit_calls)
        if runtime_diagnostics is not None
        else 0
    )
    fit_call_delta = max(0, fit_calls_after - fit_calls_before)

    if fit_failure_reason is not None:
        _assign_fallback(
            arrays,
            trace_idx,
            failure_reason=fit_failure_reason,
            table=table,
            feasible=feasible,
            trend=trend,
            merged=merged,
        )
        return _TraceFitResult(fit_plan, None, None, fit_call_delta, False)

    if not _assign_model_prediction(
        arrays,
        trace_idx,
        trend_model=trend_model,
        diagnostics=diagnostics,
        plan=fit_plan,
        offset_abs_m=offset_abs_m,
        dt=float(table.dt_scalar_sec),
        n_samples=int(table.n_samples_orig),
        runtime_fit_source=runtime_fit_source,
        runtime_diagnostics=runtime_diagnostics,
    ):
        _assign_fallback(
            arrays,
            trace_idx,
            failure_reason=PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
            table=table,
            feasible=feasible,
            trend=trend,
            merged=merged,
        )
        return _TraceFitResult(
            fit_plan,
            trend_model,
            diagnostics,
            fit_call_delta,
            False,
        )

    if diagnostics is None:
        diagnostics = _diagnostics_for_plan(
            plan=fit_plan,
            x_obs=x_obs,
            y_obs=y_obs,
            cache=fit_cache,
        )
        _assign_model_diagnostics(arrays, trace_idx, diagnostics)

    return _TraceFitResult(fit_plan, trend_model, diagnostics, fit_call_delta, True)


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
    merged: MergeResult,
    cfg: PhysicsLiteConfig,
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
        merged=merged,
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


def build_geometry_two_piece_physical_center(
    *,
    coarse_npz: Mapping[str, np.ndarray],
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    merged: MergeResult,
    cfg: PhysicsLiteConfig,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> PhysicalCenterResult:
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

    if not bool(cfg.physical_trend.enabled):
        return _build_disabled_result(table, trend)

    with (
        runtime_diagnostics.time_block('geometry_load_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        try:
            geometry = load_coarse_geometry_from_npz(coarse_npz, n_traces=n)
        except (KeyError, TypeError, ValueError):
            geometry = None

    use_geometry_offset = bool(cfg.physical_trend.use_geometry_offset)
    if use_geometry_offset and geometry is None:
        return _assign_fallback_all(
            failure_reason=PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
            table=table,
            feasible=feasible,
            trend=trend,
            merged=merged,
        )

    if use_geometry_offset:
        offset_abs_m = _as_vector(
            'geometry.offset_abs_geom_m',
            geometry.offset_abs_geom_m,
            n_traces=n,
            dtype=np.float32,
        )
        offset_signed_m = None
        offset_source = PHYSICAL_OFFSET_SOURCE_GEOMETRY
    else:
        offset_abs_m = _table_offset_abs_m(table, n_traces=n)
        offset_signed_m = _as_vector(
            'table.offset_m',
            table.offset_m,
            n_traces=n,
            dtype=np.float32,
        )
        offset_source = PHYSICAL_OFFSET_SOURCE_HEADER

    with (
        runtime_diagnostics.time_block('source_grouping_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        source_groups_from_geometry = False
        groups: tuple[SourceGroup, ...] = ()
        source_group_geometry = geometry
        if source_group_geometry is None and not use_geometry_offset:
            source_group_geometry = _load_source_group_geometry_from_npz(
                coarse_npz,
                n_traces=n,
            )
        if source_group_geometry is not None:
            coord_group_tol_m = float(cfg.physical_trend.coord_group_tol_m)
            source_xy_degenerate = is_source_xy_degenerate(
                source_group_geometry,
                table=table,
                coord_group_tol_m=coord_group_tol_m,
            )
            if source_xy_degenerate and use_geometry_offset:
                return _assign_fallback_all(
                    failure_reason=PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
                    table=table,
                    feasible=feasible,
                    trend=trend,
                    merged=merged,
                )
            if not source_xy_degenerate:
                groups = build_source_groups(
                    source_group_geometry,
                    coord_group_tol_m=coord_group_tol_m,
                )
                source_groups_from_geometry = len(groups) > 0
        if len(groups) == 0 and not use_geometry_offset:
            groups = _build_table_source_groups(table, n_traces=n)

    if len(groups) == 0:
        return _assign_fallback_all(
            failure_reason=PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
            table=table,
            feasible=feasible,
            trend=trend,
            merged=merged,
        )

    if runtime_diagnostics is not None:
        runtime_diagnostics.set_source_groups(len(groups))

    with (
        runtime_diagnostics.time_block('source_group_ordering_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        groups_by_id = {int(group.group_id): group for group in groups}
        group_id_by_trace = np.full((n,), -1, dtype=np.int32)
        for group in groups:
            group_id_by_trace[
                np.asarray(group.trace_indices, dtype=np.int64)
            ] = np.int32(group.group_id)

    pick_t_sec = _as_vector(
        'table.coarse_pick_t_sec',
        table.coarse_pick_t_sec,
        n_traces=n,
        dtype=np.float32,
    )
    with (
        runtime_diagnostics.time_block('valid_mask_build_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        with (
            runtime_diagnostics.time_block('velocity_prefilter_sec')
            if runtime_diagnostics is not None
            else nullcontext()
        ):
            valid_for_fit = _compute_physical_prefilter_mask(
                offset_abs_m=offset_abs_m,
                table=table,
                feasible=feasible,
                cfg=cfg,
            )

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
        )

    arrays = _allocate_result_arrays(table)
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
    strategy = _fit_strategy(cfg)
    fit_cache: dict[tuple[int, ...], _FitCacheEntry] = {}
    min_fit_obs = 2 * int(cfg.two_piece_ransac.min_pts)

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

        for group in groups:
            group_id = int(group.group_id)
            if not bool(is_anchor_by_id.get(group_id, False)):
                continue
            for trace_idx in np.asarray(group.trace_indices, dtype=np.int64).tolist():
                result = _fit_and_assign_trace(
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
                    merged=merged,
                    cfg=cfg,
                    strategy=strategy,
                    fit_cache=fit_cache,
                    min_fit_obs=min_fit_obs,
                    runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT,
                    runtime_diagnostics=runtime_diagnostics,
                )
                if runtime_diagnostics is not None and result.fit_call_delta > 0:
                    runtime_diagnostics.record_anchor_fit_calls(
                        result.fit_call_delta
                    )
                if (
                    result.assigned_from_model
                    and result.plan is not None
                    and result.trend_model is not None
                ):
                    key = _anchor_model_key(group_id, result.plan)
                    anchor_models.setdefault(
                        key,
                        _AnchorModelContext(
                            trend_model=result.trend_model,
                            diagnostics=result.diagnostics,
                        ),
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
            reuse_items: dict[
                tuple[int, int, int, bool],
                list[tuple[int, _ObservationPlan, _AnchorModelContext]],
            ] = {}
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
                        merged=merged,
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

                context = None
                key = None
                with (
                    runtime_diagnostics.time_block('compatible_anchor_search_sec')
                    if runtime_diagnostics is not None
                    else nullcontext()
                ):
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
                        with (
                            runtime_diagnostics.time_block('anchor_lookup_sec')
                            if runtime_diagnostics is not None
                            else nullcontext()
                        ):
                            context = anchor_models.get(key)

                if context is not None and plan is not None and key is not None:
                    reuse_items.setdefault(key, []).append((trace_idx, plan, context))
                    continue
                if runtime_diagnostics is not None:
                    runtime_diagnostics.record_no_compatible_anchor_context()

                fallback = str(
                    cfg.physical_runtime.anchor_reuse.fallback_if_no_compatible_segment
                )
                if fallback == 'full_fit':
                    fallback_full_group_ids.add(group_id)
                    _fit_and_assign_trace(
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
                        merged=merged,
                        cfg=cfg,
                        strategy=strategy,
                        fit_cache=fit_cache,
                        min_fit_obs=min_fit_obs,
                        runtime_fit_source=(
                            PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_FULL_FIT_NO_COMPATIBLE_ANCHOR
                        ),
                        runtime_diagnostics=runtime_diagnostics,
                    )
                else:
                    _fallback_no_compatible_anchor(
                        arrays=arrays,
                        trace_idx=trace_idx,
                        table=table,
                        feasible=feasible,
                        trend=trend,
                        merged=merged,
                        cfg=cfg,
                    )

            non_anchor_mode = str(cfg.physical_runtime.anchor_reuse.non_anchor_mode)
            if runtime_diagnostics is not None and reuse_items:
                runtime_diagnostics.record_reuse_contexts(len(reuse_items))
            if non_anchor_mode == 'nearest_anchor':
                for items in reuse_items.values():
                    for trace_idx, plan, context in items:
                        if _assign_model_prediction(
                            arrays,
                            trace_idx,
                            trend_model=context.trend_model,
                            diagnostics=context.diagnostics,
                            plan=plan,
                            offset_abs_m=offset_abs_m,
                            dt=dt,
                            n_samples=n_samples,
                            runtime_fit_source=(
                                PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE
                            ),
                            runtime_diagnostics=runtime_diagnostics,
                        ):
                            n_reused_predictions += 1
                        else:
                            _assign_fallback(
                                arrays,
                                trace_idx,
                                failure_reason=(
                                    PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID
                                ),
                                table=table,
                                feasible=feasible,
                                trend=trend,
                                merged=merged,
                            )
                continue

            stats_by_key: dict[tuple[int, int, int, bool], _ReuseShiftStats] = {}
            adaptive_refit = False
            for key, items in reuse_items.items():
                key_trace_indices = np.asarray(
                    [trace_idx for trace_idx, _plan, _context in items],
                    dtype=np.int64,
                )
                context = items[0][2]
                with (
                    runtime_diagnostics.time_block('t0_shift_sec')
                    if runtime_diagnostics is not None
                    else nullcontext()
                ):
                    stats = _compute_reuse_t0_shift_stats(
                        trend_model=context.trend_model,
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
                assigned_count = 0
                for trace_idx in group_trace_indices.tolist():
                    result = _fit_and_assign_trace(
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
                        merged=merged,
                        cfg=cfg,
                        strategy=strategy,
                        fit_cache=fit_cache,
                        min_fit_obs=min_fit_obs,
                        runtime_fit_source=(
                            PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT
                        ),
                        runtime_diagnostics=runtime_diagnostics,
                    )
                    if result.assigned_from_model:
                        assigned_count += 1
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
                            merged=merged,
                        )
                    continue

                use_shift = (
                    fallback_mode == 'nearest_anchor_plus_t0_shift'
                    and bool(stats.shift_valid)
                )
                shift_sec = float(stats.t0_shift_sec) if use_shift else 0.0
                for trace_idx, plan, context in items:
                    if _assign_model_prediction(
                        arrays,
                        trace_idx,
                        trend_model=context.trend_model,
                        diagnostics=context.diagnostics,
                        plan=plan,
                        offset_abs_m=offset_abs_m,
                        dt=dt,
                        n_samples=n_samples,
                        runtime_fit_source=(
                            PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE
                        ),
                        t0_shift_sec=shift_sec,
                        runtime_diagnostics=runtime_diagnostics,
                    ):
                        n_reused_predictions += 1
                        if use_shift:
                            group_shifted_count += 1
                    else:
                        _assign_fallback(
                            arrays,
                            trace_idx,
                            failure_reason=(
                                PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID
                            ),
                            table=table,
                            feasible=feasible,
                            trend=trend,
                            merged=merged,
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
        return _finalize_result(arrays)

    for trace_idx in range(n):
        _fit_and_assign_trace(
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
            merged=merged,
            cfg=cfg,
            strategy=strategy,
            fit_cache=fit_cache,
            min_fit_obs=min_fit_obs,
            runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_FULL_FIT,
            runtime_diagnostics=runtime_diagnostics,
        )

    if runtime_diagnostics is not None:
        runtime_diagnostics.set_unique_fit_contexts(len(fit_cache))

    return _finalize_result(arrays)
