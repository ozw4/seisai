from __future__ import annotations

import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch
from seisai_pick.trend.trend_fit_strategy import (
    TwoPieceIRLSAutoBreakStrategy,
    TwoPieceRansacAutoBreakStrategy,
)

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
from .physical_center_fallback import (
    _allocate_result_arrays,
    _as_bool_vector,
    _as_vector,
    _assign_fallback,
    _assign_robust_fallback,
    _build_disabled_result,
    _emit_fallback_all_and_done,
    _finalize_result_with_pending_trend_fallback,
    _PendingTrendFallback,
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
from .pick_table import CoarsePickTable
from .progress import NullProgressReporter, build_progress_reporter
from .runtime_diagnostics import PhysicalRuntimeDiagnostics
from .runtime_policy import (
    SourceXYAnchorSelectionResult,
    select_source_xy_stride_anchors,
)
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


@dataclass(frozen=True)
class _GroupObservationContext:
    group_id: int
    neighbor_group_ids: np.ndarray
    neighbor_indices: np.ndarray
    valid_obs_indices: np.ndarray
    valid_obs_key: tuple[int, ...]
    neighbor_count: int
    prefilter_valid_count: int
    side_context: _SideObservationContext | None = None


@dataclass(frozen=True)
class _ObservationPlan:
    obs_indices: np.ndarray
    obs_key: tuple[int, ...] | None
    neighbor_count: int
    prefilter_valid_count: int
    segment_id: int
    side: int
    relaxed: bool


@dataclass(frozen=True)
class _SignedOffsetSideLabels:
    side: np.ndarray
    finite: np.ndarray


@dataclass(frozen=True)
class _SideObservationContext:
    obs_indices_by_side: tuple[np.ndarray, np.ndarray, np.ndarray]
    obs_key_by_side: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    finite_count: int
    nonzero_count: int


@dataclass(frozen=True)
class _GapSegmentContext:
    segment_id_by_trace_idx: dict[int, int]
    obs_indices_by_segment_id: dict[int, np.ndarray]
    obs_key_by_segment_id: dict[int, tuple[int, ...]]


@dataclass
class _ObservationPlanCache:
    offset_signed_labels: _SignedOffsetSideLabels | None = None
    index_members: dict[tuple[int, ...], frozenset[int]] = field(
        default_factory=dict,
    )
    signed_side_context: dict[
        tuple[int, tuple[int, ...]], _SideObservationContext
    ] = field(default_factory=dict)
    geometry_side: dict[
        tuple[tuple[int, ...], int],
        tuple[np.ndarray, tuple[int, ...], int, bool],
    ] = field(default_factory=dict)
    gap_segment: dict[
        tuple[tuple[int, ...], int, float, float | None],
        tuple[np.ndarray, tuple[int, ...], int],
    ] = field(default_factory=dict)
    gap_context: dict[
        tuple[tuple[int, ...], float, float | None],
        _GapSegmentContext,
    ] = field(default_factory=dict)


@dataclass(frozen=True)
class _FitCacheEntry:
    model: object | None
    diagnostics: tuple[float, float, float, float, float, float, float] | None
    fit_failed: bool
    diagnostics_computed: bool = False
    failure_reason: int | None = None


@dataclass(frozen=True)
class _TraceFitResult:
    plan: _ObservationPlan | None
    trend_model: object | None
    diagnostics: tuple[float, float, float, float, float, float, float] | None
    x_obs: np.ndarray | None = None
    y_obs: np.ndarray | None = None


@dataclass(frozen=True)
class _TracePlanAssignment:
    trace_idx: int
    plan: _ObservationPlan
    fit_key: tuple[int, ...]
    obs_count_before_sampling: int


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
    diagnostics: tuple[float, float, float, float, float, float, float] | None
    valid_trace_indices: frozenset[int]


@dataclass(frozen=True)
class _FitTaskCfgValues:
    fit_kind: str
    n_iter: int
    inlier_th_ms: float
    irls_huber_c: float
    irls_iters: int
    min_pts: int
    n_break_cand: int
    q_lo: float
    q_hi: float
    seed: int
    slope_eps: float
    sort_offsets: bool
    min_offset_spread_m: float
    torch_num_threads_per_worker: int


@dataclass(frozen=True)
class _FitTask:
    fit_key: tuple[int, ...]
    x_obs: np.ndarray
    y_obs: np.ndarray
    w_obs: np.ndarray
    obs_count_before_sampling: int
    cfg_values: _FitTaskCfgValues


@dataclass(frozen=True)
class _FitTaskResult:
    fit_key: tuple[int, ...]
    trend_model: object | None
    diagnostics: tuple[float, float, float, float, float, float, float] | None
    fit_failed: bool
    failure_reason: int | None
    elapsed_sec: float
    obs_count: int
    obs_count_before_sampling: int
    fit_attempted: bool


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


def _indices_key(indices: np.ndarray) -> tuple[int, ...]:
    return tuple(np.asarray(indices, dtype=np.int64).tolist())


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


def _obs_key_or_build(
    obs_indices: np.ndarray,
    obs_key: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if obs_key is not None:
        return obs_key
    return _indices_key(obs_indices)


def _index_key_contains(
    *,
    obs_key: tuple[int, ...],
    trace_idx: int,
    cache: _ObservationPlanCache,
) -> bool:
    members = cache.index_members.get(obs_key)
    if members is None:
        members = frozenset(obs_key)
        cache.index_members[obs_key] = members
    return int(trace_idx) in members


def _side_slot(side: int) -> int:
    value = int(side)
    if value < -1 or value > 1:
        msg = 'side must be -1, 0, or 1'
        raise ValueError(msg)
    return value + 1


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


def _filtered_obs_key(
    *,
    obs_indices: np.ndarray,
    obs_key: tuple[int, ...],
    keep_mask: np.ndarray,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    obs = np.asarray(obs_indices, dtype=np.int64)
    mask = np.asarray(keep_mask, dtype=np.bool_)
    if mask.shape != obs.shape:
        msg = 'keep_mask must match obs_indices shape'
        raise ValueError(msg)
    if bool(np.all(mask)):
        return obs, obs_key
    filtered = obs[mask]
    with (
        runtime_diagnostics.time_block('side_segment_key_build_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        filtered_key = _indices_key(filtered)
    return filtered, filtered_key


def _build_side_observation_context(
    *,
    labels: _SignedOffsetSideLabels,
    obs_indices: np.ndarray,
    obs_key: tuple[int, ...],
    cache: _ObservationPlanCache,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> _SideObservationContext:
    cache_key = (id(labels.side), obs_key)
    cached = cache.signed_side_context.get(cache_key)
    if cached is not None:
        if runtime_diagnostics is not None:
            runtime_diagnostics.inc('n_side_context_cache_hits')
        return cached

    if runtime_diagnostics is not None:
        runtime_diagnostics.inc('n_side_context_cache_misses')
    with (
        runtime_diagnostics.time_block('side_filter_precompute_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        obs = np.asarray(obs_indices, dtype=np.int64)
        finite_count = int(np.count_nonzero(labels.finite[obs]))
        nonzero_count = int(np.count_nonzero(labels.side[obs] != 0))
        obs_by_side: list[np.ndarray] = []
        key_by_side: list[tuple[int, ...]] = []
        for side in (-1, 0, 1):
            side_obs, side_key = _filtered_obs_key(
                obs_indices=obs,
                obs_key=obs_key,
                keep_mask=labels.side[obs] == int(side),
                runtime_diagnostics=runtime_diagnostics,
            )
            obs_by_side.append(side_obs)
            key_by_side.append(side_key)
            if runtime_diagnostics is not None:
                runtime_diagnostics.record_side_obs_count(int(side_obs.size))
        context = _SideObservationContext(
            obs_indices_by_side=tuple(obs_by_side),  # type: ignore[arg-type]
            obs_key_by_side=tuple(key_by_side),  # type: ignore[arg-type]
            finite_count=finite_count,
            nonzero_count=nonzero_count,
        )
    cache.signed_side_context[cache_key] = context
    if runtime_diagnostics is not None:
        runtime_diagnostics.inc('n_side_contexts_built')
        runtime_diagnostics.inc(
            'n_side_gap_precomputed_fit_keys',
            len(set(context.obs_key_by_side)),
        )
    return context


def _obs_with_target_labeled_side(
    *,
    trace_idx: int,
    obs_indices: np.ndarray,
    obs_key: tuple[int, ...],
    labels: _SignedOffsetSideLabels,
    cache: _ObservationPlanCache,
    min_finite_count: int,
    side_context: _SideObservationContext | None = None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> tuple[np.ndarray, tuple[int, ...], int, bool]:
    trace = int(trace_idx)
    obs = np.asarray(obs_indices, dtype=np.int64)
    if side_context is not None:
        if runtime_diagnostics is not None:
            runtime_diagnostics.inc('n_side_context_lookup_calls')
        context = side_context
    else:
        context = _build_side_observation_context(
            labels=labels,
            obs_indices=obs,
            obs_key=obs_key,
            cache=cache,
            runtime_diagnostics=runtime_diagnostics,
        )
    finite_count = int(context.finite_count)
    nonzero_count = int(context.nonzero_count)
    if not _index_key_contains(obs_key=obs_key, trace_idx=trace, cache=cache):
        if bool(labels.finite[trace]):
            finite_count += 1
        if int(labels.side[trace]) != 0:
            nonzero_count += 1

    if finite_count < int(min_finite_count) or nonzero_count == 0:
        return obs, obs_key, 0, False

    target_side = int(labels.side[trace])
    with (
        runtime_diagnostics.time_block('side_filter_lookup_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        side_slot = _side_slot(target_side)
        side_obs = context.obs_indices_by_side[side_slot]
        side_obs_key = context.obs_key_by_side[side_slot]
    return side_obs, side_obs_key, target_side, True


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
    offset_signed_labels: _SignedOffsetSideLabels | None = None,
    plan_cache: _ObservationPlanCache | None = None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> dict[int, _GroupObservationContext]:
    valid_mask = np.asarray(valid_for_fit, dtype=np.bool_)
    observation_cache = (
        plan_cache if plan_cache is not None else _ObservationPlanCache()
    )
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
        valid_obs_key = _indices_key(valid_obs_indices)
        side_context = None
        if offset_signed_labels is not None and bool(
            cfg.physical_trend.segment_by_offset_sign
        ):
            side_context = _build_side_observation_context(
                labels=offset_signed_labels,
                obs_indices=valid_obs_indices,
                obs_key=valid_obs_key,
                cache=observation_cache,
                runtime_diagnostics=runtime_diagnostics,
            )
        contexts[group_id] = _GroupObservationContext(
            group_id=group_id,
            neighbor_group_ids=neighbor_group_ids,
            neighbor_indices=neighbor_indices,
            valid_obs_indices=valid_obs_indices,
            valid_obs_key=valid_obs_key,
            neighbor_count=int(neighbor_group_ids.size),
            prefilter_valid_count=int(valid_obs_indices.size),
            side_context=side_context,
        )
    return contexts


def _obs_with_target_side(
    *,
    trace_idx: int,
    obs_indices: np.ndarray,
    geometry: CoarseGeometry,
    obs_key: tuple[int, ...] | None = None,
    cache: _ObservationPlanCache | None = None,
) -> tuple[np.ndarray, tuple[int, ...], int, bool]:
    plan_cache = cache if cache is not None else _ObservationPlanCache()
    key = _obs_key_or_build(obs_indices, obs_key)
    cache_key = (key, int(trace_idx))
    cached = plan_cache.geometry_side.get(cache_key)
    if cached is not None:
        return cached

    obs = np.asarray(obs_indices, dtype=np.int64)
    context_indices = _append_index(obs_indices, trace_idx)
    signed = signed_offset_side_from_geometry(geometry, context_indices)
    if not bool(signed.reliable):
        result = (obs, key, 0, False)
        plan_cache.geometry_side[cache_key] = result
        return result

    pos = _trace_position_map(context_indices)
    target_side = int(signed.side[pos[int(trace_idx)]])
    obs_side = np.asarray(
        [int(signed.side[pos[int(obs_idx)]]) for obs_idx in obs.tolist()],
        dtype=np.int8,
    )
    side_obs, side_obs_key = _filtered_obs_key(
        obs_indices=obs,
        obs_key=key,
        keep_mask=obs_side == target_side,
    )
    result = (side_obs, side_obs_key, target_side, True)
    plan_cache.geometry_side[cache_key] = result
    return result


def _obs_with_target_signed_offset_side(
    *,
    trace_idx: int,
    obs_indices: np.ndarray,
    signed_offset_m: np.ndarray,
    zero_tol_m: float = 1.0e-6,
    obs_key: tuple[int, ...] | None = None,
    cache: _ObservationPlanCache | None = None,
    labels: _SignedOffsetSideLabels | None = None,
    finite_mask: np.ndarray | None = None,
    min_finite_count: int = 2,
    side_context: _SideObservationContext | None = None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> tuple[np.ndarray, tuple[int, ...], int, bool]:
    plan_cache = cache if cache is not None else _ObservationPlanCache()
    if labels is not None:
        plan_cache.offset_signed_labels = labels
    elif plan_cache.offset_signed_labels is None:
        plan_cache.offset_signed_labels = _signed_offset_side_labels(
            signed_offset_m,
            zero_tol_m=zero_tol_m,
            finite_mask=finite_mask,
        )
    key = _obs_key_or_build(obs_indices, obs_key)
    return _obs_with_target_labeled_side(
        trace_idx=trace_idx,
        obs_indices=obs_indices,
        obs_key=key,
        labels=plan_cache.offset_signed_labels,
        cache=plan_cache,
        min_finite_count=int(min_finite_count),
        side_context=side_context,
        runtime_diagnostics=runtime_diagnostics,
    )


def _gap_context_key(
    *,
    obs_key: tuple[int, ...],
    cfg: PhysicsLiteConfig,
) -> tuple[tuple[int, ...], float, float | None]:
    min_gap = cfg.physical_trend.min_gap_m
    return (
        obs_key,
        float(cfg.physical_trend.gap_ratio),
        None if min_gap is None else float(min_gap),
    )


def _build_gap_segment_context(
    *,
    obs_indices: np.ndarray,
    obs_key: tuple[int, ...],
    offset_abs_m: np.ndarray,
    cfg: PhysicsLiteConfig,
    cache: _ObservationPlanCache,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> _GapSegmentContext:
    cache_key = _gap_context_key(obs_key=obs_key, cfg=cfg)
    cached = cache.gap_context.get(cache_key)
    if cached is not None:
        if runtime_diagnostics is not None:
            runtime_diagnostics.inc('n_gap_context_cache_hits')
        return cached

    if runtime_diagnostics is not None:
        runtime_diagnostics.inc('n_gap_context_cache_misses')
    with (
        runtime_diagnostics.time_block('gap_segment_precompute_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        obs = np.asarray(obs_indices, dtype=np.int64)
        segment_id = split_offset_gap_segments(
            np.asarray(offset_abs_m, dtype=np.float32)[obs],
            split_by_offset_gap=bool(cfg.physical_trend.split_by_offset_gap),
            gap_ratio=float(cfg.physical_trend.gap_ratio),
            min_gap_m=cfg.physical_trend.min_gap_m,
        )
        segment_id_by_trace_idx = {
            int(trace_idx): int(seg_id)
            for trace_idx, seg_id in zip(
                obs.tolist(),
                segment_id.tolist(),
                strict=True,
            )
        }
        obs_indices_by_segment_id: dict[int, np.ndarray] = {}
        obs_key_by_segment_id: dict[int, tuple[int, ...]] = {}
        for seg_id in np.unique(segment_id).astype(np.int64).tolist():
            segment_obs, segment_key = _filtered_obs_key(
                obs_indices=obs,
                obs_key=obs_key,
                keep_mask=segment_id == int(seg_id),
                runtime_diagnostics=runtime_diagnostics,
            )
            obs_indices_by_segment_id[int(seg_id)] = segment_obs
            obs_key_by_segment_id[int(seg_id)] = segment_key
            if runtime_diagnostics is not None:
                runtime_diagnostics.record_gap_segment_obs_count(
                    int(segment_obs.size)
                )
        context = _GapSegmentContext(
            segment_id_by_trace_idx=segment_id_by_trace_idx,
            obs_indices_by_segment_id=obs_indices_by_segment_id,
            obs_key_by_segment_id=obs_key_by_segment_id,
        )
    cache.gap_context[cache_key] = context
    if runtime_diagnostics is not None:
        runtime_diagnostics.inc('n_gap_contexts_built')
        runtime_diagnostics.inc(
            'n_side_gap_precomputed_fit_keys',
            len(context.obs_key_by_segment_id),
        )
    return context


def _obs_with_target_gap_segment(
    *,
    trace_idx: int,
    obs_indices: np.ndarray,
    offset_abs_m: np.ndarray,
    cfg: PhysicsLiteConfig,
    obs_key: tuple[int, ...] | None = None,
    cache: _ObservationPlanCache | None = None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    plan_cache = cache if cache is not None else _ObservationPlanCache()
    key = _obs_key_or_build(obs_indices, obs_key)
    min_gap = cfg.physical_trend.min_gap_m
    fallback_gap_key = (
        key,
        int(trace_idx),
        float(cfg.physical_trend.gap_ratio),
        None if min_gap is None else float(min_gap),
    )
    obs = np.asarray(obs_indices, dtype=np.int64)
    if _index_key_contains(obs_key=key, trace_idx=int(trace_idx), cache=plan_cache):
        if runtime_diagnostics is not None:
            runtime_diagnostics.inc('n_gap_trace_in_obs')
            runtime_diagnostics.inc('n_gap_fast_path_calls')
        context = _build_gap_segment_context(
            obs_indices=obs,
            obs_key=key,
            offset_abs_m=offset_abs_m,
            cfg=cfg,
            cache=plan_cache,
            runtime_diagnostics=runtime_diagnostics,
        )
        with (
            runtime_diagnostics.time_block('gap_segment_lookup_sec')
            if runtime_diagnostics is not None
            else nullcontext()
        ):
            target_segment_id = int(
                context.segment_id_by_trace_idx[int(trace_idx)]
            )
            return (
                context.obs_indices_by_segment_id[target_segment_id],
                context.obs_key_by_segment_id[target_segment_id],
                target_segment_id,
            )

    if runtime_diagnostics is not None:
        runtime_diagnostics.inc('n_gap_trace_not_in_obs')
        runtime_diagnostics.inc('n_gap_fallback_calls')
    cached = plan_cache.gap_segment.get(fallback_gap_key)
    if cached is not None:
        return cached

    with (
        runtime_diagnostics.time_block('gap_segment_fallback_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
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
            [int(segment_id[pos[int(obs_idx)]]) for obs_idx in obs.tolist()],
            dtype=np.int64,
        )
        segment_obs, segment_obs_key = _filtered_obs_key(
            obs_indices=obs,
            obs_key=key,
            keep_mask=obs_segment_id == target_segment_id,
            runtime_diagnostics=runtime_diagnostics,
        )
        result = (segment_obs, segment_obs_key, target_segment_id)
    plan_cache.gap_segment[fallback_gap_key] = result
    return result


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
    plan_cache: _ObservationPlanCache | None = None,
) -> _ObservationPlan | None:
    observation_cache = (
        plan_cache if plan_cache is not None else _ObservationPlanCache()
    )
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
    valid_obs_key = group_context.valid_obs_key
    neighbor_count = int(group_context.neighbor_count)
    prefilter_valid_count = int(group_context.prefilter_valid_count)
    min_fit_obs = 2 * _fit_min_pts(cfg)

    if prefilter_valid_count < min_fit_obs:
        return _ObservationPlan(
            obs_indices=valid_obs,
            obs_key=valid_obs_key,
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=-1,
            side=0,
            relaxed=False,
        )

    side_obs = valid_obs
    side_obs_key = valid_obs_key
    side = 0
    side_reliable = False
    with (
        runtime_diagnostics.time_block('side_segment_build_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        if bool(cfg.physical_trend.segment_by_offset_sign):
            if offset_signed_m is not None:
                (
                    side_obs,
                    side_obs_key,
                    side,
                    side_reliable,
                ) = _obs_with_target_signed_offset_side(
                    trace_idx=trace_idx,
                    obs_indices=valid_obs,
                    signed_offset_m=offset_signed_m,
                    obs_key=valid_obs_key,
                    cache=observation_cache,
                    labels=observation_cache.offset_signed_labels,
                    min_finite_count=(
                        1
                        if (
                            geometry is not None
                            and bool(cfg.physical_trend.use_geometry_offset)
                            and geometry.offset_signed_geom_m is not None
                        )
                        else 2
                    ),
                    side_context=group_context.side_context,
                    runtime_diagnostics=runtime_diagnostics,
                )
            elif geometry is not None:
                (
                    side_obs,
                    side_obs_key,
                    side,
                    side_reliable,
                ) = _obs_with_target_side(
                    trace_idx=trace_idx,
                    obs_indices=valid_obs,
                    geometry=geometry,
                    obs_key=valid_obs_key,
                    cache=observation_cache,
                )

        segment_obs = side_obs
        segment_obs_key = side_obs_key
        segment_id = 0
        if bool(cfg.physical_trend.split_by_offset_gap):
            segment_obs, segment_obs_key, segment_id = _obs_with_target_gap_segment(
                trace_idx=trace_idx,
                obs_indices=side_obs,
                obs_key=side_obs_key,
                offset_abs_m=offset_abs_m,
                cfg=cfg,
                cache=observation_cache,
                runtime_diagnostics=runtime_diagnostics,
            )

    if int(segment_obs.size) >= min_fit_obs:
        return _ObservationPlan(
            obs_indices=segment_obs,
            obs_key=segment_obs_key,
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
            obs_key=side_obs_key,
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=0,
            side=side,
            relaxed=True,
        )

    if side_reliable and int(valid_obs.size) >= min_fit_obs:
        return _ObservationPlan(
            obs_indices=valid_obs,
            obs_key=valid_obs_key,
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=0,
            side=0,
            relaxed=True,
        )

    return _ObservationPlan(
        obs_indices=segment_obs,
        obs_key=segment_obs_key,
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


_PhysicalFitStrategy = TwoPieceRansacAutoBreakStrategy | TwoPieceIRLSAutoBreakStrategy


def _fit_min_pts(cfg: PhysicsLiteConfig) -> int:
    if cfg.physical_trend.fit_kind == 'two_piece_irls_autobreak':
        return int(cfg.two_piece_irls.min_pts)
    return int(cfg.two_piece_ransac.min_pts)


def _fit_strategy(cfg: PhysicsLiteConfig) -> _PhysicalFitStrategy:
    if cfg.physical_trend.fit_kind == 'two_piece_irls_autobreak':
        return TwoPieceIRLSAutoBreakStrategy(
            huber_c=float(cfg.two_piece_irls.huber_c),
            iters=int(cfg.two_piece_irls.iters),
            min_pts=int(cfg.two_piece_irls.min_pts),
            n_break_cand=int(cfg.two_piece_irls.n_break_cand),
            q_lo=float(cfg.two_piece_irls.q_lo),
            q_hi=float(cfg.two_piece_irls.q_hi),
            slope_eps=float(cfg.two_piece_irls.slope_eps),
            sort_offsets=bool(cfg.two_piece_irls.sort_offsets),
        )
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


def _confidence_weights_for_obs(coarse_pmax_obs: np.ndarray) -> np.ndarray:
    w = np.asarray(coarse_pmax_obs, dtype=np.float32)
    if w.ndim != 1:
        w = w.reshape(-1)
    good = np.isfinite(w) & (w > np.float32(0.0))
    if not bool(np.any(good)):
        return np.ones_like(w, dtype=np.float32)
    fill = np.float32(np.median(w[good].astype(np.float64, copy=False)))
    out = np.where(good, w, fill).astype(np.float32, copy=False)
    return np.clip(out, np.float32(1.0e-6), None)


def _fit_strategy_model(
    strategy: _PhysicalFitStrategy,
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    w_tensor: torch.Tensor,
):
    if isinstance(strategy, TwoPieceIRLSAutoBreakStrategy):
        return strategy.fit(x_tensor, y_tensor, w_tensor)
    return strategy.fit(x_tensor, y_tensor)


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


def _fit_key_for_obs(
    obs_indices: np.ndarray,
    *,
    precomputed_key: tuple[int, ...] | None = None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
    after_sampling: bool = False,
    count_missing_precomputed: bool = True,
) -> tuple[int, ...]:
    if precomputed_key is not None:
        if runtime_diagnostics is not None:
            runtime_diagnostics.inc('n_precomputed_fit_key_used')
        return precomputed_key

    if runtime_diagnostics is not None:
        if bool(count_missing_precomputed):
            runtime_diagnostics.inc('n_fit_key_missing_precomputed')
        runtime_diagnostics.inc('n_fit_key_built_from_indices')
        if bool(after_sampling):
            runtime_diagnostics.inc('n_fit_key_built_after_sampling')
    return _indices_key(obs_indices)


def _fit_cache_key(plan: _ObservationPlan) -> tuple[int, ...]:
    return _fit_key_for_obs(plan.obs_indices, precomputed_key=plan.obs_key)


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


def _fit_task_cfg_values(cfg: PhysicsLiteConfig) -> _FitTaskCfgValues:
    executor = cfg.physical_runtime.fit_executor
    is_irls = cfg.physical_trend.fit_kind == 'two_piece_irls_autobreak'
    return _FitTaskCfgValues(
        fit_kind=str(cfg.physical_trend.fit_kind),
        n_iter=int(cfg.two_piece_ransac.n_iter),
        inlier_th_ms=float(cfg.two_piece_ransac.inlier_th_ms),
        irls_huber_c=float(cfg.two_piece_irls.huber_c),
        irls_iters=int(cfg.two_piece_irls.iters),
        min_pts=_fit_min_pts(cfg),
        n_break_cand=(
            int(cfg.two_piece_irls.n_break_cand)
            if is_irls
            else int(cfg.two_piece_ransac.n_break_cand)
        ),
        q_lo=(
            float(cfg.two_piece_irls.q_lo)
            if is_irls
            else float(cfg.two_piece_ransac.q_lo)
        ),
        q_hi=(
            float(cfg.two_piece_irls.q_hi)
            if is_irls
            else float(cfg.two_piece_ransac.q_hi)
        ),
        seed=int(cfg.two_piece_ransac.seed),
        slope_eps=(
            float(cfg.two_piece_irls.slope_eps)
            if is_irls
            else float(cfg.two_piece_ransac.slope_eps)
        ),
        sort_offsets=(
            bool(cfg.two_piece_irls.sort_offsets)
            if is_irls
            else bool(cfg.two_piece_ransac.sort_offsets)
        ),
        min_offset_spread_m=float(cfg.physical_trend.min_offset_spread_m),
        torch_num_threads_per_worker=int(executor.torch_num_threads_per_worker),
    )


def _fit_task_from_work_item(
    work_item: _FitContextWorkItem,
    *,
    cfg_values: _FitTaskCfgValues,
) -> _FitTask:
    return _FitTask(
        fit_key=work_item.fit_key,
        x_obs=np.asarray(work_item.x_obs, dtype=np.float32),
        y_obs=np.asarray(work_item.y_obs, dtype=np.float32),
        w_obs=np.asarray(work_item.w_obs, dtype=np.float32),
        obs_count_before_sampling=int(work_item.obs_count_before_sampling),
        cfg_values=cfg_values,
    )


def _strategy_from_fit_task_cfg(
    cfg_values: _FitTaskCfgValues,
) -> _PhysicalFitStrategy:
    if cfg_values.fit_kind == 'two_piece_irls_autobreak':
        return TwoPieceIRLSAutoBreakStrategy(
            huber_c=float(cfg_values.irls_huber_c),
            iters=int(cfg_values.irls_iters),
            min_pts=int(cfg_values.min_pts),
            n_break_cand=int(cfg_values.n_break_cand),
            q_lo=float(cfg_values.q_lo),
            q_hi=float(cfg_values.q_hi),
            slope_eps=float(cfg_values.slope_eps),
            sort_offsets=bool(cfg_values.sort_offsets),
        )
    return TwoPieceRansacAutoBreakStrategy(
        n_iter=int(cfg_values.n_iter),
        inlier_th_ms=float(cfg_values.inlier_th_ms),
        min_pts=int(cfg_values.min_pts),
        n_break_cand=int(cfg_values.n_break_cand),
        q_lo=float(cfg_values.q_lo),
        q_hi=float(cfg_values.q_hi),
        seed=int(cfg_values.seed),
        slope_eps=float(cfg_values.slope_eps),
        sort_offsets=bool(cfg_values.sort_offsets),
    )


def _run_fit_task(task: _FitTask) -> _FitTaskResult:
    cfg_values = task.cfg_values
    x_obs = np.asarray(task.x_obs, dtype=np.float32)
    y_obs = np.asarray(task.y_obs, dtype=np.float32)
    w_obs = np.asarray(task.w_obs, dtype=np.float32)
    spread_failure_reason = _offset_spread_failure_reason(
        x_obs,
        min_pts=int(cfg_values.min_pts),
        min_offset_spread_m=float(cfg_values.min_offset_spread_m),
    )
    if spread_failure_reason is not None:
        return _FitTaskResult(
            fit_key=task.fit_key,
            trend_model=None,
            diagnostics=None,
            fit_failed=False,
            failure_reason=spread_failure_reason,
            elapsed_sec=0.0,
            obs_count=int(x_obs.size),
            obs_count_before_sampling=int(task.obs_count_before_sampling),
            fit_attempted=False,
        )

    strategy = _strategy_from_fit_task_cfg(cfg_values)
    start = time.perf_counter()
    trend_model = _fit_strategy_model(
        strategy,
        torch.as_tensor(x_obs, dtype=torch.float32),
        torch.as_tensor(y_obs, dtype=torch.float32),
        torch.as_tensor(w_obs, dtype=torch.float32),
    )
    elapsed = time.perf_counter() - start
    if trend_model is None:
        return _FitTaskResult(
            fit_key=task.fit_key,
            trend_model=None,
            diagnostics=None,
            fit_failed=True,
            failure_reason=PHYSICAL_MODEL_FAILURE_FIT_FAILED,
            elapsed_sec=elapsed,
            obs_count=int(x_obs.size),
            obs_count_before_sampling=int(task.obs_count_before_sampling),
            fit_attempted=True,
        )
    try:
        diagnostics = _model_diagnostics(
            trend_model,
            obs_offsets_m=x_obs,
            obs_times_sec=y_obs,
        )
    except (TypeError, ValueError, RuntimeError):
        diagnostics = None
    return _FitTaskResult(
        fit_key=task.fit_key,
        trend_model=trend_model,
        diagnostics=diagnostics,
        fit_failed=False,
        failure_reason=None,
        elapsed_sec=elapsed,
        obs_count=int(x_obs.size),
        obs_count_before_sampling=int(task.obs_count_before_sampling),
        fit_attempted=True,
    )


def _set_fit_worker_torch_num_threads(num_threads: int) -> None:
    torch.set_num_threads(int(num_threads))


def _fit_progress_fields(
    *,
    done: int,
    total: int,
    start_sec: float,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
    force: bool = False,
) -> dict[str, object]:
    elapsed = max(0.0, time.perf_counter() - float(start_sec))
    rate = (float(done) / elapsed) if elapsed > 0.0 else 0.0
    remaining = max(0, int(total) - int(done))
    eta = (float(remaining) / rate) if rate > 0.0 else 0.0
    fields: dict[str, object] = {
        'done': int(done),
        'total': int(total),
        'elapsed': elapsed,
        'rate': rate,
        'eta': eta,
        'force': force,
    }
    if runtime_diagnostics is not None:
        fields.update(
            {
                'cache_hit': int(runtime_diagnostics.n_cache_hits),
                'cache_miss': int(runtime_diagnostics.n_cache_misses),
                'n_fit_calls': int(runtime_diagnostics.n_fit_calls),
                'fit_total_sec': float(runtime_diagnostics.ransac_fit_total_sec),
            }
        )
    return fields


def _run_fit_tasks_with_executor(
    tasks: list[_FitTask],
    *,
    cfg: PhysicsLiteConfig,
    progress: object | None = None,
    progress_context: Mapping[str, object] | None = None,
    progress_start_done: int = 0,
    progress_total: int | None = None,
    progress_start_sec: float | None = None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
    progress_cache_miss_base: int = 0,
    progress_fit_calls_base: int = 0,
    progress_fit_total_sec_base: float = 0.0,
) -> dict[tuple[int, ...], _FitTaskResult]:
    reporter = (
        progress
        if progress is not None
        else build_progress_reporter(cfg.physical_runtime.progress)
    )
    context = dict(progress_context or {})
    total = len(tasks) if progress_total is None else int(progress_total)
    start_sec = (
        time.perf_counter()
        if progress_start_sec is None
        else progress_start_sec
    )
    fit_calls_done = 0
    fit_total_sec = float(progress_fit_total_sec_base)
    executor_cfg = cfg.physical_runtime.fit_executor
    if str(executor_cfg.backend) == 'thread':
        with ThreadPoolExecutor(max_workers=executor_cfg.max_workers) as executor:
            futures = [executor.submit(_run_fit_task, task) for task in tasks]
            out: dict[tuple[int, ...], _FitTaskResult] = {}
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                out[result.fit_key] = result
                if bool(result.fit_attempted):
                    fit_calls_done += 1
                    fit_total_sec += float(result.elapsed_sec)
                fields = _fit_progress_fields(
                    done=int(progress_start_done) + completed,
                    total=total,
                    start_sec=start_sec,
                    runtime_diagnostics=runtime_diagnostics,
                    force=int(progress_start_done) + completed >= total,
                )
                fields.update(
                    {
                        'cache_miss': int(progress_cache_miss_base) + completed,
                        'n_fit_calls': int(progress_fit_calls_base) + fit_calls_done,
                        'fit_total_sec': fit_total_sec,
                    }
                )
                reporter.emit(
                    'fit.progress',
                    **context,
                    **fields,
                )
            return out

    with ProcessPoolExecutor(
        max_workers=executor_cfg.max_workers,
        initializer=_set_fit_worker_torch_num_threads,
        initargs=(int(executor_cfg.torch_num_threads_per_worker),),
    ) as executor:
        futures = [executor.submit(_run_fit_task, task) for task in tasks]
        out: dict[tuple[int, ...], _FitTaskResult] = {}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            out[result.fit_key] = result
            if bool(result.fit_attempted):
                fit_calls_done += 1
                fit_total_sec += float(result.elapsed_sec)
            fields = _fit_progress_fields(
                done=int(progress_start_done) + completed,
                total=total,
                start_sec=start_sec,
                runtime_diagnostics=runtime_diagnostics,
                force=int(progress_start_done) + completed >= total,
            )
            fields.update(
                {
                    'cache_miss': int(progress_cache_miss_base) + completed,
                    'n_fit_calls': int(progress_fit_calls_base) + fit_calls_done,
                    'fit_total_sec': fit_total_sec,
                }
            )
            reporter.emit(
                'fit.progress',
                **context,
                **fields,
            )
        return out


def _fit_model_for_plan(
    *,
    strategy: _PhysicalFitStrategy,
    plan: _ObservationPlan,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    w_obs: np.ndarray,
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
            w_tensor = torch.as_tensor(w_obs, dtype=torch.float32)
            if runtime_diagnostics is None:
                trend_model = _fit_strategy_model(
                    strategy,
                    x_tensor,
                    y_tensor,
                    w_tensor,
                )
            else:
                with runtime_diagnostics.time_ransac_fit(
                    obs_count=int(np.asarray(x_obs).size),
                    obs_count_before=obs_count_before_sampling,
                ):
                    trend_model = _fit_strategy_model(
                        strategy,
                        x_tensor,
                        y_tensor,
                        w_tensor,
                    )
        except (TypeError, ValueError, RuntimeError):
            trend_model = None

        if trend_model is None:
            entry = _FitCacheEntry(
                model=None,
                diagnostics=None,
                fit_failed=True,
                failure_reason=PHYSICAL_MODEL_FAILURE_FIT_FAILED,
            )
        else:
            entry = _FitCacheEntry(
                model=trend_model,
                diagnostics=None,
                fit_failed=False,
            )
        cache[cache_key] = entry
    elif runtime_diagnostics is not None:
        runtime_diagnostics.record_cache_hit()

    if entry.failure_reason is not None:
        return None, None, entry.failure_reason
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
        failure_reason=entry.failure_reason,
    )
    return diagnostics


def _assign_model_diagnostics(
    arrays: dict[str, np.ndarray],
    trace_idx: int,
    diagnostics: tuple[float, float, float, float, float, float, float] | None,
) -> None:
    _assign_model_diagnostics_batch(
        arrays,
        np.asarray([int(trace_idx)], dtype=np.int64),
        diagnostics,
    )


def _assign_model_diagnostics_batch(
    arrays: dict[str, np.ndarray],
    trace_indices: np.ndarray,
    diagnostics: tuple[float, float, float, float, float, float, float] | None,
) -> None:
    if diagnostics is None:
        return
    indices = np.asarray(trace_indices, dtype=np.int64)
    if indices.size == 0:
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
    arrays['physical_model_break_offset_m'][indices] = np.float32(break_offset)
    arrays['physical_model_slope_near_s_per_m'][indices] = np.float32(slope_near)
    arrays['physical_model_slope_far_s_per_m'][indices] = np.float32(slope_far)
    arrays['physical_model_velocity_near_m_s'][indices] = np.float32(velocity_near)
    arrays['physical_model_velocity_far_m_s'][indices] = np.float32(velocity_far)
    arrays['physical_model_resid_p50_ms'][indices] = np.float32(resid_p50)
    arrays['physical_model_resid_p90_ms'][indices] = np.float32(resid_p90)


def _shift_for_trace_indices(
    t0_shift_sec: float | np.ndarray,
    *,
    trace_indices: np.ndarray,
    n_traces: int,
) -> float | np.ndarray:
    shift = np.asarray(t0_shift_sec, dtype=np.float64)
    if shift.ndim == 0:
        return float(shift)
    if shift.ndim != 1:
        msg = 't0_shift_sec must be scalar or 1D'
        raise ValueError(msg)
    indices = np.asarray(trace_indices, dtype=np.int64)
    if shift.shape == indices.shape:
        return shift
    if int(shift.shape[0]) == int(n_traces):
        return shift[indices]
    msg = (
        't0_shift_sec vector must match trace_indices or full trace count, '
        f'got {shift.shape[0]} for {indices.size} indices and {n_traces} traces'
    )
    raise ValueError(msg)


def _status_for_plan_batch(
    trace_indices: np.ndarray,
    plan_by_trace: Mapping[int, _ObservationPlan] | _ObservationPlan,
) -> np.ndarray:
    indices = np.asarray(trace_indices, dtype=np.int64)
    if isinstance(plan_by_trace, _ObservationPlan):
        status = (
            PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT
            if bool(plan_by_trace.relaxed)
            else PHYSICAL_MODEL_STATUS_TWO_PIECE_OK
        )
        return np.full((indices.size,), np.uint8(status), dtype=np.uint8)

    out = np.empty((indices.size,), dtype=np.uint8)
    for pos, trace_idx in enumerate(indices.tolist()):
        plan = plan_by_trace[int(trace_idx)]
        out[pos] = np.uint8(
            PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT
            if bool(plan.relaxed)
            else PHYSICAL_MODEL_STATUS_TWO_PIECE_OK
        )
    return out


def _assign_model_prediction_batch(
    arrays: dict[str, np.ndarray],
    trace_indices: np.ndarray,
    *,
    trend_model,
    diagnostics: tuple[float, float, float, float, float, float, float] | None,
    plan_by_trace: Mapping[int, _ObservationPlan] | _ObservationPlan,
    offset_abs_m: np.ndarray,
    dt: float,
    n_samples: int,
    runtime_fit_source: int,
    t0_shift_sec: float | np.ndarray = 0.0,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
) -> np.ndarray:
    indices = np.asarray(trace_indices, dtype=np.int64)
    if indices.ndim != 1:
        msg = 'trace_indices must be 1D'
        raise ValueError(msg)
    if indices.size == 0:
        return np.zeros((0,), dtype=np.bool_)

    if runtime_diagnostics is not None:
        runtime_diagnostics.inc('n_prediction_calls', int(indices.size))
        runtime_diagnostics.inc('n_prediction_batches')

    offset_arr = np.asarray(offset_abs_m, dtype=np.float32)
    with (
        runtime_diagnostics.time_block('prediction_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        try:
            physical_t_sec = _predict_model_array_sec(
                trend_model,
                offset_arr[indices],
            )
            shift = _shift_for_trace_indices(
                t0_shift_sec,
                trace_indices=indices,
                n_traces=int(offset_arr.shape[0]),
            )
            physical_t_sec = physical_t_sec + shift
        except (TypeError, ValueError, RuntimeError):
            physical_t_sec = np.full((indices.size,), np.nan, dtype=np.float64)

    valid = np.isfinite(physical_t_sec)
    if not bool(np.any(valid)):
        return valid.astype(np.bool_, copy=False)

    with (
        runtime_diagnostics.time_block('assignment_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        valid_indices = indices[valid]
        center_i = np.rint(physical_t_sec[valid] / float(dt)).astype(np.int64)
        center_i = np.clip(center_i, 0, int(n_samples) - 1).astype(np.int32)
        arrays['physical_center_i'][valid_indices] = center_i
        arrays['physical_center_t_sec'][valid_indices] = (
            center_i.astype(np.float64) * float(dt)
        ).astype(np.float32)
        arrays['physical_model_status'][valid_indices] = _status_for_plan_batch(
            valid_indices,
            plan_by_trace,
        )
        arrays['physical_model_failure_reason'][valid_indices] = np.uint8(
            PHYSICAL_MODEL_FAILURE_NONE
        )
        arrays['physical_runtime_fit_source'][valid_indices] = np.uint8(
            runtime_fit_source
        )
        _assign_model_diagnostics_batch(arrays, valid_indices, diagnostics)
    return valid.astype(np.bool_, copy=False)


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
    valid = _assign_model_prediction_batch(
        arrays,
        np.asarray([int(trace_idx)], dtype=np.int64),
        trend_model=trend_model,
        diagnostics=diagnostics,
        plan_by_trace=plan,
        offset_abs_m=offset_abs_m,
        dt=dt,
        n_samples=n_samples,
        runtime_fit_source=runtime_fit_source,
        t0_shift_sec=t0_shift_sec,
        runtime_diagnostics=runtime_diagnostics,
    )
    return bool(valid[0])


def _prepare_trace_plan_assignment(
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
    trend_provider: object | None = None,
    merged: MergeResult,
    cfg: PhysicsLiteConfig,
    observation_plan_cache: _ObservationPlanCache,
    min_fit_obs: int,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
    pending_trend_fallback: _PendingTrendFallback | None = None,
) -> _TracePlanAssignment | None:
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
            trend_provider=trend_provider,
            merged=merged,
            pending_trend_fallback=pending_trend_fallback,
        )
        return None

    plan = _build_observation_plan(
        trace_idx=trace_idx,
        target_group_id=int(group_id_by_trace[trace_idx]),
        group_context_by_id=group_context_by_id,
        geometry=geometry,
        offset_abs_m=offset_abs_m,
        offset_signed_m=offset_signed_m,
        cfg=cfg,
        runtime_diagnostics=runtime_diagnostics,
        plan_cache=observation_plan_cache,
    )
    if plan is None:
        _assign_fallback(
            arrays,
            trace_idx,
            failure_reason=PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            pending_trend_fallback=pending_trend_fallback,
        )
        return None

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
            trend_provider=trend_provider,
            merged=merged,
            pending_trend_fallback=pending_trend_fallback,
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
            pick_t_sec=pick_t_sec,
            coarse_pmax=table.coarse_pmax,
            cfg=cfg,
            min_required_obs=int(min_fit_obs),
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


def _assign_prepared_model_prediction_batch(
    *,
    arrays: dict[str, np.ndarray],
    items: list[tuple[int, _TraceFitResult]],
    offset_abs_m: np.ndarray,
    dt: float,
    n_samples: int,
    runtime_fit_source: int,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    trend_provider: object | None = None,
    merged: MergeResult,
    fit_cache: dict[tuple[int, ...], _FitCacheEntry],
    t0_shift_sec: float | np.ndarray = 0.0,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
    pending_trend_fallback: _PendingTrendFallback | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float, float, float, float] | None]:
    if not items:
        return np.zeros((0,), dtype=np.bool_), None

    trace_indices = np.asarray(
        [trace_idx for trace_idx, _result in items], dtype=np.int64
    )
    first_result = items[0][1]
    if first_result.plan is None or first_result.trend_model is None:
        msg = 'prepared model assignment requires plan and trend_model'
        raise ValueError(msg)

    plan_by_trace = {
        int(trace_idx): result.plan
        for trace_idx, result in items
        if result.plan is not None
    }
    diagnostics = first_result.diagnostics
    valid = _assign_model_prediction_batch(
        arrays,
        trace_indices,
        trend_model=first_result.trend_model,
        diagnostics=diagnostics,
        plan_by_trace=plan_by_trace,
        offset_abs_m=offset_abs_m,
        dt=dt,
        n_samples=n_samples,
        runtime_fit_source=runtime_fit_source,
        t0_shift_sec=t0_shift_sec,
        runtime_diagnostics=runtime_diagnostics,
    )

    if bool(np.any(valid)) and diagnostics is None:
        first_valid_pos = int(np.flatnonzero(valid)[0])
        first_valid_result = items[first_valid_pos][1]
        if (
            first_valid_result.plan is not None
            and first_valid_result.x_obs is not None
            and first_valid_result.y_obs is not None
        ):
            diagnostics = _diagnostics_for_plan(
                plan=first_valid_result.plan,
                x_obs=first_valid_result.x_obs,
                y_obs=first_valid_result.y_obs,
                cache=fit_cache,
            )
            _assign_model_diagnostics_batch(
                arrays,
                trace_indices[valid],
                diagnostics,
            )

    for trace_idx in trace_indices[~valid].tolist():
        _assign_fallback(
            arrays,
            int(trace_idx),
            failure_reason=PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            pending_trend_fallback=pending_trend_fallback,
        )
    return valid, diagnostics


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


def _assign_fit_context_fallback(
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


def _fit_and_assign_context_work_item(
    *,
    arrays: dict[str, np.ndarray],
    work_item: _FitContextWorkItem,
    offset_abs_m: np.ndarray,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    trend_provider: object | None = None,
    merged: MergeResult,
    cfg: PhysicsLiteConfig,
    strategy: _PhysicalFitStrategy,
    fit_cache: dict[tuple[int, ...], _FitCacheEntry],
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
    pending_trend_fallback: _PendingTrendFallback | None = None,
) -> _FitContextWorkResult:
    trend_model, diagnostics, fit_failure_reason = _fit_model_for_plan(
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
        arrays=arrays,
        work_item=work_item,
        offset_abs_m=offset_abs_m,
        table=table,
        feasible=feasible,
        trend=trend,
        trend_provider=trend_provider,
        merged=merged,
        fit_cache=fit_cache,
        trend_model=trend_model,
        diagnostics=diagnostics,
        fit_failure_reason=fit_failure_reason,
        runtime_diagnostics=runtime_diagnostics,
        pending_trend_fallback=pending_trend_fallback,
    )


def _assign_fit_context_work_item_outcome(
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
    diagnostics: tuple[float, float, float, float, float, float, float] | None,
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


def _prepare_fit_context_assignments_for_trace_indices(
    trace_indices: np.ndarray,
    *,
    arrays: dict[str, np.ndarray],
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
    trend_provider: object | None = None,
    merged: MergeResult,
    cfg: PhysicsLiteConfig,
    observation_plan_cache: _ObservationPlanCache,
    min_fit_obs: int,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
    pending_trend_fallback: _PendingTrendFallback | None = None,
) -> tuple[
    dict[tuple[int, ...], list[_TracePlanAssignment]],
    list[_TracePlanAssignment],
]:
    grouped: dict[tuple[int, ...], list[_TracePlanAssignment]] = {}
    ordered: list[_TracePlanAssignment] = []
    for trace_idx in np.asarray(trace_indices, dtype=np.int64).tolist():
        assignment = _prepare_trace_plan_assignment(
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
            trend_provider=trend_provider,
            merged=merged,
            cfg=cfg,
            observation_plan_cache=observation_plan_cache,
            min_fit_obs=min_fit_obs,
            runtime_diagnostics=runtime_diagnostics,
            pending_trend_fallback=pending_trend_fallback,
        )
        if assignment is None:
            continue
        grouped.setdefault(assignment.fit_key, []).append(assignment)
        ordered.append(assignment)
    return grouped, ordered


def _fit_and_assign_context_work_items(
    work_items: list[_FitContextWorkItem],
    *,
    arrays: dict[str, np.ndarray],
    offset_abs_m: np.ndarray,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    trend_provider: object | None = None,
    merged: MergeResult,
    cfg: PhysicsLiteConfig,
    strategy: _PhysicalFitStrategy,
    fit_cache: dict[tuple[int, ...], _FitCacheEntry],
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
    progress: object | None = None,
    progress_context: Mapping[str, object] | None = None,
    pending_trend_fallback: _PendingTrendFallback | None = None,
) -> dict[tuple[int, ...], _FitContextWorkResult]:
    reporter = progress if progress is not None else NullProgressReporter()
    context = dict(progress_context or {})
    fit_start = time.perf_counter()
    if bool(cfg.physical_runtime.fit_executor.enabled) and work_items:
        return _fit_and_assign_context_work_items_parallel(
            work_items,
            arrays=arrays,
            offset_abs_m=offset_abs_m,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            cfg=cfg,
            fit_cache=fit_cache,
            runtime_diagnostics=runtime_diagnostics,
            progress=reporter,
            progress_context=context,
            progress_start_sec=fit_start,
            pending_trend_fallback=pending_trend_fallback,
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
            arrays=arrays,
            work_item=work_item,
            offset_abs_m=offset_abs_m,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            cfg=cfg,
            strategy=strategy,
            fit_cache=fit_cache,
            runtime_diagnostics=runtime_diagnostics,
            pending_trend_fallback=pending_trend_fallback,
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


def _fit_and_assign_context_work_items_parallel(
    work_items: list[_FitContextWorkItem],
    *,
    arrays: dict[str, np.ndarray],
    offset_abs_m: np.ndarray,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    trend_provider: object | None = None,
    merged: MergeResult,
    cfg: PhysicsLiteConfig,
    fit_cache: dict[tuple[int, ...], _FitCacheEntry],
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
    progress: object | None = None,
    progress_context: Mapping[str, object] | None = None,
    progress_start_sec: float | None = None,
    pending_trend_fallback: _PendingTrendFallback | None = None,
) -> dict[tuple[int, ...], _FitContextWorkResult]:
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
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            fit_cache=fit_cache,
            trend_model=entry.model,
            diagnostics=entry.diagnostics,
            fit_failure_reason=fit_failure_reason,
            runtime_diagnostics=runtime_diagnostics,
            pending_trend_fallback=pending_trend_fallback,
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
                table=table,
                feasible=feasible,
                trend=trend,
                trend_provider=trend_provider,
                merged=merged,
                fit_cache=fit_cache,
                trend_model=task_result.trend_model,
                diagnostics=task_result.diagnostics,
                fit_failure_reason=task_result.failure_reason,
                runtime_diagnostics=runtime_diagnostics,
                pending_trend_fallback=pending_trend_fallback,
            )

    return results


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
    trend_provider: object | None = None,
    merged: MergeResult,
    cfg: PhysicsLiteConfig,
    pending_trend_fallback: _PendingTrendFallback | None = None,
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
        trend_provider=trend_provider,
        merged=merged,
        pending_trend_fallback=pending_trend_fallback,
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


def preflight_geometry_two_piece_fallback(
    *,
    coarse_npz: Mapping[str, np.ndarray],
    table: CoarsePickTable,
    cfg: PhysicsLiteConfig,
) -> PhysicalCenterFallbackPreflight:
    n = int(table.n_traces)
    use_geometry_offset = bool(cfg.physical_trend.use_geometry_offset)
    try:
        geometry = load_coarse_geometry_from_npz(coarse_npz, n_traces=n)
    except (KeyError, TypeError, ValueError):
        geometry = None

    if use_geometry_offset and geometry is None:
        return PhysicalCenterFallbackPreflight(
            status='geometry_invalid',
            reason='geometry_invalid',
            fallback_mode=str(cfg.physical_runtime.geometry_invalid_fallback),
            geometry_loaded=False,
            groups=None,
        )

    source_grouping_invalid = False
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
            source_grouping_invalid = True
        if not source_xy_degenerate:
            groups = build_source_groups(
                source_group_geometry,
                coord_group_tol_m=coord_group_tol_m,
            )
    if len(groups) == 0 and not use_geometry_offset:
        groups = _build_table_source_groups(table, n_traces=n)

    if source_grouping_invalid:
        return PhysicalCenterFallbackPreflight(
            status='geometry_invalid',
            reason='source_xy_degenerate',
            fallback_mode=str(cfg.physical_runtime.geometry_invalid_fallback),
            geometry_loaded=geometry is not None,
            groups=len(groups),
        )
    if len(groups) == 0:
        return PhysicalCenterFallbackPreflight(
            status='geometry_invalid',
            reason='source_group_empty',
            fallback_mode=str(cfg.physical_runtime.group_invalid_fallback),
            geometry_loaded=geometry is not None,
            groups=0,
        )
    return PhysicalCenterFallbackPreflight(
        status=None,
        reason=None,
        fallback_mode=None,
        geometry_loaded=geometry is not None,
        groups=len(groups),
    )


def build_geometry_two_piece_physical_center(
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
    reporter = (
        progress
        if progress is not None
        else build_progress_reporter(cfg.physical_runtime.progress)
    )
    context = dict(progress_context or {})
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
        try:
            geometry = load_coarse_geometry_from_npz(coarse_npz, n_traces=n)
        except (KeyError, TypeError, ValueError):
            geometry = None
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
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            reporter=reporter,
            context=context,
            runtime_diagnostics=runtime_diagnostics,
        )

    segment_by_offset_sign = bool(cfg.physical_trend.segment_by_offset_sign)
    offset_signed_labels = None
    if use_geometry_offset:
        offset_abs_m = _as_vector(
            'geometry.offset_abs_geom_m',
            geometry.offset_abs_geom_m,
            n_traces=n,
            dtype=np.float32,
        )
        if segment_by_offset_sign and geometry.offset_signed_geom_m is not None:
            offset_signed_m = _as_vector(
                'geometry.offset_signed_geom_m',
                geometry.offset_signed_geom_m,
                n_traces=n,
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
        offset_abs_m = _table_offset_abs_m(table, n_traces=n)
        if segment_by_offset_sign:
            offset_signed_m = _as_vector(
                'table.offset_m',
                table.offset_m,
                n_traces=n,
                dtype=np.float32,
            )
            offset_signed_labels = _signed_offset_side_labels(offset_signed_m)
        else:
            offset_signed_m = None
        offset_source = PHYSICAL_OFFSET_SOURCE_HEADER

    reporter.emit('physical-center.stage_start', **context, stage='source_grouping')
    stage_start = time.perf_counter()
    with (
        runtime_diagnostics.time_block('source_grouping_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        source_grouping_invalid = False
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
                source_grouping_invalid = True
            if not source_xy_degenerate:
                groups = build_source_groups(
                    source_group_geometry,
                    coord_group_tol_m=coord_group_tol_m,
                )
                source_groups_from_geometry = len(groups) > 0
        if len(groups) == 0 and not use_geometry_offset:
            groups = _build_table_source_groups(table, n_traces=n)
    reporter.emit(
        'physical-center.stage_done',
        **context,
        stage='source_grouping',
        elapsed=time.perf_counter() - stage_start,
        groups=len(groups),
    )

    if source_grouping_invalid:
        return _emit_fallback_all_and_done(
            status='geometry_invalid',
            reason='source_xy_degenerate',
            fallback_mode=str(cfg.physical_runtime.geometry_invalid_fallback),
            failure_reason=PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
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
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
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
        groups_by_id = {int(group.group_id): group for group in groups}
        group_id_by_trace = np.full((n,), -1, dtype=np.int32)
        for group in groups:
            group_id_by_trace[
                np.asarray(group.trace_indices, dtype=np.int64)
            ] = np.int32(group.group_id)
    reporter.emit(
        'physical-center.stage_done',
        **context,
        stage='source_group_ordering',
        elapsed=time.perf_counter() - stage_start,
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

    arrays = _allocate_result_arrays(table)
    pending_trend_fallback = _PendingTrendFallback()
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
    strategy = _fit_strategy(cfg)
    fit_cache: dict[tuple[int, ...], _FitCacheEntry] = {}
    min_fit_obs = 2 * _fit_min_pts(cfg)

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

        anchor_trace_chunks = [
            np.asarray(group.trace_indices, dtype=np.int64)
            for group in groups
            if bool(is_anchor_by_id.get(int(group.group_id), False))
        ]
        anchor_trace_indices = (
            np.concatenate(anchor_trace_chunks)
            if anchor_trace_chunks
            else np.zeros((0,), dtype=np.int64)
        )
        anchor_assignments_by_fit, anchor_assignments = (
            _prepare_fit_context_assignments_for_trace_indices(
                anchor_trace_indices,
                arrays=arrays,
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
                trend_provider=trend_provider,
                merged=merged,
                cfg=cfg,
                observation_plan_cache=observation_plan_cache,
                min_fit_obs=min_fit_obs,
                runtime_diagnostics=runtime_diagnostics,
                pending_trend_fallback=pending_trend_fallback,
            )
        )
        anchor_work_items = _build_fit_context_work_items(
            anchor_assignments_by_fit,
            offset_abs_m=offset_abs_m,
            pick_t_sec=pick_t_sec,
            coarse_pmax=table.coarse_pmax,
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
            arrays=arrays,
            offset_abs_m=offset_abs_m,
            table=table,
            feasible=feasible,
            trend=trend,
            trend_provider=trend_provider,
            merged=merged,
            cfg=cfg,
            strategy=strategy,
            fit_cache=fit_cache,
            runtime_diagnostics=runtime_diagnostics,
            progress=reporter,
            progress_context=context,
            pending_trend_fallback=pending_trend_fallback,
        )
        if runtime_diagnostics is not None:
            anchor_fit_call_delta = max(
                0,
                int(runtime_diagnostics.n_fit_calls) - anchor_fit_calls_before,
            )
            if anchor_fit_call_delta > 0:
                runtime_diagnostics.record_anchor_fit_calls(anchor_fit_call_delta)
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
        anchor_models_by_group_id: dict[
            int,
            dict[tuple[int, int, bool], _AnchorModelContext],
        ] = {}
        for model_key, anchor_context in anchor_models.items():
            anchor_group_id, side, segment_id, relaxed = model_key
            anchor_models_by_group_id.setdefault(int(anchor_group_id), {}).setdefault(
                (int(side), int(segment_id), bool(relaxed)),
                anchor_context,
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
            compatible_anchor_context_by_plan_key: dict[
                tuple[int, int, bool],
                _AnchorModelContext,
            ] = {}
            with (
                runtime_diagnostics.time_block('compatible_anchor_search_sec')
                if runtime_diagnostics is not None
                else nullcontext()
            ):
                if (
                    bool(cfg.physical_runtime.anchor_reuse.enabled)
                    and bool(distance_ok)
                ):
                    with (
                        runtime_diagnostics.time_block('anchor_lookup_sec')
                        if runtime_diagnostics is not None
                        else nullcontext()
                    ):
                        compatible_anchor_context_by_plan_key = (
                            anchor_models_by_group_id.get(nearest_anchor_id, {})
                        )
            reuse_items: dict[
                tuple[int, int, int, bool],
                list[tuple[int, _ObservationPlan, _AnchorModelContext]],
            ] = {}
            fallback_full_trace_indices: list[int] = []
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
                        trend_provider=trend_provider,
                        merged=merged,
                        pending_trend_fallback=pending_trend_fallback,
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
                    plan_cache=observation_plan_cache,
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

                anchor_context = None
                key = None
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
                    anchor_context = compatible_anchor_context_by_plan_key.get(
                        (int(plan.side), int(plan.segment_id), bool(plan.relaxed))
                    )

                if anchor_context is not None and plan is not None and key is not None:
                    reuse_items.setdefault(key, []).append(
                        (trace_idx, plan, anchor_context)
                    )
                    continue
                if runtime_diagnostics is not None:
                    runtime_diagnostics.record_no_compatible_anchor_context()

                fallback = str(
                    cfg.physical_runtime.anchor_reuse.fallback_if_no_compatible_segment
                )
                if fallback == 'full_fit':
                    fallback_full_group_ids.add(group_id)
                    fallback_full_trace_indices.append(trace_idx)
                else:
                    _fallback_no_compatible_anchor(
                        arrays=arrays,
                        trace_idx=trace_idx,
                        table=table,
                        feasible=feasible,
                        trend=trend,
                        trend_provider=trend_provider,
                        merged=merged,
                        cfg=cfg,
                        pending_trend_fallback=pending_trend_fallback,
                    )

            if fallback_full_trace_indices:
                fallback_full_assignments_by_fit, _fallback_full_assignments = (
                    _prepare_fit_context_assignments_for_trace_indices(
                        np.asarray(fallback_full_trace_indices, dtype=np.int64),
                        arrays=arrays,
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
                        trend_provider=trend_provider,
                        merged=merged,
                        cfg=cfg,
                        observation_plan_cache=observation_plan_cache,
                        min_fit_obs=min_fit_obs,
                        runtime_diagnostics=runtime_diagnostics,
                        pending_trend_fallback=pending_trend_fallback,
                    )
                )
                fallback_full_work_items = _build_fit_context_work_items(
                    fallback_full_assignments_by_fit,
                    offset_abs_m=offset_abs_m,
                    pick_t_sec=pick_t_sec,
                    coarse_pmax=table.coarse_pmax,
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
                    arrays=arrays,
                    offset_abs_m=offset_abs_m,
                    table=table,
                    feasible=feasible,
                    trend=trend,
                    trend_provider=trend_provider,
                    merged=merged,
                    cfg=cfg,
                    strategy=strategy,
                    fit_cache=fit_cache,
                    runtime_diagnostics=runtime_diagnostics,
                    progress=reporter,
                    progress_context=context,
                    pending_trend_fallback=pending_trend_fallback,
                )

            non_anchor_mode = str(cfg.physical_runtime.anchor_reuse.non_anchor_mode)
            if runtime_diagnostics is not None and reuse_items:
                runtime_diagnostics.record_reuse_contexts(len(reuse_items))
            if non_anchor_mode == 'nearest_anchor':
                for items in reuse_items.values():
                    trace_indices = np.asarray(
                        [trace_idx for trace_idx, _plan, _context in items],
                        dtype=np.int64,
                    )
                    anchor_context = items[0][2]
                    plan_by_trace = {
                        int(trace_idx): plan for trace_idx, plan, _context in items
                    }
                    valid = _assign_model_prediction_batch(
                        arrays,
                        trace_indices,
                        trend_model=anchor_context.trend_model,
                        diagnostics=anchor_context.diagnostics,
                        plan_by_trace=plan_by_trace,
                        offset_abs_m=offset_abs_m,
                        dt=dt,
                        n_samples=n_samples,
                        runtime_fit_source=(
                            PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE
                        ),
                        runtime_diagnostics=runtime_diagnostics,
                    )
                    n_reused_predictions += int(np.count_nonzero(valid))
                    for trace_idx in trace_indices[~valid].tolist():
                        _assign_fallback(
                            arrays,
                            int(trace_idx),
                            failure_reason=(PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID),
                            table=table,
                            feasible=feasible,
                            trend=trend,
                            trend_provider=trend_provider,
                            merged=merged,
                            pending_trend_fallback=pending_trend_fallback,
                        )
                continue

            stats_by_key: dict[tuple[int, int, int, bool], _ReuseShiftStats] = {}
            adaptive_refit = False
            for key, items in reuse_items.items():
                key_trace_indices = np.asarray(
                    [trace_idx for trace_idx, _plan, _context in items],
                    dtype=np.int64,
                )
                anchor_context = items[0][2]
                with (
                    runtime_diagnostics.time_block('t0_shift_sec')
                    if runtime_diagnostics is not None
                    else nullcontext()
                ):
                    stats = _compute_reuse_t0_shift_stats(
                        trend_model=anchor_context.trend_model,
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
                refit_assignments_by_fit, _refit_assignments = (
                    _prepare_fit_context_assignments_for_trace_indices(
                        group_trace_indices,
                        arrays=arrays,
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
                        trend_provider=trend_provider,
                        merged=merged,
                        cfg=cfg,
                        observation_plan_cache=observation_plan_cache,
                        min_fit_obs=min_fit_obs,
                        runtime_diagnostics=runtime_diagnostics,
                        pending_trend_fallback=pending_trend_fallback,
                    )
                )
                refit_work_items = _build_fit_context_work_items(
                    refit_assignments_by_fit,
                    offset_abs_m=offset_abs_m,
                    pick_t_sec=pick_t_sec,
                    coarse_pmax=table.coarse_pmax,
                    runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT,
                )
                reporter.emit(
                    'physical-center.contexts_built',
                    **context,
                    phase='adaptive_refit',
                    work_items=len(refit_work_items),
                    unique_keys=len(refit_assignments_by_fit),
                )
                refit_results = _fit_and_assign_context_work_items(
                    refit_work_items,
                    arrays=arrays,
                    offset_abs_m=offset_abs_m,
                    table=table,
                    feasible=feasible,
                    trend=trend,
                    trend_provider=trend_provider,
                    merged=merged,
                    cfg=cfg,
                    strategy=strategy,
                    fit_cache=fit_cache,
                    runtime_diagnostics=runtime_diagnostics,
                    progress=reporter,
                    progress_context=context,
                    pending_trend_fallback=pending_trend_fallback,
                )
                assigned_count = sum(
                    len(result.valid_trace_indices)
                    for result in refit_results.values()
                )
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
                            trend_provider=trend_provider,
                            merged=merged,
                            pending_trend_fallback=pending_trend_fallback,
                        )
                    continue

                use_shift = (
                    fallback_mode == 'nearest_anchor_plus_t0_shift'
                    and bool(stats.shift_valid)
                )
                shift_sec = float(stats.t0_shift_sec) if use_shift else 0.0
                trace_indices = np.asarray(
                    [trace_idx for trace_idx, _plan, _context in items],
                    dtype=np.int64,
                )
                anchor_context = items[0][2]
                plan_by_trace = {
                    int(trace_idx): plan for trace_idx, plan, _context in items
                }
                valid = _assign_model_prediction_batch(
                    arrays,
                    trace_indices,
                    trend_model=anchor_context.trend_model,
                    diagnostics=anchor_context.diagnostics,
                    plan_by_trace=plan_by_trace,
                    offset_abs_m=offset_abs_m,
                    dt=dt,
                    n_samples=n_samples,
                    runtime_fit_source=(
                        PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE
                    ),
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
                        table=table,
                        feasible=feasible,
                        trend=trend,
                        trend_provider=trend_provider,
                        merged=merged,
                        pending_trend_fallback=pending_trend_fallback,
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
        result = _finalize_result_with_pending_trend_fallback(
            arrays,
            pending_trend_fallback=pending_trend_fallback,
            table=table,
            feasible=feasible,
            trend=trend,
            merged=merged,
            trend_provider=trend_provider,
        )
        reporter.emit(
            'physical-center.done',
            **context,
            status='ok',
            n_traces=n,
            n_source_groups=len(groups),
            n_unique_fit_contexts=len(fit_cache),
        )
        return result

    reporter.emit(
        'physical-center.stage_start',
        **context,
        stage='fit_context_preparation',
    )
    stage_start = time.perf_counter()
    fit_context_assignments: dict[tuple[int, ...], list[_TracePlanAssignment]] = {}
    for trace_idx in range(n):
        assignment = _prepare_trace_plan_assignment(
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
            trend_provider=trend_provider,
            merged=merged,
            cfg=cfg,
            observation_plan_cache=observation_plan_cache,
            min_fit_obs=min_fit_obs,
            runtime_diagnostics=runtime_diagnostics,
            pending_trend_fallback=pending_trend_fallback,
        )
        if assignment is not None:
            fit_context_assignments.setdefault(assignment.fit_key, []).append(
                assignment
            )
    reporter.emit(
        'physical-center.stage_done',
        **context,
        stage='fit_context_preparation',
        elapsed=time.perf_counter() - stage_start,
        unique_keys=len(fit_context_assignments),
    )

    work_items = _build_fit_context_work_items(
        fit_context_assignments,
        offset_abs_m=offset_abs_m,
        pick_t_sec=pick_t_sec,
        coarse_pmax=table.coarse_pmax,
        runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_FULL_FIT,
    )
    reporter.emit(
        'physical-center.contexts_built',
        **context,
        work_items=len(work_items),
        unique_keys=len(fit_context_assignments),
    )
    _fit_and_assign_context_work_items(
        work_items,
        arrays=arrays,
        offset_abs_m=offset_abs_m,
        table=table,
        feasible=feasible,
        trend=trend,
        trend_provider=trend_provider,
        merged=merged,
        cfg=cfg,
        strategy=strategy,
        fit_cache=fit_cache,
        runtime_diagnostics=runtime_diagnostics,
        progress=reporter,
        progress_context=context,
        pending_trend_fallback=pending_trend_fallback,
    )

    if runtime_diagnostics is not None:
        runtime_diagnostics.set_unique_fit_contexts(len(fit_cache))

    result = _finalize_result_with_pending_trend_fallback(
        arrays,
        pending_trend_fallback=pending_trend_fallback,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        trend_provider=trend_provider,
    )
    reporter.emit(
        'physical-center.done',
        **context,
        status='ok',
        n_traces=n,
        n_source_groups=len(groups),
        n_unique_fit_contexts=len(fit_cache),
    )
    return result
