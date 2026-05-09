from __future__ import annotations

from collections.abc import Mapping
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
    load_coarse_geometry_from_npz,
    select_nearest_source_groups,
    signed_offset_side_from_geometry,
    split_offset_gap_segments,
)
from .merge import MergeResult
from .pick_table import CoarsePickTable
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

__all__ = [
    'PHYSICAL_OFFSET_SOURCE_GEOMETRY',
    'PHYSICAL_OFFSET_SOURCE_HEADER',
    'PHYSICAL_OFFSET_SOURCE_LABELS',
    'PHYSICAL_OFFSET_SOURCE_NONE',
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
    return _finalize_result(arrays)


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
    groups: tuple[SourceGroup, ...],
    groups_by_id: Mapping[int, SourceGroup],
    geometry: CoarseGeometry | None,
    offset_abs_m: np.ndarray,
    valid_for_fit: np.ndarray,
    cfg: PhysicsLiteConfig,
    use_neighbor_context: bool,
) -> _ObservationPlan | None:
    group_ids = _select_group_ids(
        groups=groups,
        target_group_id=target_group_id,
        cfg=cfg,
        use_neighbor_context=use_neighbor_context,
    )
    neighbor_indices = _concat_group_traces(group_ids, groups_by_id=groups_by_id)
    valid_obs = neighbor_indices[
        np.asarray(valid_for_fit, dtype=np.bool_)[neighbor_indices]
    ]
    prefilter_valid_count = int(valid_obs.size)
    min_fit_obs = 2 * int(cfg.two_piece_ransac.min_pts)

    if prefilter_valid_count < min_fit_obs:
        return _ObservationPlan(
            obs_indices=valid_obs,
            neighbor_count=int(group_ids.size),
            prefilter_valid_count=prefilter_valid_count,
            segment_id=-1,
            side=0,
            relaxed=False,
        )

    side_obs = valid_obs
    side = 0
    side_reliable = False
    if geometry is not None and bool(cfg.physical_trend.segment_by_offset_sign):
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
            neighbor_count=int(group_ids.size),
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
            neighbor_count=int(group_ids.size),
            prefilter_valid_count=prefilter_valid_count,
            segment_id=0,
            side=side,
            relaxed=True,
        )

    if side_reliable and int(valid_obs.size) >= min_fit_obs:
        return _ObservationPlan(
            obs_indices=valid_obs,
            neighbor_count=int(group_ids.size),
            prefilter_valid_count=prefilter_valid_count,
            segment_id=0,
            side=0,
            relaxed=True,
        )

    return _ObservationPlan(
        obs_indices=segment_obs,
        neighbor_count=int(group_ids.size),
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


def _predict_model_sec(trend_model, offset_m: float) -> float:
    pred = trend_model.predict(torch.tensor([float(offset_m)], dtype=torch.float32))
    pred_np = _tensor_to_numpy(pred).astype(np.float64, copy=False)
    if pred_np.shape != (1,):
        msg = f'trend model prediction must have shape (1,), got {pred_np.shape}'
        raise ValueError(msg)
    return float(pred_np[0])


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


def _fit_model_for_plan(
    *,
    strategy: TwoPieceRansacAutoBreakStrategy,
    plan: _ObservationPlan,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    cache: dict[tuple[int, ...], _FitCacheEntry],
) -> tuple[
    object | None,
    tuple[float, float, float, float, float, float, float] | None,
    int | None,
]:
    cache_key = _fit_cache_key(plan)
    entry = cache.get(cache_key)
    if entry is None:
        try:
            trend_model = strategy.fit(
                torch.as_tensor(x_obs, dtype=torch.float32),
                torch.as_tensor(y_obs, dtype=torch.float32),
            )
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

    if not bool(cfg.physical_trend.enabled):
        return _build_disabled_result(table, trend)

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
        offset_source = PHYSICAL_OFFSET_SOURCE_GEOMETRY
    else:
        offset_abs_m = _table_offset_abs_m(table, n_traces=n)
        offset_source = PHYSICAL_OFFSET_SOURCE_HEADER

    source_groups_from_geometry = False
    groups: tuple[SourceGroup, ...] = ()
    source_group_geometry = geometry
    if source_group_geometry is None and not use_geometry_offset:
        source_group_geometry = _load_source_group_geometry_from_npz(
            coarse_npz,
            n_traces=n,
        )
    if source_group_geometry is not None:
        groups = build_source_groups(
            source_group_geometry,
            coord_group_tol_m=float(cfg.physical_trend.coord_group_tol_m),
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

    groups_by_id = {int(group.group_id): group for group in groups}
    group_id_by_trace = np.full((n,), -1, dtype=np.int32)
    for group in groups:
        group_id_by_trace[np.asarray(group.trace_indices, dtype=np.int64)] = np.int32(
            group.group_id
        )

    pick_t_sec = _as_vector(
        'table.coarse_pick_t_sec',
        table.coarse_pick_t_sec,
        n_traces=n,
        dtype=np.float32,
    )
    valid_for_fit = _compute_physical_prefilter_mask(
        offset_abs_m=offset_abs_m,
        table=table,
        feasible=feasible,
        cfg=cfg,
    )

    arrays = _allocate_result_arrays(table)
    strategy = _fit_strategy(cfg)
    fit_cache: dict[tuple[int, ...], _FitCacheEntry] = {}
    min_fit_obs = 2 * int(cfg.two_piece_ransac.min_pts)

    for trace_idx in range(n):
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
            target_group_id=int(group_id_by_trace[trace_idx]),
            groups=groups,
            groups_by_id=groups_by_id,
            geometry=geometry,
            offset_abs_m=offset_abs_m,
            valid_for_fit=valid_for_fit,
            cfg=cfg,
            use_neighbor_context=source_groups_from_geometry,
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
            continue

        arrays['physical_model_neighbor_count'][trace_idx] = np.int32(
            plan.neighbor_count
        )
        arrays['physical_prefilter_valid_count'][trace_idx] = np.int32(
            plan.prefilter_valid_count
        )
        arrays['physical_model_segment_id'][trace_idx] = np.int32(plan.segment_id)
        arrays['physical_model_side'][trace_idx] = np.int8(plan.side)

        if int(plan.obs_indices.size) < min_fit_obs:
            _assign_fallback(
                arrays,
                trace_idx,
                failure_reason=PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS,
                table=table,
                feasible=feasible,
                trend=trend,
                merged=merged,
            )
            continue

        obs_indices = np.asarray(plan.obs_indices, dtype=np.int64)
        x_obs = np.asarray(offset_abs_m[obs_indices], dtype=np.float32)
        y_obs = np.asarray(pick_t_sec[obs_indices], dtype=np.float32)
        trend_model, diagnostics, fit_failure_reason = _fit_model_for_plan(
            strategy=strategy,
            plan=plan,
            x_obs=x_obs,
            y_obs=y_obs,
            cache=fit_cache,
        )

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
            continue

        try:
            physical_t_sec = _predict_model_sec(
                trend_model,
                float(offset_abs_m[trace_idx]),
            )
        except (TypeError, ValueError, RuntimeError):
            physical_t_sec = np.nan

        if not np.isfinite(physical_t_sec):
            _assign_fallback(
                arrays,
                trace_idx,
                failure_reason=PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
                table=table,
                feasible=feasible,
                trend=trend,
                merged=merged,
            )
            continue

        center_i = int(np.rint(physical_t_sec / dt))
        center_i = int(np.clip(center_i, 0, n_samples - 1))
        arrays['physical_center_i'][trace_idx] = np.int32(center_i)
        arrays['physical_center_t_sec'][trace_idx] = np.float32(center_i * dt)
        arrays['physical_model_status'][trace_idx] = np.uint8(
            PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT
            if bool(plan.relaxed)
            else PHYSICAL_MODEL_STATUS_TWO_PIECE_OK
        )

        if diagnostics is None:
            diagnostics = _diagnostics_for_plan(
                plan=plan,
                x_obs=x_obs,
                y_obs=y_obs,
                cache=fit_cache,
            )
        if diagnostics is not None:
            (
                break_offset,
                slope_near,
                slope_far,
                velocity_near,
                velocity_far,
                resid_p50,
                resid_p90,
            ) = diagnostics
            arrays['physical_model_break_offset_m'][trace_idx] = np.float32(
                break_offset
            )
            arrays['physical_model_slope_near_s_per_m'][trace_idx] = np.float32(
                slope_near
            )
            arrays['physical_model_slope_far_s_per_m'][trace_idx] = np.float32(
                slope_far
            )
            arrays['physical_model_velocity_near_m_s'][trace_idx] = np.float32(
                velocity_near
            )
            arrays['physical_model_velocity_far_m_s'][trace_idx] = np.float32(
                velocity_far
            )
            arrays['physical_model_resid_p50_ms'][trace_idx] = np.float32(resid_p50)
            arrays['physical_model_resid_p90_ms'][trace_idx] = np.float32(resid_p90)

    return _finalize_result(arrays)
