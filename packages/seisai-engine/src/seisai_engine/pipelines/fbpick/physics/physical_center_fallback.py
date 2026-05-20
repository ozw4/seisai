from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np

from .feasible import FeasibleBandResult
from .merge import MergeResult
from .physical_center_context import PhysicalCenterInputs
from .physical_center_types import (
    PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED,
    PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND,
    PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP,
    PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST,
    PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED,
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND,
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST,
    PhysicalCenterResult,
)
from .pick_table import CoarsePickTable
from .runtime_diagnostics import PhysicalRuntimeDiagnostics
from .trend import TrendResult


@dataclass(frozen=True)
class _PendingTrendFallbackRecord:
    failure_reason: int
    runtime_fit_source: int


@dataclass
class _PendingTrendFallback:
    records: dict[int, _PendingTrendFallbackRecord] = field(default_factory=dict)

    def add(
        self,
        trace_idx: int,
        *,
        failure_reason: int,
        runtime_fit_source: int,
    ) -> None:
        self.records[int(trace_idx)] = _PendingTrendFallbackRecord(
            failure_reason=int(failure_reason),
            runtime_fit_source=int(runtime_fit_source),
        )

    def clear(self) -> None:
        self.records.clear()


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
        'physical_fit_model_type': np.full((n,), '', dtype='<U16'),
        'physical_fit_selected_model': np.full((n,), '', dtype='<U16'),
        'physical_fit_relative_improvement': np.full(
            (n,),
            np.nan,
            dtype=np.float32,
        ),
        'physical_fit_single_line_cost': np.full((n,), np.nan, dtype=np.float32),
        'physical_fit_two_piece_cost': np.full((n,), np.nan, dtype=np.float32),
        'physical_fit_single_line_slope': np.full((n,), np.nan, dtype=np.float32),
        'physical_fit_single_line_t0_sec': np.full((n,), np.nan, dtype=np.float32),
        'physical_fit_two_piece_slope_near': np.full(
            (n,),
            np.nan,
            dtype=np.float32,
        ),
        'physical_fit_two_piece_slope_far': np.full(
            (n,),
            np.nan,
            dtype=np.float32,
        ),
        'physical_fit_two_piece_break_offset_m': np.full(
            (n,),
            np.nan,
            dtype=np.float32,
        ),
    }


def _finalize_result(arrays: dict[str, np.ndarray]) -> PhysicalCenterResult:
    arrays['fine_center_i'][:] = arrays['physical_center_i']
    arrays['fine_center_t_sec'][:] = arrays['physical_center_t_sec']
    return PhysicalCenterResult(**arrays)


def _existing_fallback_trend(
    trend: TrendResult,
    trend_provider: object | None = None,
) -> TrendResult:
    if trend_provider is None:
        return trend
    return trend_provider.get(reason='fallback_existing_trend')


def _use_full_trend_for_fallback_all(
    trend_provider: object | None,
    *,
    n_traces: int,
) -> bool:
    if trend_provider is None:
        return True
    mode = str(getattr(trend_provider, 'fallback_existing_trend_mode', 'full'))
    if mode == 'robust':
        return False
    if mode != 'partial':
        return True

    partial_cfg = getattr(trend_provider, '_partial_cfg', None)
    if partial_cfg is None or not bool(partial_cfg.enabled):
        return True
    too_many = int(n_traces) > int(partial_cfg.max_traces)
    provider_n_traces = getattr(trend_provider, '_n_traces', None)
    total_traces = int(provider_n_traces) if provider_n_traces is not None else 0
    if total_traces > 0:
        too_many = too_many or (
            float(n_traces) / float(total_traces) > float(partial_cfg.max_fraction)
        )
    if not too_many:
        return True
    fallback = str(partial_cfg.fallback_if_too_many)
    if fallback == 'full':
        return True
    if fallback == 'error':
        msg = (
            'partial trend fallback target count exceeds configured '
            f'limits: n_targets={int(n_traces)}'
        )
        raise RuntimeError(msg)
    record_too_many = getattr(
        trend_provider,
        'record_partial_too_many_robust_fallback',
        None,
    )
    if record_too_many is not None:
        record_too_many(int(n_traces))
    return False


def _partial_fallback_center_for_trace(
    trace_idx: int,
    *,
    n_samples: int,
    dt: float,
    trend_provider: object | None,
) -> tuple[bool, tuple[int, np.float32, int] | None]:
    if trend_provider is None:
        return False, None
    get_partial = getattr(trend_provider, 'get_partial', None)
    if get_partial is None:
        return False, None
    partial = get_partial(
        np.asarray([int(trace_idx)], dtype=np.int64),
        reason='fallback_existing_trend',
    )
    if partial is None:
        return False, None

    valid = np.asarray(partial.valid_mask, dtype=np.bool_).reshape(-1)
    center_i = np.asarray(partial.center_i, dtype=np.int64).reshape(-1)
    center_t = np.asarray(partial.center_t_sec, dtype=np.float32).reshape(-1)
    if valid.size != 1 or center_i.size != 1 or center_t.size != 1:
        msg = 'partial trend fallback must return one result per requested trace'
        raise ValueError(msg)
    sample_i = int(center_i[0])
    sample_t = float(center_t[0])
    if (
        bool(valid[0])
        and _is_valid_pick_i(sample_i, n_samples_orig=n_samples)
        and np.isfinite(sample_t)
    ):
        return True, (
            sample_i,
            np.float32(sample_i * dt),
            PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND,
        )
    return True, None


def _fallback_center_without_existing_trend(
    trace_idx: int,
    *,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    merged: MergeResult,
) -> tuple[int, np.float32, int]:
    n_samples = int(table.n_samples_orig)
    dt = float(table.dt_scalar_sec)
    idx = int(trace_idx)
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


def _fallback_center_for_trace(
    trace_idx: int,
    *,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    merged: MergeResult,
    trend_provider: object | None = None,
) -> tuple[int, np.float32, int]:
    n_samples = int(table.n_samples_orig)
    dt = float(table.dt_scalar_sec)
    idx = int(trace_idx)
    partial_attempted, partial_center = _partial_fallback_center_for_trace(
        idx,
        n_samples=n_samples,
        dt=dt,
        trend_provider=trend_provider,
    )
    if partial_center is not None:
        return partial_center

    if not partial_attempted:
        trend = _existing_fallback_trend(trend, trend_provider)

        trend_i = int(np.asarray(trend.trend_center_i, dtype=np.int64)[idx])
        trend_t = float(np.asarray(trend.trend_center_sec, dtype=np.float32)[idx])
        if _is_valid_pick_i(trend_i, n_samples_orig=n_samples) and np.isfinite(
            trend_t,
        ):
            return (
                trend_i,
                np.float32(trend_i * dt),
                PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND,
            )

    return _fallback_center_without_existing_trend(
        idx,
        table=table,
        feasible=feasible,
        merged=merged,
    )


def _should_defer_partial_existing_trend(
    trend_provider: object | None,
) -> bool:
    if trend_provider is None:
        return False
    mode = str(getattr(trend_provider, 'fallback_existing_trend_mode', 'full'))
    if mode != 'partial':
        return False
    if getattr(trend_provider, 'get_partial', None) is None:
        return False
    partial_cfg = getattr(trend_provider, '_partial_cfg', None)
    return partial_cfg is None or bool(partial_cfg.enabled)


def _assign_fallback_values(
    arrays: dict[str, np.ndarray],
    trace_idx: int,
    *,
    center_i: int,
    center_t: np.float32,
    fallback_status: int,
    failure_reason: int,
    runtime_fit_source: int,
) -> None:
    arrays['physical_center_i'][trace_idx] = np.int32(center_i)
    arrays['physical_center_t_sec'][trace_idx] = np.float32(center_t)
    arrays['physical_model_status'][trace_idx] = np.uint8(fallback_status)
    arrays['physical_model_failure_reason'][trace_idx] = np.uint8(failure_reason)
    source = int(runtime_fit_source)
    if fallback_status == PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST:
        source = PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST
    arrays['physical_runtime_fit_source'][trace_idx] = np.uint8(source)


def _assign_deferred_existing_trend_placeholder(
    arrays: dict[str, np.ndarray],
    trace_idx: int,
    *,
    failure_reason: int,
    runtime_fit_source: int,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    merged: MergeResult,
) -> None:
    center_i, center_t, _fallback_status = _fallback_center_without_existing_trend(
        trace_idx,
        table=table,
        feasible=feasible,
        merged=merged,
    )
    _assign_fallback_values(
        arrays,
        trace_idx,
        center_i=center_i,
        center_t=center_t,
        fallback_status=PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND,
        failure_reason=failure_reason,
        runtime_fit_source=runtime_fit_source,
    )


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
    trend_provider: object | None = None,
    pending_trend_fallback: _PendingTrendFallback | None = None,
) -> None:
    if (
        pending_trend_fallback is not None
        and _should_defer_partial_existing_trend(trend_provider)
    ):
        pending_trend_fallback.add(
            trace_idx,
            failure_reason=failure_reason,
            runtime_fit_source=runtime_fit_source,
        )
        _assign_deferred_existing_trend_placeholder(
            arrays,
            trace_idx,
            failure_reason=failure_reason,
            runtime_fit_source=runtime_fit_source,
            table=table,
            feasible=feasible,
            merged=merged,
        )
        return

    center_i, center_t, fallback_status = _fallback_center_for_trace(
        trace_idx,
        table=table,
        feasible=feasible,
        trend=trend,
        trend_provider=trend_provider,
        merged=merged,
    )
    _assign_fallback_values(
        arrays,
        trace_idx,
        center_i=center_i,
        center_t=center_t,
        fallback_status=fallback_status,
        failure_reason=failure_reason,
        runtime_fit_source=runtime_fit_source,
    )


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
        PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST,
    )
    arrays['physical_model_failure_reason'][trace_idx] = np.uint8(failure_reason)
    arrays['physical_runtime_fit_source'][trace_idx] = np.uint8(
        PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST,
    )


def _partial_fallback_lookup(
    partial: object,
) -> dict[int, tuple[bool, int, float]]:
    indices = np.asarray(partial.indices, dtype=np.int64).reshape(-1)
    center_i = np.asarray(partial.center_i, dtype=np.int64).reshape(-1)
    center_t = np.asarray(
        partial.center_t_sec,
        dtype=np.float32,
    ).reshape(-1)
    valid = np.asarray(partial.valid_mask, dtype=np.bool_).reshape(-1)
    if not (indices.shape == center_i.shape == center_t.shape == valid.shape):
        msg = 'partial trend fallback arrays must have matching 1D shapes'
        raise ValueError(msg)
    return {
        int(trace_idx): (bool(valid[pos]), int(center_i[pos]), float(center_t[pos]))
        for pos, trace_idx in enumerate(indices.tolist())
    }


def _apply_pending_trend_fallback(
    arrays: dict[str, np.ndarray],
    *,
    pending_trend_fallback: _PendingTrendFallback | None,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    merged: MergeResult,
    trend_provider: object | None,
) -> None:
    if pending_trend_fallback is None or not pending_trend_fallback.records:
        return

    pending_items = [
        (trace_idx, record)
        for trace_idx, record in pending_trend_fallback.records.items()
        if int(arrays['physical_model_status'][int(trace_idx)])
        == PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND
    ]
    pending_trend_fallback.clear()
    if not pending_items:
        return

    indices = np.asarray(
        [trace_idx for trace_idx, _record in pending_items],
        dtype=np.int64,
    )
    get_partial = (
        None if trend_provider is None else getattr(trend_provider, 'get_partial', None)
    )
    partial = (
        get_partial(indices, reason='fallback_existing_trend')
        if get_partial is not None
        else None
    )
    if partial is None:
        trend = _existing_fallback_trend(trend, trend_provider)
        for trace_idx, record in pending_items:
            center_i, center_t, fallback_status = _fallback_center_for_trace(
                trace_idx,
                table=table,
                feasible=feasible,
                trend=trend,
                trend_provider=None,
                merged=merged,
            )
            _assign_fallback_values(
                arrays,
                trace_idx,
                center_i=center_i,
                center_t=center_t,
                fallback_status=fallback_status,
                failure_reason=record.failure_reason,
                runtime_fit_source=record.runtime_fit_source,
            )
        return

    partial_by_trace = _partial_fallback_lookup(partial)
    n_samples = int(table.n_samples_orig)
    dt = float(table.dt_scalar_sec)
    for trace_idx, record in pending_items:
        partial_value = partial_by_trace.get(int(trace_idx))
        if partial_value is not None:
            valid, center_i, center_t_sec = partial_value
            if (
                valid
                and _is_valid_pick_i(center_i, n_samples_orig=n_samples)
                and np.isfinite(center_t_sec)
            ):
                _assign_fallback_values(
                    arrays,
                    trace_idx,
                    center_i=center_i,
                    center_t=np.float32(center_i * dt),
                    fallback_status=PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND,
                    failure_reason=record.failure_reason,
                    runtime_fit_source=record.runtime_fit_source,
                )
                continue

        center_i, center_t, fallback_status = _fallback_center_without_existing_trend(
            trace_idx,
            table=table,
            feasible=feasible,
            merged=merged,
        )
        _assign_fallback_values(
            arrays,
            trace_idx,
            center_i=center_i,
            center_t=center_t,
            fallback_status=fallback_status,
            failure_reason=record.failure_reason,
            runtime_fit_source=record.runtime_fit_source,
        )


def _finalize_result_with_pending_trend_fallback(
    arrays: dict[str, np.ndarray],
    *,
    pending_trend_fallback: _PendingTrendFallback | None,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    merged: MergeResult,
    trend_provider: object | None,
) -> PhysicalCenterResult:
    _apply_pending_trend_fallback(
        arrays,
        pending_trend_fallback=pending_trend_fallback,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        trend_provider=trend_provider,
    )
    return _finalize_result(arrays)


def _assign_fallback_all(
    *,
    failure_reason: int,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    merged: MergeResult,
    trend_provider: object | None = None,
) -> PhysicalCenterResult:
    arrays = _allocate_result_arrays(table)
    n = int(table.n_traces)
    n_samples = int(table.n_samples_orig)
    dt = float(table.dt_scalar_sec)
    use_full_trend = _use_full_trend_for_fallback_all(
        trend_provider,
        n_traces=n,
    )
    if use_full_trend:
        trend = _existing_fallback_trend(trend, trend_provider)
    robust_i = _as_vector(
        'merged.robust_pick_i',
        merged.robust_pick_i,
        n_traces=n,
        dtype=np.int64,
    )
    lo_sec = _as_vector(
        'feasible.feasible_lo_sec',
        feasible.feasible_lo_sec,
        n_traces=n,
        dtype=np.float32,
    )
    hi_sec = _as_vector(
        'feasible.feasible_hi_sec',
        feasible.feasible_hi_sec,
        n_traces=n,
        dtype=np.float32,
    )

    center_i = np.clip(robust_i, 0, n_samples - 1).astype(np.int32)
    status = np.full(
        (n,),
        np.uint8(PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST),
        dtype=np.uint8,
    )
    fit_source = np.full(
        (n,),
        np.uint8(PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST),
        dtype=np.uint8,
    )

    valid_band = np.isfinite(lo_sec) & np.isfinite(hi_sec) & (lo_sec <= hi_sec)
    if bool(np.any(valid_band)):
        lo_i = np.zeros((n,), dtype=np.int64)
        hi_i = np.full((n,), n_samples - 1, dtype=np.int64)
        with np.errstate(invalid='ignore'):
            lo_i[valid_band] = np.clip(
                np.ceil(lo_sec[valid_band] / np.float32(dt)),
                0,
                n_samples - 1,
            ).astype(np.int64)
            hi_i[valid_band] = np.clip(
                np.floor(hi_sec[valid_band] / np.float32(dt)),
                0,
                n_samples - 1,
            ).astype(np.int64)
        valid_band &= lo_i <= hi_i
        if bool(np.any(valid_band)):
            clipped_i = np.clip(robust_i, lo_i, hi_i).astype(np.int32)
            center_i[valid_band] = clipped_i[valid_band]
            status[valid_band] = np.uint8(
                PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP,
            )
            fit_source[valid_band] = np.uint8(
                PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND,
            )

    if use_full_trend:
        trend_i = _as_vector(
            'trend.trend_center_i',
            trend.trend_center_i,
            n_traces=n,
            dtype=np.int64,
        )
        trend_t = _as_vector(
            'trend.trend_center_sec',
            trend.trend_center_sec,
            n_traces=n,
            dtype=np.float32,
        )
        valid_trend = (trend_i >= 0) & (trend_i < n_samples) & np.isfinite(trend_t)
        center_i[valid_trend] = trend_i[valid_trend].astype(np.int32)
        status[valid_trend] = np.uint8(
            PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND,
        )
        fit_source[valid_trend] = np.uint8(
            PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND,
        )

    arrays['physical_center_i'][:] = center_i
    arrays['physical_center_t_sec'][:] = center_i.astype(np.float64) * dt
    arrays['physical_model_status'][:] = status
    arrays['physical_model_failure_reason'][:] = np.uint8(failure_reason)
    arrays['physical_runtime_fit_source'][:] = fit_source
    return _finalize_result(arrays)


def _assign_robust_fallback_all(
    *,
    failure_reason: int,
    table: CoarsePickTable,
    merged: MergeResult,
) -> PhysicalCenterResult:
    arrays = _allocate_result_arrays(table)
    n = int(table.n_traces)
    n_samples = int(table.n_samples_orig)
    dt = float(table.dt_scalar_sec)
    robust_i = _as_vector(
        'merged.robust_pick_i',
        merged.robust_pick_i,
        n_traces=n,
        dtype=np.int64,
    )
    center_i = np.clip(robust_i, 0, n_samples - 1).astype(np.int32)
    arrays['physical_center_i'][:] = center_i
    arrays['physical_center_t_sec'][:] = center_i.astype(np.float64) * dt
    arrays['physical_model_status'][:] = np.uint8(
        PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST,
    )
    arrays['physical_model_failure_reason'][:] = np.uint8(failure_reason)
    arrays['physical_runtime_fit_source'][:] = np.uint8(
        PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST,
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
        PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED,
    )
    arrays['physical_model_failure_reason'][:] = np.uint8(
        PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED,
    )
    arrays['physical_runtime_fit_source'][:] = np.uint8(
        PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND,
    )
    return _finalize_result(arrays)


def _assign_configured_fallback_all(
    *,
    fallback_mode: str,
    failure_reason: int,
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    merged: MergeResult,
    trend_provider: object | None = None,
) -> PhysicalCenterResult:
    if str(fallback_mode) == 'robust':
        return _assign_robust_fallback_all(
            failure_reason=failure_reason,
            table=table,
            merged=merged,
        )
    return _assign_fallback_all(
        failure_reason=failure_reason,
        table=table,
        feasible=feasible,
        trend=trend,
        trend_provider=trend_provider,
        merged=merged,
    )


def _emit_fallback_all_and_done(
    *,
    status: str,
    reason: str,
    fallback_mode: str,
    failure_reason: int,
    inputs: PhysicalCenterInputs,
    reporter: object,
    context: Mapping[str, object],
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None,
) -> PhysicalCenterResult:
    table = inputs.table
    n = int(table.n_traces)
    reporter.emit(
        'physical-center.stage_start',
        **context,
        stage='fallback_assign_all',
        reason=reason,
        fallback=fallback_mode,
    )
    stage_start = time.perf_counter()
    with (
        runtime_diagnostics.time_block('fallback_sec')
        if runtime_diagnostics is not None
        else nullcontext()
    ):
        result = _assign_configured_fallback_all(
            fallback_mode=fallback_mode,
            failure_reason=failure_reason,
            table=table,
            feasible=inputs.feasible,
            trend=inputs.trend,
            trend_provider=inputs.trend_provider,
            merged=inputs.merged,
        )
    reporter.emit(
        'physical-center.stage_done',
        **context,
        stage='fallback_assign_all',
        reason=reason,
        fallback=fallback_mode,
        elapsed=time.perf_counter() - stage_start,
        n_traces=n,
    )
    reporter.emit(
        'physical-center.done',
        **context,
        status=status,
        n_traces=n,
    )
    return result
