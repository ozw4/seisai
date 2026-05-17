"""Physical-center model prediction and assignment helpers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .physical_center_fallback import _assign_fallback
from .physical_center_fit import (
    _fit_cache_key,
    _FitCacheEntry,
    _model_diagnostics,
    _tensor_to_numpy,
)
from .physical_center_observation import _ObservationPlan
from .physical_center_types import (
    PHYSICAL_MODEL_FAILURE_NONE,
    PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
    PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT,
    PHYSICAL_MODEL_STATUS_TWO_PIECE_OK,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .feasible import FeasibleBandResult
    from .merge import MergeResult
    from .physical_center_fallback import _PendingTrendFallback
    from .pick_table import CoarsePickTable
    from .runtime_diagnostics import PhysicalRuntimeDiagnostics
    from .trend import TrendResult


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


def _predict_model_sec(trend_model: object, offset_m: float) -> float:
    pred = trend_model.predict(torch.tensor([float(offset_m)], dtype=torch.float32))
    pred_np = _tensor_to_numpy(pred).astype(np.float64, copy=False)
    if pred_np.shape != (1,):
        msg = f'trend model prediction must have shape (1,), got {pred_np.shape}'
        raise ValueError(msg)
    return float(pred_np[0])


def _predict_model_array_sec(trend_model: object, offset_m: np.ndarray) -> np.ndarray:
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


def _assign_model_prediction_batch(  # noqa: PLR0913
    arrays: dict[str, np.ndarray],
    trace_indices: np.ndarray,
    *,
    trend_model: object,
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


def _assign_model_prediction(  # noqa: PLR0913
    arrays: dict[str, np.ndarray],
    trace_idx: int,
    *,
    trend_model: object,
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


def _assign_prepared_model_prediction_batch(  # noqa: PLR0913
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
