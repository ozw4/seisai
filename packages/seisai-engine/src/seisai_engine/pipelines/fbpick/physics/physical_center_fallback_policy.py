"""Explicit physical-center fallback policy for fine-window centers."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import TYPE_CHECKING

import numpy as np

from seisai_engine.pipelines.fbpick.common import (
    FINE_WINDOW_REJECT_CENTER_OUTSIDE_PREFILTER_BAND,
)

from .physical_center_types import (
    PHYSICAL_MODEL_FAILURE_NONE,
    PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
    PHYSICAL_MODEL_STATUS_COARSE_IN_BAND_FALLBACK,
    PHYSICAL_MODEL_STATUS_LABELS,
    PHYSICAL_MODEL_STATUS_NEIGHBOR_PHYSICAL_FIT_REUSE,
    PHYSICAL_MODEL_STATUS_REJECT_PHYSICS_COARSE_OUTSIDE_BAND,
    PHYSICAL_MODEL_STATUS_REJECT_PHYSICS_NO_NEIGHBOR_FIT,
    PHYSICAL_MODEL_STATUS_REJECT_PHYSICS_NO_VALID_WINDOW,
    PHYSICAL_MODEL_STATUS_SINGLE_LINE_OK,
    PHYSICAL_MODEL_STATUS_TWO_PIECE_OK,
    PHYSICAL_RUNTIME_FIT_SOURCE_COARSE_IN_BAND_FALLBACK,
    PHYSICAL_RUNTIME_FIT_SOURCE_NEIGHBOR_PHYSICAL_FIT_REUSE,
)
from .window_constraint import evaluate_fine_window_constraint

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .config import PhysicsLiteConfig
    from .physical_center_types import PhysicalCenterResult
    from .pick_table import CoarsePickTable
    from .window_constraint import FineWindowConstraintResult

_STATUS_BY_LABEL = {
    label: code for code, label in PHYSICAL_MODEL_STATUS_LABELS.items()
}
_SELF_FIT_STATUS_CODES = {
    PHYSICAL_MODEL_STATUS_TWO_PIECE_OK,
    PHYSICAL_MODEL_STATUS_SINGLE_LINE_OK,
}


def _copy_result_arrays(physical: PhysicalCenterResult) -> dict[str, np.ndarray]:
    return {
        item.name: np.asarray(getattr(physical, item.name)).copy()
        for item in fields(physical)
    }


def _offset_abs_m(
    *,
    coarse_npz: Mapping[str, np.ndarray],
    table: CoarsePickTable,
    cfg: PhysicsLiteConfig,
) -> np.ndarray:
    if (
        bool(cfg.physical_trend.use_geometry_offset)
        and 'offset_abs_geom_m' in coarse_npz
    ):
        arr = np.asarray(coarse_npz['offset_abs_geom_m'], dtype=np.float32)
        if arr.ndim == 1 and int(arr.shape[0]) == int(table.n_traces):
            return np.abs(arr).astype(np.float32, copy=False)
    return np.abs(np.asarray(table.offset_m, dtype=np.float32))


def _source_xy_distance(
    coarse_npz: Mapping[str, np.ndarray],
    *,
    n_traces: int,
) -> np.ndarray | None:
    if 'source_x_m' not in coarse_npz or 'source_y_m' not in coarse_npz:
        return None
    x = np.asarray(coarse_npz['source_x_m'], dtype=np.float64)
    y = np.asarray(coarse_npz['source_y_m'], dtype=np.float64)
    if x.shape != (n_traces,) or y.shape != (n_traces,):
        return None
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None
    return np.stack([x, y], axis=1)


def _evaluate_centers(
    center_i: np.ndarray,
    *,
    table: CoarsePickTable,
    cfg: PhysicsLiteConfig,
) -> FineWindowConstraintResult:
    return evaluate_fine_window_constraint(
        offsets_m=table.offset_m,
        dt_sec=float(table.dt_scalar_sec),
        n_samples_orig=int(table.n_samples_orig),
        fine_center_i=np.asarray(center_i, dtype=np.int32),
        physical_prefilter=cfg.physical_prefilter,
        constraint=cfg.physical_runtime.fine_window_constraint,
    )


def _predict_single_line(
    arrays: dict[str, np.ndarray],
    source_idx: int,
    target_offset_m: float,
) -> float:
    slope = float(arrays['physical_fit_single_line_slope'][source_idx])
    t0_sec = float(arrays['physical_fit_single_line_t0_sec'][source_idx])
    if not (np.isfinite(slope) and np.isfinite(t0_sec)):
        return float('nan')
    return slope * float(target_offset_m) + t0_sec


def _predict_two_piece(
    arrays: dict[str, np.ndarray],
    *,
    source_idx: int,
    source_offset_m: float,
    target_offset_m: float,
) -> float:
    slope_near = float(arrays['physical_fit_two_piece_slope_near'][source_idx])
    slope_far = float(arrays['physical_fit_two_piece_slope_far'][source_idx])
    break_m = float(arrays['physical_fit_two_piece_break_offset_m'][source_idx])
    if not (
        np.isfinite(slope_near)
        and np.isfinite(slope_far)
        and np.isfinite(break_m)
    ):
        slope_near = float(arrays['physical_model_slope_near_s_per_m'][source_idx])
        slope_far = float(arrays['physical_model_slope_far_s_per_m'][source_idx])
        break_m = float(arrays['physical_model_break_offset_m'][source_idx])
    source_t_sec = float(arrays['physical_center_t_sec'][source_idx])
    if not (
        np.isfinite(slope_near)
        and np.isfinite(slope_far)
        and np.isfinite(break_m)
        and np.isfinite(source_t_sec)
    ):
        return float('nan')
    if float(source_offset_m) <= break_m:
        t0_sec = source_t_sec - slope_near * float(source_offset_m)
    else:
        far_intercept = source_t_sec - slope_far * float(source_offset_m)
        t_break_sec = far_intercept + slope_far * break_m
        t0_sec = t_break_sec - slope_near * break_m
    if float(target_offset_m) <= break_m:
        return t0_sec + slope_near * float(target_offset_m)
    t_break_sec = t0_sec + slope_near * break_m
    return t_break_sec + slope_far * (float(target_offset_m) - break_m)


def _predict_from_candidate(
    arrays: dict[str, np.ndarray],
    *,
    source_idx: int,
    target_offset_m: float,
    offsets_abs_m: np.ndarray,
    dt_sec: float,
) -> int | None:
    status = int(arrays['physical_model_status'][source_idx])
    if status == PHYSICAL_MODEL_STATUS_SINGLE_LINE_OK:
        pred_sec = _predict_single_line(arrays, source_idx, target_offset_m)
    elif status == PHYSICAL_MODEL_STATUS_TWO_PIECE_OK:
        pred_sec = _predict_two_piece(
            arrays,
            source_idx=source_idx,
            source_offset_m=float(offsets_abs_m[source_idx]),
            target_offset_m=target_offset_m,
        )
    else:
        return None
    if not np.isfinite(pred_sec):
        return None
    return int(np.rint(pred_sec / float(dt_sec)))


def _candidate_order(
    target_idx: int,
    candidate_indices: np.ndarray,
    *,
    source_xy: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if source_xy is not None:
        delta = source_xy[candidate_indices] - source_xy[int(target_idx)]
        distance = np.sqrt(np.sum(delta * delta, axis=1))
        order = np.lexsort((candidate_indices, distance))
        return candidate_indices[order], distance[order].astype(np.float32)
    distance = np.abs(candidate_indices.astype(np.int64) - int(target_idx))
    order = np.lexsort((candidate_indices, distance))
    return candidate_indices[order], distance[order].astype(np.float32)


def apply_physics_fallback_policy(  # noqa: C901, PLR0915
    *,
    physical: PhysicalCenterResult,
    initial_window_constraint: FineWindowConstraintResult,
    coarse_npz: Mapping[str, np.ndarray],
    table: CoarsePickTable,
    cfg: PhysicsLiteConfig,
) -> tuple[PhysicalCenterResult, FineWindowConstraintResult, dict[str, np.ndarray]]:
    """Apply self-fit, neighbor-fit reuse, coarse-in-band, then reject policy."""
    n = int(table.n_traces)
    center_source = np.full((n,), 'self_physical_fit', dtype='<U32')
    fallback_source = np.full((n,), '', dtype='<U32')
    neighbor_source_index = np.full((n,), -1, dtype=np.int32)
    neighbor_source_distance = np.full((n,), np.nan, dtype=np.float32)
    coarse_fallback_mask = np.zeros((n,), dtype=np.bool_)
    reject_mask = np.zeros((n,), dtype=np.bool_)
    reject_reason = np.full((n,), '', dtype='<U40')

    if not bool(cfg.physical_runtime.fallback_policy.enabled):
        diagnostics = {
            'physical_center_source': center_source,
            'physical_fallback_source': fallback_source,
            'physical_neighbor_source_index': neighbor_source_index,
            'physical_neighbor_source_distance': neighbor_source_distance,
            'coarse_in_band_fallback_mask': coarse_fallback_mask,
            'reject_physics_mask': reject_mask,
            'reject_physics_reason': reject_reason,
        }
        return physical, initial_window_constraint, diagnostics

    arrays = _copy_result_arrays(physical)
    status = np.asarray(arrays['physical_model_status'], dtype=np.uint8)
    failure = np.asarray(arrays['physical_model_failure_reason'], dtype=np.uint8)
    initial_valid = np.asarray(
        initial_window_constraint.fine_window_valid_mask,
        dtype=np.bool_,
    )

    candidate_labels = (
        cfg.physical_runtime.neighbor_physical_fit_reuse.candidate_statuses
    )
    candidate_statuses = {
        int(_STATUS_BY_LABEL[label])
        for label in candidate_labels
        if label in _STATUS_BY_LABEL
    } & _SELF_FIT_STATUS_CODES
    self_ok = (
        np.isin(status, np.fromiter(candidate_statuses, dtype=np.uint8))
        & (failure == np.uint8(PHYSICAL_MODEL_FAILURE_NONE))
        & initial_valid
    )
    needs_policy = ~self_ok
    center_source[needs_policy] = 'pending_fallback_policy'
    fallback_source[needs_policy] = 'pending_fallback_policy'

    offsets_abs = _offset_abs_m(coarse_npz=coarse_npz, table=table, cfg=cfg)
    source_xy = _source_xy_distance(coarse_npz, n_traces=n)
    candidate_indices = np.flatnonzero(self_ok).astype(np.int64)
    dt_sec = float(table.dt_scalar_sec)

    reuse_cfg = cfg.physical_runtime.neighbor_physical_fit_reuse
    if bool(reuse_cfg.enabled) and candidate_indices.size > 0:
        for target_idx in np.flatnonzero(needs_policy).astype(np.int64).tolist():
            ordered_candidates, ordered_distances = _candidate_order(
                int(target_idx),
                candidate_indices,
                source_xy=source_xy,
            )
            for candidate_idx, distance in zip(
                ordered_candidates.tolist(),
                ordered_distances.tolist(),
                strict=True,
            ):
                if (
                    source_xy is not None
                    and reuse_cfg.max_source_xy_distance_m is not None
                    and float(distance) > float(reuse_cfg.max_source_xy_distance_m)
                ):
                    continue
                if (
                    source_xy is None
                    and reuse_cfg.max_trace_distance is not None
                    and int(distance) > int(reuse_cfg.max_trace_distance)
                ):
                    continue
                center_i = _predict_from_candidate(
                    arrays,
                    source_idx=int(candidate_idx),
                    target_offset_m=float(offsets_abs[int(target_idx)]),
                    offsets_abs_m=offsets_abs,
                    dt_sec=dt_sec,
                )
                if center_i is None or not 0 <= center_i < int(table.n_samples_orig):
                    continue
                trial_centers = np.asarray(
                    arrays['fine_center_i'],
                    dtype=np.int32,
                ).copy()
                trial_centers[int(target_idx)] = np.int32(center_i)
                trial = _evaluate_centers(trial_centers, table=table, cfg=cfg)
                if not bool(trial.fine_window_valid_mask[int(target_idx)]):
                    continue
                arrays['physical_center_i'][int(target_idx)] = np.int32(center_i)
                arrays['physical_center_t_sec'][int(target_idx)] = np.float32(
                    center_i * float(table.dt_scalar_sec)
                )
                arrays['fine_center_i'][int(target_idx)] = np.int32(center_i)
                arrays['fine_center_t_sec'][int(target_idx)] = np.float32(
                    center_i * float(table.dt_scalar_sec)
                )
                arrays['physical_model_status'][int(target_idx)] = np.uint8(
                    PHYSICAL_MODEL_STATUS_NEIGHBOR_PHYSICAL_FIT_REUSE
                )
                arrays['physical_model_failure_reason'][int(target_idx)] = np.uint8(
                    PHYSICAL_MODEL_FAILURE_NONE
                )
                arrays['physical_runtime_fit_source'][int(target_idx)] = np.uint8(
                    PHYSICAL_RUNTIME_FIT_SOURCE_NEIGHBOR_PHYSICAL_FIT_REUSE
                )
                center_source[int(target_idx)] = 'neighbor_physical_fit_reuse'
                fallback_source[int(target_idx)] = 'neighbor_physical_fit_reuse'
                neighbor_source_index[int(target_idx)] = np.int32(candidate_idx)
                neighbor_source_distance[int(target_idx)] = np.float32(distance)
                needs_policy[int(target_idx)] = False
                break

    remaining = np.flatnonzero(needs_policy).astype(np.int64)
    if remaining.size > 0:
        coarse_centers = np.asarray(table.coarse_pick_i, dtype=np.int32)
        coarse_eval = _evaluate_centers(coarse_centers, table=table, cfg=cfg)
        for target_idx in remaining.tolist():
            idx = int(target_idx)
            if bool(coarse_eval.fine_window_valid_mask[idx]):
                center_i = int(coarse_centers[idx])
                arrays['physical_center_i'][idx] = np.int32(center_i)
                arrays['physical_center_t_sec'][idx] = np.float32(
                    center_i * float(table.dt_scalar_sec)
                )
                arrays['fine_center_i'][idx] = np.int32(center_i)
                arrays['fine_center_t_sec'][idx] = np.float32(
                    center_i * float(table.dt_scalar_sec)
                )
                arrays['physical_model_status'][idx] = np.uint8(
                    PHYSICAL_MODEL_STATUS_COARSE_IN_BAND_FALLBACK
                )
                arrays['physical_model_failure_reason'][idx] = np.uint8(
                    PHYSICAL_MODEL_FAILURE_NONE
                )
                arrays['physical_runtime_fit_source'][idx] = np.uint8(
                    PHYSICAL_RUNTIME_FIT_SOURCE_COARSE_IN_BAND_FALLBACK
                )
                center_source[idx] = 'coarse_in_band_fallback'
                fallback_source[idx] = 'coarse_in_band_fallback'
                coarse_fallback_mask[idx] = True
                needs_policy[idx] = False
                continue

            reason = int(coarse_eval.fine_window_reject_reason[idx])
            reject_status = (
                PHYSICAL_MODEL_STATUS_REJECT_PHYSICS_COARSE_OUTSIDE_BAND
                if reason == FINE_WINDOW_REJECT_CENTER_OUTSIDE_PREFILTER_BAND
                else PHYSICAL_MODEL_STATUS_REJECT_PHYSICS_NO_VALID_WINDOW
            )
            if bool(reuse_cfg.enabled) and candidate_indices.size == 0:
                reject_status = PHYSICAL_MODEL_STATUS_REJECT_PHYSICS_NO_NEIGHBOR_FIT
            arrays['physical_model_status'][idx] = np.uint8(reject_status)
            arrays['physical_model_failure_reason'][idx] = np.uint8(
                PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID
            )
            center_source[idx] = 'reject'
            fallback_source[idx] = 'reject'
            reject_mask[idx] = True
            reject_reason[idx] = PHYSICAL_MODEL_STATUS_LABELS[int(reject_status)]

    final_physical = replace(physical, **arrays)
    final_window = _evaluate_centers(
        np.asarray(final_physical.fine_center_i, dtype=np.int32),
        table=table,
        cfg=cfg,
    )
    final_window_valid = np.asarray(final_window.fine_window_valid_mask, dtype=np.bool_)
    rejected = np.isin(
        np.asarray(final_physical.physical_model_status, dtype=np.uint8),
        np.asarray(
            [
                PHYSICAL_MODEL_STATUS_REJECT_PHYSICS_NO_VALID_WINDOW,
                PHYSICAL_MODEL_STATUS_REJECT_PHYSICS_COARSE_OUTSIDE_BAND,
                PHYSICAL_MODEL_STATUS_REJECT_PHYSICS_NO_NEIGHBOR_FIT,
            ],
            dtype=np.uint8,
        ),
    )
    final_window_valid[rejected] = False
    final_window = replace(final_window, fine_window_valid_mask=final_window_valid)
    reject_mask[rejected] = True
    reject_reason[(rejected) & (reject_reason == '')] = 'reject_physics_no_valid_window'

    diagnostics = {
        'physical_center_source': center_source,
        'physical_fallback_source': fallback_source,
        'physical_neighbor_source_index': neighbor_source_index,
        'physical_neighbor_source_distance': neighbor_source_distance,
        'coarse_in_band_fallback_mask': coarse_fallback_mask,
        'reject_physics_mask': reject_mask,
        'reject_physics_reason': reject_reason,
    }
    return final_physical, final_window, diagnostics
