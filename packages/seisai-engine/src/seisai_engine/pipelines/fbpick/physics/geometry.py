from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from seisai_engine.pipelines.fbpick.common import COARSE_GEOMETRY_OPTIONAL_KEYS

__all__ = [
    "CoarseGeometry",
    "SignedOffsetResult",
    "SourceGroup",
    "build_source_groups",
    "estimate_signed_offset_side",
    "load_coarse_geometry_from_npz",
    "select_nearest_source_groups",
    "split_offset_gap_segments",
]


@dataclass(frozen=True)
class CoarseGeometry:
    source_x_m: np.ndarray
    source_y_m: np.ndarray
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    offset_abs_geom_m: np.ndarray
    geometry_valid_mask: np.ndarray


@dataclass(frozen=True)
class SourceGroup:
    group_id: int
    source_key_x: int
    source_key_y: int
    source_x_m: float
    source_y_m: float
    trace_indices: np.ndarray


@dataclass(frozen=True)
class SignedOffsetResult:
    signed_offset_m: np.ndarray
    side: np.ndarray
    reliable: bool


def _coerce_n_traces(n_traces: int) -> int:
    n = int(n_traces)
    if n < 0:
        msg = "n_traces must be non-negative"
        raise ValueError(msg)
    return n


def _coerce_float32_vector(name: str, value, *, n_traces: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1 or int(arr.shape[0]) != int(n_traces):
        msg = f"{name} must be 1D with length n_traces"
        raise ValueError(msg)
    return arr


def _coerce_bool_vector(name: str, value, *, n_traces: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.bool_)
    if arr.ndim != 1 or int(arr.shape[0]) != int(n_traces):
        msg = f"{name} must be 1D with length n_traces"
        raise ValueError(msg)
    return arr


def _validate_positive_finite(name: str, value: float) -> float:
    out = float(value)
    if (not math.isfinite(out)) or out <= 0.0:
        msg = f"{name} must be finite and > 0"
        raise ValueError(msg)
    return out


def _validate_nonnegative_finite(name: str, value: float) -> float:
    out = float(value)
    if (not math.isfinite(out)) or out < 0.0:
        msg = f"{name} must be finite and >= 0"
        raise ValueError(msg)
    return out


def _validate_gap_cfg(
    *,
    gap_ratio: float,
    min_gap_m: float | None,
) -> tuple[float, float | None]:
    ratio = float(gap_ratio)
    if (not math.isfinite(ratio)) or ratio <= 1.0:
        msg = "gap_ratio must be > 1.0"
        raise ValueError(msg)
    if min_gap_m is None:
        return ratio, None
    min_gap = float(min_gap_m)
    if (not math.isfinite(min_gap)) or min_gap <= 0.0:
        msg = "min_gap_m must be None or > 0"
        raise ValueError(msg)
    return ratio, min_gap


def _validate_geometry_finite_where_valid(geometry: CoarseGeometry) -> None:
    valid = np.asarray(geometry.geometry_valid_mask, dtype=np.bool_)
    if not np.any(valid):
        return
    for key in (
        "source_x_m",
        "source_y_m",
        "receiver_x_m",
        "receiver_y_m",
        "offset_abs_geom_m",
    ):
        arr = np.asarray(getattr(geometry, key))
        if not np.all(np.isfinite(arr[valid])):
            msg = f"{key} must be finite where geometry_valid_mask is True"
            raise ValueError(msg)


def load_coarse_geometry_from_npz(
    coarse_npz: Mapping[str, np.ndarray],
    *,
    n_traces: int,
) -> CoarseGeometry | None:
    n = _coerce_n_traces(n_traces)
    present = [key for key in COARSE_GEOMETRY_OPTIONAL_KEYS if key in coarse_npz]
    if not present:
        return None
    missing = [key for key in COARSE_GEOMETRY_OPTIONAL_KEYS if key not in coarse_npz]
    if missing:
        msg = f"coarse npz missing optional geometry keys: {missing}"
        raise KeyError(msg)

    geometry = CoarseGeometry(
        source_x_m=_coerce_float32_vector(
            "source_x_m",
            coarse_npz["source_x_m"],
            n_traces=n,
        ),
        source_y_m=_coerce_float32_vector(
            "source_y_m",
            coarse_npz["source_y_m"],
            n_traces=n,
        ),
        receiver_x_m=_coerce_float32_vector(
            "receiver_x_m",
            coarse_npz["receiver_x_m"],
            n_traces=n,
        ),
        receiver_y_m=_coerce_float32_vector(
            "receiver_y_m",
            coarse_npz["receiver_y_m"],
            n_traces=n,
        ),
        offset_abs_geom_m=_coerce_float32_vector(
            "offset_abs_geom_m",
            coarse_npz["offset_abs_geom_m"],
            n_traces=n,
        ),
        geometry_valid_mask=_coerce_bool_vector(
            "geometry_valid_mask",
            coarse_npz["geometry_valid_mask"],
            n_traces=n,
        ),
    )
    _validate_geometry_finite_where_valid(geometry)
    return geometry


def _source_groupable_mask(geometry: CoarseGeometry) -> np.ndarray:
    return (
        np.asarray(geometry.geometry_valid_mask, dtype=np.bool_)
        & np.isfinite(np.asarray(geometry.source_x_m))
        & np.isfinite(np.asarray(geometry.source_y_m))
    )


def _source_key(
    source_x_m: float, source_y_m: float, *, coord_group_tol_m: float
) -> tuple[int, int]:
    return (
        int(round(float(source_x_m) / coord_group_tol_m)),
        int(round(float(source_y_m) / coord_group_tol_m)),
    )


def build_source_groups(
    geometry: CoarseGeometry,
    *,
    coord_group_tol_m: float,
) -> tuple[SourceGroup, ...]:
    tol = _validate_positive_finite("coord_group_tol_m", coord_group_tol_m)
    valid = _source_groupable_mask(geometry)
    grouped: dict[tuple[int, int], list[int]] = {}

    for trace_idx in np.flatnonzero(valid):
        key = _source_key(
            float(geometry.source_x_m[trace_idx]),
            float(geometry.source_y_m[trace_idx]),
            coord_group_tol_m=tol,
        )
        grouped.setdefault(key, []).append(int(trace_idx))

    groups: list[SourceGroup] = []
    for group_id, ((key_x, key_y), indices) in enumerate(grouped.items()):
        trace_indices = np.asarray(indices, dtype=np.int64)
        groups.append(
            SourceGroup(
                group_id=int(group_id),
                source_key_x=int(key_x),
                source_key_y=int(key_y),
                source_x_m=float(
                    np.mean(
                        np.asarray(geometry.source_x_m[trace_indices], dtype=np.float64)
                    )
                ),
                source_y_m=float(
                    np.mean(
                        np.asarray(geometry.source_y_m[trace_indices], dtype=np.float64)
                    )
                ),
                trace_indices=trace_indices,
            )
        )
    return tuple(groups)


def select_nearest_source_groups(
    groups: Sequence[SourceGroup],
    *,
    target_group_id: int,
    k_neighbors: int,
    max_source_distance_m: float | None,
    include_self: bool,
) -> np.ndarray:
    k = int(k_neighbors)
    if k <= 0:
        msg = "k_neighbors must be positive"
        raise ValueError(msg)
    max_distance = None
    if max_source_distance_m is not None:
        max_distance = _validate_nonnegative_finite(
            "max_source_distance_m",
            max_source_distance_m,
        )

    by_id = {int(group.group_id): group for group in groups}
    target_id = int(target_group_id)
    if target_id not in by_id:
        msg = f"target_group_id not found: {target_id}"
        raise ValueError(msg)
    target = by_id[target_id]

    ranked: list[tuple[float, int]] = []
    for group in groups:
        group_id = int(group.group_id)
        if group_id == target_id and not include_self:
            continue
        dx = float(group.source_x_m) - float(target.source_x_m)
        dy = float(group.source_y_m) - float(target.source_y_m)
        distance = math.hypot(dx, dy)
        if (
            max_distance is not None
            and group_id != target_id
            and distance > max_distance
        ):
            continue
        ranked.append((distance, group_id))

    if include_self:
        ranked.sort(key=lambda item: (item[1] != target_id, item[0], item[1]))
    else:
        ranked.sort(key=lambda item: (item[0], item[1]))
    return np.asarray([group_id for _, group_id in ranked[:k]], dtype=np.int64)


def _coerce_trace_indices(trace_indices: np.ndarray, *, n_traces: int) -> np.ndarray:
    indices = np.asarray(trace_indices)
    if indices.ndim != 1:
        msg = "trace_indices must be 1D"
        raise ValueError(msg)
    if not np.issubdtype(indices.dtype, np.integer):
        msg = "trace_indices must have integer dtype"
        raise TypeError(msg)
    out = indices.astype(np.int64, copy=False)
    if np.any(out < 0) or np.any(out >= int(n_traces)):
        msg = "trace_indices must be within [0, n_traces)"
        raise ValueError(msg)
    return out


def _selected_valid_geometry_mask(
    geometry: CoarseGeometry, indices: np.ndarray
) -> np.ndarray:
    return (
        np.asarray(geometry.geometry_valid_mask[indices], dtype=np.bool_)
        & np.isfinite(np.asarray(geometry.source_x_m[indices]))
        & np.isfinite(np.asarray(geometry.source_y_m[indices]))
        & np.isfinite(np.asarray(geometry.receiver_x_m[indices]))
        & np.isfinite(np.asarray(geometry.receiver_y_m[indices]))
    )


def _orient_principal_direction(u: np.ndarray) -> np.ndarray:
    out = np.asarray(u, dtype=np.float64).copy()
    dominant = int(np.argmax(np.abs(out)))
    if out[dominant] < 0.0:
        out *= -1.0
    return out


def estimate_signed_offset_side(
    geometry: CoarseGeometry,
    trace_indices: np.ndarray,
    *,
    min_receiver_spread_m: float = 1.0e-3,
    zero_tol_m: float = 1.0e-6,
) -> SignedOffsetResult:
    n_traces = int(np.asarray(geometry.geometry_valid_mask).shape[0])
    indices = _coerce_trace_indices(trace_indices, n_traces=n_traces)
    min_spread = _validate_positive_finite(
        "min_receiver_spread_m", min_receiver_spread_m
    )
    zero_tol = _validate_nonnegative_finite("zero_tol_m", zero_tol_m)

    signed = np.zeros((indices.size,), dtype=np.float32)
    side = np.zeros((indices.size,), dtype=np.int8)
    if indices.size == 0:
        return SignedOffsetResult(signed_offset_m=signed, side=side, reliable=False)

    valid = _selected_valid_geometry_mask(geometry, indices)
    if int(np.count_nonzero(valid)) < 2:
        return SignedOffsetResult(signed_offset_m=signed, side=side, reliable=False)

    valid_indices = indices[valid]
    receiver_xy = np.stack(
        [
            np.asarray(geometry.receiver_x_m[valid_indices], dtype=np.float64),
            np.asarray(geometry.receiver_y_m[valid_indices], dtype=np.float64),
        ],
        axis=1,
    )
    centered_receiver_xy = receiver_xy - np.mean(receiver_xy, axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered_receiver_xy, full_matrices=False)
    if singular_values.size == 0 or float(singular_values[0]) <= 0.0:
        return SignedOffsetResult(signed_offset_m=signed, side=side, reliable=False)

    u = _orient_principal_direction(vh[0])
    receiver_projection = centered_receiver_xy @ u
    if float(np.ptp(receiver_projection)) <= min_spread:
        return SignedOffsetResult(signed_offset_m=signed, side=side, reliable=False)

    source_xy = np.stack(
        [
            np.asarray(geometry.source_x_m[valid_indices], dtype=np.float64),
            np.asarray(geometry.source_y_m[valid_indices], dtype=np.float64),
        ],
        axis=1,
    )
    signed_valid = np.sum((receiver_xy - source_xy) * u[None, :], axis=1)
    signed[valid] = signed_valid.astype(np.float32)
    if not np.any(np.abs(signed_valid) > zero_tol):
        return SignedOffsetResult(signed_offset_m=signed, side=side, reliable=False)

    side[valid] = np.where(
        np.abs(signed_valid) <= zero_tol,
        0,
        np.where(signed_valid < 0.0, -1, 1),
    ).astype(np.int8)
    return SignedOffsetResult(signed_offset_m=signed, side=side, reliable=True)


def split_offset_gap_segments(
    offset_abs_geom_m: np.ndarray,
    *,
    split_by_offset_gap: bool,
    gap_ratio: float,
    min_gap_m: float | None = None,
) -> np.ndarray:
    offsets = np.asarray(offset_abs_geom_m, dtype=np.float64)
    if offsets.ndim != 1:
        msg = "offset_abs_geom_m must be 1D"
        raise ValueError(msg)
    segment_id = np.zeros((offsets.size,), dtype=np.int64)
    if offsets.size <= 1 or not bool(split_by_offset_gap):
        return segment_id
    if not np.all(np.isfinite(offsets)):
        msg = "offset_abs_geom_m must be finite"
        raise ValueError(msg)

    ratio, min_gap = _validate_gap_cfg(
        gap_ratio=gap_ratio,
        min_gap_m=min_gap_m,
    )
    order = np.argsort(offsets, kind="stable")
    sorted_offsets = offsets[order]
    sorted_segment_id = np.zeros((offsets.size,), dtype=np.int64)
    diff = np.abs(np.diff(sorted_offsets))
    positive = diff[diff > 0.0]
    if positive.size == 0:
        return segment_id

    threshold = float(np.median(positive)) * ratio
    if min_gap is not None:
        threshold = max(threshold, min_gap)

    gap_after = np.flatnonzero(diff > threshold)
    start = 0
    current_segment_id = 0
    for pos in gap_after:
        stop = int(pos) + 1
        sorted_segment_id[start:stop] = current_segment_id
        current_segment_id += 1
        start = stop
    sorted_segment_id[start:] = current_segment_id
    segment_id[order] = sorted_segment_id
    return segment_id
