from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .geometry import SourceGroup

__all__ = [
    'SourceXYAnchorSelectionResult',
    'order_source_groups_source_xy',
    'select_source_xy_stride_anchors',
]


@dataclass(frozen=True)
class SourceXYAnchorSelectionResult:
    group_ids: np.ndarray
    ordered_group_ids: tuple[int, ...]
    anchor_group_ids: tuple[int, ...]
    is_anchor: np.ndarray
    nearest_anchor_group_id: np.ndarray
    source_distance_m: np.ndarray


def _source_xy(groups: Sequence[SourceGroup]) -> np.ndarray:
    xy = np.asarray(
        [(float(group.source_x_m), float(group.source_y_m)) for group in groups],
        dtype=np.float64,
    )
    if xy.ndim != 2 or xy.shape[1] != 2:
        msg = 'source group XY must have shape (n_groups, 2)'
        raise ValueError(msg)
    if not np.all(np.isfinite(xy)):
        msg = 'source group XY must be finite'
        raise ValueError(msg)
    return xy


def _dominant_source_line_projection(xy: np.ndarray) -> np.ndarray | None:
    if int(xy.shape[0]) < 2:
        return None
    centered = xy - np.mean(xy, axis=0, keepdims=True)
    if not np.any(centered):
        return None
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0 or float(singular_values[0]) <= 0.0:
        return None
    if (
        singular_values.size > 1
        and float(singular_values[0]) <= float(singular_values[1]) * (1.0 + 1.0e-6)
    ):
        return None
    direction = np.asarray(vh[0], dtype=np.float64)
    dominant_axis = int(np.argmax(np.abs(direction)))
    if float(direction[dominant_axis]) < 0.0:
        direction *= -1.0
    return np.sum(xy * direction[None, :], axis=1)


def order_source_groups_source_xy(
    groups: Sequence[SourceGroup],
) -> tuple[SourceGroup, ...]:
    if len(groups) == 0:
        return ()
    xy = _source_xy(groups)
    projection = _dominant_source_line_projection(xy)
    if projection is None:
        ranked = [
            (
                float(group.source_x_m),
                float(group.source_y_m),
                int(group.group_id),
                idx,
            )
            for idx, group in enumerate(groups)
        ]
    else:
        ranked = [
            (
                float(projection[idx]),
                float(group.source_x_m),
                float(group.source_y_m),
                int(group.group_id),
                idx,
            )
            for idx, group in enumerate(groups)
        ]
    ranked.sort()
    return tuple(groups[int(item[-1])] for item in ranked)


def _stride_order_indices(
    *,
    n_groups: int,
    stride: int,
    include_first: bool,
    include_last: bool,
) -> tuple[int, ...]:
    n = int(n_groups)
    if n < 0:
        msg = 'n_groups must be non-negative'
        raise ValueError(msg)
    step = int(stride)
    if step <= 0:
        msg = 'anchor_stride_source_groups must be > 0'
        raise ValueError(msg)
    if n == 0:
        return ()

    start = 0 if bool(include_first) else step
    selected = list(range(start, n, step))
    if not bool(include_first):
        selected = [idx for idx in selected if idx != 0]
    if bool(include_last) and (n - 1) not in selected:
        selected.append(n - 1)
    selected = sorted({int(idx) for idx in selected if 0 <= int(idx) < n})
    if not selected:
        msg = 'anchor selection produced no anchor groups'
        raise ValueError(msg)
    return tuple(selected)


def select_source_xy_stride_anchors(
    groups: Sequence[SourceGroup],
    *,
    anchor_stride_source_groups: int,
    include_first: bool,
    include_last: bool,
) -> SourceXYAnchorSelectionResult:
    ordered = order_source_groups_source_xy(groups)
    order_indices = _stride_order_indices(
        n_groups=len(ordered),
        stride=int(anchor_stride_source_groups),
        include_first=bool(include_first),
        include_last=bool(include_last),
    )
    anchor_group_ids = tuple(int(ordered[idx].group_id) for idx in order_indices)
    anchor_ids = set(anchor_group_ids)

    group_ids = np.asarray([int(group.group_id) for group in groups], dtype=np.int32)
    is_anchor = np.asarray(
        [int(group.group_id) in anchor_ids for group in groups],
        dtype=np.bool_,
    )
    nearest_anchor_group_id = np.full((len(groups),), -1, dtype=np.int32)
    source_distance_m = np.full((len(groups),), np.nan, dtype=np.float32)

    anchors_by_id = {
        int(group.group_id): group
        for group in ordered
        if int(group.group_id) in anchor_ids
    }
    for pos, group in enumerate(groups):
        ranked: list[tuple[float, int]] = []
        for anchor_id, anchor in anchors_by_id.items():
            dx = float(anchor.source_x_m) - float(group.source_x_m)
            dy = float(anchor.source_y_m) - float(group.source_y_m)
            distance = math.hypot(dx, dy)
            ranked.append((distance, int(anchor_id)))
        ranked.sort(key=lambda item: (item[0], item[1]))
        if ranked:
            distance, anchor_id = ranked[0]
            nearest_anchor_group_id[pos] = np.int32(anchor_id)
            source_distance_m[pos] = np.float32(distance)

    return SourceXYAnchorSelectionResult(
        group_ids=group_ids,
        ordered_group_ids=tuple(int(group.group_id) for group in ordered),
        anchor_group_ids=anchor_group_ids,
        is_anchor=is_anchor,
        nearest_anchor_group_id=nearest_anchor_group_id,
        source_distance_m=source_distance_m,
    )
