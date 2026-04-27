from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    'TraceAnchorSelection',
    'TraceSegment',
    'select_trace_anchors',
    'split_trace_segments_by_offset_gap',
]


@dataclass(frozen=True)
class TraceSegment:
    segment_id: int
    start_pos: int
    stop_pos: int
    n_traces: int
    n_anchor_rows: int = 0


@dataclass(frozen=True)
class TraceAnchorSelection:
    anchor_raw_indices: np.ndarray
    anchor_source_pos: np.ndarray
    anchor_offsets_m: np.ndarray
    trace_valid: np.ndarray
    segment_id: np.ndarray
    segments: tuple[TraceSegment, ...]
    anchor_bin_start_pos: np.ndarray | None = None
    anchor_bin_stop_pos: np.ndarray | None = None


def _validate_offsets(offsets_m: np.ndarray) -> np.ndarray:
    offsets = np.asarray(offsets_m, dtype=np.float64)
    if offsets.ndim != 1:
        msg = f'offsets_m must be 1D, got shape={offsets.shape}'
        raise ValueError(msg)
    if offsets.size == 0:
        msg = 'offsets_m must be non-empty'
        raise ValueError(msg)
    if not np.all(np.isfinite(offsets)):
        msg = 'offsets_m must be finite'
        raise ValueError(msg)
    return offsets


def _validate_gap_cfg(
    *,
    gap_ratio: float,
    min_gap_m: float | None,
) -> tuple[float, float | None]:
    ratio = float(gap_ratio)
    if (not math.isfinite(ratio)) or ratio <= 1.0:
        msg = 'gap_ratio must be > 1.0'
        raise ValueError(msg)
    if min_gap_m is None:
        return ratio, None
    min_gap = float(min_gap_m)
    if (not math.isfinite(min_gap)) or min_gap <= 0.0:
        msg = 'min_gap_m must be None or > 0'
        raise ValueError(msg)
    return ratio, min_gap


def _make_segments(boundaries: list[int]) -> tuple[TraceSegment, ...]:
    segments: list[TraceSegment] = []
    for segment_id, (start, stop) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        segments.append(
            TraceSegment(
                segment_id=int(segment_id),
                start_pos=int(start),
                stop_pos=int(stop),
                n_traces=int(stop - start),
            )
        )
    return tuple(segments)


def split_trace_segments_by_offset_gap(
    offsets_m: np.ndarray,
    *,
    gap_ratio: float,
    min_gap_m: float | None = None,
) -> tuple[TraceSegment, ...]:
    offsets = _validate_offsets(offsets_m)
    ratio, min_gap = _validate_gap_cfg(gap_ratio=gap_ratio, min_gap_m=min_gap_m)
    n_traces = int(offsets.size)
    if n_traces == 1:
        return _make_segments([0, 1])

    diff = np.abs(np.diff(offsets))
    positive = diff[diff > 0.0]
    if positive.size == 0:
        return _make_segments([0, n_traces])

    threshold = float(np.median(positive)) * ratio
    if min_gap is not None:
        threshold = max(threshold, min_gap)

    gap_after = np.flatnonzero(diff > threshold)
    if gap_after.size == 0:
        return _make_segments([0, n_traces])

    boundaries = [0]
    boundaries.extend(int(i) + 1 for i in gap_after)
    boundaries.append(n_traces)
    return _make_segments(boundaries)


def _with_anchor_rows(
    segments: tuple[TraceSegment, ...],
    counts: np.ndarray,
) -> tuple[TraceSegment, ...]:
    return tuple(
        TraceSegment(
            segment_id=segment.segment_id,
            start_pos=segment.start_pos,
            stop_pos=segment.stop_pos,
            n_traces=segment.n_traces,
            n_anchor_rows=int(count),
        )
        for segment, count in zip(segments, counts, strict=True)
    )


def _allocate_anchor_rows(
    segments: tuple[TraceSegment, ...],
    *,
    target_rows: int,
) -> tuple[TraceSegment, ...]:
    target = int(target_rows)
    if target <= 0:
        msg = 'target_rows must be positive'
        raise ValueError(msg)

    lengths = np.asarray([s.n_traces for s in segments], dtype=np.int64)
    if np.any(lengths <= 0):
        msg = 'segments must be non-empty'
        raise ValueError(msg)
    total = int(lengths.sum())
    if target > total:
        msg = 'target_rows must be <= total traces'
        raise ValueError(msg)

    n_segments = int(lengths.size)
    counts = np.zeros(n_segments, dtype=np.int64)
    if n_segments > target:
        order = sorted(range(n_segments), key=lambda i: (-int(lengths[i]), int(i)))
        for i in order[:target]:
            counts[i] = 1
        return _with_anchor_rows(segments, counts)

    ideal = lengths.astype(np.float64) * (float(target) / float(total))
    counts = np.floor(ideal).astype(np.int64)
    counts = np.maximum(counts, 1)
    counts = np.minimum(counts, lengths)

    while int(counts.sum()) < target:
        candidates = [i for i in range(n_segments) if counts[i] < lengths[i]]
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda i: (float(ideal[i] - counts[i]), int(lengths[i]), -int(i)),
        )
        counts[best] += 1

    while int(counts.sum()) > target:
        candidates = [i for i in range(n_segments) if counts[i] > 1]
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda i: (float(counts[i] - ideal[i]), -int(lengths[i]), -int(i)),
        )
        counts[best] -= 1

    if int(counts.sum()) != target:
        msg = 'failed to allocate requested trace anchor rows'
        raise RuntimeError(msg)
    if np.any(counts > lengths):
        msg = 'allocated anchor rows exceed segment length'
        raise RuntimeError(msg)
    return _with_anchor_rows(segments, counts)


def _segment_bins(segment: TraceSegment) -> list[tuple[int, int]]:
    if segment.n_anchor_rows <= 0:
        return []
    edges = np.floor(
        np.linspace(0, segment.n_traces, segment.n_anchor_rows + 1, dtype=np.float64)
    ).astype(np.int64)
    bins: list[tuple[int, int]] = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        start = int(segment.start_pos + left)
        stop = int(segment.start_pos + right)
        if stop <= start:
            msg = 'trace anchor bin is empty'
            raise RuntimeError(msg)
        bins.append((start, stop))
    return bins


def _select_from_bin(
    start: int,
    stop: int,
    *,
    mode: str,
    rng: np.random.Generator | None,
) -> int:
    if mode == 'center':
        return int((int(start) + int(stop)) // 2)
    if mode == 'random':
        if rng is None:
            rng = np.random.default_rng()
        return int(rng.integers(int(start), int(stop)))
    msg = 'mode must be "random" or "center"'
    raise ValueError(msg)


def _validate_raw_indices(raw_indices: np.ndarray, *, expected_size: int) -> np.ndarray:
    indices = np.asarray(raw_indices)
    if indices.ndim != 1:
        msg = f'raw_indices must be 1D, got shape={indices.shape}'
        raise ValueError(msg)
    if int(indices.size) != int(expected_size):
        msg = 'raw_indices and offsets_m must have the same length'
        raise ValueError(msg)
    if not np.issubdtype(indices.dtype, np.integer):
        msg = 'raw_indices must have integer dtype'
        raise TypeError(msg)
    indices = indices.astype(np.int64, copy=False)
    if np.any(indices < 0):
        msg = 'raw_indices must be non-negative'
        raise ValueError(msg)
    return indices


def _empty_selection_arrays(trace_len: int) -> tuple[np.ndarray, ...]:
    return (
        -np.ones(trace_len, dtype=np.int64),
        -np.ones(trace_len, dtype=np.int64),
        np.zeros(trace_len, dtype=np.float32),
        np.zeros(trace_len, dtype=np.bool_),
        -np.ones(trace_len, dtype=np.int64),
        -np.ones(trace_len, dtype=np.int64),
        -np.ones(trace_len, dtype=np.int64),
    )


def select_trace_anchors(
    raw_indices: np.ndarray,
    offsets_m: np.ndarray,
    trace_len: int,
    mode: str,
    *,
    gap_ratio: float = 5.0,
    min_gap_m: float | None = None,
    rng: np.random.Generator | None = None,
) -> TraceAnchorSelection:
    h_out = int(trace_len)
    if h_out <= 0:
        msg = 'trace_len must be positive'
        raise ValueError(msg)

    mode_norm = str(mode)
    if mode_norm not in ('random', 'center'):
        msg = 'mode must be "random" or "center"'
        raise ValueError(msg)
    if mode_norm == 'random' and rng is None:
        rng = np.random.default_rng()

    offsets = _validate_offsets(offsets_m)
    indices = _validate_raw_indices(raw_indices, expected_size=int(offsets.size))
    segments = split_trace_segments_by_offset_gap(
        offsets,
        gap_ratio=gap_ratio,
        min_gap_m=min_gap_m,
    )

    (
        anchor_raw_indices,
        anchor_source_pos,
        anchor_offsets_m,
        trace_valid,
        segment_id,
        anchor_bin_start_pos,
        anchor_bin_stop_pos,
    ) = _empty_selection_arrays(h_out)

    n_traces = int(indices.size)
    if n_traces < h_out:
        row = 0
        counts = np.zeros(len(segments), dtype=np.int64)
        for segment in segments:
            for source_pos in range(segment.start_pos, segment.stop_pos):
                anchor_raw_indices[row] = indices[source_pos]
                anchor_source_pos[row] = source_pos
                anchor_offsets_m[row] = np.float32(offsets[source_pos])
                trace_valid[row] = True
                segment_id[row] = segment.segment_id
                anchor_bin_start_pos[row] = source_pos
                anchor_bin_stop_pos[row] = source_pos + 1
                row += 1
            counts[segment.segment_id] = segment.n_traces
        return TraceAnchorSelection(
            anchor_raw_indices=anchor_raw_indices,
            anchor_source_pos=anchor_source_pos,
            anchor_offsets_m=anchor_offsets_m,
            trace_valid=trace_valid,
            segment_id=segment_id,
            segments=_with_anchor_rows(segments, counts),
            anchor_bin_start_pos=anchor_bin_start_pos,
            anchor_bin_stop_pos=anchor_bin_stop_pos,
        )

    allocated_segments = _allocate_anchor_rows(segments, target_rows=h_out)
    row = 0
    for segment in allocated_segments:
        for bin_start, bin_stop in _segment_bins(segment):
            source_pos = _select_from_bin(
                bin_start,
                bin_stop,
                mode=mode_norm,
                rng=rng,
            )
            anchor_raw_indices[row] = indices[source_pos]
            anchor_source_pos[row] = source_pos
            anchor_offsets_m[row] = np.float32(offsets[source_pos])
            trace_valid[row] = True
            segment_id[row] = segment.segment_id
            anchor_bin_start_pos[row] = bin_start
            anchor_bin_stop_pos[row] = bin_stop
            row += 1

    if row != h_out:
        msg = 'trace anchor selection did not fill trace_len rows'
        raise RuntimeError(msg)

    return TraceAnchorSelection(
        anchor_raw_indices=anchor_raw_indices,
        anchor_source_pos=anchor_source_pos,
        anchor_offsets_m=anchor_offsets_m,
        trace_valid=trace_valid,
        segment_id=segment_id,
        segments=allocated_segments,
        anchor_bin_start_pos=anchor_bin_start_pos,
        anchor_bin_stop_pos=anchor_bin_stop_pos,
    )
