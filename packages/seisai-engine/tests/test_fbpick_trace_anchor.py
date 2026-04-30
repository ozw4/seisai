from __future__ import annotations

import numpy as np
import pytest

from seisai_engine.pipelines.fbpick.coarse import (
    select_trace_anchors,
    split_trace_segments_by_offset_gap,
)


def test_split_trace_segments_by_offset_gap_detects_large_jumps() -> None:
    offsets = np.asarray([0, 10, 20, 300, 310, 320, 1000], dtype=np.float32)

    segments = split_trace_segments_by_offset_gap(
        offsets,
        gap_ratio=5.0,
        min_gap_m=None,
    )

    assert [(s.start_pos, s.stop_pos, s.n_traces) for s in segments] == [
        (0, 3, 3),
        (3, 6, 3),
        (6, 7, 1),
    ]
    assert [s.n_anchor_rows for s in segments] == [0, 0, 0]


def test_split_trace_segments_by_offset_gap_matches_fbpick_gap_example() -> None:
    offsets = np.asarray([0, 10, 20, 30, 1000, 1010, 1020], dtype=np.float32)

    segments = split_trace_segments_by_offset_gap(
        offsets,
        gap_ratio=5.0,
        min_gap_m=None,
    )

    assert [(s.start_pos, s.stop_pos) for s in segments] == [(0, 4), (4, 7)]


def test_split_trace_segments_by_offset_gap_keeps_flat_offsets_together() -> None:
    segments = split_trace_segments_by_offset_gap(
        np.zeros(4, dtype=np.float32),
        gap_ratio=5.0,
        min_gap_m=10.0,
    )

    assert [(s.start_pos, s.stop_pos, s.n_traces) for s in segments] == [(0, 4, 4)]


def test_split_trace_segments_by_offset_gap_rejects_nonfinite_offsets() -> None:
    with pytest.raises(ValueError, match='offsets_m must be finite'):
        split_trace_segments_by_offset_gap(
            np.asarray([0.0, np.nan], dtype=np.float32),
            gap_ratio=5.0,
            min_gap_m=None,
        )


def test_select_trace_anchors_pads_when_trace_count_is_shorter_than_output() -> None:
    selection = select_trace_anchors(
        np.asarray([10, 11, 12], dtype=np.int64),
        np.asarray([100.0, 200.0, 300.0], dtype=np.float32),
        5,
        'center',
        gap_ratio=5.0,
        min_gap_m=None,
    )

    np.testing.assert_array_equal(
        selection.anchor_raw_indices,
        np.asarray([10, 11, 12, -1, -1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        selection.anchor_source_pos,
        np.asarray([0, 1, 2, -1, -1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        selection.trace_valid,
        np.asarray([True, True, True, False, False], dtype=np.bool_),
    )
    np.testing.assert_array_equal(selection.segment_id[3:], np.asarray([-1, -1]))
    np.testing.assert_allclose(selection.anchor_offsets_m[3:], np.asarray([0.0, 0.0]))
    assert [s.n_anchor_rows for s in selection.segments] == [3]


def test_select_trace_anchors_center_mode_never_bins_across_gap() -> None:
    raw_indices = np.arange(10, dtype=np.int64)
    offsets = np.asarray([0, 10, 20, 30, 40, 1000, 1010, 1020, 1030, 1040])

    selection = select_trace_anchors(
        raw_indices,
        offsets,
        4,
        'center',
        gap_ratio=5.0,
        min_gap_m=None,
    )

    np.testing.assert_array_equal(
        selection.anchor_source_pos,
        np.asarray([1, 3, 6, 8], dtype=np.int64),
    )
    np.testing.assert_array_equal(selection.segment_id, np.asarray([0, 0, 1, 1]))
    assert [s.n_anchor_rows for s in selection.segments] == [2, 2]
    assert selection.anchor_bin_start_pos is not None
    assert selection.anchor_bin_stop_pos is not None
    assert all(
        start >= 0 and stop <= 5
        for start, stop in zip(
            selection.anchor_bin_start_pos[:2],
            selection.anchor_bin_stop_pos[:2],
            strict=True,
        )
    )
    assert all(
        start >= 5 and stop <= 10
        for start, stop in zip(
            selection.anchor_bin_start_pos[2:],
            selection.anchor_bin_stop_pos[2:],
            strict=True,
        )
    )


def test_select_trace_anchors_random_mode_stays_inside_gap_segments() -> None:
    raw_indices = np.arange(7, dtype=np.int64)
    offsets = np.asarray([0, 10, 20, 30, 1000, 1010, 1020], dtype=np.float32)

    selection = select_trace_anchors(
        raw_indices,
        offsets,
        4,
        'random',
        gap_ratio=5.0,
        min_gap_m=None,
        rng=np.random.default_rng(0),
    )

    assert [s.n_anchor_rows for s in selection.segments] == [2, 2]
    np.testing.assert_array_equal(selection.segment_id, np.asarray([0, 0, 1, 1]))
    assert selection.anchor_bin_start_pos is not None
    assert selection.anchor_bin_stop_pos is not None
    assert all(
        0 <= start < stop <= 4
        for start, stop in zip(
            selection.anchor_bin_start_pos[:2],
            selection.anchor_bin_stop_pos[:2],
            strict=True,
        )
    )
    assert all(
        4 <= start < stop <= 7
        for start, stop in zip(
            selection.anchor_bin_start_pos[2:],
            selection.anchor_bin_stop_pos[2:],
            strict=True,
        )
    )
    assert np.all(selection.anchor_source_pos[:2] < 4)
    assert np.all(selection.anchor_source_pos[2:] >= 4)


def test_select_trace_anchors_prefers_long_segments_when_there_are_too_many() -> None:
    raw_indices = np.arange(11, dtype=np.int64)
    offsets = np.asarray(
        [0, 10, 20, 1000, 1010, 2000, 2010, 3000, 3010, 4000, 4010],
        dtype=np.float32,
    )

    selection = select_trace_anchors(
        raw_indices,
        offsets,
        3,
        'center',
        gap_ratio=1.5,
        min_gap_m=50.0,
    )

    np.testing.assert_array_equal(selection.anchor_source_pos, np.asarray([1, 4, 6]))
    assert [s.n_anchor_rows for s in selection.segments] == [1, 1, 1, 0, 0]


def test_select_trace_anchors_random_mode_is_seed_reproducible() -> None:
    raw_indices = np.arange(50, dtype=np.int64)
    offsets = np.arange(50, dtype=np.float32) * 10.0

    first = select_trace_anchors(
        raw_indices,
        offsets,
        5,
        'random',
        gap_ratio=5.0,
        min_gap_m=None,
        rng=np.random.default_rng(123),
    )
    second = select_trace_anchors(
        raw_indices,
        offsets,
        5,
        'random',
        gap_ratio=5.0,
        min_gap_m=None,
        rng=np.random.default_rng(123),
    )
    different_seed = select_trace_anchors(
        raw_indices,
        offsets,
        5,
        'random',
        gap_ratio=5.0,
        min_gap_m=None,
        rng=np.random.default_rng(124),
    )

    np.testing.assert_array_equal(first.anchor_source_pos, second.anchor_source_pos)
    assert not np.array_equal(first.anchor_source_pos, different_seed.anchor_source_pos)
    assert np.all(first.trace_valid)
