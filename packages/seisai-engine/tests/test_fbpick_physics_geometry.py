from __future__ import annotations

import numpy as np
import pytest

from seisai_engine.pipelines.fbpick.physics import (
    CoarseGeometry,
    SourceGroup,
    build_source_groups,
    estimate_signed_offset_side,
    load_coarse_geometry_from_npz,
    select_nearest_source_groups,
    split_offset_gap_segments,
)


def _geometry(
    *,
    source_x_m: np.ndarray,
    source_y_m: np.ndarray,
    receiver_x_m: np.ndarray | None = None,
    receiver_y_m: np.ndarray | None = None,
    offset_abs_geom_m: np.ndarray | None = None,
    geometry_valid_mask: np.ndarray | None = None,
) -> CoarseGeometry:
    n_traces = int(source_x_m.shape[0])
    if receiver_x_m is None:
        receiver_x_m = source_x_m.copy()
    if receiver_y_m is None:
        receiver_y_m = source_y_m.copy()
    if offset_abs_geom_m is None:
        offset_abs_geom_m = np.zeros((n_traces,), dtype=np.float32)
    if geometry_valid_mask is None:
        geometry_valid_mask = np.ones((n_traces,), dtype=np.bool_)
    return CoarseGeometry(
        source_x_m=np.asarray(source_x_m, dtype=np.float32),
        source_y_m=np.asarray(source_y_m, dtype=np.float32),
        receiver_x_m=np.asarray(receiver_x_m, dtype=np.float32),
        receiver_y_m=np.asarray(receiver_y_m, dtype=np.float32),
        offset_abs_geom_m=np.asarray(offset_abs_geom_m, dtype=np.float32),
        geometry_valid_mask=np.asarray(geometry_valid_mask, dtype=np.bool_),
    )


def test_load_coarse_geometry_from_npz_normalizes_arrays() -> None:
    coarse = {
        "source_x_m": np.array([10.0, 10.5], dtype=np.float64),
        "source_y_m": np.array([20.0, 20.5], dtype=np.float64),
        "receiver_x_m": np.array([15.0, 16.0], dtype=np.float64),
        "receiver_y_m": np.array([25.0, 26.0], dtype=np.float64),
        "offset_abs_geom_m": np.array([7.0, 8.0], dtype=np.float64),
        "geometry_valid_mask": np.array([1, 0], dtype=np.int32),
    }

    geometry = load_coarse_geometry_from_npz(coarse, n_traces=2)

    assert geometry is not None
    assert geometry.source_x_m.dtype == np.float32
    assert geometry.geometry_valid_mask.dtype == np.bool_
    np.testing.assert_array_equal(
        geometry.geometry_valid_mask,
        np.array([True, False], dtype=np.bool_),
    )


def test_load_coarse_geometry_from_npz_returns_none_without_geometry() -> None:
    coarse = {
        "coarse_pick_i": np.array([10, 20], dtype=np.int32),
        "coarse_pmax": np.array([0.8, 0.9], dtype=np.float32),
    }

    assert load_coarse_geometry_from_npz(coarse, n_traces=2) is None


def test_load_coarse_geometry_from_npz_rejects_partial_geometry() -> None:
    with pytest.raises(KeyError, match="missing optional geometry keys"):
        load_coarse_geometry_from_npz(
            {"source_x_m": np.array([10.0], dtype=np.float32)},
            n_traces=1,
        )


def test_build_source_groups_uses_coordinate_tolerance_and_excludes_invalid() -> None:
    geometry = _geometry(
        source_x_m=np.array([10.1, 10.8, 14.2, 99.0], dtype=np.float32),
        source_y_m=np.array([20.2, 20.7, 20.1, 99.0], dtype=np.float32),
        geometry_valid_mask=np.array([True, True, True, False], dtype=np.bool_),
    )

    groups = build_source_groups(geometry, coord_group_tol_m=2.0)

    assert len(groups) == 2
    assert groups[0].group_id == 0
    assert (groups[0].source_key_x, groups[0].source_key_y) == (5, 10)
    np.testing.assert_array_equal(
        groups[0].trace_indices, np.array([0, 1], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        groups[1].trace_indices, np.array([2], dtype=np.int64)
    )


def test_select_nearest_source_groups_respects_self_max_distance_and_order() -> None:
    groups = (
        SourceGroup(0, 0, 0, 0.0, 0.0, np.array([0], dtype=np.int64)),
        SourceGroup(1, 0, 0, 10.0, 0.0, np.array([1], dtype=np.int64)),
        SourceGroup(2, 0, 0, 3.0, 4.0, np.array([2], dtype=np.int64)),
        SourceGroup(3, 0, 0, 100.0, 0.0, np.array([3], dtype=np.int64)),
    )

    with_self = select_nearest_source_groups(
        groups,
        target_group_id=0,
        k_neighbors=3,
        max_source_distance_m=None,
        include_self=True,
    )
    without_self = select_nearest_source_groups(
        groups,
        target_group_id=0,
        k_neighbors=3,
        max_source_distance_m=6.0,
        include_self=False,
    )

    np.testing.assert_array_equal(with_self, np.array([0, 2, 1], dtype=np.int64))
    np.testing.assert_array_equal(without_self, np.array([2], dtype=np.int64))


def test_select_nearest_source_groups_prioritizes_self_with_same_xy_neighbor() -> None:
    groups = (
        SourceGroup(0, 0, 0, 0.0, 0.0, np.array([0], dtype=np.int64)),
        SourceGroup(1, 0, 0, 0.0, 0.0, np.array([1], dtype=np.int64)),
        SourceGroup(2, 0, 0, 5.0, 0.0, np.array([2], dtype=np.int64)),
    )

    selected = select_nearest_source_groups(
        groups,
        target_group_id=1,
        k_neighbors=2,
        max_source_distance_m=None,
        include_self=True,
    )

    np.testing.assert_array_equal(selected, np.array([1, 0], dtype=np.int64))


def test_estimate_signed_offset_side_splits_straight_receiver_line() -> None:
    geometry = _geometry(
        source_x_m=np.zeros((5,), dtype=np.float32),
        source_y_m=np.zeros((5,), dtype=np.float32),
        receiver_x_m=np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32),
        receiver_y_m=np.zeros((5,), dtype=np.float32),
    )

    result = estimate_signed_offset_side(
        geometry,
        np.arange(5, dtype=np.int64),
        min_receiver_spread_m=1.0e-3,
        zero_tol_m=1.0e-5,
    )

    assert result.reliable is True
    np.testing.assert_allclose(
        result.signed_offset_m,
        np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.side, np.array([-1, -1, 0, 1, 1], dtype=np.int8)
    )


def test_estimate_signed_offset_side_marks_degenerate_receiver_pca_unreliable() -> None:
    geometry = _geometry(
        source_x_m=np.zeros((3,), dtype=np.float32),
        source_y_m=np.zeros((3,), dtype=np.float32),
        receiver_x_m=np.ones((3,), dtype=np.float32),
        receiver_y_m=np.ones((3,), dtype=np.float32),
    )

    result = estimate_signed_offset_side(geometry, np.arange(3, dtype=np.int64))

    assert result.reliable is False
    np.testing.assert_array_equal(result.side, np.zeros((3,), dtype=np.int8))


def test_split_offset_gap_segments_sorts_offsets_and_maps_segments_back() -> None:
    segment_id = split_offset_gap_segments(
        np.array([200.0, 10.0, 500.0, 30.0, 210.0, 20.0], dtype=np.float32),
        split_by_offset_gap=True,
        gap_ratio=5.0,
        min_gap_m=None,
    )

    np.testing.assert_array_equal(
        segment_id, np.array([1, 0, 2, 0, 1, 0], dtype=np.int64)
    )


def test_split_offset_gap_segments_can_be_disabled() -> None:
    segment_id = split_offset_gap_segments(
        np.array([10.0, 1000.0], dtype=np.float32),
        split_by_offset_gap=False,
        gap_ratio=5.0,
        min_gap_m=None,
    )

    np.testing.assert_array_equal(segment_id, np.zeros((2,), dtype=np.int64))
