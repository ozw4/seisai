from __future__ import annotations

import numpy as np
import pytest

from seisai_engine.pipelines.fbpick.coarse import (
    build_coarse_fb_labels_for_anchors,
    build_coarse_time_grid,
    build_time_channel,
    project_coarse_indices_to_raw_time,
    project_fb_indices_to_coarse_time,
    resample_waveform_time_axis,
)


def test_build_coarse_time_grid_uses_endpoint_aligned_contract() -> None:
    grid = build_coarse_time_grid(
        raw_time_len=11,
        coarse_time_len=6,
        dt_sec=0.002,
    )

    assert grid.raw_time_len == 11
    assert grid.coarse_time_len == 6
    assert grid.raw_to_coarse_factor == pytest.approx(0.5)
    assert grid.coarse_to_raw_factor == pytest.approx(2.0)
    assert grid.dt_eff_sec == pytest.approx(0.004)
    assert grid.time_view_sec.dtype == np.float32
    np.testing.assert_allclose(
        grid.time_view_sec,
        np.linspace(0.0, 0.02, 6, dtype=np.float32),
    )


def test_project_fb_indices_to_coarse_time_preserves_ignore_and_endpoints() -> None:
    raw = np.asarray([0, 2, 10, -1], dtype=np.int64)

    projected = project_fb_indices_to_coarse_time(
        raw,
        raw_time_len=11,
        coarse_time_len=6,
    )

    np.testing.assert_array_equal(
        projected,
        np.asarray([0, 1, 5, -1], dtype=np.int64),
    )


def test_project_coarse_indices_to_raw_time_preserves_ignore_and_endpoints() -> None:
    coarse = np.asarray([0, 3, 5, -1], dtype=np.int64)

    projected = project_coarse_indices_to_raw_time(
        coarse,
        raw_time_len=11,
        coarse_time_len=6,
    )

    np.testing.assert_array_equal(
        projected,
        np.asarray([0, 6, 10, -1], dtype=np.int64),
    )


def test_project_indices_rejects_non_integer_and_out_of_range_values() -> None:
    with pytest.raises(ValueError, match='integer-valued'):
        project_fb_indices_to_coarse_time(
            np.asarray([1.5], dtype=np.float32),
            raw_time_len=11,
            coarse_time_len=6,
        )

    with pytest.raises(ValueError, match='\\[0, 10\\]'):
        project_fb_indices_to_coarse_time(
            np.asarray([11], dtype=np.int64),
            raw_time_len=11,
            coarse_time_len=6,
        )

    with pytest.raises(ValueError, match='\\[0, 5\\]'):
        project_coarse_indices_to_raw_time(
            np.asarray([6], dtype=np.int64),
            raw_time_len=11,
            coarse_time_len=6,
        )


def test_build_time_channel_repeats_seconds_or_normalized_time() -> None:
    time_view = np.asarray([0.0, 0.25, 0.5], dtype=np.float32)

    seconds = build_time_channel(time_view, trace_len=2, normalize=False)
    normalized = build_time_channel(time_view, trace_len=2, normalize=True)

    assert seconds.shape == (2, 3)
    assert seconds.dtype == np.float32
    np.testing.assert_allclose(seconds[0], time_view)
    np.testing.assert_allclose(seconds[1], time_view)
    np.testing.assert_allclose(normalized[0], np.asarray([0.0, 0.5, 1.0]))


def test_build_coarse_fb_labels_for_anchors_ignores_invalid_trace_rows() -> None:
    labels = build_coarse_fb_labels_for_anchors(
        np.asarray([0, 10, 9999, -1], dtype=np.int64),
        np.asarray([True, True, False, True], dtype=np.bool_),
        raw_time_len=11,
        coarse_time_len=6,
    )

    np.testing.assert_array_equal(
        labels,
        np.asarray([0, 5, -1, -1], dtype=np.int64),
    )


def test_resample_waveform_time_axis_returns_float32_endpoint_aligned_shape() -> None:
    waveform = np.stack(
        [
            np.linspace(-1.0, 1.0, 17, dtype=np.float32),
            np.linspace(2.0, -2.0, 17, dtype=np.float32),
        ],
        axis=0,
    )

    out = resample_waveform_time_axis(waveform, out_time_len=9)

    assert out.shape == (2, 9)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, 0], waveform[:, 0])
    np.testing.assert_allclose(out[:, -1], waveform[:, -1])
    assert np.all(np.isfinite(out))


def test_resample_waveform_time_axis_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match='Hvalid'):
        resample_waveform_time_axis(np.zeros((0, 4), dtype=np.float32), out_time_len=4)

    with pytest.raises(ValueError, match='time axis length'):
        resample_waveform_time_axis(np.zeros((2, 1), dtype=np.float32), out_time_len=4)

    with pytest.raises(ValueError, match='out_time_len'):
        resample_waveform_time_axis(np.zeros((2, 4), dtype=np.float32), out_time_len=1)

    bad = np.zeros((2, 4), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        resample_waveform_time_axis(bad, out_time_len=4)
