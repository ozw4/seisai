import numpy as np

from seisai_pick.detection.peak_detection import detect_event_peaks


def test_detect_event_peaks_unimodal_returns_single_peak() -> None:
    s_t = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    peaks = detect_event_peaks(
        s_t,
        min_score=0.0,
        min_distance=0,
        smooth_window=1,
    )

    assert peaks.dtype == np.int64
    assert np.array_equal(peaks, np.array([1], dtype=np.int64))


def test_detect_event_peaks_min_score_filters_out_peak() -> None:
    s_t = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    peaks = detect_event_peaks(
        s_t,
        min_score=1.1,
        min_distance=0,
        smooth_window=1,
    )

    assert peaks.dtype == np.int64
    assert peaks.size == 0


def test_detect_event_peaks_min_distance_keeps_higher_peak() -> None:
    s_t = np.array([0.0, 1.0, 0.0, 0.9, 0.0], dtype=np.float32)

    peaks = detect_event_peaks(
        s_t,
        min_score=0.0,
        min_distance=2,
        smooth_window=1,
    )

    assert np.array_equal(peaks, np.array([1], dtype=np.int64))


def test_detect_event_peaks_tie_within_min_distance_keeps_earlier() -> None:
    s_t = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    peaks = detect_event_peaks(
        s_t,
        min_score=0.0,
        min_distance=2,
        smooth_window=1,
    )

    assert np.array_equal(peaks, np.array([1], dtype=np.int64))


def test_detect_event_peaks_plateau_keeps_start_only() -> None:
    s_t = np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)

    peaks = detect_event_peaks(
        s_t,
        min_score=0.0,
        min_distance=0,
        smooth_window=1,
    )

    assert np.array_equal(peaks, np.array([1], dtype=np.int64))


def test_detect_event_peaks_smoothing_window_preserves_length_behavior() -> None:
    s_t = np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=np.float32)

    peaks = detect_event_peaks(
        s_t,
        min_score=0.2,
        min_distance=0,
        smooth_window=3,
    )

    assert peaks.dtype == np.int64
    assert np.array_equal(peaks, np.array([2], dtype=np.int64))
