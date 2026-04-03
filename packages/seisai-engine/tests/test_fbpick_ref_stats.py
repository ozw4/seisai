from __future__ import annotations

import numpy as np
import pytest

from seisai_engine.pipelines.fbpick.common import (
    compute_ref_stats,
    compute_ref_stats_from_records,
)


def test_compute_ref_stats_returns_p99_and_uses_abs_offset() -> None:
    time_abs_sec = np.asarray([0.1, 0.2, 0.5, 1.0, 4.0], dtype=np.float32)
    offset_m = np.asarray([-10.0, 20.0, -30.0, 40.0, -80.0], dtype=np.float32)

    stats = compute_ref_stats(time_abs_sec=time_abs_sec, offset_m=offset_m)

    assert stats['time_ref_sec'] == pytest.approx(
        float(np.percentile(time_abs_sec, 99))
    )
    assert stats['offset_ref_m'] == pytest.approx(
        float(np.percentile(np.abs(offset_m), 99))
    )


def test_compute_ref_stats_from_records_concatenates_iterable_dicts() -> None:
    records = [
        {
            'time_abs_sec': np.asarray([0.25, 0.5], dtype=np.float32),
            'offset_m': np.asarray([-100.0, 150.0], dtype=np.float32),
        },
        {
            'time_abs_sec': [1.0, 2.5, 3.0],
            'offset_m': [-50.0, 200.0, -250.0],
        },
    ]

    stats = compute_ref_stats_from_records(records)

    all_time = np.asarray([0.25, 0.5, 1.0, 2.5, 3.0], dtype=np.float64)
    all_offset = np.asarray([-100.0, 150.0, -50.0, 200.0, -250.0], dtype=np.float64)
    assert stats['time_ref_sec'] == pytest.approx(float(np.percentile(all_time, 99)))
    assert stats['offset_ref_m'] == pytest.approx(
        float(np.percentile(np.abs(all_offset), 99))
    )


def test_compute_ref_stats_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match='time_abs_sec must be non-empty'):
        compute_ref_stats(
            time_abs_sec=np.asarray([], dtype=np.float32),
            offset_m=np.asarray([1.0], dtype=np.float32),
        )
