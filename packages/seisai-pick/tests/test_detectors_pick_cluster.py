import numpy as np
import pytest
from seisai_pick import detectors
from seisai_pick.detectors import (
    _sliding_sum_same,
    _sliding_unique_trace_count,
    detect_event_pick_cluster,
)


def test_detect_event_pick_cluster_default_matches_trigger_mode() -> None:
    rng = np.random.default_rng(123)
    x_ht = rng.normal(size=(3, 64)).astype(np.float32)

    implicit = detect_event_pick_cluster(
        x_ht,
        0.001,
        sta_ms=2.0,
        lta_ms=8.0,
        min_on_ms=1.0,
        win_ms=6.0,
    )
    explicit = detect_event_pick_cluster(
        x_ht,
        0.001,
        sta_ms=2.0,
        lta_ms=8.0,
        min_on_ms=1.0,
        win_ms=6.0,
        cluster_count_mode='trigger',
    )

    assert implicit[0] == explicit[0]
    assert np.array_equal(implicit[1], explicit[1])
    assert np.array_equal(implicit[2], explicit[2])


def test_sliding_unique_trace_count_deduplicates_repeated_trace_picks() -> None:
    pick_mask = np.zeros((1, 12), dtype=np.bool_)
    pick_mask[0, 4] = True
    pick_mask[0, 6] = True
    pick_hist = pick_mask.sum(axis=0).astype(np.int32)

    trigger_counts = _sliding_sum_same(pick_hist, 5)
    unique_counts = _sliding_unique_trace_count(pick_mask, 5)

    assert int(trigger_counts.max()) == 2
    assert int(unique_counts.max()) == 1


def test_sliding_unique_trace_count_counts_multiple_traces() -> None:
    pick_mask = np.zeros((2, 12), dtype=np.bool_)
    pick_mask[0, 4] = True
    pick_mask[1, 6] = True

    unique_counts = _sliding_unique_trace_count(pick_mask, 5)

    assert int(unique_counts.max()) == 2


def test_sliding_unique_trace_count_keeps_separated_same_trace_at_one() -> None:
    pick_mask = np.zeros((1, 20), dtype=np.bool_)
    pick_mask[0, 3] = True
    pick_mask[0, 14] = True

    unique_counts = _sliding_unique_trace_count(pick_mask, 5)

    assert int(unique_counts.max()) == 1


def test_detect_event_pick_cluster_unique_trace_mode_runs_real_detector() -> None:
    rng = np.random.default_rng(456)
    x_ht = rng.normal(size=(3, 48)).astype(np.float32)

    is_event, pick_hist, cluster = detect_event_pick_cluster(
        x_ht,
        0.001,
        sta_ms=1.0,
        lta_ms=4.0,
        min_on_ms=1.0,
        win_ms=4.0,
        cluster_count_mode='unique_trace',
    )

    assert isinstance(is_event, bool)
    assert pick_hist.shape == (48,)
    assert cluster.shape == (48,)
    assert np.all(cluster <= x_ht.shape[0])


def test_detect_event_pick_cluster_uses_selected_count_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pick_hist = np.zeros(12, dtype=np.int32)
    pick_hist[[4, 6]] = 1
    pick_mask = np.zeros((1, 12), dtype=np.bool_)
    pick_mask[0, [4, 6]] = True

    def fake_hist(*_args) -> np.ndarray:
        return pick_hist.copy()

    def fake_hist_and_mask(*_args) -> tuple[np.ndarray, np.ndarray]:
        return pick_hist.copy(), pick_mask.copy()

    monkeypatch.setattr(detectors, '_stalta_pick_hist', fake_hist)
    monkeypatch.setattr(
        detectors, '_stalta_pick_hist_and_trace_mask', fake_hist_and_mask
    )

    x_ht = np.zeros((1, 12), dtype=np.float32)
    trigger_event, trigger_hist, trigger_cluster = detect_event_pick_cluster(
        x_ht,
        1.0,
        sta_ms=1000.0,
        lta_ms=2000.0,
        win_ms=5000.0,
        min_traces=2,
        cluster_count_mode='trigger',
    )
    unique_event, unique_hist, unique_cluster = detect_event_pick_cluster(
        x_ht,
        1.0,
        sta_ms=1000.0,
        lta_ms=2000.0,
        win_ms=5000.0,
        min_traces=2,
        cluster_count_mode='unique_trace',
    )

    assert trigger_event is True
    assert unique_event is False
    assert np.array_equal(trigger_hist, pick_hist)
    assert np.array_equal(unique_hist, pick_hist)
    assert int(trigger_cluster.max()) == 2
    assert int(unique_cluster.max()) == 1


def test_detect_event_pick_cluster_rejects_invalid_count_mode() -> None:
    x_ht = np.zeros((1, 8), dtype=np.float32)

    with pytest.raises(ValueError, match='cluster_count_mode'):
        detect_event_pick_cluster(
            x_ht,
            0.001,
            sta_ms=1.0,
            lta_ms=2.0,
            cluster_count_mode='bad',
        )
