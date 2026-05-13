from __future__ import annotations

import numpy as np
import pytest
from seisai_dataset import noise_decider
from seisai_dataset.noise_decider import EventDetectConfig, decide_noise


def test_event_detect_config_forwards_cluster_count_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_pick_cluster(*_args, cluster_count_mode: str, **_kwargs):
        captured['cluster_count_mode'] = cluster_count_mode
        return False, np.zeros(8, dtype=np.int32), np.zeros(8, dtype=np.int32)

    def fake_majority(*_args, **_kwargs):
        return False, np.zeros(8, dtype=np.int32)

    monkeypatch.setattr(noise_decider, 'detect_event_pick_cluster', fake_pick_cluster)
    monkeypatch.setattr(noise_decider, 'detect_event_stalta_majority', fake_majority)

    cfg = EventDetectConfig(use_envelope=False, cluster_count_mode='unique_trace')
    decision = decide_noise(np.zeros((2, 8), dtype=np.float32), 0.001, cfg)

    assert decision.is_noise is True
    assert captured['cluster_count_mode'] == 'unique_trace'


def test_decide_noise_rejects_invalid_cluster_count_mode() -> None:
    cfg = EventDetectConfig(use_envelope=False, cluster_count_mode='bad')

    with pytest.raises(ValueError, match='cluster_count_mode'):
        decide_noise(np.zeros((2, 8), dtype=np.float32), 0.001, cfg)
