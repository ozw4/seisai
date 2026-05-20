from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seisai_engine.pipelines.fbpick.common import (
    FINAL_REQUIRED_KEYS,
    REASON_MASK_LOW_SCORE,
    ROBUST_SOURCE_COARSE_OBSERVED,
    ROBUST_SOURCE_THEORETICAL,
    ROBUST_SOURCE_TREND_FILL,
    build_fbpick_final_payload,
    load_fbpick_final_npz,
    save_fbpick_final_npz,
    validate_fbpick_final_payload,
    validate_fine_result_payload,
)


def _make_lineage() -> np.ndarray:
    return np.asarray(
        '{"iter_id":"i0","source_model_id":"fine-model","cfg_hash":"cfg","git_sha":"deadbeef"}'
    )


def _make_coarse_payload() -> dict[str, np.ndarray]:
    n_traces = 4
    coarse_pick_i = np.array([105, 125, 145, 165], dtype=np.int32)
    return {
        'dt_sec': np.asarray(0.004, dtype=np.float32),
        'n_samples_orig': np.asarray(512, dtype=np.int32),
        'n_traces': np.asarray(n_traces, dtype=np.int32),
        'ffid_values': np.array([10, 10, 10, 10], dtype=np.int32),
        'chno_values': np.array([1, 2, 3, 4], dtype=np.int32),
        'offsets_m': np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
        'trace_indices': np.arange(n_traces, dtype=np.int64),
        'coarse_pick_i': coarse_pick_i,
        'coarse_pick_t_sec': coarse_pick_i.astype(np.float32) * np.float32(0.004),
        'coarse_pmax': np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        'coarse_prob_summary': np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        'lineage': np.asarray('{"stage":"coarse"}'),
    }


def _make_robust_payload() -> dict[str, np.ndarray]:
    n_traces = 4
    robust_pick_i = np.array([105, 125, 145, 165], dtype=np.int32)
    return {
        'dt_sec': np.asarray(0.004, dtype=np.float32),
        'n_samples_orig': np.asarray(512, dtype=np.int32),
        'n_traces': np.asarray(n_traces, dtype=np.int32),
        'ffid_values': np.array([10, 10, 10, 10], dtype=np.int32),
        'chno_values': np.array([1, 2, 3, 4], dtype=np.int32),
        'offsets_m': np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
        'trace_indices': np.arange(n_traces, dtype=np.int64),
        'robust_pick_i': robust_pick_i,
        'robust_pick_t_sec': robust_pick_i.astype(np.float32) * np.float32(0.004),
        'robust_conf': np.array([0.8, 0.4, 0.9, 0.6], dtype=np.float32),
        'robust_source': np.array(
            [
                ROBUST_SOURCE_COARSE_OBSERVED,
                ROBUST_SOURCE_THEORETICAL,
                ROBUST_SOURCE_COARSE_OBSERVED,
                ROBUST_SOURCE_TREND_FILL,
            ],
            dtype=np.uint8,
        ),
        'used_theoretical_mask': np.array([False, True, False, False], dtype=np.bool_),
        'reason_mask': np.array([0, 0, REASON_MASK_LOW_SCORE, 0], dtype=np.uint8),
        'conf_prob1': np.array([0.8, 0.4, 0.9, 0.6], dtype=np.float32),
        'conf_trend1': np.array([0.8, 0.4, 0.9, 0.6], dtype=np.float32),
        'conf_rs1': np.array([0.8, 0.4, 0.9, 0.6], dtype=np.float32),
        'lineage': np.asarray('{"stage":"robust"}'),
    }


def _make_fine_payload() -> dict[str, np.ndarray]:
    window_start_i = np.array([-23, -3, 17, 37], dtype=np.int32)
    fine_pick_local_i = np.array([128, 129, 10, 200], dtype=np.int32)
    fine_pick_local_f = fine_pick_local_i.astype(np.float32)
    final_pick_f = window_start_i.astype(np.float32) + fine_pick_local_f
    final_pick_i = (
        window_start_i.astype(np.int64) + fine_pick_local_i.astype(np.int64)
    ).astype(np.int32)
    final_pick_t_sec = final_pick_f * np.float32(0.004)
    fine_pmax = np.array([0.95, 0.2, 0.7, 0.9], dtype=np.float32)
    payload = {
        'dt_sec': np.asarray(0.004, dtype=np.float32),
        'n_samples_orig': np.asarray(512, dtype=np.int32),
        'n_traces': np.asarray(4, dtype=np.int32),
        'trace_indices': np.arange(4, dtype=np.int64),
        'fine_pick_local_i': fine_pick_local_i,
        'fine_pick_local_f': fine_pick_local_f,
        'fine_pmax': fine_pmax,
        'final_pick_i': final_pick_i,
        'final_pick_f': final_pick_f,
        'final_pick_t_sec': final_pick_t_sec,
        'final_conf': fine_pmax.copy(),
        'window_start_i': window_start_i,
        'window_end_i': (window_start_i + 256).astype(np.int32),
        'fine_window_valid_mask': np.ones(4, dtype=np.bool_),
    }
    validate_fine_result_payload(payload)
    return payload


def _make_final_payload() -> dict[str, np.ndarray]:
    return build_fbpick_final_payload(
        coarse_payload=_make_coarse_payload(),
        robust_payload=_make_robust_payload(),
        fine_payload=_make_fine_payload(),
        high_conf_threshold=0.5,
        lineage=_make_lineage(),
    )


def test_validate_fbpick_final_payload_requires_keys_and_exact_dtypes() -> None:
    payload = _make_final_payload()

    missing = dict(payload)
    missing.pop('final_conf')
    with pytest.raises(KeyError):
        validate_fbpick_final_payload(missing, high_conf_threshold=0.5)

    wrong_dtype = dict(payload)
    wrong_dtype['final_conf'] = payload['final_conf'].astype(np.float64)
    with pytest.raises(ValueError, match='final_conf dtype must be float32'):
        validate_fbpick_final_payload(wrong_dtype, high_conf_threshold=0.5)


def test_save_and_load_fbpick_final_npz_round_trip(tmp_path: Path) -> None:
    payload = _make_final_payload()
    out_path = save_fbpick_final_npz(tmp_path / 'roundtrip.fbpick_final.npz', **payload)
    loaded = load_fbpick_final_npz(out_path)

    assert set(FINAL_REQUIRED_KEYS).issubset(loaded.keys())
    for key, value in payload.items():
        if key == 'lineage':
            assert np.asarray(loaded[key]).item() == np.asarray(value).item()
        else:
            np.testing.assert_array_equal(loaded[key], value)


def test_build_fbpick_final_payload_merges_and_applies_v0_rules() -> None:
    payload = _make_final_payload()

    np.testing.assert_array_equal(
        payload['coarse_pick_i'],
        np.array([105, 125, 145, 165], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        payload['robust_pick_i'],
        np.array([105, 125, 145, 165], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        payload['window_start_i'],
        np.array([-23, -3, 17, 37], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        payload['window_end_i'],
        np.array([232, 252, 272, 292], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        payload['final_pick_i'],
        np.array([105, 126, 27, 237], dtype=np.int32),
    )
    np.testing.assert_allclose(
        payload['final_pick_f'],
        np.array([105.0, 126.0, 27.0, 237.0], dtype=np.float32),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        payload['final_pick_t_sec'],
        np.array([0.42, 0.504, 0.108, 0.948], dtype=np.float32),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        payload['final_conf'],
        np.array([0.8, 0.2, 0.7, 0.6], dtype=np.float32),
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(
        payload['reject_mask'],
        np.array([False, True, True, True], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        payload['high_conf_mask'],
        np.array([True, False, False, False], dtype=np.bool_),
    )


def test_build_fbpick_final_payload_rejects_trace_indices_mismatch() -> None:
    coarse = _make_coarse_payload()
    robust = _make_robust_payload()
    fine = _make_fine_payload()
    fine['trace_indices'] = np.array([0, 1, 99, 3], dtype=np.int64)

    with pytest.raises(ValueError, match='trace_indices must match across'):
        build_fbpick_final_payload(
            coarse_payload=coarse,
            robust_payload=robust,
            fine_payload=fine,
            high_conf_threshold=0.5,
            lineage=_make_lineage(),
        )


@pytest.mark.parametrize(
    ('key', 'value', 'message'),
    [
        ('n_traces', np.asarray(5, dtype=np.int32), 'n_traces must match across'),
        ('dt_sec', np.asarray(0.002, dtype=np.float32), 'dt_sec must match across'),
        (
            'n_samples_orig',
            np.asarray(1024, dtype=np.int32),
            'n_samples_orig must match across',
        ),
    ],
)
def test_build_fbpick_final_payload_rejects_common_scalar_mismatches(
    key: str,
    value: np.ndarray,
    message: str,
) -> None:
    coarse = _make_coarse_payload()
    robust = _make_robust_payload()
    fine = _make_fine_payload()
    robust[key] = value

    with pytest.raises(ValueError, match=message):
        build_fbpick_final_payload(
            coarse_payload=coarse,
            robust_payload=robust,
            fine_payload=fine,
            high_conf_threshold=0.5,
            lineage=_make_lineage(),
        )
