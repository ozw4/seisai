from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('segyio')

import stage2_make_psn512_windows as st2


def test_percentile_threshold_ignores_non_finite() -> None:
    x = np.asarray([np.nan, np.inf, -np.inf, 1.0, 3.0], dtype=np.float32)
    th = st2._percentile_threshold(x, frac=0.5)
    assert th == 2.0

    th2 = st2._percentile_threshold(
        np.asarray([np.nan, np.inf], dtype=np.float32), frac=0.5
    )
    assert np.isnan(th2)


def test_build_keep_mask_with_explicit_thresholds_sets_reason_bits() -> None:
    cfg = replace(st2.DEFAULT_STAGE2_CFG, half_win=128)

    pick_final_i = np.asarray([10, 0, 20, 30, 40], dtype=np.int64)
    trend_center_i = np.asarray([12.0, 12.0, np.nan, 30.0, 1000.0], dtype=np.float32)
    n_samples_in = 2000

    scores = {
        'conf_prob1': np.asarray([0.6, 0.1, 0.2, 0.4, 0.9], dtype=np.float32),
        'conf_trend1': np.asarray([0.6, 0.1, 0.2, 0.6, 0.9], dtype=np.float32),
        'conf_rs1': np.asarray([0.4, 0.9, 0.9, 0.6, 0.9], dtype=np.float32),
    }
    thresholds = {'conf_prob1': 0.5, 'conf_trend1': 0.5, 'conf_rs1': 0.5}

    keep, thresholds_used, reason, base_valid = st2.build_keep_mask(
        pick_final_i=pick_final_i,
        trend_center_i=trend_center_i,
        n_samples_in=n_samples_in,
        scores=scores,
        thresholds=thresholds,
        cfg=cfg,
    )

    assert base_valid.tolist() == [True, False, False, True, False]
    assert keep.tolist() == [False, False, False, False, False]
    assert thresholds_used == thresholds

    assert reason.dtype == np.uint8

    # invalid pick -> bit0 must be set (and bit2 is also set by implementation)
    assert (int(reason[1]) & (1 << 0)) != 0
    assert (int(reason[1]) & (1 << 2)) != 0

    # trend missing -> bit1 must be set (and bit2 is also set by implementation)
    assert (int(reason[2]) & (1 << 1)) != 0
    assert (int(reason[2]) & (1 << 2)) != 0

    # pick outside window -> bit2 must be set
    assert (int(reason[4]) & (1 << 2)) != 0

    # base_valid で落ちたスコア由来のビット（これは排他的にしたいなら exact でもOK）
    assert int(reason[0]) == (1 << 5)  # conf_rs low
    assert int(reason[3]) == (1 << 3)  # conf_prob low


def test_build_keep_mask_computes_thresholds_when_none() -> None:
    cfg = replace(st2.DEFAULT_STAGE2_CFG, half_win=128, drop_low_frac=0.25)

    pick_final_i = np.asarray([10, 20, 30, 40], dtype=np.int64)
    trend_center_i = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    n_samples_in = 100

    scores = {
        'conf_prob1': np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        'conf_trend1': np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        'conf_rs1': np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    }

    keep, thresholds_used, reason, base_valid = st2.build_keep_mask(
        pick_final_i=pick_final_i,
        trend_center_i=trend_center_i,
        n_samples_in=n_samples_in,
        scores=scores,
        thresholds=None,
        cfg=cfg,
    )

    assert base_valid.tolist() == [True, True, True, True]

    expected_th = float(
        np.nanpercentile(np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32), 25.0)
    )
    assert thresholds_used == {
        'conf_prob1': expected_th,
        'conf_trend1': expected_th,
        'conf_rs1': expected_th,
    }

    assert keep.tolist() == [False, True, True, True]
    assert int(reason[0]) == ((1 << 3) | (1 << 4) | (1 << 5))


def test_build_keep_mask_missing_threshold_key_raises() -> None:
    cfg = replace(st2.DEFAULT_STAGE2_CFG, half_win=128)

    pick_final_i = np.asarray([10, 20], dtype=np.int64)
    trend_center_i = np.asarray([10.0, 20.0], dtype=np.float32)
    n_samples_in = 100

    scores = {
        'conf_prob1': np.asarray([1.0, 1.0], dtype=np.float32),
        'conf_trend1': np.asarray([1.0, 1.0], dtype=np.float32),
        'conf_rs1': np.asarray([1.0, 1.0], dtype=np.float32),
    }
    thresholds = {'conf_prob1': 0.5, 'conf_rs1': 0.5}

    with pytest.raises(KeyError) as ei:
        st2.build_keep_mask(
            pick_final_i=pick_final_i,
            trend_center_i=trend_center_i,
            n_samples_in=n_samples_in,
            scores=scores,
            thresholds=thresholds,
            cfg=cfg,
        )
    assert 'conf_trend1' in str(ei.value)
