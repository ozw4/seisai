from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('segyio')

import stage4_psn512_infer_to_raw as st4


def test_post_trough_apply_mask_from_offsets_min_max() -> None:
    cfg = replace(
        st4.DEFAULT_STAGE4_CFG,
        post_trough_offs_abs_min_m=100.0,
        post_trough_offs_abs_max_m=200.0,
    )
    offs = np.asarray([-50.0, 100.0, 150.0, 250.0, np.nan], dtype=np.float32)
    m = st4._post_trough_apply_mask_from_offsets(offs, cfg=cfg)
    assert m.tolist() == [False, True, True, False, False]


def test_shift_pick_to_preceding_trough_after_pick_mode() -> None:
    x = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    p_out, peak_i, trough_i, _ath, peak_amp, trough_amp, reason = (
        st4._shift_pick_to_preceding_trough_1d(
            x,
            p=4,
            max_shift=2,
            scan_ahead=8,
            smooth_win=1,
            a_th=0.5,
            peak_search='after_pick',
        )
    )
    assert (p_out, peak_i, trough_i, reason) == (5, 6, 5, 'shifted')
    assert peak_amp > 0.9
    assert trough_amp < -0.9


def test_shift_pick_to_preceding_trough_before_pick_mode() -> None:
    x = np.asarray([0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    p_out, peak_i, trough_i, _ath, peak_amp, trough_amp, reason = (
        st4._shift_pick_to_preceding_trough_1d(
            x,
            p=5,
            max_shift=4,
            scan_ahead=4,
            smooth_win=1,
            a_th=0.5,
            peak_search='before_pick',
        )
    )
    assert (p_out, peak_i, trough_i, reason) == (2, 3, 2, 'shifted')
    assert peak_amp > 0.9
    assert trough_amp < -0.9


def test_post_trough_adjust_picks_vectorized_behavior() -> None:
    raw = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    picks = np.asarray([4, 0], dtype=np.int32)

    out = st4._post_trough_adjust_picks(
        picks,
        raw,
        max_shift=2,
        scan_ahead=8,
        smooth_win=1,
        a_th=0.5,
        peak_search='after_pick',
        apply_mask=None,
        debug=False,
        debug_label='t',
        debug_max_examples=0,
    )
    assert out.tolist() == [5, 0]


def test_align_post_trough_shifts_corrects_outlier_shift() -> None:
    p_in = np.asarray([10, 10, 10, 10, 10], dtype=np.int32)
    p_post = np.asarray([12, 12, 20, 12, 12], dtype=np.int32)

    # apply_mask は「補正対象」だけでなく「近傍medianに使うdの有効/無効」も決める。
    # outlier(2)を補正するには近傍(1,3)が support として有効である必要がある。
    # 端(0,4)は巻き込みを防ぐため無効化する。
    apply_mask = np.asarray([False, True, True, True, False], dtype=bool)

    out = st4._align_post_trough_shifts_to_neighbors(
        p_in,
        p_post,
        peak_search='after_pick',
        radius=1,  # outlier(2)の近傍が [2,2] になり、median=2 に安定
        min_support=2,  # outlierのみ補正され、1/3はsupport不足で補正されない
        max_dev=1,
        max_shift=16,
        propagate_zero=False,
        zero_pin_tol=2,
        apply_mask=apply_mask,
        debug=False,
        debug_label='t',
    )
    assert out.tolist() == [12, 12, 12, 12, 12]


def test_align_post_trough_shifts_propagates_zero_when_enabled() -> None:
    p_in = np.asarray([10, 10, 10], dtype=np.int32)
    p_post = np.asarray([12, 12, 10], dtype=np.int32)  # last shift=0

    out = st4._align_post_trough_shifts_to_neighbors(
        p_in,
        p_post,
        peak_search='after_pick',
        radius=1,
        min_support=1,
        max_dev=1,
        max_shift=16,
        propagate_zero=True,
        zero_pin_tol=0,
        apply_mask=None,
        debug=False,
        debug_label='t',
    )
    assert out.tolist() == [12, 12, 12]
