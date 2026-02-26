from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('seisai_pick')

import jogsarar_shared as js


def test_valid_pick_mask_basic() -> None:
    picks = np.asarray([np.nan, np.inf, -1.0, 0.0, 1.0, 5.0, 9.0], dtype=np.float32)

    m0 = js.valid_pick_mask(picks, n_samples=None, zero_is_invalid=True)
    assert m0.tolist() == [False, False, False, False, True, True, True]

    m1 = js.valid_pick_mask(picks, n_samples=None, zero_is_invalid=False)
    assert m1.tolist() == [False, False, False, True, True, True, True]


def test_valid_pick_mask_with_n_samples() -> None:
    picks = np.asarray([0.0, 1.0, 9.0, 10.0, 100.0], dtype=np.float32)
    m = js.valid_pick_mask(picks, n_samples=10, zero_is_invalid=False)
    assert m.tolist() == [True, True, True, False, False]


def test_build_pick_aligned_window_happy_path_and_fill() -> None:
    wave = np.asarray(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [10, 11, 12, 13, 14, 15, 16, 17],
            [20, 21, 22, 23, 24, 25, 26, 27],
        ],
        dtype=np.float32,
    )
    picks = np.asarray([3.0, 0.0, 1.0], dtype=np.float32)

    out = js.build_pick_aligned_window(wave, picks=picks, pre=2, post=3, fill=-9.0)
    assert out.shape == (3, 5)

    assert out[0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert out[1].tolist() == [-9.0, -9.0, -9.0, -9.0, -9.0]
    assert out[2].tolist() == [-9.0, 20.0, 21.0, 22.0, 23.0]


def test_build_groups_by_key_stable_order() -> None:
    values = np.asarray([2, 1, 2, 1, 1], dtype=np.int64)
    uniq, inv, groups = js.build_groups_by_key(values)

    assert uniq.tolist() == [1, 2]
    assert inv.tolist() == [1, 0, 1, 0, 0]
    assert [g.tolist() for g in groups] == [[1, 3, 4], [0, 2]]


def test_build_key_to_indices_matches_groups() -> None:
    values = np.asarray([7, 7, 3, 7, 3], dtype=np.int64)
    d = js.build_key_to_indices(values)

    assert set(d.keys()) == {3, 7}
    assert d[7].tolist() == [0, 1, 3]
    assert d[3].tolist() == [2, 4]
