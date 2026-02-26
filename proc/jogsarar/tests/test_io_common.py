from __future__ import annotations

import numpy as np
import pytest
from common import npz_io


def test_require_npz_key_missing_raises_keyerror(tmp_path) -> None:
    p = tmp_path / 'x.npz'
    np.savez_compressed(p, present=np.asarray(1, dtype=np.int32))
    with np.load(p, allow_pickle=False) as z:
        with pytest.raises(KeyError):
            npz_io.require_npz_key(z, 'missing', context='unit')


def test_npz_scalar_int_type_and_ndim_validation(tmp_path) -> None:
    p = tmp_path / 'x.npz'
    np.savez_compressed(
        p,
        ok=np.asarray(7, dtype=np.int32),
        bad_type=np.asarray(7.0, dtype=np.float32),
        bad_ndim=np.asarray([1, 2], dtype=np.int32),
    )
    with np.load(p, allow_pickle=False) as z:
        assert npz_io.npz_scalar_int(z, 'ok', context='unit') == 7
        with pytest.raises(TypeError):
            npz_io.npz_scalar_int(z, 'bad_type', context='unit')
        with pytest.raises(ValueError):
            npz_io.npz_scalar_int(z, 'bad_ndim', context='unit')


def test_npz_scalar_float_type_and_ndim_validation(tmp_path) -> None:
    p = tmp_path / 'x.npz'
    np.savez_compressed(
        p,
        ok=np.asarray(0.25, dtype=np.float32),
        bad_type=np.asarray(3, dtype=np.int32),
        bad_ndim=np.asarray([0.1, 0.2], dtype=np.float32),
    )
    with np.load(p, allow_pickle=False) as z:
        assert npz_io.npz_scalar_float(z, 'ok', context='unit') == pytest.approx(0.25)
        with pytest.raises(TypeError):
            npz_io.npz_scalar_float(z, 'bad_type', context='unit')
        with pytest.raises(ValueError):
            npz_io.npz_scalar_float(z, 'bad_ndim', context='unit')


def test_npz_scalar_str_type_and_ndim_validation(tmp_path) -> None:
    p = tmp_path / 'x.npz'
    np.savez_compressed(
        p,
        ok=np.asarray(b'abc'),
        bad_type=np.asarray(3, dtype=np.int32),
        bad_ndim=np.asarray(['a', 'b']),
    )
    with np.load(p, allow_pickle=False) as z:
        assert npz_io.npz_scalar_str(z, 'ok', context='unit') == 'abc'
        with pytest.raises(TypeError):
            npz_io.npz_scalar_str(z, 'bad_type', context='unit')
        with pytest.raises(ValueError):
            npz_io.npz_scalar_str(z, 'bad_ndim', context='unit')


def test_npz_1d_length_mismatch_raises_valueerror(tmp_path) -> None:
    p = tmp_path / 'x.npz'
    np.savez_compressed(p, arr=np.asarray([1, 2, 3], dtype=np.int32))
    with np.load(p, allow_pickle=False) as z:
        with pytest.raises(ValueError):
            npz_io.npz_1d(z, 'arr', context='unit', n=2, dtype=np.int64)


class _FakeTraceAccessor:
    def __init__(self, raw: np.ndarray) -> None:
        self.raw = raw


class _FakeSegy:
    def __init__(self, raw: np.ndarray) -> None:
        self.trace = _FakeTraceAccessor(raw)


def test_is_contiguous_basic_cases() -> None:
    pytest.importorskip('segyio')
    from common import segy_io

    assert segy_io.is_contiguous(np.asarray([], dtype=np.int64))
    assert segy_io.is_contiguous(np.asarray([3], dtype=np.int64))
    assert segy_io.is_contiguous(np.asarray([2, 3, 4], dtype=np.int64))
    assert not segy_io.is_contiguous(np.asarray([2, 4, 5], dtype=np.int64))


def test_load_traces_by_indices_contiguous_and_non_contiguous() -> None:
    pytest.importorskip('segyio')
    from common import segy_io

    raw = np.arange(40, dtype=np.float64).reshape(5, 8)
    fake = _FakeSegy(raw)

    idx_cont = np.asarray([1, 2, 3], dtype=np.int64)
    got_cont = segy_io.load_traces_by_indices(fake, idx_cont)
    assert got_cont.shape == (3, 8)
    assert got_cont.ndim == 2
    assert got_cont.dtype == np.float32
    np.testing.assert_allclose(got_cont, raw[1:4].astype(np.float32))

    idx_non = np.asarray([3, 1, 4], dtype=np.int64)
    got_non = segy_io.load_traces_by_indices(fake, idx_non)
    assert got_non.shape == (3, 8)
    assert got_non.ndim == 2
    assert got_non.dtype == np.float32
    np.testing.assert_allclose(got_non, raw[idx_non].astype(np.float32))

    idx_single = np.asarray([2], dtype=np.int64)
    got_single = segy_io.load_traces_by_indices(fake, idx_single)
    assert got_single.shape == (1, 8)
    assert got_single.ndim == 2
    assert got_single.dtype == np.float32
