import numpy as np
import pytest
from seisai_dataset.phase_pick_io import (
    csr_first_positive,
    invalidate_s_by_first,
    load_phase_pick_csr_npz,
    pad_csr,
    subset_csr,
    subset_pad_first_invalidate,
    validate_csr,
)


def test_validate_csr_rejects_invalid_inputs():
    # indptr must be 1D
    with pytest.raises(ValueError):
        validate_csr(indptr=np.array([[0, 1]]), data=np.array([1], dtype=np.int64))

    # indptr[0] must be 0
    with pytest.raises(ValueError):
        validate_csr(
            indptr=np.array([1, 1], dtype=np.int64),
            data=np.array([1], dtype=np.int64),
        )

    # indptr must be monotonic non-decreasing
    with pytest.raises(ValueError):
        validate_csr(
            indptr=np.array([0, 2, 1], dtype=np.int64),
            data=np.array([1, 2], dtype=np.int64),
        )

    # indptr[-1] must equal len(data)
    with pytest.raises(ValueError):
        validate_csr(
            indptr=np.array([0, 1, 1], dtype=np.int64),
            data=np.array([1, 2], dtype=np.int64),
        )

    # data must be 1D
    with pytest.raises(ValueError):
        validate_csr(
            indptr=np.array([0, 0], dtype=np.int64),
            data=np.array([[1]], dtype=np.int64),
        )

    # data must be integer dtype
    with pytest.raises(ValueError):
        validate_csr(
            indptr=np.array([0, 1], dtype=np.int64),
            data=np.array([1.0], dtype=np.float32),
        )


def test_subset_returns_expected_csr_and_first():
    # 4 traces (rows)
    p_indptr = np.array([0, 2, 2, 3, 5], dtype=np.int64)
    p_data = np.array([10, 5, 2, 0, 7], dtype=np.int64)  # includes invalid 0

    s_indptr = np.array([0, 1, 2, 2, 3], dtype=np.int64)
    s_data = np.array([12, 3, 6], dtype=np.int64)

    indices = np.array([3, 0, 2], dtype=np.int64)

    p_ip, p_d = subset_csr(indptr=p_indptr, data=p_data, indices=indices, name='p')
    s_ip, s_d = subset_csr(indptr=s_indptr, data=s_data, indices=indices, name='s')

    np.testing.assert_array_equal(p_ip, np.array([0, 2, 4, 5], dtype=np.int64))
    np.testing.assert_array_equal(p_d, np.array([0, 7, 10, 5, 2], dtype=np.int64))

    np.testing.assert_array_equal(s_ip, np.array([0, 1, 2, 2], dtype=np.int64))
    np.testing.assert_array_equal(s_d, np.array([6, 12], dtype=np.int64))

    p_first = csr_first_positive(indptr=p_ip, data=p_d)
    s_first = csr_first_positive(indptr=s_ip, data=s_d)
    np.testing.assert_array_equal(p_first, np.array([7, 5, 2], dtype=np.int64))
    np.testing.assert_array_equal(s_first, np.array([6, 12, 0], dtype=np.int64))


def test_pad_adds_empty_traces_and_first_becomes_zero():
    p_ip = np.array([0, 2, 4, 5], dtype=np.int64)
    p_d = np.array([0, 7, 10, 5, 2], dtype=np.int64)

    s_ip = np.array([0, 1, 2, 2], dtype=np.int64)
    s_d = np.array([6, 12], dtype=np.int64)

    p_ip2, p_d2 = pad_csr(indptr=p_ip, data=p_d, n_traces=5, name='p')
    s_ip2, s_d2 = pad_csr(indptr=s_ip, data=s_d, n_traces=5, name='s')

    np.testing.assert_array_equal(p_ip2, np.array([0, 2, 4, 5, 5, 5], dtype=np.int64))
    np.testing.assert_array_equal(p_d2, p_d)

    np.testing.assert_array_equal(s_ip2, np.array([0, 1, 2, 2, 2, 2], dtype=np.int64))
    np.testing.assert_array_equal(s_d2, s_d)

    p_first = csr_first_positive(indptr=p_ip2, data=p_d2)
    s_first = csr_first_positive(indptr=s_ip2, data=s_d2)
    np.testing.assert_array_equal(p_first, np.array([7, 5, 2, 0, 0], dtype=np.int64))
    np.testing.assert_array_equal(s_first, np.array([6, 12, 0, 0, 0], dtype=np.int64))


def test_s_first_lt_p_first_invalidates_s_slice_and_first():
    # H=2
    p_ip = np.array([0, 1, 2], dtype=np.int64)
    p_d = np.array([10, 5], dtype=np.int64)

    s_ip = np.array([0, 1, 2], dtype=np.int64)
    s_d = np.array([7, 9], dtype=np.int64)

    p_first = csr_first_positive(indptr=p_ip, data=p_d)
    s_first = csr_first_positive(indptr=s_ip, data=s_d)

    s_ip2, s_d2, s_first2 = invalidate_s_by_first(
        p_first=p_first,
        s_indptr=s_ip,
        s_data=s_d,
        s_first=s_first,
    )

    np.testing.assert_array_equal(s_ip2, np.array([0, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(s_d2, np.array([9], dtype=np.int64))
    np.testing.assert_array_equal(s_first2, np.array([0, 9], dtype=np.int64))


def test_non_positive_picks_are_invalid_for_first():
    indptr = np.array([0, 3, 3], dtype=np.int64)  # H=2
    data = np.array([-3, 0, -1], dtype=np.int64)
    first = csr_first_positive(indptr=indptr, data=data)
    np.testing.assert_array_equal(first, np.array([0, 0], dtype=np.int64))


def test_helpers_reject_non_integer_dtype_inputs():
    # subset_csr: float inputs must not be silently cast
    with pytest.raises(ValueError):
        subset_csr(
            indptr=np.array([0, 1], dtype=np.float32),
            data=np.array([1], dtype=np.int64),
            indices=np.array([0], dtype=np.int64),
        )
    with pytest.raises(ValueError):
        subset_csr(
            indptr=np.array([0, 1], dtype=np.int64),
            data=np.array([1.0], dtype=np.float32),
            indices=np.array([0], dtype=np.int64),
        )
    with pytest.raises(ValueError):
        subset_csr(
            indptr=np.array([0, 1], dtype=np.int64),
            data=np.array([1], dtype=np.int64),
            indices=np.array([0.0], dtype=np.float32),
        )

    # csr_first_positive: float data must not be silently cast
    with pytest.raises(ValueError):
        csr_first_positive(
            indptr=np.array([0, 1], dtype=np.int64),
            data=np.array([1.0], dtype=np.float32),
        )

    # invalidate_s_by_first: non-integer s_first must be rejected
    with pytest.raises(ValueError):
        invalidate_s_by_first(
            p_first=np.array([1, 2], dtype=np.int64),
            s_indptr=np.array([0, 0, 0], dtype=np.int64),
            s_data=np.array([], dtype=np.int64),
            s_first=np.array([0.0, 0.0], dtype=np.float32),
        )


def test_load_phase_pick_csr_npz_missing_keys_dtype_and_validate(tmp_path):
    # missing key(s)
    path_missing = tmp_path / 'missing_keys.npz'
    np.savez(
        path_missing,
        p_indptr=np.array([0, 0], dtype=np.int32),
        p_data=np.array([], dtype=np.int32),
        s_indptr=np.array([0, 0], dtype=np.int32),
    )
    with pytest.raises(ValueError, match='missing key'):
        _ = load_phase_pick_csr_npz(path_missing)

    # dtype conversion to int64 (and validate linkage)
    path_ok = tmp_path / 'ok.npz'
    np.savez(
        path_ok,
        p_indptr=np.array([0, 2, 2], dtype=np.int32),
        p_data=np.array([5, 0], dtype=np.int32),
        s_indptr=np.array([0, 1, 1], dtype=np.int32),
        s_data=np.array([7], dtype=np.int32),
    )
    csr = load_phase_pick_csr_npz(path_ok)
    assert csr.p_indptr.dtype == np.int64
    assert csr.p_data.dtype == np.int64
    assert csr.s_indptr.dtype == np.int64
    assert csr.s_data.dtype == np.int64

    # validate propagation: indptr[-1] must equal len(data)
    path_bad = tmp_path / 'bad.npz'
    np.savez(
        path_bad,
        p_indptr=np.array([0, 1], dtype=np.int32),
        p_data=np.array([1, 2], dtype=np.int32),
        s_indptr=np.array([0, 0], dtype=np.int32),
        s_data=np.array([], dtype=np.int32),
    )
    with pytest.raises(ValueError, match=r'indptr\[-1\].*len\(data\)'):
        _ = load_phase_pick_csr_npz(path_bad)


def test_subset_pad_first_invalidate_end_to_end():
    # 4 traces total
    # P picks:
    # t0: [10]
    # t1: [5, 0]  (includes invalid 0)
    # t2: []
    # t3: [2]
    p_indptr = np.array([0, 1, 3, 3, 4], dtype=np.int64)
    p_data = np.array([10, 5, 0, 2], dtype=np.int64)

    # S picks:
    # t0: [12]
    # t1: [3]     (will be invalidated because 3 < 5)
    # t2: []
    # t3: [0, 6]  (includes invalid 0, first should be 6)
    s_indptr = np.array([0, 1, 2, 2, 4], dtype=np.int64)
    s_data = np.array([12, 3, 0, 6], dtype=np.int64)

    indices = np.array([1, 3, 0], dtype=np.int64)
    out = subset_pad_first_invalidate(
        p_indptr=p_indptr,
        p_data=p_data,
        s_indptr=s_indptr,
        s_data=s_data,
        indices=indices,
        subset_traces=5,
    )

    np.testing.assert_array_equal(
        out.p_indptr, np.array([0, 2, 3, 4, 4, 4], dtype=np.int64)
    )
    np.testing.assert_array_equal(out.p_data, np.array([5, 0, 2, 10], dtype=np.int64))
    np.testing.assert_array_equal(
        out.p_first, np.array([5, 2, 10, 0, 0], dtype=np.int64)
    )

    # First trace's S gets invalidated (slice length becomes 0) and s_first becomes 0 there.
    np.testing.assert_array_equal(
        out.s_indptr, np.array([0, 0, 2, 3, 3, 3], dtype=np.int64)
    )
    np.testing.assert_array_equal(out.s_data, np.array([0, 6, 12], dtype=np.int64))
    np.testing.assert_array_equal(
        out.s_first, np.array([0, 6, 12, 0, 0], dtype=np.int64)
    )
