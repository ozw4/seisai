import numpy as np
from seisai_dataset.builder.builder import PhasePSNMap


def _csr_from_rows(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    lengths = np.array([len(r) for r in rows], dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    data = np.array([v for r in rows for v in r], dtype=np.int64)
    return indptr, data


def test_psn_map_basic_contract_and_normalization() -> None:
    H, W = 3, 32
    rng = np.random.default_rng(0)

    p_rows = [[5], [0, -1, 10], []]  # <=0 are invalid
    s_rows = [[], [12], []]
    p_indptr, p_data = _csr_from_rows(p_rows)
    s_indptr, s_data = _csr_from_rows(s_rows)

    sample = {
        'x_view': rng.normal(size=(H, W)).astype(np.float32),
        'meta': {'trace_valid': np.ones(H, dtype=np.bool_)},
        'p_indptr': p_indptr,
        'p_data': p_data,
        's_indptr': s_indptr,
        's_data': s_data,
    }

    op = PhasePSNMap(dst='psn_map', sigma=2.0)
    op(sample)

    y = sample['psn_map']
    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float32
    assert y.shape == (3, H, W)
    assert np.all(y >= 0.0)

    sum_hw = y.sum(axis=0)
    np.testing.assert_allclose(
        sum_hw, np.ones((H, W), dtype=np.float32), atol=1e-6, rtol=0
    )

    label_valid = sample['label_valid']
    assert label_valid.dtype == np.bool_
    np.testing.assert_array_equal(
        label_valid, np.array([True, True, False], dtype=np.bool_)
    )


def test_psn_map_out_of_range_picks_drop_and_trace_becomes_invalid() -> None:
    H, W = 2, 8
    rng = np.random.default_rng(0)

    # trace0: picks are invalid after projection (<=0 or >=W)
    p_rows = [[0, -1, W, W + 1], [2]]
    s_rows = [[], []]
    p_indptr, p_data = _csr_from_rows(p_rows)
    s_indptr, s_data = _csr_from_rows(s_rows)

    sample = {
        'x_view': rng.normal(size=(H, W)).astype(np.float32),
        'meta': {'trace_valid': np.ones(H, dtype=np.bool_)},
        'p_indptr': p_indptr,
        'p_data': p_data,
        's_indptr': s_indptr,
        's_data': s_data,
    }

    op = PhasePSNMap(dst='psn_map', sigma=1.5)
    op(sample)

    label_valid = sample['label_valid']
    np.testing.assert_array_equal(label_valid, np.array([False, True], dtype=np.bool_))

    y = sample['psn_map']
    # For trace0, no picks -> Noise=1 everywhere (after normalization).
    np.testing.assert_allclose(y[2, 0], np.ones(W, dtype=np.float32), atol=1e-6, rtol=0)


def test_psn_map_overlap_p_and_s_is_renormalized() -> None:
    H, W = 1, 16
    rng = np.random.default_rng(0)

    p_indptr, p_data = _csr_from_rows([[5]])
    s_indptr, s_data = _csr_from_rows([[5]])

    sample = {
        'x_view': rng.normal(size=(H, W)).astype(np.float32),
        'meta': {'trace_valid': np.ones(H, dtype=np.bool_)},
        'p_indptr': p_indptr,
        'p_data': p_data,
        's_indptr': s_indptr,
        's_data': s_data,
    }

    op = PhasePSNMap(dst='psn_map', sigma=1.0)
    op(sample)

    y = sample['psn_map']
    # At the shared peak bin, P and S should split and Noise should be 0.
    assert float(y[0, 0, 5]) == 0.5
    assert float(y[1, 0, 5]) == 0.5
    assert float(y[2, 0, 5]) == 0.0


def test_psn_map_label_valid_respects_trace_valid() -> None:
    H, W = 2, 16
    rng = np.random.default_rng(0)

    p_indptr, p_data = _csr_from_rows([[5], [6]])
    s_indptr, s_data = _csr_from_rows([[], []])

    sample = {
        'x_view': rng.normal(size=(H, W)).astype(np.float32),
        'meta': {'trace_valid': np.array([True, False], dtype=np.bool_)},
        'p_indptr': p_indptr,
        'p_data': p_data,
        's_indptr': s_indptr,
        's_data': s_data,
    }

    op = PhasePSNMap(dst='psn_map', sigma=2.0)
    op(sample)

    label_valid = sample['label_valid']
    np.testing.assert_array_equal(label_valid, np.array([True, False], dtype=np.bool_))


def test_psn_map_accepts_3d_x_view_channels_first() -> None:
    H, W = 2, 16
    rng = np.random.default_rng(0)

    p_indptr, p_data = _csr_from_rows([[5], []])
    s_indptr, s_data = _csr_from_rows([[], [6]])

    sample = {
        'x_view': rng.normal(size=(2, H, W)).astype(np.float32),  # (C,H,W)
        'meta': {'trace_valid': np.ones(H, dtype=np.bool_)},
        'p_indptr': p_indptr,
        'p_data': p_data,
        's_indptr': s_indptr,
        's_data': s_data,
    }

    op = PhasePSNMap(dst='psn_map', sigma=2.0)
    op(sample)

    y = sample['psn_map']
    assert y.shape == (3, H, W)
