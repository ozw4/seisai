import numpy as np
import pytest
from seisai_transforms.view_projection import project_pick_csr_view


def _csr_from_rows(
    rows: list[list[int]], *, dtype=np.int64
) -> tuple[np.ndarray, np.ndarray]:
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    lengths = np.array([len(r) for r in rows], dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    data = np.array([v for r in rows for v in r], dtype=dtype)
    return indptr, data


def test_project_pick_csr_view_hflip_only() -> None:
    H, W = 4, 100
    rows = [[11], [22, 23], [33], [44, 45, 46]]
    indptr, data = _csr_from_rows(rows)

    indptr_v, data_v = project_pick_csr_view(
        indptr, data, H=H, W=W, meta={'hflip': True}
    )

    exp_rows = [[44, 45, 46], [33], [22, 23], [11]]
    exp_indptr, exp_data = _csr_from_rows(exp_rows)
    np.testing.assert_array_equal(indptr_v, exp_indptr)
    np.testing.assert_array_equal(data_v, exp_data)


def test_project_pick_csr_view_factor_h_only_expand() -> None:
    # factor_h=2.0, H=5 の最近傍リサンプル: src indices -> [1,2,2,3,3]
    H, W = 5, 100
    rows = [[10], [20], [30], [40], [50]]
    indptr, data = _csr_from_rows(rows)

    indptr_v, data_v = project_pick_csr_view(
        indptr, data, H=H, W=W, meta={'factor_h': 2.0}
    )

    exp_rows = [[20], [30], [30], [40], [40]]
    exp_indptr, exp_data = _csr_from_rows(exp_rows)
    np.testing.assert_array_equal(indptr_v, exp_indptr)
    np.testing.assert_array_equal(data_v, exp_data)


def test_project_pick_csr_view_factor_and_start_only() -> None:
    H, W = 2, 20
    rows = [[1, 2, 3], [10]]
    indptr, data = _csr_from_rows(rows)

    indptr_v, data_v = project_pick_csr_view(
        indptr, data, H=H, W=W, meta={'factor': 2.0, 'start': 2}
    )

    exp_rows = [[2, 4], [18]]  # [1,2,3] -> [0,2,4] drop 0
    exp_indptr, exp_data = _csr_from_rows(exp_rows)
    np.testing.assert_array_equal(indptr_v, exp_indptr)
    np.testing.assert_array_equal(data_v, exp_data)


def test_project_pick_csr_view_combined_hflip_factor_h_factor_start() -> None:
    H, W = 5, 20
    rows = [[1], [2], [3], [4], [5]]
    indptr, data = _csr_from_rows(rows)

    meta = {'hflip': True, 'factor_h': 0.5, 'factor': 2.0, 'start': 1}
    indptr_v, data_v = project_pick_csr_view(indptr, data, H=H, W=W, meta=meta)

    # H direction:
    # factor_h=0.5 -> j=[0,0,2,4,4]; hflip -> src_map=H-1-j=[4,4,2,0,0]
    # picks: [5,5,3,1,1]
    # T direction: round(p*2)-1 -> [9,9,5,1,1]
    exp_rows = [[9], [9], [5], [1], [1]]
    exp_indptr, exp_data = _csr_from_rows(exp_rows)
    np.testing.assert_array_equal(indptr_v, exp_indptr)
    np.testing.assert_array_equal(data_v, exp_data)


def test_project_pick_csr_view_drop_out_of_range_and_non_positive() -> None:
    H, W = 2, 5
    rows = [[0, -1, 5, 6], [2, 5]]  # trace0 becomes empty, trace1 keeps [2]
    indptr, data = _csr_from_rows(rows)

    indptr_v, data_v = project_pick_csr_view(indptr, data, H=H, W=W, meta={})

    exp_rows = [[], [2]]
    exp_indptr, exp_data = _csr_from_rows(exp_rows)
    np.testing.assert_array_equal(indptr_v, exp_indptr)
    np.testing.assert_array_equal(data_v, exp_data)


def test_project_pick_csr_view_invalid_meta_raises() -> None:
    H, W = 1, 10
    indptr, data = _csr_from_rows([[1]])

    with pytest.raises(ValueError):
        project_pick_csr_view(indptr, data, H=H, W=W, meta={'factor': 0.0})
