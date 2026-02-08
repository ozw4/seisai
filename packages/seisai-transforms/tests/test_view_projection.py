import numpy as np
import pytest
from seisai_transforms.view_projection import (
    project_fb_idx_view,
    project_offsets_view,
    project_time_view,
)


def test_project_fb_idx_view_identity_and_invalid_zero():
    H, W = 3, 10
    fb = np.array([0, 1, 5], dtype=np.int64)  # 0 は無効扱い(<=0 が -1)
    out = project_fb_idx_view(fb, H=H, W=W, meta={})
    assert out.shape == (H,)
    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, np.array([-1, 1, 5], dtype=np.int64))


def test_project_fb_idx_view_start_and_factor():
    H, W = 3, 40
    fb = np.array([10, 11, 12], dtype=np.int64)
    meta = {'start': 2, 'factor': 2.0}  # round(fb*2)-2
    out = project_fb_idx_view(fb, H=H, W=W, meta=meta)
    np.testing.assert_array_equal(out, np.array([18, 20, 22], dtype=np.int64))


def test_project_fb_idx_view_hflip():
    H, W = 4, 10
    fb = np.array([1, 2, 3, 4], dtype=np.int64)
    out = project_fb_idx_view(fb, H=H, W=W, meta={'hflip': True})
    np.testing.assert_array_equal(out, np.array([4, 3, 2, 1], dtype=np.int64))


def test_project_fb_idx_view_factor_h_nearest_resample_propagates_invalid():
    # factor_h=2.0, H=5 の最近傍リサンプル:
    # src indices -> [1,2,2,3,3]
    H, W = 5, 100
    fb = np.array([10, -1, 12, 13, 14], dtype=np.int64)
    out = project_fb_idx_view(fb, H=H, W=W, meta={'factor_h': 2.0})
    np.testing.assert_array_equal(out, np.array([-1, 12, 12, 13, 13], dtype=np.int64))


def test_project_fb_idx_view_factor_h_shrink_nearest_resample():
    # factor_h=0.5, H=5 の最近傍リサンプル:
    # src indices -> [-2,0,2,4,6] -> clip -> [0,0,2,4,4]
    H, W = 5, 100
    fb = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    out = project_fb_idx_view(fb, H=H, W=W, meta={'factor_h': 0.5})
    np.testing.assert_array_equal(out, np.array([10, 10, 12, 14, 14], dtype=np.int64))


def test_project_fb_idx_view_length_mismatch_raises():
    with pytest.raises(ValueError):
        project_fb_idx_view(np.array([1, 2, 3], dtype=np.int64), H=4, W=10, meta={})


def test_project_offsets_view_identity_and_hflip():
    H = 3
    offsets = np.array([0.0, 10.0, 20.0], dtype=np.float32)
    out = project_offsets_view(offsets, H=H, meta={'hflip': True})
    assert out.shape == (H,)
    assert out.dtype == np.float32
    np.testing.assert_allclose(
        out, np.array([20.0, 10.0, 0.0], dtype=np.float32), atol=0, rtol=0
    )


def test_project_offsets_view_factor_h_linear_resample_expand():
    # offsets=[0,10,20,30,40], factor_h=2.0 の中心固定リサンプル:
    # src -> [1,1.5,2,2.5,3] なので [10,15,20,25,30]
    H = 5
    offsets = np.array([0, 10, 20, 30, 40], dtype=np.float32)
    out = project_offsets_view(offsets, H=H, meta={'factor_h': 2.0})
    np.testing.assert_allclose(
        out, np.array([10, 15, 20, 25, 30], dtype=np.float32), atol=1e-6, rtol=0
    )


def test_project_offsets_view_factor_h_linear_resample_shrink():
    # factor_h=0.5: src -> [-2,0,2,4,6] -> clip -> [0,0,2,4,4]
    H = 5
    offsets = np.array([0, 10, 20, 30, 40], dtype=np.float32)
    out = project_offsets_view(offsets, H=H, meta={'factor_h': 0.5})
    np.testing.assert_allclose(
        out, np.array([0, 0, 20, 40, 40], dtype=np.float32), atol=1e-6, rtol=0
    )


def test_project_offsets_view_invalid_factor_h_raises():
    with pytest.raises(ValueError):
        project_offsets_view(
            np.array([0, 1, 2], dtype=np.float32), H=3, meta={'factor_h': 0.0}
        )


def test_project_offsets_view_invalid_dim_raises():
    with pytest.raises(ValueError):
        project_offsets_view(np.zeros((2, 2), dtype=np.float32), H=2, meta={})


def test_project_time_view_factor_and_start():
    # time_1d の dt0 を使って tv = t0 + (arange(W)+start) * (dt0/factor)
    time_raw = (np.arange(8, dtype=np.float64) * 0.002).astype(
        np.float64
    )  # dt0=0.002, t0=0
    H, W = 5, 4
    meta = {'factor': 2.0, 'start': 3}
    out = project_time_view(time_raw, H=H, W=W, meta=meta)
    assert out.shape == (W,)
    assert out.dtype == np.float32
    np.testing.assert_allclose(
        out, np.array([0.003, 0.004, 0.005, 0.006], dtype=np.float32), atol=1e-7, rtol=0
    )


@pytest.mark.parametrize(
    'time_1d,H,W,meta',
    [
        (np.array([0.0], dtype=np.float32), 1, 4, {}),  # len < 2
        (np.zeros((2, 2), dtype=np.float32), 2, 4, {}),  # not 1D
        (np.array([0.0, 0.1], dtype=np.float32), 1, 4, {'factor': 0.0}),  # factor<=0
        (np.array([0.0, 0.1], dtype=np.float32), 1, 4, {'start': -1}),  # start<0
    ],
)
def test_project_time_view_invalid_inputs_raise(time_1d, H, W, meta):
    with pytest.raises(ValueError):
        project_time_view(time_1d, H=H, W=W, meta=meta)
