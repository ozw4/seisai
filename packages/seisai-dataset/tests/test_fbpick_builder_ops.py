import numpy as np
import pytest

from seisai_dataset import FBGaussMapMs, NormalizeOffsetByConst, NormalizeTimeByConst
from seisai_dataset.builder import FBGaussMap


def _expected_fb_map(*, H: int, W: int, fb_idx_view: np.ndarray, sigma: float) -> np.ndarray:
    sample = {
        'x_view': np.zeros((H, W), dtype=np.float32),
        'meta': {
            'fb_idx_view': np.asarray(fb_idx_view, dtype=np.int64),
        },
    }
    FBGaussMap(dst='expected', sigma=float(sigma), src='fb_idx_view')(sample)
    return sample['expected']


def test_fb_gauss_map_ms_prefers_dt_eff_sec_and_zeros_invalid_rows() -> None:
    H, W = 3, 33
    sample = {
        'x_view': np.zeros((H, W), dtype=np.float32),
        'meta': {
            'fb_idx_view': np.array([12, 12, 40], dtype=np.int64),
            'trace_valid': np.array([True, False, True], dtype=np.bool_),
            'dt_sec': 0.010,
            'dt_eff_sec': 0.002,
        },
    }

    op = FBGaussMapMs(
        dst='y_fb_map',
        src='fb_idx_view',
        sigma_ms=10.0,
        sigma_samples_min=0.25,
        sigma_samples_max=100.0,
    )
    op(sample)

    expected = _expected_fb_map(
        H=H,
        W=W,
        fb_idx_view=np.array([12, 12, 40], dtype=np.int64),
        sigma=5.0,
    )
    wrong_sigma = _expected_fb_map(
        H=H,
        W=W,
        fb_idx_view=np.array([12, 12, 40], dtype=np.int64),
        sigma=1.0,
    )

    np.testing.assert_allclose(sample['y_fb_map'][0], expected[0], atol=1e-6)
    assert not np.allclose(sample['y_fb_map'][0], wrong_sigma[0], atol=1e-6)
    np.testing.assert_allclose(sample['y_fb_map'][1], np.zeros((W,), dtype=np.float32))
    np.testing.assert_allclose(sample['y_fb_map'][2], np.zeros((W,), dtype=np.float32))


@pytest.mark.parametrize(
    ('meta', 'expected_sigma'),
    [
        ({'dt_sec': 0.004}, 4.0),
        ({'dt_eff_sec': 0.0001, 'dt_sec': 0.004}, 40.0),
    ],
)
def test_fb_gauss_map_ms_clamps_sigma_samples(
    meta: dict[str, float],
    expected_sigma: float,
) -> None:
    H, W = 2, 25
    sample = {
        'x_view': np.zeros((H, W), dtype=np.float32),
        'meta': {
            'fb_idx_view': np.array([8, 14], dtype=np.int64),
            'trace_valid': np.array([True, True], dtype=np.bool_),
            **meta,
        },
    }

    op = FBGaussMapMs(
        dst='y_fb_map',
        src='fb_idx_view',
        sigma_ms=10.0,
        sigma_samples_min=4.0,
        sigma_samples_max=40.0,
    )
    op(sample)

    expected = _expected_fb_map(
        H=H,
        W=W,
        fb_idx_view=np.array([8, 14], dtype=np.int64),
        sigma=expected_sigma,
    )
    np.testing.assert_allclose(sample['y_fb_map'], expected, atol=1e-6)


def test_normalize_time_and_offset_by_const_apply_clipping() -> None:
    sample = {
        'time_ch_raw': np.array([[0.0, 2.0, 8.0]], dtype=np.float32),
        'offset_ch_raw': np.array([[-500.0, 1000.0, -4000.0]], dtype=np.float32),
    }

    NormalizeTimeByConst(
        src='time_ch_raw',
        dst='time_ch',
        ref_sec=4.0,
        clip_lo=0.0,
        clip_hi=1.5,
    )(sample)
    NormalizeOffsetByConst(
        src='offset_ch_raw',
        dst='offset_ch',
        ref_m=1000.0,
        use_abs=True,
        clip_lo=0.0,
        clip_hi=1.5,
    )(sample)

    np.testing.assert_allclose(
        sample['time_ch'],
        np.array([[0.0, 0.5, 1.5]], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        sample['offset_ch'],
        np.array([[0.5, 1.0, 1.5]], dtype=np.float32),
        atol=1e-6,
    )
