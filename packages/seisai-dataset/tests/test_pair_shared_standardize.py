import numpy as np

from seisai_dataset.segy_gather_pair_dataset import _standardize_pair_from_input


def test_standardize_pair_from_input_uses_input_stats() -> None:
    x_in = np.array([[1, 2, 3], [0, 0, 0]], dtype=np.float32)
    x_tg = np.array([[10, 20, 30], [5, 5, 5]], dtype=np.float32)
    eps = 1e-8

    x_in_std, x_tg_std, mean, std = _standardize_pair_from_input(
        x_in, x_tg, eps=eps
    )

    xf = x_in.astype(np.float32, copy=False)
    exp_mean = xf.mean(axis=-1)
    exp_std = xf.std(axis=-1) + np.float32(eps)
    exp_in = (xf - exp_mean[:, None]) / exp_std[:, None]
    exp_tg = (x_tg.astype(np.float32, copy=False) - exp_mean[:, None]) / exp_std[
        :, None
    ]

    assert np.allclose(x_in_std, exp_in, rtol=0.0, atol=1e-6)
    assert np.allclose(x_tg_std, exp_tg, rtol=0.0, atol=1e-6)
    assert std[1] == np.float32(eps)


def test_standardize_pair_from_input_dtype_and_shape() -> None:
    x_in = np.zeros((3, 4), dtype=np.float64)
    x_tg = np.ones((3, 4), dtype=np.float64)
    eps = 1e-8

    x_in_std, x_tg_std, mean, std = _standardize_pair_from_input(
        x_in, x_tg, eps=eps
    )

    assert mean.dtype == np.float32
    assert std.dtype == np.float32
    assert mean.shape == (3,)
    assert std.shape == (3,)
    assert x_in_std.dtype == np.float32
    assert x_tg_std.dtype == np.float32
