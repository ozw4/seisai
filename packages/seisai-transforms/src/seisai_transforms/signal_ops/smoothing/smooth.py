import numpy as np
from seisai_utils.validator import validate_numpy


def smooth_1d_np(
    x: np.ndarray,
    window: int,
) -> np.ndarray:
    """1次元配列の移動平均による平滑化。.

    パラメータ
    ----------
    x_t : np.ndarray
        形状 (T,) の 1D 配列。
    window : int
        平滑化窓長(サンプル数)。1 以下ならコピーして返す。

    戻り値
    -------
    y_t : np.ndarray
        平滑化後の 1D 配列(形状 (T,))。
    """
    validate_numpy(x, allowed_ndims=(1,), name='x')
    if window <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    if window > x.size:
        msg = 'window must be <= length of x'
        raise ValueError(msg)

    x = np.asarray(x, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode='same')
