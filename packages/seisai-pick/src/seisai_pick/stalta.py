"""STA/LTA (Short-Term Average / Long-Term Average) utilities.

This module provides Numba-accelerated STA/LTA ratio computations for:
- 1D traces (`stalta_1d`)
- 2D arrays processed trace-by-trace in parallel (`stalta_2d`)
- Arbitrary-dimensional inputs via a convenience wrapper (`stalta`) using `StaltaParams`
"""

from typing import Any, NamedTuple

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


# === 1D: ユーザーの逐次和アルゴリズム(そのまま採用) =====================
@njit(cache=True, fastmath=True)
def stalta_1d(
    trc: NDArray[Any], ns: int = 10, nl: int = 100, eps: float = 1e-12
) -> NDArray[np.float64]:
    """Compute the STA/LTA ratio for a single 1D trace.

    Parameters
    ----------
    trc : numpy.ndarray
            1D input trace (time series).
    ns : int, default=10
            STA window length in samples.
    nl : int, default=100
            LTA window length in samples (must be >= ns).
    eps : float, default=1e-12
            Small value to avoid division by zero.

    Returns
    -------
    numpy.ndarray
            STA/LTA ratio array of the same length as `trc` with dtype float64.

    """
    tmax = trc.size
    R = np.zeros(tmax, dtype=np.float64)
    s_sta = 0.0
    s_lta = 0.0
    for i in range(tmax):
        x2 = float(trc[i]) * float(trc[i])
        s_sta += x2
        s_lta += x2
        if i >= ns:
            y = float(trc[i - ns])
            s_sta -= y * y
            sta = s_sta / ns
        else:
            sta = s_sta / (i + 1)
        if i >= nl:
            y = float(trc[i - nl])
            s_lta -= y * y
            lta = s_lta / nl
        else:
            lta = s_lta / (i + 1)
        R[i] = sta / lta if lta > eps else 0.0
    return R


# === (H,T): 2Dを各トレース独立に並列処理(時間=最後の軸) ==================
@njit(cache=True, fastmath=True, parallel=True)
def stalta_2d(
    x_2d: NDArray[Any], ns: int = 10, nl: int = 100, eps: float = 1e-12
) -> NDArray[np.float64]:
    """Compute STA/LTA ratio for a 2D array trace-by-trace in parallel.

    Parameters
    ----------
    x_2d : numpy.ndarray
            Input array of shape (H, T), where T is the time axis (last axis).
    ns : int, default=10
            STA window length in samples.
    nl : int, default=100
            LTA window length in samples (must be >= ns).
    eps : float, default=1e-12
            Small value to avoid division by zero.

    Returns
    -------
    numpy.ndarray
            STA/LTA ratio array of shape (H, T) with dtype float64.

    """
    H, T = x_2d.shape
    out = np.empty((H, T), dtype=np.float64)
    for h in prange(H):
        out[h, :] = stalta_1d(x_2d[h], ns, nl, eps)
    return out


# === 汎用ラッパ: 任意次元 (..., T) に対応 / axis で時間軸を指定 =================
class StaltaParams(NamedTuple):
    """Parameters for STA/LTA computation.

    Attributes
    ----------
    ns : int
            STA window length in samples.
    nl : int
            LTA window length in samples (must be >= ns).
    eps : float
            Small value to avoid division by zero.
    axis : int
            Time axis index in the input array.
    out_dtype : Any | None
            Output dtype; if None, an appropriate dtype is chosen automatically.

    """

    ns: int = 10
    nl: int = 100
    eps: float = 1e-12
    axis: int = -1
    out_dtype: Any | None = None


def stalta(x: Any, params: StaltaParams | None = None, **kwargs: Any) -> NDArray[Any]:
    """Compute STA/LTA ratio along a time axis.

    Parameters
    ----------
    x : array_like
            Input array with time along `axis`.
    params : StaltaParams | None
            Grouped parameters for STA/LTA computation; if None, defaults are used.
    **kwargs : Any
            Backward-compatible overrides for: ns, nl, eps, axis, out_dtype.

    Returns
    -------
    numpy.ndarray
            STA/LTA ratio array with the same shape as `x`.

    Raises
    ------
    ValueError
            If window sizes are invalid (ns>0, nl>=ns).
    TypeError
            If unexpected keyword arguments are provided.

    """
    if params is None:
        params = StaltaParams()

    if kwargs:
        allowed = set(StaltaParams._fields)
        unknown = set(kwargs) - allowed
        if unknown:
            msg = f'Unexpected keyword argument(s): {sorted(unknown)}'
            raise TypeError(msg)
        data = params._asdict()
        data.update(kwargs)
        params = StaltaParams(**data)

    ns = params.ns
    nl = params.nl
    eps = params.eps
    axis = params.axis
    out_dtype = params.out_dtype

    # ガード(サイレント劣化禁止)
    if not (
        isinstance(ns, int) and isinstance(nl, int) and ns > 0 and nl > 0 and nl >= ns
    ):
        msg = f'Invalid window sizes: ns={ns}, nl={nl}. Require integers, ns>0, nl>=ns.'
        raise ValueError(msg)

    x = np.asarray(x)
    # 1) 時間軸を末尾へ
    x_last = np.moveaxis(x, axis, -1)
    T = x_last.shape[-1]
    lead_shape = x_last.shape[:-1]

    # 2) 2D (N, T) へまとめる(C連続&float64でJITに優しい形に)
    x_2d = x_last.reshape(-1, T)
    x_2d = np.ascontiguousarray(x_2d, dtype=np.float64)

    # 3) Numba で各トレース処理
    r_2d = stalta_2d(x_2d, ns=ns, nl=nl, eps=eps)

    # 4) 元の形に戻す→元の axis に戻す
    r_last = r_2d.reshape((*lead_shape, T))
    r = np.moveaxis(r_last, -1, axis)

    # 5) 出力 dtype
    if out_dtype is None:
        out_dtype = x.dtype if x.dtype in (np.float32, np.float64) else np.float64
    return r.astype(out_dtype, copy=False)
