from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from seisai_utils.validator import validate_array
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

# ===== helpers =====


def _percentile(a: np.ndarray, q: float, axis: int, keepdims: bool) -> np.ndarray:
    assert isinstance(a, np.ndarray)
    assert 0.0 <= float(q) <= 100.0
    return np.percentile(a, float(q), axis=axis, keepdims=keepdims, method='linear')


def _robust_loc_scale(seg_nw: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """seg_nw: (N, Wseg).

    Returns:
      mu_n:   (N,)
      sigma_n:(N,)

    """
    assert isinstance(seg_nw, np.ndarray)
    assert seg_nw.ndim == 2
    if method == 'mad':
        mu = np.median(seg_nw, axis=1)
        mad = np.median(np.abs(seg_nw - mu[:, None]), axis=1)
        sigma = 1.4826 * mad
        return mu, sigma
    if method == 'iqr':
        mu = np.median(seg_nw, axis=1)
        q75 = _percentile(seg_nw, 75.0, axis=1, keepdims=False)
        q25 = _percentile(seg_nw, 25.0, axis=1, keepdims=False)
        sigma = (q75 - q25) / 1.349
        return mu, sigma
    msg = "method must be 'mad' or 'iqr'"
    raise ValueError(msg)


def _build_anchors(
    W: int, win: int, hop: int, causal: bool
) -> tuple[np.ndarray, Callable[[int], tuple[int, int]]]:
    assert int(W) > 0
    assert int(win) > 0
    assert int(hop) > 0
    W = int(W)
    win = int(win)
    hop = int(hop)

    if causal:
        anchors = np.arange(win // 2, W, hop, dtype=np.int64)

        def _bounds(t: int) -> tuple[int, int]:
            end = int(t)
            start = max(0, end - win)
            return start, end
    else:
        anchors = np.arange(0, W, hop, dtype=np.int64)

        def _bounds(center: int) -> tuple[int, int]:
            half = win // 2
            start = max(0, int(center) - half)
            end = min(W, start + win)
            start = max(0, end - win)
            return start, end

    if anchors.size == 0 or anchors[0] != 0:
        anchors = np.insert(anchors, 0, 0)
    if anchors[-1] != (W - 1):
        anchors = np.append(anchors, W - 1)
    return anchors, _bounds


def _interp_rows(
    anchors: np.ndarray,  # (K,)
    values_nk: np.ndarray,  # (N, K)
    tgrid: np.ndarray,  # (T,)
) -> np.ndarray:  # (N, T)
    assert anchors.ndim == 1
    assert values_nk.ndim == 2
    assert values_nk.shape[1] == anchors.size
    assert tgrid.ndim == 1

    idx = np.searchsorted(anchors, tgrid, side='right') - 1
    idx = np.clip(idx, 0, anchors.size - 2)

    x0 = anchors[idx]
    x1 = anchors[idx + 1]
    dx = x1 - x0
    dx[dx == 0] = 1.0

    t = tgrid.astype(np.float64, copy=False)
    alpha = (t - x0) / dx

    v0 = values_nk[:, idx]
    v1 = values_nk[:, idx + 1]
    alpha2d = alpha[None, :]

    return v0 + (v1 - v0) * alpha2d


# ===== main =====
def robust_agc_np(
    x: np.ndarray,
    *,
    win: int = 6000,
    hop: int | None = None,
    method: str = 'mad',
    gamma: float = 0.75,
    eps: float = 1e-8,
    clamp_pct: tuple[float, float] = (5.0, 95.0),
    causal: bool = False,
    chunk: int = 1_000_000,
    return_meta: bool = False,
    use_tqdm: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    validate_array(x, allowed_ndims=(1, 2, 3, 4), name='x', backend='numpy')

    Hdims = x.shape[:-1]
    W = int(x.shape[-1])

    win = int(win)
    if win <= 0:
        msg = 'win must be positive'
        raise ValueError(msg)
    hop = int(hop if hop is not None else max(1, win // 4))
    if hop <= 0:
        msg = 'hop must be positive'
        raise ValueError(msg)
    if method not in ('mad', 'iqr'):
        msg = "method must be 'mad' or 'iqr'"
        raise ValueError(msg)
    if not (0.0 < float(gamma) <= 1.0):
        msg = 'gamma must be in (0, 1]'
        raise ValueError(msg)
    p_lo, p_hi = float(clamp_pct[0]), float(clamp_pct[1])
    if not (0.0 <= p_lo < p_hi <= 100.0):
        msg = 'clamp_pct must satisfy 0 <= lo < hi <= 100'
        raise ValueError(msg)
    chunk = int(chunk)
    if chunk <= 0:
        msg = 'chunk must be positive'
        raise ValueError(msg)

    N = int(np.prod(Hdims, dtype=np.int64)) if Hdims else 1
    x_nw = x.reshape(N, W)

    anchors, bounds = _build_anchors(W, win, hop, causal)
    K = int(anchors.size)

    x_nw64 = x_nw.astype(np.float64, copy=False)
    mu_k = np.empty((N, K), dtype=np.float64)
    sg_k = np.empty((N, K), dtype=np.float64)

    if use_tqdm:
        it_stats = tqdm(
            range(K),
            desc='robust_agc_np: stats',
            total=K,
        )
    else:
        it_stats = range(K)

    for j in it_stats:
        t_anchor = int(anchors[j])
        s, e = bounds(t_anchor)
        if e <= s:
            e = min(W, s + 1)
        seg = x_nw64[:, s:e]
        mu, sigma = _robust_loc_scale(seg, method)
        mu_k[:, j] = mu
        sg_k[:, j] = sigma

    p5 = _percentile(sg_k, p_lo, axis=1, keepdims=True)
    p95 = _percentile(sg_k, p_hi, axis=1, keepdims=True)
    sg_k = np.clip(sg_k, p5, p95)
    sg_k = np.maximum(sg_k, float(eps))

    y_nw = np.empty_like(x_nw, dtype=x.dtype)

    n_chunks = (W + chunk - 1) // chunk
    if use_tqdm:
        it_chunks = tqdm(
            range(n_chunks),
            desc='robust_agc_np: apply',
            total=n_chunks,
        )
    else:
        it_chunks = range(n_chunks)

    for ci in it_chunks:
        t0 = ci * chunk
        t1 = min(W, t0 + chunk)
        tgrid = np.arange(t0, t1, dtype=np.float64)

        mu_seg = _interp_rows(anchors, mu_k, tgrid)
        sg_seg = _interp_rows(anchors, sg_k, tgrid)

        y_chunk = (
            (x_nw64[:, t0:t1] - mu_seg) / (sg_seg + float(eps)) ** float(gamma)
        ).astype(x.dtype, copy=False)
        y_nw[:, t0:t1] = y_chunk

    y = y_nw.reshape(*Hdims, W)

    if return_meta:
        meta = {
            'anchors': anchors,
            'mu_k': mu_k.reshape(*Hdims, K),
            'sigma_k': sg_k.reshape(*Hdims, K),
            'gamma': float(gamma),
            'method': method,
            'win': int(win),
            'hop': int(hop),
            'clamp_pct': (p_lo, p_hi),
            'causal': bool(causal),
            'eps': float(eps),
            'chunk': int(chunk),
            'use_tqdm': bool(use_tqdm),
        }
        return y, meta

    return y
