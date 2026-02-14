from __future__ import annotations

from typing import Any

import numpy as np

_STD_EPS = 1e-8
_DEN_EPS = 1e-12


def _as_float_2d(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        msg = f"{name} must be 2D (n_traces, n_samples), got shape={arr.shape}"
        raise ValueError(msg)
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        msg = f"{name} must have positive shape, got {arr.shape}"
        raise ValueError(msg)
    return arr


def _as_bool_1d(name: str, x: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(x, dtype=bool)
    if arr.ndim != 1 or arr.shape[0] != n:
        msg = f"{name} must be 1D of length {n}, got shape={arr.shape}"
        raise ValueError(msg)
    return arr


def _local_reference_with_meta(
    X: np.ndarray, valid_mask: np.ndarray, k_neighbors: int, method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _as_float_2d("X", X)
    n_traces, n_samples = x.shape
    valid = _as_bool_1d("valid_mask", valid_mask, n_traces)

    if k_neighbors <= 0:
        msg = f"k_neighbors must be positive, got {k_neighbors}"
        raise ValueError(msg)
    if method != "median":
        msg = f"unsupported method={method!r}; supported: 'median'"
        raise ValueError(msg)

    ref = np.zeros((n_traces, n_samples), dtype=np.float32)
    valid_ref = np.zeros(n_traces, dtype=bool)
    neighbor_count = np.zeros(n_traces, dtype=np.int32)

    for i in range(n_traces):
        left = max(0, i - k_neighbors)
        right = min(n_traces, i + k_neighbors + 1)
        idx = np.arange(left, right, dtype=np.int32)
        idx = idx[idx != i]
        if idx.size == 0:
            continue
        idx = idx[valid[idx]]
        neighbor_count[i] = int(idx.size)
        if idx.size == 0:
            continue
        ref[i] = np.median(x[idx], axis=0).astype(np.float32, copy=False)
        valid_ref[i] = True

    return ref, valid_ref, neighbor_count


def make_local_reference(
    X: np.ndarray, valid_mask: np.ndarray, k_neighbors: int, method: str = "median"
) -> np.ndarray:
    """Build local reference waveforms from neighboring traces.

    Notes:
        - For traces without valid neighbors, reference is all-zero.
        - Use internal helper `_local_reference_with_meta` if valid-ref mask is needed.

    """
    ref, _, _ = _local_reference_with_meta(X, valid_mask, k_neighbors, method)
    return ref


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64, copy=False)
    bb = b.astype(np.float64, copy=False)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    den = np.sqrt(np.dot(aa, aa) * np.dot(bb, bb))
    if den <= _DEN_EPS:
        return 0.0
    return float(np.dot(aa, bb) / den)


def _find_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 1:
        msg = f"mask must be 1D, got shape={m.shape}"
        raise ValueError(msg)

    idx = np.flatnonzero(m)
    if idx.size == 0:
        return []

    segments: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for cur_v in idx[1:]:
        cur = int(cur_v)
        if cur != prev + 1:
            segments.append((start, prev + 1))
            start = cur
        prev = cur
    segments.append((start, prev + 1))
    return segments


def _build_taper_window(
    n: int, taper: str | None = "hann", taper_power: float = 1.0
) -> np.ndarray:
    if n <= 0:
        msg = f"n must be positive, got {n}"
        raise ValueError(msg)
    if taper_power <= 0.0:
        msg = f"taper_power must be positive, got {taper_power}"
        raise ValueError(msg)

    if taper is None:
        win = np.ones(n, dtype=np.float64)
    elif taper == "hann":
        win = np.hanning(n).astype(np.float64, copy=False)
    else:
        msg = f"unsupported taper={taper!r}; supported: None, 'hann'"
        raise ValueError(msg)

    if taper_power != 1.0:
        win = np.power(win, float(taper_power))
    return win


def _estimate_shift_ncc_with_score(
    trace: np.ndarray,
    ref: np.ndarray,
    max_lag: int,
    mode: str = "diff",
    *,
    subsample: bool = False,
    taper: str | None = "hann",
    taper_power: float = 1.0,
    lag_penalty: float = 0.0,
    lag_penalty_power: float = 1.0,
) -> tuple[float, float, float]:
    tr = np.asarray(trace, dtype=np.float32)
    rf = np.asarray(ref, dtype=np.float32)

    if tr.ndim != 1 or rf.ndim != 1:
        msg = f"trace/ref must be 1D, got trace={tr.shape}, ref={rf.shape}"
        raise ValueError(msg)
    if tr.shape[0] != rf.shape[0]:
        msg = f"trace/ref length mismatch: {tr.shape[0]} != {rf.shape[0]}"
        raise ValueError(msg)
    if max_lag < 0:
        msg = f"max_lag must be >= 0, got {max_lag}"
        raise ValueError(msg)
    if lag_penalty < 0.0:
        msg = f"lag_penalty must be non-negative, got {lag_penalty}"
        raise ValueError(msg)
    if lag_penalty_power <= 0.0:
        msg = f"lag_penalty_power must be positive, got {lag_penalty_power}"
        raise ValueError(msg)
    if not np.all(np.isfinite(tr)) or not np.all(np.isfinite(rf)):
        msg = "trace/ref must be finite"
        raise ValueError(msg)

    if mode == "diff":
        tr_p = np.diff(tr)
        rf_p = np.diff(rf)
    elif mode == "raw":
        tr_p = tr
        rf_p = rf
    else:
        msg = f"mode must be 'diff' or 'raw', got {mode!r}"
        raise ValueError(msg)

    n = tr_p.shape[0]
    if n <= 0:
        msg = "effective signal length must be positive"
        raise ValueError(msg)

    win = _build_taper_window(n, taper=taper, taper_power=taper_power)
    tr_w = tr_p.astype(np.float64, copy=False) * win
    rf_w = rf_p.astype(np.float64, copy=False) * win

    eff_lag = min(int(max_lag), n - 1)
    lags = np.arange(-eff_lag, eff_lag + 1, dtype=np.int32)
    corrs = np.empty(lags.shape[0], dtype=np.float64)

    for j, lag in enumerate(lags):
        if lag >= 0:
            a = tr_w[lag:]
            b = rf_w[: n - lag]
        else:
            k = -lag
            a = tr_w[: n - k]
            b = rf_w[k:]
        corrs[j] = _ncc(a, b)

    scores = corrs.copy()
    if eff_lag > 0 and lag_penalty > 0.0:
        p = np.power(np.abs(lags) / float(eff_lag), float(lag_penalty_power))
        scores = scores - float(lag_penalty) * p

    score_max = float(np.max(scores))
    cands = np.flatnonzero(np.isclose(scores, score_max, rtol=0.0, atol=1e-14))
    if cands.size == 1:
        best_idx = int(cands[0])
    else:
        cand_lags = lags[cands]
        order = np.lexsort((cand_lags, np.abs(cand_lags)))
        best_idx = int(cands[order[0]])

    best_lag = float(lags[best_idx])
    if subsample and 0 < best_idx < scores.shape[0] - 1:
        ym1 = scores[best_idx - 1]
        y0 = scores[best_idx]
        yp1 = scores[best_idx + 1]
        den = ym1 - 2.0 * y0 + yp1
        if abs(den) > _DEN_EPS:
            frac = 0.5 * (ym1 - yp1) / den
            frac = float(np.clip(frac, -1.0, 1.0))
            best_lag = best_lag + frac

    c_at_best = float(corrs[best_idx])
    score_at_best = float(scores[best_idx])
    return best_lag, c_at_best, score_at_best


def estimate_shift_ncc(
    trace: np.ndarray,
    ref: np.ndarray,
    max_lag: int,
    mode: str = "diff",
    *,
    subsample: bool = False,
    taper: str | None = "hann",
    taper_power: float = 1.0,
    lag_penalty: float = 0.0,
    lag_penalty_power: float = 1.0,
) -> tuple[float, float]:
    """Estimate residual lag by NCC.

    Sign convention:
        Positive returned lag means the input `trace` is delayed (later in time)
        relative to `ref`. Therefore, to align waveform samples, apply shift `-lag`.
    """
    best_lag, cmax, _ = _estimate_shift_ncc_with_score(
        trace,
        ref,
        max_lag=max_lag,
        mode=mode,
        subsample=subsample,
        taper=taper,
        taper_power=taper_power,
        lag_penalty=lag_penalty,
        lag_penalty_power=lag_penalty_power,
    )
    return best_lag, cmax


def _solve_tridiag(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    n = diag.shape[0]
    if n == 1:
        if abs(float(diag[0])) <= _DEN_EPS:
            msg = "singular system in tridiagonal solver"
            raise ValueError(msg)
        return np.array([rhs[0] / diag[0]], dtype=np.float64)

    cp = np.empty(n - 1, dtype=np.float64)
    dp = np.empty(n, dtype=np.float64)

    b0 = float(diag[0])
    if abs(b0) <= _DEN_EPS:
        msg = "singular system in tridiagonal solver"
        raise ValueError(msg)
    cp[0] = upper[0] / b0
    dp[0] = rhs[0] / b0

    for i in range(1, n):
        den = float(diag[i]) - float(lower[i - 1]) * cp[i - 1]
        if abs(den) <= _DEN_EPS:
            msg = "singular system in tridiagonal solver"
            raise ValueError(msg)
        if i < n - 1:
            cp[i] = upper[i] / den
        dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / den

    out = np.empty(n, dtype=np.float64)
    out[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        out[i] = dp[i] - cp[i] * out[i + 1]
    return out


def _smooth_wls(
    delta_raw: np.ndarray,
    weights: np.ndarray,
    lam: float,
    propagate: bool = True,
) -> np.ndarray:
    if lam < 0.0:
        msg = f"lam must be >= 0, got {lam}"
        raise ValueError(msg)

    n = delta_raw.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if np.all(weights <= 0.0):
        return np.zeros(n, dtype=np.float32)
    if n == 1:
        return np.array([delta_raw[0] if weights[0] > 0.0 else 0.0], dtype=np.float32)

    if propagate:
        if lam == 0.0:
            out = np.zeros(n, dtype=np.float64)
            m = weights > 0.0
            out[m] = delta_raw[m]
            return out.astype(np.float32, copy=False)

        diag = weights.copy()
        diag[0] += lam
        diag[-1] += lam
        if n > 2:
            diag[1:-1] += 2.0 * lam

        lower = np.full(n - 1, -lam, dtype=np.float64)
        upper = np.full(n - 1, -lam, dtype=np.float64)
        rhs = weights * delta_raw
        out = _solve_tridiag(lower, diag, upper, rhs)
        return out.astype(np.float32, copy=False)

    out = np.zeros(n, dtype=np.float64)
    active = weights > 0.0
    for start, end in _find_true_segments(active):
        m = end - start
        if m == 1:
            out[start] = delta_raw[start]
            continue
        if lam == 0.0:
            out[start:end] = delta_raw[start:end]
            continue

        w_seg = weights[start:end].copy()
        d_seg = delta_raw[start:end]
        diag = w_seg.copy()
        diag[0] += lam
        diag[-1] += lam
        if m > 2:
            diag[1:-1] += 2.0 * lam
        lower = np.full(m - 1, -lam, dtype=np.float64)
        upper = np.full(m - 1, -lam, dtype=np.float64)
        rhs = w_seg * d_seg
        out[start:end] = _solve_tridiag(lower, diag, upper, rhs)
    return out.astype(np.float32, copy=False)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.shape != weights.shape:
        msg = f"values/weights shape mismatch: {values.shape} != {weights.shape}"
        raise ValueError(msg)
    if np.any(weights < 0.0):
        msg = "weights must be non-negative"
        raise ValueError(msg)
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0

    order = np.argsort(values, kind="mergesort")
    v = values[order]
    w = weights[order]
    csum = np.cumsum(w)
    idx = int(np.searchsorted(csum, 0.5 * total, side="left"))
    idx = int(np.clip(idx, 0, v.shape[0] - 1))
    return float(v[idx])


def _smooth_wmedian(
    delta_raw: np.ndarray,
    weights: np.ndarray,
    window: int,
    propagate: bool = True,
) -> np.ndarray:
    if window <= 0 or window % 2 == 0:
        msg = f"window must be positive odd integer, got {window}"
        raise ValueError(msg)
    n = delta_raw.shape[0]
    half = window // 2
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if not propagate and weights[i] <= 0.0:
            continue
        left = max(0, i - half)
        right = min(n, i + half + 1)
        vals = delta_raw[left:right]
        w = weights[left:right]
        if not propagate:
            m = w > 0.0
            vals = vals[m]
            w = w[m]
        if vals.size == 0:
            out[i] = 0.0
            continue
        out[i] = _weighted_median(vals, w)
    return out


def smooth_shifts(
    delta_raw: np.ndarray,
    weights: np.ndarray,
    method: str = "wls",
    lam: float = 5.0,
    window: int = 7,
    propagate: bool = True,
) -> np.ndarray:
    d = np.asarray(delta_raw, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    if d.ndim != 1 or w.ndim != 1 or d.shape[0] != w.shape[0]:
        msg = f"delta_raw/weights must be 1D with same length, got {d.shape}, {w.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(w)):
        msg = "delta_raw/weights must be finite"
        raise ValueError(msg)
    if np.any(w < 0.0):
        msg = "weights must be non-negative"
        raise ValueError(msg)

    if method == "wls":
        return _smooth_wls(d, w, lam=lam, propagate=propagate)
    if method == "wmedian":
        return _smooth_wmedian(d, w, window=window, propagate=propagate)

    msg = f"unsupported method={method!r}; supported: 'wls', 'wmedian'"
    raise ValueError(msg)


def apply_shift_linear(
    X: np.ndarray, shifts: np.ndarray, fill: float = 0.0
) -> np.ndarray:
    """Apply non-circular per-trace shift with linear interpolation.

    Positive shift means moving the waveform later (to larger sample index).
    """
    x = _as_float_2d("X", X)
    n_traces, n_samples = x.shape
    s = np.asarray(shifts, dtype=np.float32)
    if s.ndim != 1 or s.shape[0] != n_traces:
        msg = f"shifts must be 1D of length {n_traces}, got shape={s.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(s)):
        msg = "shifts must be finite"
        raise ValueError(msg)

    base = np.arange(n_samples, dtype=np.float32)
    out = np.empty_like(x, dtype=np.float32)
    for i in range(n_traces):
        query = base - float(s[i])
        out[i] = np.interp(query, base, x[i], left=fill, right=fill)
    return out


def _compute_valid_mask(X: np.ndarray, std_eps: float = _STD_EPS) -> np.ndarray:
    amp = np.max(np.abs(X), axis=1)
    std = np.std(X, axis=1)
    return np.isfinite(amp) & np.isfinite(std) & (amp > 0.0) & (std >= std_eps)


def refine_firstbreak_residual_statics(
    X: np.ndarray,
    max_lag: int,
    k_neighbors: int,
    n_iter: int = 4,
    mode: str = "diff",
    c_th: float = 0.3,
    smooth_method: str = "wls",
    lam: float = 5.0,
    subsample: bool = True,
    propagate_low_corr: bool = False,
    taper: str | None = "hann",
    taper_power: float = 1.0,
    lag_penalty: float = 0.0,
    lag_penalty_power: float = 1.0,
    *,
    window: int = 7,
    std_eps: float = _STD_EPS,
    update_tol: float | None = None,
) -> dict[str, Any]:
    """Iterative local-reference residual statics refinement.

    Args:
        X: First-break pre-aligned windows, shape (n_traces, n_samples).
        max_lag: Integer search half-window for NCC.
        k_neighbors: Local reference uses neighbors in [i-K, i+K], excluding self.
        n_iter: Number of iterations.
        mode: Correlation mode ('diff' or 'raw').
        c_th: Correlation threshold to build confidence weights in [0,1).
        smooth_method: 'wls' or 'wmedian'.
        lam: Smoothness coefficient for WLS.
        subsample: If True, use 3-point parabolic lag refinement.
        propagate_low_corr: If False, disallow updates where weight is zero.
        taper: Correlation taper type. Supported: None, 'hann'.
        taper_power: Exponent applied to taper window.
        lag_penalty: Penalty magnitude for larger |lag| selection.
        lag_penalty_power: Exponent for lag penalty curve.
        window: Weighted median filter window for smooth_method='wmedian'.
        std_eps: Trace std threshold to define invalid traces.
        update_tol: Optional stopping threshold for mean absolute update.

    Returns:
        dict with keys:
            - delta_pick (float32, shape n_traces):
                Pick correction in samples. Positive means later pick.
                Invalid traces are 0.0.
            - cmax (float32, shape n_traces):
                Raw NCC at selected lag for each trace.
            - score (float32, shape n_traces):
                Final-iteration selection score after lag penalty.
            - valid_mask (bool, shape n_traces):
                True for valid input traces.
            - delta_trace (float32, shape n_traces):
                Cumulative trace shift actually applied to waveforms for alignment.
                Positive means waveform moved later. This is opposite sign of delta_pick.
            - history (list[dict[str, float | int]]):
                Iteration logs.

    """
    x0 = _as_float_2d("X", X)

    if max_lag < 0:
        msg = f"max_lag must be >= 0, got {max_lag}"
        raise ValueError(msg)
    if k_neighbors <= 0:
        msg = f"k_neighbors must be positive, got {k_neighbors}"
        raise ValueError(msg)
    if n_iter <= 0:
        msg = f"n_iter must be positive, got {n_iter}"
        raise ValueError(msg)
    if not (0.0 <= c_th < 1.0):
        msg = f"c_th must satisfy 0 <= c_th < 1, got {c_th}"
        raise ValueError(msg)
    if lag_penalty < 0.0:
        msg = f"lag_penalty must be non-negative, got {lag_penalty}"
        raise ValueError(msg)
    if lag_penalty_power <= 0.0:
        msg = f"lag_penalty_power must be positive, got {lag_penalty_power}"
        raise ValueError(msg)
    if taper_power <= 0.0:
        msg = f"taper_power must be positive, got {taper_power}"
        raise ValueError(msg)
    if taper not in (None, "hann"):
        msg = f"unsupported taper={taper!r}; supported: None, 'hann'"
        raise ValueError(msg)
    if std_eps < 0.0:
        msg = f"std_eps must be non-negative, got {std_eps}"
        raise ValueError(msg)
    if update_tol is not None and update_tol < 0.0:
        msg = f"update_tol must be non-negative, got {update_tol}"
        raise ValueError(msg)

    n_traces, _ = x0.shape
    valid_mask = _compute_valid_mask(x0, std_eps=std_eps)

    x_work = x0.copy()
    delta_pick = np.zeros(n_traces, dtype=np.float64)
    delta_trace = np.zeros(n_traces, dtype=np.float64)
    cmax_last = np.zeros(n_traces, dtype=np.float64)
    score_last = np.zeros(n_traces, dtype=np.float64)
    history: list[dict[str, float | int]] = []

    for i_iter in range(n_iter):
        ref, valid_ref, neighbor_count = _local_reference_with_meta(
            x_work, valid_mask, k_neighbors, method="median"
        )
        valid_ref &= np.std(ref, axis=1) >= std_eps

        delta_raw = np.zeros(n_traces, dtype=np.float64)
        cmax = np.zeros(n_traces, dtype=np.float64)
        score = np.zeros(n_traces, dtype=np.float64)

        for i in range(n_traces):
            if not valid_mask[i] or not valid_ref[i]:
                continue
            lag_i, cmax_i, score_i = _estimate_shift_ncc_with_score(
                x_work[i],
                ref[i],
                max_lag=max_lag,
                mode=mode,
                subsample=subsample,
                taper=taper,
                taper_power=taper_power,
                lag_penalty=lag_penalty,
                lag_penalty_power=lag_penalty_power,
            )
            delta_raw[i] = lag_i
            cmax[i] = cmax_i
            score[i] = score_i

        weights = np.zeros(n_traces, dtype=np.float64)
        good = valid_mask & valid_ref & (cmax >= c_th)
        if np.any(good):
            weights[good] = (cmax[good] - c_th) / (1.0 - c_th)
            weights[good] = np.clip(weights[good], 0.0, 1.0)

        max_neighbors = 2 * k_neighbors
        if max_neighbors > 0:
            neigh_scale = np.clip(
                neighbor_count.astype(np.float64) / float(max_neighbors), 0.0, 1.0
            )
            weights *= neigh_scale

        delta_smooth = smooth_shifts(
            delta_raw,
            weights,
            method=smooth_method,
            lam=lam,
            window=window,
            propagate=propagate_low_corr,
        ).astype(np.float64, copy=False)
        delta_smooth[~valid_mask] = 0.0

        # IMPORTANT:
        # - delta_pick > 0 means pick should be later.
        # - To align waveform samples, trace must be shifted by the opposite sign.
        shift_to_align = -delta_smooth
        x_work = apply_shift_linear(x_work, shift_to_align, fill=0.0)

        delta_pick += delta_smooth
        delta_trace += shift_to_align
        cmax_last = cmax
        score_last = score

        inlier = valid_mask & valid_ref
        history.append(
            {
                "iter": int(i_iter + 1),
                "n_valid": int(np.sum(valid_mask)),
                "n_ref_valid": int(np.sum(inlier)),
                "n_weighted": int(np.sum(weights > 0.0)),
                "mean_cmax": float(np.mean(cmax[inlier])) if np.any(inlier) else 0.0,
                "mean_score": float(np.mean(score[inlier])) if np.any(inlier) else 0.0,
                "mean_abs_raw": (
                    float(np.mean(np.abs(delta_raw[inlier]))) if np.any(inlier) else 0.0
                ),
                "mean_abs_update": (
                    float(np.mean(np.abs(delta_smooth[valid_mask])))
                    if np.any(valid_mask)
                    else 0.0
                ),
            }
        )

        if (
            update_tol is not None
            and history[-1]["mean_abs_update"] <= float(update_tol)
        ):
            break

    out_delta_pick = delta_pick.astype(np.float32, copy=False)
    out_delta_pick[~valid_mask] = 0.0
    out_cmax = cmax_last.astype(np.float32, copy=False)
    out_cmax[~valid_mask] = 0.0
    out_score = score_last.astype(np.float32, copy=False)
    out_score[~valid_mask] = 0.0
    out_delta_trace = delta_trace.astype(np.float32, copy=False)
    out_delta_trace[~valid_mask] = 0.0

    return {
        "delta_pick": out_delta_pick,
        "cmax": out_cmax,
        "score": out_score,
        "valid_mask": valid_mask,
        "delta_trace": out_delta_trace,
        "history": history,
    }


__all__ = [
    "apply_shift_linear",
    "estimate_shift_ncc",
    "make_local_reference",
    "refine_firstbreak_residual_statics",
    "smooth_shifts",
]
