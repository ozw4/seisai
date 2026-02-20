from __future__ import annotations

import numpy as np


def lmo_correct_picks(
    picks_i: np.ndarray,
    offsets_m: np.ndarray,
    dt_sec: float,
    velocity_m_s: float,
    *,
    reference_offset_m: float = 0.0,
    abs_offset: bool = True,
) -> np.ndarray:
    """Linear moveout (LMO) correction for pick indices.

    Standard definition:
      t0 = t - x / v
    where x is offset [m], v is velocity [m/s], t is time [s].

    Args:
        picks_i: (n_traces,) pick sample indices (float/int). NaN allowed.
        offsets_m: (n_traces,) offsets in meters. Can be signed or absolute.
        dt_sec: sample interval in seconds.
        velocity_m_s: linear velocity [m/s].
        reference_offset_m: reference offset [m] (default 0).
        abs_offset: if True, use abs(offset - reference_offset).

    Returns:
        (n_traces,) corrected pick sample indices (float32).

    """
    p = np.asarray(picks_i, dtype=np.float64)
    off = np.asarray(offsets_m, dtype=np.float64)

    if p.ndim != 1 or off.ndim != 1:
        msg = f'picks_i/offsets_m must be 1D, got {p.shape}, {off.shape}'
        raise ValueError(msg)
    if p.shape != off.shape:
        msg = f'picks_i/offsets_m shape mismatch: {p.shape} vs {off.shape}'
        raise ValueError(msg)

    dt = float(dt_sec)
    if not (dt > 0.0):
        msg = f'dt_sec must be > 0, got {dt_sec}'
        raise ValueError(msg)

    v = float(velocity_m_s)
    if not (v > 0.0):
        msg = f'velocity_m_s must be > 0, got {velocity_m_s}'
        raise ValueError(msg)

    x = off - float(reference_offset_m)
    if bool(abs_offset):
        x = np.abs(x)

    shift_samples = (x / v) / dt  # (n_traces,)
    out = p.copy()
    m = np.isfinite(out)
    out[m] = out[m] - shift_samples[m]
    return out.astype(np.float32, copy=False)


def apply_lmo_linear(
    traces: np.ndarray,
    offsets_m: np.ndarray,
    dt_sec: float,
    velocity_m_s: float,
    *,
    reference_offset_m: float = 0.0,
    abs_offset: bool = True,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Apply linear moveout (LMO) to traces by resampling each trace.

    If an event follows t(x) = t0 + x/v, then LMO-corrected trace aligns it to t0.
    Mapping used:
      y_out(t) = y_in(t + x/v)

    Args:
        traces: (n_traces, n_samples) array.
        offsets_m: (n_traces,) offsets in meters (signed or absolute).
        dt_sec: sample interval in seconds.
        velocity_m_s: linear velocity [m/s].
        reference_offset_m: reference offset [m] (default 0).
        abs_offset: if True, use abs(offset - reference_offset).
        fill_value: value for out-of-range samples after shift.

    Returns:
        (n_traces, n_samples) float32 array (LMO-corrected).

    """
    y = np.asarray(traces)
    if y.ndim != 2:
        msg = f'traces must be 2D (n_traces, n_samples), got {y.shape}'
        raise ValueError(msg)

    n_tr, n_samp = int(y.shape[0]), int(y.shape[1])
    off = np.asarray(offsets_m, dtype=np.float64)
    if off.ndim != 1 or off.shape[0] != n_tr:
        msg = f'offsets_m must be (n_traces,), got {off.shape}, n_traces={n_tr}'
        raise ValueError(msg)

    dt = float(dt_sec)
    if not (dt > 0.0):
        msg = f'dt_sec must be > 0, got {dt_sec}'
        raise ValueError(msg)

    v = float(velocity_m_s)
    if not (v > 0.0):
        msg = f'velocity_m_s must be > 0, got {velocity_m_s}'
        raise ValueError(msg)

    x = off - float(reference_offset_m)
    if bool(abs_offset):
        x = np.abs(x)

    shift_samples = (x / v) / dt  # (n_traces,)

    # base sample index grid
    s = np.arange(n_samp, dtype=np.float64)

    out = np.empty((n_tr, n_samp), dtype=np.float32)
    fv = float(fill_value)

    y_f = y.astype(np.float32, copy=False)
    for i in range(n_tr):
        si = s + float(shift_samples[i])
        out[i] = np.interp(
            si, s, y_f[i].astype(np.float64, copy=False), left=fv, right=fv
        ).astype(np.float32, copy=False)

    return out
