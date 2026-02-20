from __future__ import annotations

import numpy as np


def _lmo_shift_samples(
    offsets_m: np.ndarray, *, dt_sec: float, vel_mps: float
) -> np.ndarray:
    if vel_mps <= 0.0:
        msg = f'LMO velocity must be positive, got {vel_mps}'
        raise ValueError(msg)
    if dt_sec <= 0.0:
        msg = f'dt_sec must be positive, got {dt_sec}'
        raise ValueError(msg)
    off = np.asarray(offsets_m, dtype=np.float32)
    return (np.abs(off) / float(vel_mps)) / float(dt_sec)  # (H,) float samples


def apply_lmo_linear(
    wave_hw: np.ndarray,  # (H,W)
    offsets_m: np.ndarray,  # (H,)
    *,
    dt_sec: float,
    vel_mps: float,
    bulk_shift_samples: float = 0.0,
    fill: float = 0.0,
) -> np.ndarray:
    w = np.asarray(wave_hw, dtype=np.float32)
    if w.ndim != 2:
        msg = f'wave_hw must be 2D (H,W), got {w.shape}'
        raise ValueError(msg)
    H, W = w.shape

    shifts = _lmo_shift_samples(offsets_m, dt_sec=dt_sec, vel_mps=vel_mps)
    if shifts.shape != (H,):
        msg = f'offsets_m must be (H,), got {np.asarray(offsets_m).shape}, H={H}'
        raise ValueError(msg)

    xi = np.arange(W, dtype=np.float32)
    out = np.empty_like(w)
    for i in range(H):
        src = xi - float(shifts[i]) + float(bulk_shift_samples)
        out[i] = np.interp(xi, src, w[i], left=fill, right=fill)
    return out


def lmo_correct_picks(
    picks: np.ndarray,
    offsets_m: np.ndarray,
    *,
    dt_sec: float,
    vel_mps: float,
) -> np.ndarray:
    p = np.asarray(picks, dtype=np.float32)
    shifts = _lmo_shift_samples(offsets_m, dt_sec=dt_sec, vel_mps=vel_mps).astype(
        np.float32, copy=False
    )
    if p.shape != shifts.shape:
        msg = f'picks must be (H,), got {p.shape}, shifts={shifts.shape}'
        raise ValueError(msg)
    out = p - shifts
    out[~np.isfinite(p)] = np.nan
    return out
