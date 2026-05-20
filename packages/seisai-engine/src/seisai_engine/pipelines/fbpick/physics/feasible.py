from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import PhysicalBandCfg, PhysicsFeasibleBandCfg
    from .pick_table import CoarsePickTable

__all__ = [
    'FeasibleBandResult',
    'compute_feasible_band',
    'compute_physical_band_result',
    'compute_velocity_t0_band_from_arrays',
]


@dataclass(frozen=True)
class FeasibleBandResult:
    feasible_mask: np.ndarray
    feasible_lo_sec: np.ndarray
    feasible_hi_sec: np.ndarray


def _coarse_pick_times(table: CoarsePickTable) -> np.ndarray:
    if table.n_traces <= 0:
        msg = 'table.n_traces must be positive'
        raise ValueError(msg)

    pick_t_sec = np.asarray(table.coarse_pick_t_sec, dtype=np.float32)
    if pick_t_sec.shape != (table.n_traces,):
        msg = (
            f'coarse_pick_t_sec must have shape {(table.n_traces,)}, '
            f'got {pick_t_sec.shape}'
        )
        raise ValueError(msg)
    return pick_t_sec


def compute_velocity_t0_band_from_arrays(
    *,
    offset_m: np.ndarray,
    pick_t_sec: np.ndarray,
    vmin_m_s: float,
    vmax_m_s: float,
    t0_lo_ms: float,
    t0_hi_ms: float,
) -> FeasibleBandResult:
    offset_arr = np.asarray(offset_m, dtype=np.float32)
    pick_arr = np.asarray(pick_t_sec, dtype=np.float32)

    if offset_arr.ndim != 1:
        msg = 'offset_m must be a 1D array'
        raise ValueError(msg)
    if pick_arr.ndim != 1:
        msg = 'pick_t_sec must be a 1D array'
        raise ValueError(msg)
    if offset_arr.shape != pick_arr.shape:
        msg = (
            'offset_m and pick_t_sec must have the same shape, '
            f'got {offset_arr.shape} and {pick_arr.shape}'
        )
        raise ValueError(msg)

    vmin = float(vmin_m_s)
    vmax = float(vmax_m_s)
    t0_lo = float(t0_lo_ms)
    t0_hi = float(t0_hi_ms)
    if not np.isfinite(vmin) or vmin <= 0.0:
        msg = 'vmin_m_s must be finite and > 0'
        raise ValueError(msg)
    if not np.isfinite(vmax) or vmax < vmin:
        msg = 'vmax_m_s must be finite and >= vmin_m_s'
        raise ValueError(msg)
    if not np.isfinite(t0_lo) or not np.isfinite(t0_hi) or t0_lo > t0_hi:
        msg = 't0_lo_ms and t0_hi_ms must be finite and t0_lo_ms <= t0_hi_ms'
        raise ValueError(msg)

    if not np.all(np.isfinite(offset_arr)):
        msg = 'offset_m must be finite'
        raise ValueError(msg)
    if not np.all(np.isfinite(pick_arr)):
        msg = 'pick_t_sec must be finite'
        raise ValueError(msg)

    offset_abs_m = np.abs(offset_arr)
    t0_lo_sec = np.float32(t0_lo * 1.0e-3)
    t0_hi_sec = np.float32(t0_hi * 1.0e-3)
    lo = (offset_abs_m / np.float32(vmax) + t0_lo_sec).astype(
        np.float32,
        copy=False,
    )
    hi = (offset_abs_m / np.float32(vmin) + t0_hi_sec).astype(
        np.float32,
        copy=False,
    )
    feasible_mask = (pick_arr >= lo) & (pick_arr <= hi)
    return FeasibleBandResult(
        feasible_mask=feasible_mask.astype(np.bool_, copy=False),
        feasible_lo_sec=lo,
        feasible_hi_sec=hi,
    )


def compute_feasible_band(
    table: CoarsePickTable,
    cfg: PhysicsFeasibleBandCfg,
) -> FeasibleBandResult:
    return compute_velocity_t0_band_from_arrays(
        offset_m=table.offset_m,
        pick_t_sec=_coarse_pick_times(table),
        vmin_m_s=cfg.vmin_mask,
        vmax_m_s=cfg.vmax_mask,
        t0_lo_ms=cfg.t0_lo_ms,
        t0_hi_ms=cfg.t0_hi_ms,
    )


def compute_physical_band_result(
    table: CoarsePickTable,
    cfg: PhysicalBandCfg,
) -> FeasibleBandResult:
    return compute_velocity_t0_band_from_arrays(
        offset_m=table.offset_m,
        pick_t_sec=_coarse_pick_times(table),
        vmin_m_s=cfg.vmin_m_s,
        vmax_m_s=cfg.vmax_m_s,
        t0_lo_ms=cfg.t0_lo_ms,
        t0_hi_ms=cfg.t0_hi_ms,
    )
