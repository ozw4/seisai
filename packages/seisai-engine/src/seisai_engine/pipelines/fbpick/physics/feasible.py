from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PhysicsFeasibleBandCfg
from .pick_table import CoarsePickTable

__all__ = ['FeasibleBandResult', 'compute_feasible_band']


@dataclass(frozen=True)
class FeasibleBandResult:
    feasible_mask: np.ndarray
    feasible_lo_sec: np.ndarray
    feasible_hi_sec: np.ndarray


def compute_feasible_band(
    table: CoarsePickTable,
    cfg: PhysicsFeasibleBandCfg,
) -> FeasibleBandResult:
    if table.n_traces <= 0:
        msg = 'table.n_traces must be positive'
        raise ValueError(msg)

    offset_abs_m = np.abs(np.asarray(table.offset_m, dtype=np.float32))
    if not np.all(np.isfinite(offset_abs_m)):
        msg = 'offset_m must be finite after abs()'
        raise ValueError(msg)

    pick_t_sec = np.asarray(table.coarse_pick_t_sec, dtype=np.float32)
    if pick_t_sec.shape != (table.n_traces,):
        msg = f'coarse_pick_t_sec must have shape {(table.n_traces,)}, got {pick_t_sec.shape}'
        raise ValueError(msg)

    t0_lo_sec = np.float32(float(cfg.t0_lo_ms) * 1.0e-3)
    t0_hi_sec = np.float32(float(cfg.t0_hi_ms) * 1.0e-3)
    lo = (offset_abs_m / np.float32(cfg.vmax_mask) + t0_lo_sec).astype(
        np.float32,
        copy=False,
    )
    hi = (offset_abs_m / np.float32(cfg.vmin_mask) + t0_hi_sec).astype(
        np.float32,
        copy=False,
    )
    feasible_mask = (pick_t_sec >= lo) & (pick_t_sec <= hi)
    return FeasibleBandResult(
        feasible_mask=feasible_mask.astype(np.bool_, copy=False),
        feasible_lo_sec=lo,
        feasible_hi_sec=hi,
    )
