from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seisai_pick.score.confidence_from_trend_resid import (
    trace_confidence_from_trend_resid_gaussian,
)

from .config import PhysicsLiteConfig
from .feasible import FeasibleBandResult
from .pick_table import CoarsePickTable
from .trend import TrendResult

__all__ = ['ConfidenceResult', 'compute_confidence_terms']


@dataclass(frozen=True)
class ConfidenceResult:
    conf_prob1: np.ndarray
    conf_trend1: np.ndarray
    conf_rs1: np.ndarray
    total_score: np.ndarray


def compute_confidence_terms(
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    cfg: PhysicsLiteConfig,
) -> ConfidenceResult:
    if trend.trend_center_sec.shape != (table.n_traces,):
        msg = (
            f'trend.trend_center_sec must have shape {(table.n_traces,)}, '
            f'got {trend.trend_center_sec.shape}'
        )
        raise ValueError(msg)

    conf_prob1 = np.clip(
        np.asarray(table.coarse_pmax, dtype=np.float32),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    valid = np.asarray(feasible.feasible_mask, dtype=np.bool_) & np.isfinite(
        trend.trend_center_sec
    )
    conf_trend_gauss = np.asarray(
        trace_confidence_from_trend_resid_gaussian(
            np.asarray(table.coarse_pick_t_sec, dtype=np.float32),
            np.asarray(trend.trend_center_sec, dtype=np.float32),
            valid,
            sigma_ms=float(cfg.trend.trend_sigma_ms),
        ),
        dtype=np.float32,
    )
    conf_trend1 = np.clip(
        conf_trend_gauss,
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    conf_rs1 = np.ones((table.n_traces,), dtype=np.float32)
    total_score = np.clip(
        conf_prob1 * conf_trend1 * conf_rs1,
        0.0,
        1.0,
    ).astype(np.float32, copy=False)

    return ConfidenceResult(
        conf_prob1=conf_prob1,
        conf_trend1=conf_trend1,
        conf_rs1=conf_rs1,
        total_score=total_score,
    )
