from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seisai_engine.pipelines.fbpick.common import (
    REASON_MASK_FILLED_FROM_TREND,
    REASON_MASK_INFEASIBLE,
    REASON_MASK_LOW_SCORE,
    ROBUST_SOURCE_COARSE_OBSERVED,
    ROBUST_SOURCE_TREND_FILL,
)

from .confidence import ConfidenceResult
from .config import PhysicsLiteConfig
from .feasible import FeasibleBandResult
from .pick_table import CoarsePickTable
from .trend import TrendResult

__all__ = ['MergeResult', 'apply_keep_reject_fill']


@dataclass(frozen=True)
class MergeResult:
    keep_mask: np.ndarray
    reject_mask: np.ndarray
    score_threshold: np.float32
    robust_pick_i: np.ndarray
    robust_pick_t_sec: np.ndarray
    robust_conf: np.ndarray
    robust_source: np.ndarray
    used_theoretical_mask: np.ndarray
    reason_mask: np.ndarray


def _lower_quantile_threshold(values: np.ndarray, frac: float) -> float:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if int(finite.size) == 0:
        msg = 'cannot compute threshold from empty finite values'
        raise ValueError(msg)
    return float(np.quantile(finite, float(frac), method='linear'))


def _merge_theoretical_placeholder(
    *,
    robust_pick_i: np.ndarray,
    robust_pick_t_sec: np.ndarray,
    robust_conf: np.ndarray,
    robust_source: np.ndarray,
    theoretical_pick_i: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Future hook for theoretical replacement.

    physics-lite currently keeps this merge point fixed and emits
    used_theoretical_mask=False for every trace.
    """
    if theoretical_pick_i is not None:
        msg = 'theoretical replacement is not wired in physics-lite'
        raise NotImplementedError(msg)
    used_theoretical_mask = np.zeros(robust_source.shape, dtype=np.bool_)
    return (
        robust_pick_i.astype(np.int32, copy=False),
        robust_pick_t_sec.astype(np.float32, copy=False),
        robust_conf.astype(np.float32, copy=False),
        robust_source.astype(np.uint8, copy=False),
        used_theoretical_mask,
    )


def apply_keep_reject_fill(
    table: CoarsePickTable,
    feasible: FeasibleBandResult,
    trend: TrendResult,
    confidence: ConfidenceResult,
    cfg: PhysicsLiteConfig,
) -> MergeResult:
    feasible_mask = np.asarray(feasible.feasible_mask, dtype=np.bool_)
    if not bool(np.any(feasible_mask)):
        msg = 'keep/reject requires at least one feasible trace'
        raise ValueError(msg)

    score_threshold = np.float32(
        _lower_quantile_threshold(
            confidence.total_score[feasible_mask],
            frac=float(cfg.keep_reject.drop_low_frac),
        )
    )
    low_score_mask = feasible_mask & (
        confidence.total_score < np.float32(score_threshold)
    )
    reject_mask = (~feasible_mask) | low_score_mask
    keep_mask = ~reject_mask

    reason_mask = np.zeros((table.n_traces,), dtype=np.uint8)
    reason_mask[~feasible_mask] |= np.uint8(REASON_MASK_INFEASIBLE)
    reason_mask[low_score_mask] |= np.uint8(REASON_MASK_LOW_SCORE)
    reason_mask[reject_mask] |= np.uint8(REASON_MASK_FILLED_FROM_TREND)

    robust_pick_i = np.asarray(table.coarse_pick_i, dtype=np.int32).copy()
    robust_pick_t_sec = np.asarray(table.coarse_pick_t_sec, dtype=np.float32).copy()
    robust_conf = np.asarray(confidence.total_score, dtype=np.float32).copy()
    robust_source = np.full(
        (table.n_traces,),
        np.uint8(ROBUST_SOURCE_COARSE_OBSERVED),
        dtype=np.uint8,
    )

    robust_pick_i[reject_mask] = np.asarray(trend.trend_center_i, dtype=np.int32)[
        reject_mask
    ]
    robust_pick_t_sec[reject_mask] = np.asarray(trend.trend_center_sec, dtype=np.float32)[
        reject_mask
    ]
    robust_conf[reject_mask] = np.clip(
        np.asarray(confidence.conf_trend1, dtype=np.float32)[reject_mask]
        * np.asarray(confidence.conf_rs1, dtype=np.float32)[reject_mask],
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    robust_source[reject_mask] = np.uint8(ROBUST_SOURCE_TREND_FILL)

    (
        robust_pick_i,
        robust_pick_t_sec,
        robust_conf,
        robust_source,
        used_theoretical_mask,
    ) = _merge_theoretical_placeholder(
        robust_pick_i=robust_pick_i,
        robust_pick_t_sec=robust_pick_t_sec,
        robust_conf=robust_conf,
        robust_source=robust_source,
    )

    return MergeResult(
        keep_mask=keep_mask.astype(np.bool_, copy=False),
        reject_mask=reject_mask.astype(np.bool_, copy=False),
        score_threshold=score_threshold,
        robust_pick_i=robust_pick_i,
        robust_pick_t_sec=robust_pick_t_sec,
        robust_conf=np.clip(robust_conf, 0.0, 1.0).astype(np.float32, copy=False),
        robust_source=robust_source.astype(np.uint8, copy=False),
        used_theoretical_mask=used_theoretical_mask.astype(np.bool_, copy=False),
        reason_mask=reason_mask.astype(np.uint8, copy=False),
    )
