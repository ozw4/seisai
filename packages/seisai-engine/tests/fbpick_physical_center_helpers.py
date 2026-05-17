from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from seisai_engine.pipelines.fbpick.physics.config import load_physics_lite_config
from seisai_engine.pipelines.fbpick.physics.feasible import FeasibleBandResult
from seisai_engine.pipelines.fbpick.physics.merge import MergeResult
from seisai_engine.pipelines.fbpick.physics.pick_table import CoarsePickTable
from seisai_engine.pipelines.fbpick.physics.trend import TrendResult
from seisai_pick.trend.trend_fit_strategy import PiecewiseLinearTrend

PHYSICAL_CENTER_RESULT_DTYPE_CONTRACT = {
    'physical_center_i': np.int32,
    'physical_center_t_sec': np.float32,
    'fine_center_i': np.int32,
    'fine_center_t_sec': np.float32,
    'physical_model_status': np.uint8,
    'physical_model_failure_reason': np.uint8,
    'physical_offset_source': np.uint8,
    'physical_model_break_offset_m': np.float32,
    'physical_model_slope_near_s_per_m': np.float32,
    'physical_model_slope_far_s_per_m': np.float32,
    'physical_model_velocity_near_m_s': np.float32,
    'physical_model_velocity_far_m_s': np.float32,
    'physical_model_neighbor_count': np.int32,
    'physical_prefilter_valid_count': np.int32,
    'physical_model_segment_id': np.int32,
    'physical_model_side': np.int8,
    'physical_model_resid_p50_ms': np.float32,
    'physical_model_resid_p90_ms': np.float32,
    'physical_anchor_group_id': np.int32,
    'physical_anchor_is_anchor': np.bool_,
    'physical_anchor_nearest_anchor_group_id': np.int32,
    'physical_anchor_source_distance_m': np.float32,
    'physical_runtime_t0_shift_ms': np.float32,
    'physical_runtime_reuse_resid_p50_ms': np.float32,
    'physical_runtime_reuse_resid_p90_ms': np.float32,
    'physical_runtime_reuse_valid_count': np.int32,
    'physical_runtime_refit_mask': np.bool_,
    'physical_runtime_fit_source': np.uint8,
}

class RecordingProgressReporter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


def two_piece_time_sec(offsets_m: np.ndarray) -> np.ndarray:
    offsets = np.asarray(offsets_m, dtype=np.float32)
    break_m = np.float32(1000.0)
    near_slope = np.float32(1.0 / 1000.0)
    far_slope = np.float32(1.0 / 3000.0)
    t0 = np.float32(0.02)
    t_break = t0 + near_slope * break_m
    return np.where(
        offsets <= break_m,
        t0 + near_slope * offsets,
        t_break + far_slope * (offsets - break_m),
    ).astype(np.float32)


def make_inputs(
    *,
    offsets_m: np.ndarray,
    pick_i: np.ndarray | None = None,
    pmax: np.ndarray | None = None,
    dt_sec: float = 0.001,
    n_samples_orig: int = 5000,
    with_geometry: bool = True,
) -> tuple[
    dict[str, np.ndarray],
    CoarsePickTable,
    FeasibleBandResult,
    TrendResult,
    MergeResult,
]:
    offsets = np.asarray(offsets_m, dtype=np.float32)
    n_traces = int(offsets.shape[0])
    if pick_i is None:
        pick_i = np.rint(two_piece_time_sec(offsets) / np.float32(dt_sec)).astype(
            np.int32
        )
    else:
        pick_i = np.asarray(pick_i, dtype=np.int32)
    pmax_arr = (
        np.full((n_traces,), 0.95, dtype=np.float32)
        if pmax is None
        else np.asarray(pmax, dtype=np.float32)
    )
    pick_t_sec = pick_i.astype(np.float32) * np.float32(dt_sec)
    trace_indices = np.arange(n_traces, dtype=np.int64)

    table = CoarsePickTable(
        n_traces=n_traces,
        n_samples_orig=int(n_samples_orig),
        dt_scalar_sec=float(dt_sec),
        shot_id=np.ones((n_traces,), dtype=np.int32),
        trace_id=trace_indices,
        ffid=np.ones((n_traces,), dtype=np.int32),
        chno=np.arange(1, n_traces + 1, dtype=np.int32),
        offset_m=offsets,
        dt_sec=np.full((n_traces,), np.float32(dt_sec), dtype=np.float32),
        coarse_pick_i=pick_i,
        coarse_pick_t_sec=pick_t_sec,
        coarse_pmax=pmax_arr,
    )
    feasible = FeasibleBandResult(
        feasible_mask=np.ones((n_traces,), dtype=np.bool_),
        feasible_lo_sec=np.zeros((n_traces,), dtype=np.float32),
        feasible_hi_sec=np.full(
            (n_traces,),
            np.float32((n_samples_orig - 1) * dt_sec),
            dtype=np.float32,
        ),
    )
    trend_i = np.clip(pick_i + 7, 0, n_samples_orig - 1).astype(np.int32)
    trend_sec = trend_i.astype(np.float32) * np.float32(dt_sec)
    trend = TrendResult(
        seed_mask=np.ones((n_traces,), dtype=np.bool_),
        seed_threshold=np.float32(0.0),
        local_center_sec=trend_sec.copy(),
        local_center_valid=np.ones((n_traces,), dtype=np.bool_),
        local_discard_mask=np.zeros((n_traces,), dtype=np.bool_),
        global_center_sec=trend_sec.copy(),
        trend_center_sec=trend_sec,
        trend_center_i=trend_i,
        filled_mask=np.zeros((n_traces,), dtype=np.bool_),
    )
    merged = MergeResult(
        keep_mask=np.ones((n_traces,), dtype=np.bool_),
        reject_mask=np.zeros((n_traces,), dtype=np.bool_),
        score_threshold=np.float32(0.0),
        robust_pick_i=pick_i.copy(),
        robust_pick_t_sec=pick_t_sec.copy(),
        robust_conf=pmax_arr.copy(),
        robust_source=np.zeros((n_traces,), dtype=np.uint8),
        used_theoretical_mask=np.zeros((n_traces,), dtype=np.bool_),
        reason_mask=np.zeros((n_traces,), dtype=np.uint8),
    )
    coarse_npz: dict[str, np.ndarray] = {}
    if with_geometry:
        coarse_npz = {
            'source_x_m': np.zeros((n_traces,), dtype=np.float32),
            'source_y_m': np.zeros((n_traces,), dtype=np.float32),
            'receiver_x_m': offsets.astype(np.float32, copy=True),
            'receiver_y_m': np.zeros((n_traces,), dtype=np.float32),
            'offset_abs_geom_m': np.abs(offsets).astype(np.float32, copy=False),
            'geometry_valid_mask': np.ones((n_traces,), dtype=np.bool_),
        }
    return coarse_npz, table, feasible, trend, merged


def physical_cfg(extra: dict[str, dict[str, object]] | None = None):
    raw: dict[str, dict[str, object]] = {
        'physical_trend': {
            'enabled': True,
            'split_by_offset_gap': True,
            'gap_ratio': 5.0,
        },
        'neighbor_context': {
            'enabled': True,
            'k_neighbors': 1,
            'include_self': True,
        },
        'two_piece_ransac': {
            'n_iter': 80,
            'inlier_th_ms': 3.0,
            'min_pts': 4,
            'n_break_cand': 16,
            'seed': 3,
        },
    }
    if extra is not None:
        for block, values in extra.items():
            raw.setdefault(block, {}).update(values)
    return load_physics_lite_config(raw)

def fake_piecewise_model() -> PiecewiseLinearTrend:
    return PiecewiseLinearTrend(
        edges=torch.tensor([0.0, 1000.0, 2500.0], dtype=torch.float32),
        coef=torch.tensor(
            [
                [0.001, 0.02],
                [1.0 / 3000.0, 1.02 - 1000.0 / 3000.0],
            ],
            dtype=torch.float32,
        ),
    )


class ConstantTrendModel:
    def __init__(self, value_sec: float) -> None:
        self.value_sec = float(value_sec)
        self.edges = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        self.coef = torch.tensor(
            [[0.001, self.value_sec], [0.001, self.value_sec]],
            dtype=torch.float32,
        )

    def predict(self, x_abs: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (int(x_abs.numel()),),
            self.value_sec,
            dtype=torch.float32,
        )


class LinearTrendModel:
    def __init__(self, slope_sec_per_m: float, intercept_sec: float) -> None:
        self.slope_sec_per_m = float(slope_sec_per_m)
        self.intercept_sec = float(intercept_sec)
        self.edges = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        self.coef = torch.tensor(
            [
                [self.slope_sec_per_m, self.intercept_sec],
                [self.slope_sec_per_m, self.intercept_sec],
            ],
            dtype=torch.float32,
        )

    def predict(self, x_abs: torch.Tensor) -> torch.Tensor:
        return (
            x_abs.to(dtype=torch.float32) * np.float32(self.slope_sec_per_m)
            + np.float32(self.intercept_sec)
        )


def fit_linear_model(x_abs: torch.Tensor, y_sec: torch.Tensor) -> LinearTrendModel:
    x_np = x_abs.detach().cpu().numpy().astype(np.float64, copy=False)
    y_np = y_sec.detach().cpu().numpy().astype(np.float64, copy=False)
    slope, intercept = np.polyfit(x_np, y_np, deg=1)
    return LinearTrendModel(float(slope), float(intercept))


def with_invalid_trend_centers(trend: TrendResult) -> TrendResult:
    n_traces = int(np.asarray(trend.trend_center_i).shape[0])
    return replace(
        trend,
        trend_center_i=np.full((n_traces,), -1, dtype=np.int32),
        trend_center_sec=np.full((n_traces,), np.nan, dtype=np.float32),
    )
