from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from seisai_engine.pipelines.fbpick.physics.config import load_physics_lite_config
from seisai_engine.pipelines.fbpick.physics.feasible import FeasibleBandResult
from seisai_engine.pipelines.fbpick.physics.merge import MergeResult
from seisai_engine.pipelines.fbpick.physics.physical_center import (
    PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND,
    PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP,
    PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT,
    PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST,
    PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED,
    PHYSICAL_MODEL_STATUS_TWO_PIECE_OK,
    build_geometry_two_piece_physical_center,
)
from seisai_engine.pipelines.fbpick.physics.pick_table import CoarsePickTable
from seisai_engine.pipelines.fbpick.physics.trend import TrendResult
from seisai_pick.trend.trend_fit_strategy import (
    PiecewiseLinearTrend,
    TwoPieceRansacAutoBreakStrategy,
)


def _two_piece_time_sec(offsets_m: np.ndarray) -> np.ndarray:
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


def _make_inputs(
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
        pick_i = np.rint(_two_piece_time_sec(offsets) / np.float32(dt_sec)).astype(
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


def _physical_cfg(extra: dict[str, dict[str, object]] | None = None):
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


def _fake_piecewise_model() -> PiecewiseLinearTrend:
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


def _with_invalid_trend_centers(trend: TrendResult) -> TrendResult:
    n_traces = int(np.asarray(trend.trend_center_i).shape[0])
    return replace(
        trend,
        trend_center_i=np.full((n_traces,), -1, dtype=np.int32),
        trend_center_sec=np.full((n_traces,), np.nan, dtype=np.float32),
    )


def test_physical_disabled_returns_existing_trend_center() -> None:
    inputs = _make_inputs(offsets_m=np.linspace(50.0, 1200.0, 12, dtype=np.float32))
    coarse_npz, table, feasible, trend, merged = inputs

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=load_physics_lite_config({}),
    )

    np.testing.assert_array_equal(result.physical_center_i, trend.trend_center_i)
    np.testing.assert_array_equal(result.fine_center_i, trend.trend_center_i)
    assert np.all(
        result.physical_model_status
        == np.uint8(PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED)
    )


def test_physical_center_calls_existing_two_piece_ransac(monkeypatch) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    inputs = _make_inputs(offsets_m=np.linspace(50.0, 1600.0, 12, dtype=np.float32))
    coarse_npz, table, feasible, trend, merged = inputs

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg({'two_piece_ransac': {'min_pts': 3}}),
    )

    assert calls
    assert np.any(result.physical_model_status == PHYSICAL_MODEL_STATUS_TWO_PIECE_OK)


def test_synthetic_two_piece_trend_predicts_physical_centers() -> None:
    offsets = np.linspace(50.0, 2000.0, 28, dtype=np.float32)
    coarse_npz, table, feasible, trend, merged = _make_inputs(offsets_m=offsets)

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(
            {'two_piece_ransac': {'n_iter': 200, 'n_break_cand': 64}}
        ),
    )

    np.testing.assert_allclose(
        result.physical_center_i,
        table.coarse_pick_i,
        atol=6,
        rtol=0,
    )
    np.testing.assert_array_equal(result.fine_center_i, result.physical_center_i)
    assert np.all(result.physical_model_status == PHYSICAL_MODEL_STATUS_TWO_PIECE_OK)
    assert np.all(np.isfinite(result.physical_model_break_offset_m))
    assert np.all(result.physical_model_velocity_near_m_s > 0.0)


def test_physical_prefilter_removes_velocity_and_low_pmax_outliers(monkeypatch) -> None:
    offsets = np.linspace(100.0, 1000.0, 12, dtype=np.float32)
    pick_i = np.rint((0.02 + offsets / 1500.0) / 0.001).astype(np.int32)
    pick_i[3] = 3000
    pmax = np.full((12,), 0.95, dtype=np.float32)
    pmax[5] = 0.1
    seen: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        seen.append(x_abs.detach().cpu().numpy().copy())
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=offsets,
        pick_i=pick_i,
        pmax=pmax,
    )
    cfg = _physical_cfg(
        {
            'physical_prefilter': {
                'vmin_m_s': 1000.0,
                'vmax_m_s': 2000.0,
                't0_lo_ms': 0.0,
                't0_hi_ms': 50.0,
                'pmax_min': 0.5,
            },
            'two_piece_ransac': {'min_pts': 3},
        }
    )

    build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=cfg,
    )

    used_offsets = np.unique(np.concatenate(seen))
    assert not np.any(np.isclose(used_offsets, offsets[3]))
    assert not np.any(np.isclose(used_offsets, offsets[5]))


def test_geometry_missing_falls_back_to_existing_trend_without_crashing() -> None:
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=np.linspace(50.0, 1200.0, 12, dtype=np.float32),
        with_geometry=False,
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(),
    )

    np.testing.assert_array_equal(result.fine_center_i, trend.trend_center_i)
    assert np.all(
        result.physical_model_status
        == np.uint8(PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND)
    )


def test_fallback_status_reports_feasible_clip_when_trend_is_unusable() -> None:
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=np.linspace(50.0, 1200.0, 12, dtype=np.float32),
        with_geometry=False,
    )
    trend = _with_invalid_trend_centers(trend)
    feasible = replace(
        feasible,
        feasible_lo_sec=np.full(
            (table.n_traces,),
            np.float32(0.100),
            dtype=np.float32,
        ),
        feasible_hi_sec=np.full(
            (table.n_traces,),
            np.float32(0.120),
            dtype=np.float32,
        ),
    )
    merged = replace(
        merged,
        robust_pick_i=np.full((table.n_traces,), 300, dtype=np.int32),
        robust_pick_t_sec=np.full(
            (table.n_traces,),
            np.float32(0.300),
            dtype=np.float32,
        ),
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(),
    )

    assert np.all(
        result.physical_model_status
        == np.uint8(PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP)
    )
    assert np.all(result.fine_center_t_sec >= feasible.feasible_lo_sec)
    assert np.all(result.fine_center_t_sec <= feasible.feasible_hi_sec)


def test_fallback_status_reports_robust_when_trend_and_feasible_clip_are_unusable() -> None:
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=np.linspace(50.0, 1200.0, 12, dtype=np.float32),
        with_geometry=False,
    )
    trend = _with_invalid_trend_centers(trend)
    feasible = replace(
        feasible,
        feasible_lo_sec=np.full((table.n_traces,), np.nan, dtype=np.float32),
        feasible_hi_sec=np.full((table.n_traces,), np.nan, dtype=np.float32),
    )
    merged = replace(
        merged,
        robust_pick_i=np.full(
            (table.n_traces,),
            table.n_samples_orig + 20,
            dtype=np.int32,
        ),
        robust_pick_t_sec=np.full(
            (table.n_traces,),
            np.float32(99.0),
            dtype=np.float32,
        ),
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(),
    )

    assert np.all(
        result.physical_model_status
        == np.uint8(PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST)
    )
    assert np.all(result.fine_center_i == table.n_samples_orig - 1)


def test_insufficient_observations_falls_back_inside_sample_range() -> None:
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=np.linspace(50.0, 500.0, 6, dtype=np.float32),
    )
    cfg = _physical_cfg({'two_piece_ransac': {'min_pts': 4}})

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=cfg,
    )

    assert np.all(
        result.physical_model_status
        == np.uint8(PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND)
    )
    assert np.all(result.fine_center_i >= 0)
    assert np.all(result.fine_center_i < table.n_samples_orig)


def test_segment_relaxation_path_uses_relaxed_fit(monkeypatch) -> None:
    offsets = np.array([10.0, 20.0, 30.0, 1000.0, 1010.0, 1020.0], dtype=np.float32)
    pick_i = np.rint((0.02 + offsets / 2000.0) / 0.001).astype(np.int32)

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        assert int(x_abs.numel()) == 6
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=offsets,
        pick_i=pick_i,
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg({'two_piece_ransac': {'min_pts': 2}}),
    )

    assert np.all(
        result.physical_model_status
        == np.uint8(PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT)
    )


def test_physical_center_diagnostic_arrays_are_save_friendly() -> None:
    offsets = np.linspace(50.0, 2000.0, 20, dtype=np.float32)
    coarse_npz, table, feasible, trend, merged = _make_inputs(offsets_m=offsets)
    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg({'two_piece_ransac': {'min_pts': 3}}),
    )

    expected_dtypes = {
        'physical_center_i': np.int32,
        'physical_center_t_sec': np.float32,
        'fine_center_i': np.int32,
        'fine_center_t_sec': np.float32,
        'physical_model_status': np.uint8,
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
    }
    for field, dtype in expected_dtypes.items():
        arr = getattr(result, field)
        assert arr.shape == (table.n_traces,)
        assert arr.dtype == np.dtype(dtype)
