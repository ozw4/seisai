from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from seisai_engine.pipelines.fbpick.physics.config import load_physics_lite_config
from seisai_engine.pipelines.fbpick.physics.feasible import FeasibleBandResult
from seisai_engine.pipelines.fbpick.physics.merge import MergeResult
from seisai_engine.pipelines.fbpick.physics.physical_center import (
    PHYSICAL_MODEL_FAILURE_FIT_FAILED,
    PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
    PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS,
    PHYSICAL_MODEL_FAILURE_NONE,
    PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED,
    PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
    PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND,
    PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP,
    PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT,
    PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST,
    PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED,
    PHYSICAL_MODEL_STATUS_TWO_PIECE_OK,
    PHYSICAL_OFFSET_SOURCE_HEADER,
    PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT,
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_FULL_FIT_NO_COMPATIBLE_ANCHOR,
    PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE,
    build_geometry_two_piece_physical_center,
)
from seisai_engine.pipelines.fbpick.physics.pick_table import CoarsePickTable
from seisai_engine.pipelines.fbpick.physics.runtime_diagnostics import (
    PhysicalRuntimeDiagnostics,
)
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


class _ConstantTrendModel:
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
    assert np.all(
        result.physical_model_failure_reason
        == np.uint8(PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED)
    )


def test_physical_center_calls_existing_two_piece_ransac(monkeypatch) -> None:
    calls: list[np.ndarray] = []

    class CountingModel:
        def __init__(self) -> None:
            self._model = _fake_piecewise_model()
            self.edges = self._model.edges
            self.coef = self._model.coef
            self.predict_call_sizes: list[int] = []

        def predict(self, x_abs: torch.Tensor) -> torch.Tensor:
            self.predict_call_sizes.append(int(x_abs.numel()))
            return self._model.predict(x_abs)

    model = CountingModel()

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return model

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    inputs = _make_inputs(offsets_m=np.linspace(50.0, 1600.0, 12, dtype=np.float32))
    coarse_npz, table, feasible, trend, merged = inputs
    diagnostics = PhysicalRuntimeDiagnostics()

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg({'two_piece_ransac': {'min_pts': 3}}),
        runtime_diagnostics=diagnostics,
    )

    assert len(calls) == 1
    assert diagnostics.n_fit_calls == 1
    assert diagnostics.n_cache_misses == 1
    assert diagnostics.n_cache_hits == int(table.n_traces) - 1
    assert model.predict_call_sizes.count(1) == int(table.n_traces)
    assert model.predict_call_sizes.count(int(table.n_traces)) == 1
    assert np.any(result.physical_model_status == PHYSICAL_MODEL_STATUS_TWO_PIECE_OK)


def test_all_zero_geometry_spread_falls_back_before_two_piece_fit(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=np.zeros((12,), dtype=np.float32),
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg({'two_piece_ransac': {'min_pts': 3}}),
    )

    assert calls == []
    assert not np.any(
        result.physical_model_status == PHYSICAL_MODEL_STATUS_TWO_PIECE_OK
    )
    assert np.all(
        result.physical_model_failure_reason
        == np.uint8(PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS)
    )
    assert np.all(result.fine_center_i >= 0)
    assert np.all(result.fine_center_i < table.n_samples_orig)


def test_constant_geometry_offsets_fall_back_even_with_enough_observations(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=np.linspace(50.0, 1200.0, 12, dtype=np.float32),
    )
    coarse_npz['offset_abs_geom_m'] = np.full((12,), 100.0, dtype=np.float32)

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(
            {
                'physical_prefilter': {'enabled': False},
                'two_piece_ransac': {'min_pts': 3},
            }
        ),
    )

    assert calls == []
    assert not np.any(
        result.physical_model_status == PHYSICAL_MODEL_STATUS_TWO_PIECE_OK
    )
    assert np.all(
        result.physical_model_failure_reason
        == np.uint8(PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS)
    )


def test_collapsed_source_xy_across_multiple_shots_rejects_geometry_fit(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=np.linspace(50.0, 600.0, 12, dtype=np.float32),
    )
    shot_id = np.asarray([101] * 6 + [102] * 6, dtype=np.int32)
    table = replace(table, shot_id=shot_id, ffid=shot_id.copy())

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(
            {
                'physical_trend': {'split_by_offset_gap': False},
                'physical_prefilter': {'enabled': False},
                'two_piece_ransac': {'min_pts': 3},
            }
        ),
    )

    assert calls == []
    assert not np.any(
        result.physical_model_status == PHYSICAL_MODEL_STATUS_TWO_PIECE_OK
    )
    assert np.all(
        result.physical_model_failure_reason
        == np.uint8(PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID)
    )


def test_single_shot_collapsed_source_xy_still_allows_geometry_fit(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=np.linspace(50.0, 600.0, 12, dtype=np.float32),
    )
    shot_id = np.full((12,), 101, dtype=np.int32)
    table = replace(table, shot_id=shot_id, ffid=shot_id.copy())

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(
            {
                'physical_trend': {'split_by_offset_gap': False},
                'physical_prefilter': {'enabled': False},
                'two_piece_ransac': {'min_pts': 3},
            }
        ),
    )

    assert len(calls) == 1
    assert np.all(result.physical_model_status == PHYSICAL_MODEL_STATUS_TWO_PIECE_OK)
    assert np.all(result.physical_model_failure_reason == PHYSICAL_MODEL_FAILURE_NONE)


def test_physical_center_fits_once_per_unique_observation_segment(monkeypatch) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    offsets = np.asarray(
        [
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            1000.0,
            1010.0,
            1020.0,
            1030.0,
            1040.0,
            1050.0,
        ],
        dtype=np.float32,
    )
    pick_i = np.rint((0.02 + offsets / 2000.0) / 0.001).astype(np.int32)
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
        cfg=_physical_cfg({'two_piece_ransac': {'min_pts': 3}}),
    )

    assert len(calls) == 2
    assert sorted(int(call.size) for call in calls) == [6, 6]
    assert np.all(result.physical_model_status == PHYSICAL_MODEL_STATUS_TWO_PIECE_OK)


def test_physical_center_uses_saved_signed_offset_for_side_segmentation(
    monkeypatch,
) -> None:
    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    offsets = np.linspace(50.0, 600.0, 12, dtype=np.float32)
    coarse_npz, table, feasible, trend, merged = _make_inputs(offsets_m=offsets)
    expected_side = np.asarray([-1] * 6 + [1] * 6, dtype=np.int8)
    coarse_npz['offset_signed_geom_m'] = (
        expected_side.astype(np.float32) * offsets
    ).astype(np.float32)

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(
            {
                'physical_trend': {'split_by_offset_gap': False},
                'two_piece_ransac': {'min_pts': 3},
            }
        ),
    )

    np.testing.assert_array_equal(result.physical_model_side, expected_side)


def test_physical_center_falls_back_to_pca_side_when_saved_signed_absent(
    monkeypatch,
) -> None:
    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    offsets = np.linspace(50.0, 600.0, 12, dtype=np.float32)
    coarse_npz, table, feasible, trend, merged = _make_inputs(offsets_m=offsets)
    expected_side = np.asarray([-1] * 6 + [1] * 6, dtype=np.int8)
    receiver_x = expected_side.astype(np.float32) * offsets
    coarse_npz['receiver_x_m'] = receiver_x.astype(np.float32)
    coarse_npz['offset_abs_geom_m'] = np.abs(receiver_x).astype(np.float32)

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(
            {
                'physical_trend': {'split_by_offset_gap': False},
                'two_piece_ransac': {'min_pts': 3},
            }
        ),
    )

    np.testing.assert_array_equal(result.physical_model_side, expected_side)


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
    assert np.all(result.physical_model_failure_reason == PHYSICAL_MODEL_FAILURE_NONE)
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
    assert np.all(
        result.physical_model_failure_reason
        == np.uint8(PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID)
    )


def test_physical_center_uses_header_offsets_when_geometry_offset_disabled(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=np.linspace(50.0, 600.0, 12, dtype=np.float32),
        with_geometry=False,
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(
            {
                'physical_trend': {
                    'use_geometry_offset': False,
                    'split_by_offset_gap': False,
                },
                'two_piece_ransac': {'min_pts': 3},
            }
        ),
    )

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], np.abs(table.offset_m), rtol=0, atol=0)
    assert np.all(result.physical_model_status == PHYSICAL_MODEL_STATUS_TWO_PIECE_OK)
    assert np.all(result.physical_model_failure_reason == PHYSICAL_MODEL_FAILURE_NONE)
    assert np.all(
        result.physical_offset_source == np.uint8(PHYSICAL_OFFSET_SOURCE_HEADER)
    )


def test_constant_header_offsets_fall_back_when_geometry_offset_disabled(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=np.full((12,), 100.0, dtype=np.float32),
        with_geometry=False,
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(
            {
                'physical_trend': {'use_geometry_offset': False},
                'physical_prefilter': {'enabled': False},
                'two_piece_ransac': {'min_pts': 3},
            }
        ),
    )

    assert calls == []
    assert not np.any(
        result.physical_model_status == PHYSICAL_MODEL_STATUS_TWO_PIECE_OK
    )
    assert np.all(
        result.physical_model_failure_reason
        == np.uint8(PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS)
    )
    assert np.all(
        result.physical_offset_source == np.uint8(PHYSICAL_OFFSET_SOURCE_HEADER)
    )


def test_header_offset_path_uses_source_xy_groups_without_geometry_offsets(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    offsets = np.linspace(50.0, 600.0, 12, dtype=np.float32)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=offsets,
        with_geometry=False,
    )
    coarse_npz['source_x_m'] = np.asarray([0.0] * 6 + [1000.0] * 6, dtype=np.float32)
    coarse_npz['source_y_m'] = np.zeros((12,), dtype=np.float32)
    coarse_npz['geometry_valid_mask'] = np.ones((12,), dtype=np.bool_)

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(
            {
                'physical_trend': {
                    'use_geometry_offset': False,
                    'split_by_offset_gap': False,
                },
                'two_piece_ransac': {'min_pts': 2},
            }
        ),
    )

    assert sorted(int(call.size) for call in calls) == [6, 6]
    assert np.all(result.physical_model_neighbor_count == np.int32(1))
    assert np.all(
        result.physical_offset_source == np.uint8(PHYSICAL_OFFSET_SOURCE_HEADER)
    )


def test_header_offset_path_uses_table_groups_when_source_xy_degenerate(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    offsets = np.linspace(50.0, 600.0, 12, dtype=np.float32)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=offsets,
        with_geometry=False,
    )
    shot_id = np.asarray([101] * 6 + [102] * 6, dtype=np.int32)
    table = replace(table, shot_id=shot_id, ffid=shot_id.copy())
    coarse_npz['source_x_m'] = np.zeros((12,), dtype=np.float32)
    coarse_npz['source_y_m'] = np.zeros((12,), dtype=np.float32)
    coarse_npz['geometry_valid_mask'] = np.ones((12,), dtype=np.bool_)

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(
            {
                'physical_trend': {
                    'use_geometry_offset': False,
                    'split_by_offset_gap': False,
                },
                'physical_prefilter': {'enabled': False},
                'two_piece_ransac': {'min_pts': 2},
            }
        ),
    )

    assert sorted(int(call.size) for call in calls) == [6, 6]
    assert np.all(result.physical_model_neighbor_count == np.int32(1))
    assert np.all(result.physical_model_status == PHYSICAL_MODEL_STATUS_TWO_PIECE_OK)
    assert np.all(
        result.physical_offset_source == np.uint8(PHYSICAL_OFFSET_SOURCE_HEADER)
    )


def test_header_offset_sign_segmentation_uses_table_offsets_without_geometry(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    offsets = np.asarray(
        [-600.0, -500.0, -400.0, -300.0, 100.0, 200.0, 300.0, 400.0],
        dtype=np.float32,
    )
    pick_i = np.rint(_two_piece_time_sec(np.abs(offsets)) / np.float32(0.001)).astype(
        np.int32
    )
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=offsets,
        pick_i=pick_i,
        with_geometry=False,
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(
            {
                'physical_trend': {
                    'use_geometry_offset': False,
                    'segment_by_offset_sign': True,
                    'split_by_offset_gap': False,
                },
                'physical_prefilter': {'enabled': False},
                'two_piece_ransac': {'min_pts': 2},
            }
        ),
    )

    call_offsets = sorted(
        tuple(np.sort(np.rint(call).astype(np.int32)).tolist()) for call in calls
    )
    assert call_offsets == [(100, 200, 300, 400), (300, 400, 500, 600)]
    np.testing.assert_array_equal(
        result.physical_model_side,
        np.asarray([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.int8),
    )
    assert np.all(
        result.physical_offset_source == np.uint8(PHYSICAL_OFFSET_SOURCE_HEADER)
    )


def test_header_offset_sign_segmentation_ignores_geometry_signed_offsets(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    offsets = np.asarray(
        [-600.0, -500.0, -400.0, -300.0, 100.0, 200.0, 300.0, 400.0],
        dtype=np.float32,
    )
    pick_i = np.rint(_two_piece_time_sec(np.abs(offsets)) / np.float32(0.001)).astype(
        np.int32
    )
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=offsets,
        pick_i=pick_i,
        with_geometry=True,
    )
    coarse_npz['offset_signed_geom_m'] = np.asarray(
        [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        dtype=np.float32,
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(
            {
                'physical_trend': {
                    'use_geometry_offset': False,
                    'segment_by_offset_sign': True,
                    'split_by_offset_gap': False,
                },
                'physical_prefilter': {'enabled': False},
                'two_piece_ransac': {'min_pts': 2},
            }
        ),
    )

    call_offsets = sorted(
        tuple(np.sort(np.rint(call).astype(np.int32)).tolist()) for call in calls
    )
    assert call_offsets == [(100, 200, 300, 400), (300, 400, 500, 600)]
    np.testing.assert_array_equal(
        result.physical_model_side,
        np.asarray([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.int8),
    )
    assert np.all(
        result.physical_offset_source == np.uint8(PHYSICAL_OFFSET_SOURCE_HEADER)
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
    assert np.all(
        result.physical_model_failure_reason
        == np.uint8(PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS)
    )
    assert np.all(result.fine_center_i >= 0)
    assert np.all(result.fine_center_i < table.n_samples_orig)


def test_fit_failed_failure_reason_preserves_fallback_status(monkeypatch) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=np.linspace(50.0, 1200.0, 12, dtype=np.float32),
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg({'two_piece_ransac': {'min_pts': 3}}),
    )

    assert np.all(
        result.physical_model_status
        == np.uint8(PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND)
    )
    assert np.all(
        result.physical_model_failure_reason
        == np.uint8(PHYSICAL_MODEL_FAILURE_FIT_FAILED)
    )
    assert len(calls) == 1


def test_prediction_invalid_failure_reason_preserves_fallback_status(monkeypatch) -> None:
    class NanPredictModel:
        def predict(self, x_abs: torch.Tensor) -> torch.Tensor:
            return torch.full((int(x_abs.numel()),), float('nan'), dtype=torch.float32)

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        return NanPredictModel()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=np.linspace(50.0, 1200.0, 12, dtype=np.float32),
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg({'two_piece_ransac': {'min_pts': 3}}),
    )

    assert np.all(
        result.physical_model_status
        == np.uint8(PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND)
    )
    assert np.all(
        result.physical_model_failure_reason
        == np.uint8(PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID)
    )


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
    assert np.all(result.physical_model_failure_reason == PHYSICAL_MODEL_FAILURE_NONE)


def test_anchor_source_xy_reuses_nearest_anchor_without_non_anchor_fit(
    monkeypatch,
) -> None:
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        x_np = x_abs.detach().cpu().numpy().copy()
        y_np = y_sec.detach().cpu().numpy().copy()
        calls.append((x_np, y_np))
        return _ConstantTrendModel(float(y_np[0]))

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)

    n_groups = 4
    traces_per_group = 4
    dt_sec = 0.001
    group_offsets = np.asarray([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    offsets = np.tile(group_offsets, n_groups).astype(np.float32)
    source_x = np.repeat(
        np.arange(n_groups, dtype=np.float32) * np.float32(100.0),
        traces_per_group,
    )
    group_base_i = np.repeat(
        np.asarray([100, 200, 300, 400], dtype=np.int32),
        traces_per_group,
    )
    pick_i = group_base_i + np.tile(
        np.arange(traces_per_group, dtype=np.int32),
        n_groups,
    )
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=offsets,
        pick_i=pick_i,
        dt_sec=dt_sec,
        n_samples_orig=1000,
    )
    coarse_npz['source_x_m'] = source_x.astype(np.float32)
    coarse_npz['source_y_m'] = np.zeros_like(source_x, dtype=np.float32)
    coarse_npz['receiver_x_m'] = (source_x + offsets).astype(np.float32)
    coarse_npz['receiver_y_m'] = np.zeros_like(source_x, dtype=np.float32)
    coarse_npz['offset_abs_geom_m'] = offsets.astype(np.float32)
    coarse_npz['geometry_valid_mask'] = np.ones_like(offsets, dtype=np.bool_)

    diagnostics = PhysicalRuntimeDiagnostics()
    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(
            {
                'physical_trend': {
                    'segment_by_offset_sign': False,
                    'split_by_offset_gap': False,
                },
                'neighbor_context': {'enabled': False},
                'physical_prefilter': {'enabled': False},
                'two_piece_ransac': {'min_pts': 2},
                'physical_runtime': {
                    'fit_policy': 'anchor_source_xy',
                    'anchor_selection': {
                        'enabled': True,
                        'anchor_stride_source_groups': 2,
                        'include_first': True,
                        'include_last': False,
                    },
                    'anchor_reuse': {'enabled': True},
                },
            }
        ),
        runtime_diagnostics=diagnostics,
    )

    assert len(calls) == 2
    assert diagnostics.n_fit_calls == 2
    assert diagnostics.n_anchor_fit_calls == 2
    assert diagnostics.n_reused_predictions == 2 * traces_per_group
    assert diagnostics.n_fallback_full_fit_no_compatible_anchor == 0
    np.testing.assert_array_equal(
        result.physical_runtime_fit_source.reshape(n_groups, traces_per_group)[:, 0],
        np.asarray([1, 2, 1, 2], dtype=np.uint8),
    )
    assert np.all(
        result.physical_runtime_fit_source[:traces_per_group]
        == np.uint8(PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT)
    )
    assert np.all(
        result.physical_runtime_fit_source[traces_per_group : 2 * traces_per_group]
        == np.uint8(PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE)
    )
    np.testing.assert_array_equal(
        result.physical_center_i[traces_per_group : 2 * traces_per_group],
        np.full((traces_per_group,), pick_i[0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.physical_center_i[3 * traces_per_group : 4 * traces_per_group],
        np.full((traces_per_group,), pick_i[2 * traces_per_group], dtype=np.int32),
    )


def test_anchor_source_xy_full_fit_fallback_when_anchor_side_missing(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        y_np = y_sec.detach().cpu().numpy().copy()
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _ConstantTrendModel(float(y_np[0]))

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)

    traces_per_group = 4
    offsets = np.tile(
        np.asarray([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
        2,
    )
    pick_i = np.asarray([100, 101, 102, 103, 300, 301, 302, 303], dtype=np.int32)
    source_x = np.repeat(np.asarray([0.0, 100.0], dtype=np.float32), traces_per_group)
    signed_offset = np.asarray([-1.0] * traces_per_group + [1.0] * traces_per_group)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=offsets,
        pick_i=pick_i,
        dt_sec=0.001,
        n_samples_orig=1000,
    )
    coarse_npz['source_x_m'] = source_x.astype(np.float32)
    coarse_npz['source_y_m'] = np.zeros_like(source_x, dtype=np.float32)
    coarse_npz['receiver_x_m'] = (source_x + signed_offset * offsets).astype(
        np.float32
    )
    coarse_npz['receiver_y_m'] = np.zeros_like(source_x, dtype=np.float32)
    coarse_npz['offset_abs_geom_m'] = offsets.astype(np.float32)
    coarse_npz['offset_signed_geom_m'] = (signed_offset * offsets).astype(np.float32)
    coarse_npz['geometry_valid_mask'] = np.ones_like(offsets, dtype=np.bool_)

    diagnostics = PhysicalRuntimeDiagnostics()
    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(
            {
                'physical_trend': {
                    'segment_by_offset_sign': True,
                    'split_by_offset_gap': False,
                },
                'neighbor_context': {'enabled': False},
                'physical_prefilter': {'enabled': False},
                'two_piece_ransac': {'min_pts': 2},
                'physical_runtime': {
                    'fit_policy': 'anchor_source_xy',
                    'anchor_selection': {
                        'enabled': True,
                        'anchor_stride_source_groups': 2,
                        'include_first': True,
                        'include_last': False,
                    },
                    'anchor_reuse': {
                        'enabled': True,
                        'fallback_if_no_compatible_segment': 'full_fit',
                    },
                },
            }
        ),
        runtime_diagnostics=diagnostics,
    )

    assert len(calls) == 2
    assert diagnostics.n_fit_calls == 2
    assert diagnostics.n_anchor_fit_calls == 1
    assert diagnostics.n_reused_predictions == 0
    assert diagnostics.n_fallback_full_fit_no_compatible_anchor == 1
    assert np.all(
        result.physical_runtime_fit_source[:traces_per_group]
        == np.uint8(PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT)
    )
    assert np.all(
        result.physical_runtime_fit_source[traces_per_group:]
        == np.uint8(
            PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_FULL_FIT_NO_COMPATIBLE_ANCHOR
        )
    )
    np.testing.assert_array_equal(
        result.physical_center_i[traces_per_group:],
        np.full((traces_per_group,), pick_i[traces_per_group], dtype=np.int32),
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
        'physical_runtime_fit_source': np.uint8,
    }
    for field, dtype in expected_dtypes.items():
        arr = getattr(result, field)
        assert arr.shape == (table.n_traces,)
        assert arr.dtype == np.dtype(dtype)
