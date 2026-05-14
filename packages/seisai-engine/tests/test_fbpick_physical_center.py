from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from seisai_engine.pipelines.fbpick.physics import (
    physical_center as physical_center_mod,
)
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
    PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT,
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


def _source_groups_for_context_tests() -> tuple[
    physical_center_mod.SourceGroup,
    ...,
]:
    return (
        physical_center_mod.SourceGroup(
            group_id=0,
            source_key_x=0,
            source_key_y=0,
            source_x_m=0.0,
            source_y_m=0.0,
            trace_indices=np.asarray([0, 1], dtype=np.int64),
        ),
        physical_center_mod.SourceGroup(
            group_id=1,
            source_key_x=100,
            source_key_y=0,
            source_x_m=100.0,
            source_y_m=0.0,
            trace_indices=np.asarray([2, 3], dtype=np.int64),
        ),
        physical_center_mod.SourceGroup(
            group_id=2,
            source_key_x=250,
            source_key_y=0,
            source_x_m=250.0,
            source_y_m=0.0,
            trace_indices=np.asarray([4, 5], dtype=np.int64),
        ),
        physical_center_mod.SourceGroup(
            group_id=3,
            source_key_x=600,
            source_key_y=0,
            source_x_m=600.0,
            source_y_m=0.0,
            trace_indices=np.asarray([6, 7], dtype=np.int64),
        ),
    )


def test_group_observation_contexts_match_existing_selection_and_filtering() -> None:
    groups = _source_groups_for_context_tests()
    groups_by_id = {int(group.group_id): group for group in groups}
    valid_for_fit = np.asarray(
        [True, False, True, True, False, True, True, False],
        dtype=np.bool_,
    )
    cfg = _physical_cfg(
        {
            'physical_trend': {
                'segment_by_offset_sign': False,
                'split_by_offset_gap': False,
            },
            'neighbor_context': {
                'enabled': True,
                'k_neighbors': 3,
                'include_self': False,
                'max_source_distance_m': 260.0,
            },
            'two_piece_ransac': {'min_pts': 2},
        }
    )

    contexts = physical_center_mod._build_group_observation_contexts(
        groups=groups,
        groups_by_id=groups_by_id,
        valid_for_fit=valid_for_fit,
        cfg=cfg,
        use_neighbor_context=True,
    )

    assert set(contexts) == {0, 1, 2, 3}
    for group in groups:
        group_id = int(group.group_id)
        expected_group_ids = physical_center_mod._select_group_ids(
            groups=groups,
            target_group_id=group_id,
            cfg=cfg,
            use_neighbor_context=True,
        )
        expected_neighbor_indices = physical_center_mod._concat_group_traces(
            expected_group_ids,
            groups_by_id=groups_by_id,
        )
        expected_valid_obs = expected_neighbor_indices[
            valid_for_fit[expected_neighbor_indices]
        ]

        context = contexts[group_id]
        assert context.group_id == group_id
        assert context.neighbor_count == int(expected_group_ids.size)
        assert context.prefilter_valid_count == int(expected_valid_obs.size)
        np.testing.assert_array_equal(
            context.neighbor_group_ids,
            expected_group_ids,
        )
        np.testing.assert_array_equal(
            context.neighbor_indices,
            expected_neighbor_indices,
        )
        np.testing.assert_array_equal(
            context.valid_obs_indices,
            expected_valid_obs,
        )

    plan = physical_center_mod._build_observation_plan(
        trace_idx=4,
        target_group_id=2,
        group_context_by_id=contexts,
        geometry=None,
        offset_abs_m=np.arange(8, dtype=np.float32),
        offset_signed_m=None,
        cfg=cfg,
    )

    assert plan is not None
    np.testing.assert_array_equal(plan.obs_indices, contexts[2].valid_obs_indices)
    assert plan.neighbor_count == contexts[2].neighbor_count
    assert plan.prefilter_valid_count == contexts[2].prefilter_valid_count


def test_group_observation_contexts_use_self_group_when_neighbor_disabled() -> None:
    groups = _source_groups_for_context_tests()
    groups_by_id = {int(group.group_id): group for group in groups}
    valid_for_fit = np.asarray(
        [True, False, True, True, False, True, True, False],
        dtype=np.bool_,
    )
    cfg = _physical_cfg({'neighbor_context': {'enabled': False}})

    contexts = physical_center_mod._build_group_observation_contexts(
        groups=groups,
        groups_by_id=groups_by_id,
        valid_for_fit=valid_for_fit,
        cfg=cfg,
        use_neighbor_context=True,
    )

    for group in groups:
        group_id = int(group.group_id)
        trace_indices = np.asarray(group.trace_indices, dtype=np.int64)
        expected_valid_obs = trace_indices[valid_for_fit[trace_indices]]
        context = contexts[group_id]
        np.testing.assert_array_equal(
            context.neighbor_group_ids,
            np.asarray([group_id], dtype=np.int64),
        )
        np.testing.assert_array_equal(context.neighbor_indices, trace_indices)
        np.testing.assert_array_equal(context.valid_obs_indices, expected_valid_obs)
        assert context.neighbor_count == 1
        assert context.prefilter_valid_count == int(expected_valid_obs.size)


def test_group_observation_contexts_precompute_signed_side_sets() -> None:
    groups = _source_groups_for_context_tests()
    groups_by_id = {int(group.group_id): group for group in groups}
    valid_for_fit = np.asarray(
        [True, False, True, True, False, True, True, False],
        dtype=np.bool_,
    )
    cfg = _physical_cfg(
        {
            'physical_trend': {
                'segment_by_offset_sign': True,
                'split_by_offset_gap': False,
            },
            'neighbor_context': {'enabled': False},
            'two_piece_ransac': {'min_pts': 2},
        }
    )
    labels = physical_center_mod._signed_offset_side_labels(
        np.asarray([-3.0, 2.0, -1.0, 0.0, 4.0, 5.0, -6.0, 7.0], dtype=np.float32)
    )
    cache = physical_center_mod._ObservationPlanCache(offset_signed_labels=labels)
    diagnostics = PhysicalRuntimeDiagnostics(detailed_timing=True)

    contexts = physical_center_mod._build_group_observation_contexts(
        groups=groups,
        groups_by_id=groups_by_id,
        valid_for_fit=valid_for_fit,
        cfg=cfg,
        use_neighbor_context=True,
        offset_signed_labels=labels,
        plan_cache=cache,
        runtime_diagnostics=diagnostics,
    )

    for context in contexts.values():
        assert context.side_context is not None
        obs = context.valid_obs_indices
        for side in (-1, 0, 1):
            expected = obs[labels.side[obs] == side]
            np.testing.assert_array_equal(
                context.side_context.obs_indices_by_side[side + 1],
                expected,
            )
            assert context.side_context.obs_key_by_side[side + 1] == tuple(
                expected.tolist()
            )

    summary = diagnostics.to_summary()
    assert summary['n_side_contexts_built'] == len(contexts)
    assert summary['side_filter_precompute_sec'] >= 0.0
    assert summary['side_obs_count_p50'] >= 0.0


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


class _LinearTrendModel:
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


def _fit_linear_model(x_abs: torch.Tensor, y_sec: torch.Tensor) -> _LinearTrendModel:
    x_np = x_abs.detach().cpu().numpy().astype(np.float64, copy=False)
    y_np = y_sec.detach().cpu().numpy().astype(np.float64, copy=False)
    slope, intercept = np.polyfit(x_np, y_np, deg=1)
    return _LinearTrendModel(float(slope), float(intercept))


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
    assert diagnostics.n_prediction_calls == int(table.n_traces)
    assert diagnostics.n_prediction_batches == 1
    assert model.predict_call_sizes.count(1) == 0
    assert model.predict_call_sizes.count(int(table.n_traces)) == 2
    assert np.any(result.physical_model_status == PHYSICAL_MODEL_STATUS_TWO_PIECE_OK)


def test_parallel_cached_context_hit_accounting_excludes_owner_trace() -> None:
    diagnostics = PhysicalRuntimeDiagnostics()
    plan = physical_center_mod._ObservationPlan(
        obs_indices=np.arange(4, dtype=np.int64),
        neighbor_count=1,
        prefilter_valid_count=4,
        segment_id=0,
        side=0,
        relaxed=False,
    )
    work_item = physical_center_mod._FitContextWorkItem(
        fit_key=(0, 1, 2, 3, 0, 0, 0),
        fit_plan=plan,
        obs_count_before_sampling=4,
        trace_indices=np.arange(4, dtype=np.int64),
        runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT,
        assignments=(),
        x_obs=np.arange(4, dtype=np.float32),
        y_obs=np.arange(4, dtype=np.float32),
    )

    physical_center_mod._record_cached_context_hits(
        runtime_diagnostics=diagnostics,
        work_item=work_item,
    )
    physical_center_mod._record_cached_context_hits(
        runtime_diagnostics=diagnostics,
        work_item=replace(
            work_item,
            trace_indices=np.asarray([0], dtype=np.int64),
        ),
    )

    assert diagnostics.n_cache_hits == 3


def test_fit_task_cache_preserves_specific_prefit_failure_reason() -> None:
    plan = physical_center_mod._ObservationPlan(
        obs_indices=np.arange(4, dtype=np.int64),
        neighbor_count=1,
        prefilter_valid_count=4,
        segment_id=0,
        side=0,
        relaxed=False,
    )
    task_result = physical_center_mod._FitTaskResult(
        fit_key=physical_center_mod._fit_cache_key(plan),
        trend_model=None,
        diagnostics=None,
        fit_failed=False,
        failure_reason=PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
        elapsed_sec=0.0,
        obs_count=4,
        obs_count_before_sampling=4,
        fit_attempted=False,
    )
    entry = physical_center_mod._cache_entry_from_fit_task_result(task_result)
    assert entry is not None
    assert entry.failure_reason == PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID

    cache = {physical_center_mod._fit_cache_key(plan): entry}
    strategy = TwoPieceRansacAutoBreakStrategy(min_pts=3)
    model, diagnostics, failure_reason = physical_center_mod._fit_model_for_plan(
        strategy=strategy,
        plan=plan,
        x_obs=np.asarray([0.0, 20.0, 40.0, 60.0], dtype=np.float32),
        y_obs=np.asarray([0.0, 0.02, 0.04, 0.06], dtype=np.float32),
        min_pts=3,
        min_offset_spread_m=1.0,
        cache=cache,
    )

    assert model is None
    assert diagnostics is None
    assert failure_reason == PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID


def test_prefit_task_failure_does_not_record_ransac_fit_call() -> None:
    diagnostics = PhysicalRuntimeDiagnostics()
    plan = physical_center_mod._ObservationPlan(
        obs_indices=np.arange(4, dtype=np.int64),
        neighbor_count=1,
        prefilter_valid_count=4,
        segment_id=0,
        side=0,
        relaxed=False,
    )
    work_item = physical_center_mod._FitContextWorkItem(
        fit_key=physical_center_mod._fit_cache_key(plan),
        fit_plan=plan,
        obs_count_before_sampling=4,
        trace_indices=np.arange(4, dtype=np.int64),
        runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT,
        assignments=(),
        x_obs=np.arange(4, dtype=np.float32),
        y_obs=np.arange(4, dtype=np.float32),
    )
    task_result = physical_center_mod._FitTaskResult(
        fit_key=work_item.fit_key,
        trend_model=None,
        diagnostics=None,
        fit_failed=False,
        failure_reason=PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
        elapsed_sec=0.0,
        obs_count=4,
        obs_count_before_sampling=4,
        fit_attempted=False,
    )

    physical_center_mod._record_new_fit_task_diagnostics(
        runtime_diagnostics=diagnostics,
        work_item=work_item,
        task_result=task_result,
    )

    assert diagnostics.n_cache_misses == 1
    assert diagnostics.n_fit_calls == 0
    assert diagnostics.ransac_fit_total_sec == 0.0
    assert diagnostics.n_cache_hits == 3


def test_assign_model_prediction_batch_matches_single_trace_assignment() -> None:
    _coarse_npz, table, _feasible, _trend, _merged = _make_inputs(
        offsets_m=np.asarray([100.0, 200.0, 300.0], dtype=np.float32),
        dt_sec=0.001,
        n_samples_orig=1000,
    )
    plan = physical_center_mod._ObservationPlan(
        obs_indices=np.arange(3, dtype=np.int64),
        neighbor_count=1,
        prefilter_valid_count=3,
        segment_id=0,
        side=0,
        relaxed=True,
    )
    diagnostics = (150.0, 0.001, 0.002, 1000.0, 500.0, 1.5, 2.5)
    model = _LinearTrendModel(0.001, 0.0)
    single_arrays = physical_center_mod._allocate_result_arrays(table)
    batch_arrays = physical_center_mod._allocate_result_arrays(table)

    assert physical_center_mod._assign_model_prediction(
        single_arrays,
        1,
        trend_model=model,
        diagnostics=diagnostics,
        plan=plan,
        offset_abs_m=table.offset_m,
        dt=float(table.dt_scalar_sec),
        n_samples=int(table.n_samples_orig),
        runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT,
        t0_shift_sec=0.01,
    )
    valid = physical_center_mod._assign_model_prediction_batch(
        batch_arrays,
        np.asarray([1], dtype=np.int64),
        trend_model=model,
        diagnostics=diagnostics,
        plan_by_trace=plan,
        offset_abs_m=table.offset_m,
        dt=float(table.dt_scalar_sec),
        n_samples=int(table.n_samples_orig),
        runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT,
        t0_shift_sec=0.01,
    )

    np.testing.assert_array_equal(valid, np.asarray([True], dtype=np.bool_))
    for key in single_arrays:
        np.testing.assert_array_equal(batch_arrays[key], single_arrays[key])


def test_assign_model_prediction_batch_handles_vector_shift_and_invalid() -> None:
    class PartiallyNanModel(_LinearTrendModel):
        def predict(self, x_abs: torch.Tensor) -> torch.Tensor:
            pred = super().predict(x_abs)
            return torch.where(
                x_abs == torch.tensor(300.0, dtype=torch.float32),
                torch.full_like(pred, float('nan')),
                pred,
            )

    _coarse_npz, table, _feasible, _trend, _merged = _make_inputs(
        offsets_m=np.asarray([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
        dt_sec=0.001,
        n_samples_orig=1000,
    )
    plan = physical_center_mod._ObservationPlan(
        obs_indices=np.arange(4, dtype=np.int64),
        neighbor_count=1,
        prefilter_valid_count=4,
        segment_id=0,
        side=0,
        relaxed=False,
    )
    diagnostics = (200.0, 0.001, 0.002, 1000.0, 500.0, 1.0, 2.0)
    trace_indices = np.arange(4, dtype=np.int64)
    shifts = np.asarray([0.0, 0.01, 0.0, 0.02], dtype=np.float64)
    model = PartiallyNanModel(0.001, 0.0)
    per_trace_arrays = physical_center_mod._allocate_result_arrays(table)
    batch_arrays = physical_center_mod._allocate_result_arrays(table)

    per_trace_valid = []
    for trace_idx in trace_indices.tolist():
        per_trace_valid.append(
            physical_center_mod._assign_model_prediction(
                per_trace_arrays,
                int(trace_idx),
                trend_model=model,
                diagnostics=diagnostics,
                plan=plan,
                offset_abs_m=table.offset_m,
                dt=float(table.dt_scalar_sec),
                n_samples=int(table.n_samples_orig),
                runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT,
                t0_shift_sec=float(shifts[int(trace_idx)]),
            )
        )
    batch_valid = physical_center_mod._assign_model_prediction_batch(
        batch_arrays,
        trace_indices,
        trend_model=model,
        diagnostics=diagnostics,
        plan_by_trace={int(idx): plan for idx in trace_indices.tolist()},
        offset_abs_m=table.offset_m,
        dt=float(table.dt_scalar_sec),
        n_samples=int(table.n_samples_orig),
        runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT,
        t0_shift_sec=shifts,
    )

    np.testing.assert_array_equal(
        batch_valid,
        np.asarray(per_trace_valid, dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        batch_valid,
        np.asarray([True, True, False, True], dtype=np.bool_),
    )
    for key in per_trace_arrays:
        np.testing.assert_array_equal(batch_arrays[key], per_trace_arrays[key])
    assert np.isnan(batch_arrays['physical_model_break_offset_m'][2])
    np.testing.assert_array_equal(
        batch_arrays['physical_model_break_offset_m'][batch_valid],
        np.full((3,), np.float32(diagnostics[0]), dtype=np.float32),
    )


def test_observation_sampling_limits_observations_before_ransac(monkeypatch) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    inputs = _make_inputs(offsets_m=np.linspace(50.0, 2500.0, 1000, dtype=np.float32))
    coarse_npz, table, feasible, trend, merged = inputs
    diagnostics = PhysicalRuntimeDiagnostics()

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(
            {
                'physical_runtime': {
                    'observation_sampling': {
                        'enabled': True,
                        'max_obs_per_fit': 100,
                        'n_offset_bins': 200,
                        'bin_pick': 'pmax_max',
                        'min_obs_per_fit_after_sampling': 8,
                        'preserve_edge_bins': True,
                    }
                },
                'two_piece_ransac': {'min_pts': 4},
            }
        ),
        runtime_diagnostics=diagnostics,
    )

    assert len(calls) == 1
    assert int(calls[0].size) <= 100
    summary = diagnostics.to_summary()
    assert summary['observation_sampling_enabled'] == 1
    assert summary['obs_count_before_p50'] == 1000.0
    assert summary['obs_count_after_p50'] <= 100.0
    assert summary['obs_downsample_rate_p50'] > 0.0
    assert np.any(result.physical_model_status == PHYSICAL_MODEL_STATUS_TWO_PIECE_OK)


def test_observation_sampling_does_not_use_unsampled_fallback_for_ransac(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    inputs = _make_inputs(offsets_m=np.linspace(50.0, 2500.0, 1000, dtype=np.float32))
    coarse_npz, table, feasible, trend, merged = inputs
    diagnostics = PhysicalRuntimeDiagnostics()

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(
            {
                'physical_runtime': {
                    'observation_sampling': {
                        'enabled': True,
                        'max_obs_per_fit': 100,
                        'n_offset_bins': 4,
                        'min_obs_per_fit_after_sampling': 8,
                    }
                },
                'two_piece_ransac': {'min_pts': 3},
            }
        ),
        runtime_diagnostics=diagnostics,
    )

    assert calls == []
    assert diagnostics.n_fit_calls == 0
    assert np.all(
        result.physical_model_failure_reason
        == np.uint8(PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS)
    )


def test_observation_sampling_is_deterministic_for_pmax_max() -> None:
    obs_indices = np.arange(1000, dtype=np.int64)
    offsets = np.linspace(10.0, 5000.0, obs_indices.size, dtype=np.float32)
    pmax = np.linspace(0.1, 0.9, obs_indices.size, dtype=np.float32)
    cfg = _physical_cfg(
        {
            'physical_runtime': {
                'observation_sampling': {
                    'enabled': True,
                    'max_obs_per_fit': 100,
                    'n_offset_bins': 200,
                    'bin_pick': 'pmax_max',
                }
            }
        }
    )

    first = physical_center_mod._sample_observation_indices_for_fit(
        obs_indices=obs_indices,
        offset_abs_m=offsets,
        pick_t_sec=_two_piece_time_sec(offsets),
        coarse_pmax=pmax,
        cfg=cfg,
    )
    second = physical_center_mod._sample_observation_indices_for_fit(
        obs_indices=obs_indices,
        offset_abs_m=offsets,
        pick_t_sec=_two_piece_time_sec(offsets),
        coarse_pmax=pmax,
        cfg=cfg,
    )

    assert int(first.size) <= 100
    np.testing.assert_array_equal(first, second)


def test_observation_sampling_preserves_edge_bins() -> None:
    obs_indices = np.arange(1000, dtype=np.int64)
    offsets = np.linspace(10.0, 5000.0, obs_indices.size, dtype=np.float32)
    pmax = np.zeros((obs_indices.size,), dtype=np.float32)
    pmax[0] = 1.0
    pmax[-1] = 1.0
    cfg = _physical_cfg(
        {
            'physical_runtime': {
                'observation_sampling': {
                    'enabled': True,
                    'max_obs_per_fit': 20,
                    'n_offset_bins': 100,
                    'bin_pick': 'pmax_max',
                    'preserve_edge_bins': True,
                }
            }
        }
    )

    sampled = physical_center_mod._sample_observation_indices_for_fit(
        obs_indices=obs_indices,
        offset_abs_m=offsets,
        pick_t_sec=_two_piece_time_sec(offsets),
        coarse_pmax=pmax,
        cfg=cfg,
    )

    assert int(sampled.size) <= 20
    assert int(sampled[0]) == 0
    assert int(sampled[-1]) == int(obs_indices[-1])


def test_observation_sampling_keeps_small_observation_sets_unchanged() -> None:
    obs_indices = np.arange(50, dtype=np.int64)
    offsets = np.linspace(10.0, 500.0, obs_indices.size, dtype=np.float32)
    cfg = _physical_cfg(
        {
            'physical_runtime': {
                'observation_sampling': {
                    'enabled': True,
                    'max_obs_per_fit': 100,
                    'n_offset_bins': 20,
                }
            }
        }
    )

    sampled = physical_center_mod._sample_observation_indices_for_fit(
        obs_indices=obs_indices,
        offset_abs_m=offsets,
        pick_t_sec=_two_piece_time_sec(offsets),
        coarse_pmax=None,
        cfg=cfg,
    )

    np.testing.assert_array_equal(sampled, obs_indices)


def test_observation_sampling_marks_insufficient_when_too_few_bins() -> None:
    obs_indices = np.arange(1000, dtype=np.int64)
    offsets = np.linspace(10.0, 5000.0, obs_indices.size, dtype=np.float32)
    cfg = _physical_cfg(
        {
            'physical_runtime': {
                'observation_sampling': {
                    'enabled': True,
                    'max_obs_per_fit': 100,
                    'n_offset_bins': 4,
                    'min_obs_per_fit_after_sampling': 8,
                }
            }
        }
    )

    sampled = physical_center_mod._sample_observation_indices_for_fit(
        obs_indices=obs_indices,
        offset_abs_m=offsets,
        pick_t_sec=_two_piece_time_sec(offsets),
        coarse_pmax=None,
        cfg=cfg,
    )

    assert sampled.dtype == np.int64
    assert int(sampled.size) == 0


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
    fit_model_calls: list[tuple[int, ...]] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fake_piecewise_model()

    original_fit_model_for_plan = physical_center_mod._fit_model_for_plan

    def counting_fit_model_for_plan(**kwargs):
        fit_model_calls.append(physical_center_mod._fit_cache_key(kwargs['plan']))
        return original_fit_model_for_plan(**kwargs)

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    monkeypatch.setattr(
        physical_center_mod,
        '_fit_model_for_plan',
        counting_fit_model_for_plan,
    )
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

    assert len(fit_model_calls) == 2
    assert len(calls) == 2
    assert sorted(int(call.size) for call in calls) == [6, 6]
    assert diagnostics.n_fit_calls == 2
    assert diagnostics.n_cache_misses == 2
    assert diagnostics.n_cache_hits == 10
    assert diagnostics.n_unique_fit_contexts == 2
    assert np.all(result.physical_model_status == PHYSICAL_MODEL_STATUS_TWO_PIECE_OK)


def test_thread_fit_executor_matches_serial_unique_contexts(monkeypatch) -> None:
    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        return _fit_linear_model(x_abs, y_sec)

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
    base_cfg = {
        'physical_prefilter': {'enabled': False},
        'two_piece_ransac': {'min_pts': 3},
    }
    serial_diagnostics = PhysicalRuntimeDiagnostics()
    parallel_diagnostics = PhysicalRuntimeDiagnostics()
    original_torch_threads = torch.get_num_threads()
    caller_torch_threads = 2 if original_torch_threads != 2 else 1
    worker_torch_threads = 1 if caller_torch_threads != 1 else 2

    serial = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=_physical_cfg(base_cfg),
        runtime_diagnostics=serial_diagnostics,
    )
    torch.set_num_threads(caller_torch_threads)
    try:
        parallel = build_geometry_two_piece_physical_center(
            coarse_npz=coarse_npz,
            table=table,
            feasible=feasible,
            trend=trend,
            merged=merged,
            cfg=_physical_cfg(
                {
                    **base_cfg,
                    'physical_runtime': {
                        'fit_executor': {
                            'enabled': True,
                            'backend': 'thread',
                            'max_workers': 2,
                            'torch_num_threads_per_worker': worker_torch_threads,
                            'chunksize': 1,
                        }
                    },
                }
            ),
            runtime_diagnostics=parallel_diagnostics,
        )
        assert torch.get_num_threads() == caller_torch_threads
    finally:
        torch.set_num_threads(original_torch_threads)

    for field in (
        'physical_center_i',
        'fine_center_i',
        'physical_model_status',
        'physical_model_failure_reason',
    ):
        np.testing.assert_array_equal(getattr(parallel, field), getattr(serial, field))
    for field in (
        'physical_model_break_offset_m',
        'physical_model_slope_near_s_per_m',
        'physical_model_slope_far_s_per_m',
        'physical_model_velocity_near_m_s',
        'physical_model_velocity_far_m_s',
        'physical_model_resid_p50_ms',
        'physical_model_resid_p90_ms',
    ):
        np.testing.assert_allclose(
            getattr(parallel, field),
            getattr(serial, field),
            equal_nan=True,
            atol=1.0e-6,
            rtol=0.0,
        )
    assert parallel_diagnostics.n_fit_calls == serial_diagnostics.n_fit_calls == 2
    assert parallel_diagnostics.n_cache_misses == serial_diagnostics.n_cache_misses
    assert parallel_diagnostics.n_cache_hits == serial_diagnostics.n_cache_hits
    summary = parallel_diagnostics.to_summary()
    assert summary['fit_executor_enabled'] == 1
    assert summary['fit_executor_backend'] == 'thread'
    assert summary['fit_executor_max_workers'] == 2
    assert summary['fit_executor_tasks'] == 2
    assert summary['fit_executor_wall_sec'] >= 0.0


def test_thread_fit_executor_propagates_worker_fit_runtime_error(monkeypatch) -> None:
    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        raise RuntimeError('synthetic worker fit failure')

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    offsets = np.linspace(50.0, 1600.0, 12, dtype=np.float32)
    coarse_npz, table, feasible, trend, merged = _make_inputs(offsets_m=offsets)

    with pytest.raises(RuntimeError, match='synthetic worker fit failure'):
        build_geometry_two_piece_physical_center(
            coarse_npz=coarse_npz,
            table=table,
            feasible=feasible,
            trend=trend,
            merged=merged,
            cfg=_physical_cfg(
                {
                    'two_piece_ransac': {'min_pts': 3},
                    'physical_runtime': {
                        'fit_executor': {
                            'enabled': True,
                            'backend': 'thread',
                            'max_workers': 2,
                        }
                    },
                }
            ),
        )


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


def _legacy_append_index(indices: np.ndarray, trace_idx: int) -> np.ndarray:
    return physical_center_mod._stable_unique(
        np.concatenate(
            [
                np.asarray(indices, dtype=np.int64),
                np.asarray([int(trace_idx)], dtype=np.int64),
            ]
        )
    )


def _legacy_trace_position_map(indices: np.ndarray) -> dict[int, int]:
    return {
        int(trace_idx): int(pos)
        for pos, trace_idx in enumerate(np.asarray(indices, dtype=np.int64).tolist())
    }


def _legacy_obs_with_target_side(
    *,
    trace_idx: int,
    obs_indices: np.ndarray,
    geometry,
) -> tuple[np.ndarray, int, bool]:
    context_indices = _legacy_append_index(obs_indices, trace_idx)
    signed = physical_center_mod.signed_offset_side_from_geometry(
        geometry,
        context_indices,
    )
    obs = np.asarray(obs_indices, dtype=np.int64)
    if not bool(signed.reliable):
        return obs, 0, False

    pos = _legacy_trace_position_map(context_indices)
    target_side = int(signed.side[pos[int(trace_idx)]])
    obs_side = np.asarray(
        [int(signed.side[pos[int(obs_idx)]]) for obs_idx in obs.tolist()],
        dtype=np.int8,
    )
    return obs[obs_side == target_side], target_side, True


def _legacy_obs_with_target_signed_offset_side(
    *,
    trace_idx: int,
    obs_indices: np.ndarray,
    signed_offset_m: np.ndarray,
    zero_tol_m: float = 1.0e-6,
) -> tuple[np.ndarray, int, bool]:
    context_indices = _legacy_append_index(obs_indices, trace_idx)
    signed = np.asarray(signed_offset_m, dtype=np.float32)[context_indices]
    finite = np.isfinite(signed)
    obs = np.asarray(obs_indices, dtype=np.int64)
    if int(np.count_nonzero(finite)) < 2:
        return obs, 0, False

    zero_tol = float(zero_tol_m)
    if zero_tol < 0.0 or not np.isfinite(zero_tol):
        msg = 'zero_tol_m must be finite and >= 0'
        raise ValueError(msg)

    side = np.zeros((context_indices.size,), dtype=np.int8)
    signed_valid = np.asarray(signed[finite], dtype=np.float64)
    if not np.any(np.abs(signed_valid) > zero_tol):
        return obs, 0, False

    side[finite] = np.where(
        np.abs(signed_valid) <= zero_tol,
        0,
        np.where(signed_valid < 0.0, -1, 1),
    ).astype(np.int8)
    pos = _legacy_trace_position_map(context_indices)
    target_side = int(side[pos[int(trace_idx)]])
    obs_side = np.asarray(
        [int(side[pos[int(obs_idx)]]) for obs_idx in obs.tolist()],
        dtype=np.int8,
    )
    return obs[obs_side == target_side], target_side, True


def _legacy_obs_with_target_gap_segment(
    *,
    trace_idx: int,
    obs_indices: np.ndarray,
    offset_abs_m: np.ndarray,
    cfg,
) -> tuple[np.ndarray, int]:
    context_indices = _legacy_append_index(obs_indices, trace_idx)
    segment_id = physical_center_mod.split_offset_gap_segments(
        np.asarray(offset_abs_m, dtype=np.float32)[context_indices],
        split_by_offset_gap=bool(cfg.physical_trend.split_by_offset_gap),
        gap_ratio=float(cfg.physical_trend.gap_ratio),
        min_gap_m=cfg.physical_trend.min_gap_m,
    )
    pos = _legacy_trace_position_map(context_indices)
    target_segment_id = int(segment_id[pos[int(trace_idx)]])
    obs = np.asarray(obs_indices, dtype=np.int64)
    obs_segment_id = np.asarray(
        [int(segment_id[pos[int(obs_idx)]]) for obs_idx in obs.tolist()],
        dtype=np.int64,
    )
    return obs[obs_segment_id == target_segment_id], target_segment_id


def _legacy_build_observation_plan(
    *,
    trace_idx: int,
    target_group_id: int,
    group_context_by_id,
    geometry,
    offset_abs_m: np.ndarray,
    offset_signed_m: np.ndarray | None,
    cfg,
    runtime_diagnostics=None,
    plan_cache=None,
):
    group_context = group_context_by_id.get(int(target_group_id))
    if group_context is None:
        msg = f'observation context not found for group_id={int(target_group_id)}'
        raise ValueError(msg)

    valid_obs = group_context.valid_obs_indices
    neighbor_count = int(group_context.neighbor_count)
    prefilter_valid_count = int(group_context.prefilter_valid_count)
    min_fit_obs = 2 * int(cfg.two_piece_ransac.min_pts)

    if prefilter_valid_count < min_fit_obs:
        return physical_center_mod._ObservationPlan(
            obs_indices=valid_obs,
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=-1,
            side=0,
            relaxed=False,
        )

    side_obs = valid_obs
    side = 0
    side_reliable = False
    if bool(cfg.physical_trend.segment_by_offset_sign):
        if offset_signed_m is not None:
            side_obs, side, side_reliable = (
                _legacy_obs_with_target_signed_offset_side(
                    trace_idx=trace_idx,
                    obs_indices=valid_obs,
                    signed_offset_m=offset_signed_m,
                )
            )
        elif geometry is not None:
            side_obs, side, side_reliable = _legacy_obs_with_target_side(
                trace_idx=trace_idx,
                obs_indices=valid_obs,
                geometry=geometry,
            )

    segment_obs = side_obs
    segment_id = 0
    if bool(cfg.physical_trend.split_by_offset_gap):
        segment_obs, segment_id = _legacy_obs_with_target_gap_segment(
            trace_idx=trace_idx,
            obs_indices=side_obs,
            offset_abs_m=offset_abs_m,
            cfg=cfg,
        )

    if int(segment_obs.size) >= min_fit_obs:
        return physical_center_mod._ObservationPlan(
            obs_indices=segment_obs,
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=segment_id,
            side=side,
            relaxed=False,
        )

    if (
        bool(cfg.physical_trend.split_by_offset_gap)
        and int(side_obs.size) >= min_fit_obs
    ):
        return physical_center_mod._ObservationPlan(
            obs_indices=side_obs,
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=0,
            side=side,
            relaxed=True,
        )

    if side_reliable and int(valid_obs.size) >= min_fit_obs:
        return physical_center_mod._ObservationPlan(
            obs_indices=valid_obs,
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=0,
            side=0,
            relaxed=True,
        )

    return physical_center_mod._ObservationPlan(
        obs_indices=segment_obs,
        neighbor_count=neighbor_count,
        prefilter_valid_count=prefilter_valid_count,
        segment_id=segment_id,
        side=side,
        relaxed=False,
    )


def test_physical_center_cached_observation_plan_matches_legacy_outputs(
    monkeypatch,
) -> None:
    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        return _fit_linear_model(x_abs, y_sec)

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    side = np.asarray([-1.0] * 8 + [1.0] * 8, dtype=np.float32)
    offsets = np.tile(
        np.asarray(
            [100.0, 110.0, 120.0, 130.0, 1000.0, 1010.0, 1020.0, 1030.0],
            dtype=np.float32,
        ),
        2,
    )
    pick_i = np.rint(_two_piece_time_sec(offsets) / np.float32(0.001)).astype(
        np.int32
    )
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=offsets,
        pick_i=pick_i,
    )
    coarse_npz['receiver_x_m'] = (side * offsets).astype(np.float32)
    coarse_npz['offset_abs_geom_m'] = offsets.astype(np.float32)
    coarse_npz['offset_signed_geom_m'] = (side * offsets).astype(np.float32)

    cfg = _physical_cfg(
        {
            'physical_trend': {
                'segment_by_offset_sign': True,
                'split_by_offset_gap': True,
            },
            'physical_prefilter': {'enabled': False},
            'neighbor_context': {'enabled': True, 'k_neighbors': 1},
            'two_piece_ransac': {'min_pts': 2},
        }
    )

    diagnostics = PhysicalRuntimeDiagnostics(detailed_timing=True)
    cached = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=cfg,
        runtime_diagnostics=diagnostics,
    )

    monkeypatch.setattr(
        physical_center_mod,
        '_build_observation_plan',
        _legacy_build_observation_plan,
    )
    legacy = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=cfg,
    )

    for field in (
        'physical_center_i',
        'physical_model_status',
        'physical_model_failure_reason',
        'physical_model_segment_id',
        'physical_model_side',
        'physical_prefilter_valid_count',
    ):
        np.testing.assert_array_equal(getattr(cached, field), getattr(legacy, field))

    summary = diagnostics.to_summary()
    assert summary['n_side_contexts_built'] > 0
    assert summary['n_gap_fast_path_calls'] > 0
    assert summary['n_gap_fallback_calls'] == 0


def test_gap_segment_context_fast_path_and_fallback_match_legacy() -> None:
    cfg = _physical_cfg(
        {
            'physical_trend': {
                'segment_by_offset_sign': False,
                'split_by_offset_gap': True,
                'gap_ratio': 5.0,
            },
            'two_piece_ransac': {'min_pts': 2},
        }
    )
    offset_abs_m = np.asarray(
        [100.0, 110.0, 120.0, 1000.0, 1010.0, 1020.0, 2000.0],
        dtype=np.float32,
    )
    obs = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    obs_key = tuple(obs.tolist())
    cache = physical_center_mod._ObservationPlanCache()
    diagnostics = PhysicalRuntimeDiagnostics(detailed_timing=True)

    fast_obs, fast_key, fast_segment_id = (
        physical_center_mod._obs_with_target_gap_segment(
            trace_idx=4,
            obs_indices=obs,
            obs_key=obs_key,
            offset_abs_m=offset_abs_m,
            cfg=cfg,
            cache=cache,
            runtime_diagnostics=diagnostics,
        )
    )
    legacy_fast_obs, legacy_fast_segment_id = _legacy_obs_with_target_gap_segment(
        trace_idx=4,
        obs_indices=obs,
        offset_abs_m=offset_abs_m,
        cfg=cfg,
    )
    np.testing.assert_array_equal(fast_obs, legacy_fast_obs)
    assert fast_key == tuple(legacy_fast_obs.tolist())
    assert fast_segment_id == legacy_fast_segment_id

    fallback_obs, fallback_key, fallback_segment_id = (
        physical_center_mod._obs_with_target_gap_segment(
            trace_idx=6,
            obs_indices=obs,
            obs_key=obs_key,
            offset_abs_m=offset_abs_m,
            cfg=cfg,
            cache=cache,
            runtime_diagnostics=diagnostics,
        )
    )
    legacy_fallback_obs, legacy_fallback_segment_id = (
        _legacy_obs_with_target_gap_segment(
            trace_idx=6,
            obs_indices=obs,
            offset_abs_m=offset_abs_m,
            cfg=cfg,
        )
    )
    np.testing.assert_array_equal(fallback_obs, legacy_fallback_obs)
    assert fallback_key == tuple(legacy_fallback_obs.tolist())
    assert fallback_segment_id == legacy_fallback_segment_id

    summary = diagnostics.to_summary()
    assert summary['n_gap_contexts_built'] == 1
    assert summary['n_gap_fast_path_calls'] == 1
    assert summary['n_gap_fallback_calls'] == 1
    assert summary['n_gap_trace_in_obs'] == 1
    assert summary['n_gap_trace_not_in_obs'] == 1


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


def test_anchor_source_xy_t0_shift_estimates_constant_target_shift(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fit_linear_model(x_abs, y_sec)

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)

    traces_per_group = 4
    offsets = np.tile(
        np.asarray([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
        2,
    )
    anchor_pick_i = np.asarray([100, 200, 300, 400], dtype=np.int32)
    pick_i = np.concatenate([anchor_pick_i, anchor_pick_i + 10]).astype(np.int32)
    source_x = np.repeat(np.asarray([0.0, 100.0], dtype=np.float32), traces_per_group)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=offsets,
        pick_i=pick_i,
        dt_sec=0.001,
        n_samples_orig=1000,
    )
    coarse_npz['source_x_m'] = source_x.astype(np.float32)
    coarse_npz['receiver_x_m'] = (source_x + offsets).astype(np.float32)

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
                    'anchor_reuse': {
                        'enabled': True,
                        'non_anchor_mode': 'nearest_anchor_plus_t0_shift',
                    },
                    't0_shift': {'min_valid_for_t0_shift': 4},
                },
            }
        ),
        runtime_diagnostics=diagnostics,
    )

    assert len(calls) == 1
    assert diagnostics.n_fit_calls == 1
    assert diagnostics.n_t0_shifted_groups == 1
    assert diagnostics.n_t0_shifted_predictions == traces_per_group
    assert diagnostics.n_reused_predictions == traces_per_group
    np.testing.assert_array_equal(result.physical_center_i, pick_i)
    np.testing.assert_allclose(
        result.physical_runtime_t0_shift_ms[traces_per_group:],
        np.full((traces_per_group,), 10.0, dtype=np.float32),
        atol=1.0e-4,
    )
    np.testing.assert_array_equal(
        result.physical_runtime_reuse_valid_count[traces_per_group:],
        np.full((traces_per_group,), traces_per_group, dtype=np.int32),
    )


def test_anchor_source_xy_t0_shift_clipping_limits_reuse_shift(
    monkeypatch,
) -> None:
    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        return _fit_linear_model(x_abs, y_sec)

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)

    traces_per_group = 4
    offsets = np.tile(
        np.asarray([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
        2,
    )
    anchor_pick_i = np.asarray([100, 200, 300, 400], dtype=np.int32)
    pick_i = np.concatenate([anchor_pick_i, anchor_pick_i + 100]).astype(np.int32)
    source_x = np.repeat(np.asarray([0.0, 100.0], dtype=np.float32), traces_per_group)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=offsets,
        pick_i=pick_i,
        dt_sec=0.001,
        n_samples_orig=1000,
    )
    coarse_npz['source_x_m'] = source_x.astype(np.float32)
    coarse_npz['receiver_x_m'] = (source_x + offsets).astype(np.float32)

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
                    'anchor_reuse': {
                        'enabled': True,
                        'non_anchor_mode': 'nearest_anchor_plus_t0_shift',
                    },
                    't0_shift': {
                        'min_valid_for_t0_shift': 4,
                        't0_shift_clip_ms': 60.0,
                    },
                },
            }
        ),
    )

    expected = np.concatenate([anchor_pick_i, anchor_pick_i + 60]).astype(np.int32)
    np.testing.assert_array_equal(result.physical_center_i, expected)
    np.testing.assert_allclose(
        result.physical_runtime_t0_shift_ms[traces_per_group:],
        np.full((traces_per_group,), 60.0, dtype=np.float32),
        atol=1.0e-4,
    )


def test_anchor_source_xy_adaptive_refit_reduces_bad_reuse_tail(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return _fit_linear_model(x_abs, y_sec)

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)

    traces_per_group = 4
    offsets = np.tile(
        np.asarray([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
        2,
    )
    anchor_pick_i = np.asarray([100, 200, 300, 400], dtype=np.int32)
    target_pick_i = np.asarray([120, 240, 360, 480], dtype=np.int32)
    pick_i = np.concatenate([anchor_pick_i, target_pick_i]).astype(np.int32)
    source_x = np.repeat(np.asarray([0.0, 100.0], dtype=np.float32), traces_per_group)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=offsets,
        pick_i=pick_i,
        dt_sec=0.001,
        n_samples_orig=1000,
    )
    coarse_npz['source_x_m'] = source_x.astype(np.float32)
    coarse_npz['receiver_x_m'] = (source_x + offsets).astype(np.float32)

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
                    'anchor_reuse': {
                        'enabled': True,
                        'non_anchor_mode': 'nearest_anchor_plus_t0_shift',
                    },
                    't0_shift': {'min_valid_for_t0_shift': 4},
                    'adaptive_refit': {
                        'enabled': True,
                        'resid_p90_ms_gt': 5.0,
                        'median_abs_shift_ms_gt': 1000.0,
                        'min_valid_for_resid_check': 4,
                    },
                },
            }
        ),
        runtime_diagnostics=diagnostics,
    )

    assert len(calls) == 2
    assert diagnostics.n_adaptive_refit_calls == 1
    assert diagnostics.n_adaptive_refit_success == 1
    assert diagnostics.n_adaptive_refit_failed == 0
    assert diagnostics.n_reused_predictions == 0
    np.testing.assert_array_equal(result.physical_center_i, pick_i)
    assert np.all(result.physical_runtime_refit_mask[traces_per_group:])
    assert np.all(
        result.physical_runtime_fit_source[traces_per_group:]
        == np.uint8(PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT)
    )


def test_anchor_source_xy_adaptive_refit_failure_falls_back_to_t0_shift(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        if len(calls) == 1:
            return _fit_linear_model(x_abs, y_sec)
        raise RuntimeError('synthetic refit failure')

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)

    traces_per_group = 4
    offsets = np.tile(
        np.asarray([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
        2,
    )
    anchor_pick_i = np.asarray([100, 200, 300, 400], dtype=np.int32)
    pick_i = np.concatenate([anchor_pick_i, anchor_pick_i + 10]).astype(np.int32)
    source_x = np.repeat(np.asarray([0.0, 100.0], dtype=np.float32), traces_per_group)
    coarse_npz, table, feasible, trend, merged = _make_inputs(
        offsets_m=offsets,
        pick_i=pick_i,
        dt_sec=0.001,
        n_samples_orig=1000,
    )
    coarse_npz['source_x_m'] = source_x.astype(np.float32)
    coarse_npz['receiver_x_m'] = (source_x + offsets).astype(np.float32)

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
                    'anchor_reuse': {
                        'enabled': True,
                        'non_anchor_mode': 'nearest_anchor_plus_t0_shift',
                    },
                    't0_shift': {'min_valid_for_t0_shift': 4},
                    'adaptive_refit': {
                        'enabled': True,
                        'resid_p90_ms_gt': 1000.0,
                        'median_abs_shift_ms_gt': 5.0,
                        'min_valid_for_resid_check': 4,
                        'fallback_if_refit_fails': 'nearest_anchor_plus_t0_shift',
                    },
                },
            }
        ),
        runtime_diagnostics=diagnostics,
    )

    assert len(calls) == 2
    assert diagnostics.n_adaptive_refit_calls == 1
    assert diagnostics.n_adaptive_refit_success == 0
    assert diagnostics.n_adaptive_refit_failed == 1
    assert diagnostics.n_t0_shifted_groups == 1
    assert diagnostics.n_t0_shifted_predictions == traces_per_group
    assert diagnostics.n_reused_predictions == traces_per_group
    np.testing.assert_array_equal(result.physical_center_i, pick_i)
    assert np.all(result.physical_runtime_refit_mask[traces_per_group:])
    assert np.all(
        result.physical_runtime_fit_source[traces_per_group:]
        == np.uint8(PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE)
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
        'physical_runtime_t0_shift_ms': np.float32,
        'physical_runtime_reuse_resid_p50_ms': np.float32,
        'physical_runtime_reuse_resid_p90_ms': np.float32,
        'physical_runtime_reuse_valid_count': np.int32,
        'physical_runtime_refit_mask': np.bool_,
        'physical_runtime_fit_source': np.uint8,
    }
    for field, dtype in expected_dtypes.items():
        arr = getattr(result, field)
        assert arr.shape == (table.n_traces,)
        assert arr.dtype == np.dtype(dtype)
