from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from fbpick_physical_center_helpers import (
    LinearTrendModel,
    RecordingProgressReporter,
    fake_piecewise_model,
    fit_linear_model,
    make_inputs,
    physical_cfg,
    two_piece_time_sec,
    with_invalid_trend_centers,
)
from seisai_engine.pipelines.fbpick.physics import (
    physical_center_context_fit as context_fit_mod,
)
from seisai_engine.pipelines.fbpick.physics import (
    physical_center_fallback as fallback_mod,
)
from seisai_engine.pipelines.fbpick.physics import (
    physical_center_fit as fit_mod,
)
from seisai_engine.pipelines.fbpick.physics import (
    physical_center_full_fit as full_fit_mod,
)
from seisai_engine.pipelines.fbpick.physics import (
    physical_center_observation as observation_mod,
)
from seisai_engine.pipelines.fbpick.physics import (
    physical_center_prediction as prediction_mod,
)
from seisai_engine.pipelines.fbpick.physics.config import load_physics_lite_config
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
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND,
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST,
    build_geometry_two_piece_physical_center,
)
from seisai_engine.pipelines.fbpick.physics.runtime_diagnostics import (
    PhysicalRuntimeDiagnostics,
)
from seisai_pick.trend.trend_fit_strategy import (
    TwoPieceIRLSAutoBreakStrategy,
    TwoPieceRansacAutoBreakStrategy,
)


def test_physical_disabled_returns_existing_trend_center() -> None:
    inputs = make_inputs(offsets_m=np.linspace(50.0, 1200.0, 12, dtype=np.float32))
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
            self._model = fake_piecewise_model()
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
    inputs = make_inputs(offsets_m=np.linspace(50.0, 1600.0, 12, dtype=np.float32))
    coarse_npz, table, feasible, trend, merged = inputs
    diagnostics = PhysicalRuntimeDiagnostics()

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg({'two_piece_ransac': {'min_pts': 3}}),
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


def test_physical_center_can_use_two_piece_irls_with_pmax_weights(monkeypatch) -> None:
    calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def fake_fit(
        self,
        x_abs: torch.Tensor,
        y_sec: torch.Tensor,
        w_conf: torch.Tensor,
    ):
        calls.append(
            (
                x_abs.detach().cpu().numpy().copy(),
                y_sec.detach().cpu().numpy().copy(),
                w_conf.detach().cpu().numpy().copy(),
            )
        )
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceIRLSAutoBreakStrategy, 'fit', fake_fit)
    pmax = np.linspace(0.1, 0.95, 12, dtype=np.float32)
    inputs = make_inputs(
        offsets_m=np.linspace(50.0, 1600.0, 12, dtype=np.float32),
        pmax=pmax,
    )
    coarse_npz, table, feasible, trend, merged = inputs

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg(
            {
                'physical_trend': {'fit_kind': 'two_piece_irls_autobreak'},
                'two_piece_irls': {'min_pts': 3, 'n_break_cand': 8, 'iters': 3},
            }
        ),
    )

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][2], pmax)
    assert np.any(result.physical_model_status == PHYSICAL_MODEL_STATUS_TWO_PIECE_OK)


def test_parallel_cached_context_hit_accounting_excludes_owner_trace() -> None:
    diagnostics = PhysicalRuntimeDiagnostics()
    plan = observation_mod._ObservationPlan(
        obs_indices=np.arange(4, dtype=np.int64),
        obs_key=(0, 1, 2, 3),
        neighbor_count=1,
        prefilter_valid_count=4,
        segment_id=0,
        side=0,
        relaxed=False,
    )
    work_item = context_fit_mod._FitContextWorkItem(
        fit_key=(0, 1, 2, 3, 0, 0, 0),
        fit_plan=plan,
        obs_count_before_sampling=4,
        trace_indices=np.arange(4, dtype=np.int64),
        runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT,
        assignments=(),
        x_obs=np.arange(4, dtype=np.float32),
        y_obs=np.arange(4, dtype=np.float32),
        w_obs=np.ones((4,), dtype=np.float32),
    )

    context_fit_mod._record_cached_context_hits(
        runtime_diagnostics=diagnostics,
        work_item=work_item,
    )
    context_fit_mod._record_cached_context_hits(
        runtime_diagnostics=diagnostics,
        work_item=replace(
            work_item,
            trace_indices=np.asarray([0], dtype=np.int64),
        ),
    )

    assert diagnostics.n_cache_hits == 3


def test_fit_key_for_obs_uses_precomputed_key_without_rebuilding(
    monkeypatch,
) -> None:
    def fail_indices_key(_indices: np.ndarray) -> tuple[int, ...]:
        raise AssertionError('_indices_key should not be called')

    monkeypatch.setattr(fit_mod, '_indices_key', fail_indices_key)
    diagnostics = PhysicalRuntimeDiagnostics()

    key = fit_mod._fit_key_for_obs(
        np.asarray([10, 11], dtype=np.int64),
        precomputed_key=(10, 11),
        runtime_diagnostics=diagnostics,
    )

    assert key == (10, 11)
    assert diagnostics.n_precomputed_fit_key_used == 1
    assert diagnostics.n_fit_key_built_from_indices == 0


def test_fit_key_for_obs_records_index_build_and_sampling_counters() -> None:
    diagnostics = PhysicalRuntimeDiagnostics()

    key = fit_mod._fit_key_for_obs(
        np.asarray([2, 4], dtype=np.int64),
        runtime_diagnostics=diagnostics,
    )
    sampled_key = fit_mod._fit_key_for_obs(
        np.asarray([4], dtype=np.int64),
        runtime_diagnostics=diagnostics,
        after_sampling=True,
        count_missing_precomputed=False,
    )

    assert key == (2, 4)
    assert sampled_key == (4,)
    assert diagnostics.n_fit_key_built_from_indices == 2
    assert diagnostics.n_fit_key_built_after_sampling == 1
    assert diagnostics.n_fit_key_missing_precomputed == 1


def test_fit_task_cache_preserves_specific_prefit_failure_reason() -> None:
    plan = observation_mod._ObservationPlan(
        obs_indices=np.arange(4, dtype=np.int64),
        obs_key=(0, 1, 2, 3),
        neighbor_count=1,
        prefilter_valid_count=4,
        segment_id=0,
        side=0,
        relaxed=False,
    )
    task_result = fit_mod._FitTaskResult(
        fit_key=fit_mod._fit_cache_key(plan),
        trend_model=None,
        diagnostics=None,
        fit_failed=False,
        failure_reason=PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
        elapsed_sec=0.0,
        obs_count=4,
        obs_count_before_sampling=4,
        fit_attempted=False,
    )
    entry = context_fit_mod._cache_entry_from_fit_task_result(task_result)
    assert entry is not None
    assert entry.failure_reason == PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID

    cache = {fit_mod._fit_cache_key(plan): entry}
    strategy = TwoPieceRansacAutoBreakStrategy(min_pts=3)
    model, diagnostics, failure_reason = fit_mod._fit_model_for_plan(
        strategy=strategy,
        plan=plan,
        x_obs=np.asarray([0.0, 20.0, 40.0, 60.0], dtype=np.float32),
        y_obs=np.asarray([0.0, 0.02, 0.04, 0.06], dtype=np.float32),
        w_obs=np.ones((4,), dtype=np.float32),
        min_pts=3,
        min_offset_spread_m=1.0,
        cache=cache,
    )

    assert model is None
    assert diagnostics is None
    assert failure_reason == PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID


def test_prefit_task_failure_does_not_record_ransac_fit_call() -> None:
    diagnostics = PhysicalRuntimeDiagnostics()
    plan = observation_mod._ObservationPlan(
        obs_indices=np.arange(4, dtype=np.int64),
        obs_key=(0, 1, 2, 3),
        neighbor_count=1,
        prefilter_valid_count=4,
        segment_id=0,
        side=0,
        relaxed=False,
    )
    work_item = context_fit_mod._FitContextWorkItem(
        fit_key=fit_mod._fit_cache_key(plan),
        fit_plan=plan,
        obs_count_before_sampling=4,
        trace_indices=np.arange(4, dtype=np.int64),
        runtime_fit_source=PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT,
        assignments=(),
        x_obs=np.arange(4, dtype=np.float32),
        y_obs=np.arange(4, dtype=np.float32),
        w_obs=np.ones((4,), dtype=np.float32),
    )
    task_result = fit_mod._FitTaskResult(
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

    context_fit_mod._record_new_fit_task_diagnostics(
        runtime_diagnostics=diagnostics,
        work_item=work_item,
        task_result=task_result,
    )

    assert diagnostics.n_cache_misses == 1
    assert diagnostics.n_fit_calls == 0
    assert diagnostics.ransac_fit_total_sec == 0.0
    assert diagnostics.n_cache_hits == 3


def test_assign_model_prediction_batch_matches_single_trace_assignment() -> None:
    _coarse_npz, table, _feasible, _trend, _merged = make_inputs(
        offsets_m=np.asarray([100.0, 200.0, 300.0], dtype=np.float32),
        dt_sec=0.001,
        n_samples_orig=1000,
    )
    plan = observation_mod._ObservationPlan(
        obs_indices=np.arange(3, dtype=np.int64),
        obs_key=(0, 1, 2),
        neighbor_count=1,
        prefilter_valid_count=3,
        segment_id=0,
        side=0,
        relaxed=True,
    )
    diagnostics = (150.0, 0.001, 0.002, 1000.0, 500.0, 1.5, 2.5)
    model = LinearTrendModel(0.001, 0.0)
    single_arrays = fallback_mod._allocate_result_arrays(table)
    batch_arrays = fallback_mod._allocate_result_arrays(table)

    assert prediction_mod._assign_model_prediction(
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
    valid = prediction_mod._assign_model_prediction_batch(
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
    class PartiallyNanModel(LinearTrendModel):
        def predict(self, x_abs: torch.Tensor) -> torch.Tensor:
            pred = super().predict(x_abs)
            return torch.where(
                x_abs == torch.tensor(300.0, dtype=torch.float32),
                torch.full_like(pred, float('nan')),
                pred,
            )

    _coarse_npz, table, _feasible, _trend, _merged = make_inputs(
        offsets_m=np.asarray([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
        dt_sec=0.001,
        n_samples_orig=1000,
    )
    plan = observation_mod._ObservationPlan(
        obs_indices=np.arange(4, dtype=np.int64),
        obs_key=(0, 1, 2, 3),
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
    per_trace_arrays = fallback_mod._allocate_result_arrays(table)
    batch_arrays = fallback_mod._allocate_result_arrays(table)

    per_trace_valid = []
    for trace_idx in trace_indices.tolist():
        per_trace_valid.append(
            prediction_mod._assign_model_prediction(
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
    batch_valid = prediction_mod._assign_model_prediction_batch(
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
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    inputs = make_inputs(offsets_m=np.linspace(50.0, 2500.0, 1000, dtype=np.float32))
    coarse_npz, table, feasible, trend, merged = inputs
    diagnostics = PhysicalRuntimeDiagnostics()

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg(
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
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    inputs = make_inputs(offsets_m=np.linspace(50.0, 2500.0, 1000, dtype=np.float32))
    coarse_npz, table, feasible, trend, merged = inputs
    diagnostics = PhysicalRuntimeDiagnostics()

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg(
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
    cfg = physical_cfg(
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

    first = fit_mod._sample_observation_indices_for_fit(
        obs_indices=obs_indices,
        offset_abs_m=offsets,
        pick_t_sec=two_piece_time_sec(offsets),
        coarse_pmax=pmax,
        cfg=cfg,
    )
    second = fit_mod._sample_observation_indices_for_fit(
        obs_indices=obs_indices,
        offset_abs_m=offsets,
        pick_t_sec=two_piece_time_sec(offsets),
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
    cfg = physical_cfg(
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

    sampled = fit_mod._sample_observation_indices_for_fit(
        obs_indices=obs_indices,
        offset_abs_m=offsets,
        pick_t_sec=two_piece_time_sec(offsets),
        coarse_pmax=pmax,
        cfg=cfg,
    )

    assert int(sampled.size) <= 20
    assert int(sampled[0]) == 0
    assert int(sampled[-1]) == int(obs_indices[-1])


def test_observation_sampling_keeps_small_observation_sets_unchanged() -> None:
    obs_indices = np.arange(50, dtype=np.int64)
    offsets = np.linspace(10.0, 500.0, obs_indices.size, dtype=np.float32)
    cfg = physical_cfg(
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

    sampled = fit_mod._sample_observation_indices_for_fit(
        obs_indices=obs_indices,
        offset_abs_m=offsets,
        pick_t_sec=two_piece_time_sec(offsets),
        coarse_pmax=None,
        cfg=cfg,
    )

    np.testing.assert_array_equal(sampled, obs_indices)


def test_observation_sampling_marks_insufficient_when_too_few_bins() -> None:
    obs_indices = np.arange(1000, dtype=np.int64)
    offsets = np.linspace(10.0, 5000.0, obs_indices.size, dtype=np.float32)
    cfg = physical_cfg(
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

    sampled = fit_mod._sample_observation_indices_for_fit(
        obs_indices=obs_indices,
        offset_abs_m=offsets,
        pick_t_sec=two_piece_time_sec(offsets),
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
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=np.zeros((12,), dtype=np.float32),
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg({'two_piece_ransac': {'min_pts': 3}}),
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
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=np.linspace(50.0, 1200.0, 12, dtype=np.float32),
    )
    coarse_npz['offset_abs_geom_m'] = np.full((12,), 100.0, dtype=np.float32)

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg(
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
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = make_inputs(
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
        cfg=physical_cfg(
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
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = make_inputs(
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
        cfg=physical_cfg(
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
        return fake_piecewise_model()

    original_fit_model_for_plan = full_fit_mod._fit_model_for_plan

    def counting_fit_model_for_plan(**kwargs):
        fit_model_calls.append(fit_mod._fit_cache_key(kwargs['plan']))
        return original_fit_model_for_plan(**kwargs)

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    monkeypatch.setattr(
        full_fit_mod,
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
    coarse_npz, table, feasible, trend, merged = make_inputs(
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
        cfg=physical_cfg({'two_piece_ransac': {'min_pts': 3}}),
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
        return fit_linear_model(x_abs, y_sec)

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
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=offsets,
        pick_i=pick_i,
    )
    base_cfg = {
        'physical_prefilter': {'enabled': False},
        'two_piece_ransac': {'min_pts': 3},
    }
    serial_diagnostics = PhysicalRuntimeDiagnostics()
    parallel_diagnostics = PhysicalRuntimeDiagnostics()
    progress = RecordingProgressReporter()
    original_torch_threads = torch.get_num_threads()
    caller_torch_threads = 2 if original_torch_threads != 2 else 1
    worker_torch_threads = 1 if caller_torch_threads != 1 else 2

    serial = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg(base_cfg),
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
            cfg=physical_cfg(
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
            progress=progress,
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
    fit_progress_events = [
        fields for event, fields in progress.events if event == 'fit.progress'
    ]
    assert fit_progress_events
    assert fit_progress_events[-1]['done'] == 2
    assert fit_progress_events[-1]['total'] == 2


def test_thread_fit_executor_propagates_worker_fit_runtime_error(monkeypatch) -> None:
    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        raise RuntimeError('synthetic worker fit failure')

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    offsets = np.linspace(50.0, 1600.0, 12, dtype=np.float32)
    coarse_npz, table, feasible, trend, merged = make_inputs(offsets_m=offsets)

    with pytest.raises(RuntimeError, match='synthetic worker fit failure'):
        build_geometry_two_piece_physical_center(
            coarse_npz=coarse_npz,
            table=table,
            feasible=feasible,
            trend=trend,
            merged=merged,
            cfg=physical_cfg(
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

def test_synthetic_two_piece_trend_predicts_physical_centers() -> None:
    offsets = np.linspace(50.0, 2000.0, 28, dtype=np.float32)
    coarse_npz, table, feasible, trend, merged = make_inputs(offsets_m=offsets)

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg(
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
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=offsets,
        pick_i=pick_i,
        pmax=pmax,
    )
    cfg = physical_cfg(
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
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=np.linspace(50.0, 1200.0, 12, dtype=np.float32),
        with_geometry=False,
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg(),
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


def test_assign_fallback_all_vectorized_matches_scalar_fallback() -> None:
    _coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=np.linspace(50.0, 1200.0, 4, dtype=np.float32),
        with_geometry=False,
    )
    trend = replace(
        trend,
        trend_center_i=np.asarray([10, -1, -1, table.n_samples_orig], dtype=np.int32),
        trend_center_sec=np.asarray([0.010, np.nan, np.nan, 1.0], dtype=np.float32),
    )
    feasible = replace(
        feasible,
        feasible_lo_sec=np.asarray([np.nan, 0.100, np.nan, 0.300], dtype=np.float32),
        feasible_hi_sec=np.asarray([np.nan, 0.120, np.nan, 0.200], dtype=np.float32),
    )
    merged = replace(
        merged,
        robust_pick_i=np.asarray(
            [300, 300, -5, table.n_samples_orig + 20],
            dtype=np.int32,
        ),
    )

    result = fallback_mod._assign_fallback_all(
        failure_reason=PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
    )
    expected = [
        fallback_mod._fallback_center_for_trace(
            idx,
            table=table,
            feasible=feasible,
            trend=trend,
            merged=merged,
        )
        for idx in range(int(table.n_traces))
    ]

    np.testing.assert_array_equal(
        result.physical_center_i,
        np.asarray(
            [center_i for center_i, _center_t, _status in expected],
            dtype=np.int32,
        ),
    )
    np.testing.assert_allclose(
        result.physical_center_t_sec,
        np.asarray([center_t for _center_i, center_t, _status in expected]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        result.physical_model_status,
        np.asarray(
            [status for _center_i, _center_t, status in expected],
            dtype=np.uint8,
        ),
    )
    np.testing.assert_array_equal(
        result.physical_runtime_fit_source,
        np.asarray(
            [
                PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND,
                PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND,
                PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST,
                PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST,
            ],
            dtype=np.uint8,
        ),
    )


def test_geometry_invalid_fallback_large_trace_count_completes() -> None:
    n_traces = 100_000
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=np.linspace(50.0, 1200.0, n_traces, dtype=np.float32),
        with_geometry=False,
    )
    trend = with_invalid_trend_centers(trend)
    feasible = replace(
        feasible,
        feasible_lo_sec=np.full((n_traces,), np.nan, dtype=np.float32),
        feasible_hi_sec=np.full((n_traces,), np.nan, dtype=np.float32),
    )
    diagnostics = PhysicalRuntimeDiagnostics()

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg(),
        runtime_diagnostics=diagnostics,
    )

    assert diagnostics.n_fit_calls == 0
    assert result.physical_center_i.shape == (n_traces,)
    assert np.all(
        result.physical_model_status
        == np.uint8(PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST)
    )


def test_geometry_invalid_done_logged_after_fallback_assign_all() -> None:
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=np.linspace(50.0, 1200.0, 12, dtype=np.float32),
        with_geometry=False,
    )
    progress = RecordingProgressReporter()

    build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg(),
        progress=progress,
    )

    fallback_start = next(
        idx
        for idx, (event, fields) in enumerate(progress.events)
        if event == 'physical-center.stage_start'
        and fields.get('stage') == 'fallback_assign_all'
    )
    fallback_done = next(
        idx
        for idx, (event, fields) in enumerate(progress.events)
        if event == 'physical-center.stage_done'
        and fields.get('stage') == 'fallback_assign_all'
    )
    physical_done = next(
        idx
        for idx, (event, fields) in enumerate(progress.events)
        if event == 'physical-center.done'
        and fields.get('status') == 'geometry_invalid'
    )

    assert fallback_start < fallback_done < physical_done


def test_geometry_invalid_can_use_robust_fallback() -> None:
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=np.linspace(50.0, 1200.0, 12, dtype=np.float32),
        with_geometry=False,
    )
    trend = with_invalid_trend_centers(trend)
    merged = replace(merged, robust_pick_i=(table.coarse_pick_i + 3).astype(np.int32))

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg(
            {'physical_runtime': {'geometry_invalid_fallback': 'robust'}}
        ),
    )

    np.testing.assert_array_equal(result.fine_center_i, merged.robust_pick_i)
    assert np.all(
        result.physical_model_status
        == np.uint8(PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST)
    )
    assert np.all(
        result.physical_runtime_fit_source
        == np.uint8(PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST)
    )


def test_physical_center_uses_header_offsets_when_geometry_offset_disabled(
    monkeypatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        calls.append(x_abs.detach().cpu().numpy().copy())
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=np.linspace(50.0, 600.0, 12, dtype=np.float32),
        with_geometry=False,
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg(
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
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=np.full((12,), 100.0, dtype=np.float32),
        with_geometry=False,
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg(
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
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    offsets = np.linspace(50.0, 600.0, 12, dtype=np.float32)
    coarse_npz, table, feasible, trend, merged = make_inputs(
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
        cfg=physical_cfg(
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
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    offsets = np.linspace(50.0, 600.0, 12, dtype=np.float32)
    coarse_npz, table, feasible, trend, merged = make_inputs(
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
        cfg=physical_cfg(
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
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    offsets = np.asarray(
        [-600.0, -500.0, -400.0, -300.0, 100.0, 200.0, 300.0, 400.0],
        dtype=np.float32,
    )
    pick_i = np.rint(two_piece_time_sec(np.abs(offsets)) / np.float32(0.001)).astype(
        np.int32
    )
    coarse_npz, table, feasible, trend, merged = make_inputs(
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
        cfg=physical_cfg(
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
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    offsets = np.asarray(
        [-600.0, -500.0, -400.0, -300.0, 100.0, 200.0, 300.0, 400.0],
        dtype=np.float32,
    )
    pick_i = np.rint(two_piece_time_sec(np.abs(offsets)) / np.float32(0.001)).astype(
        np.int32
    )
    coarse_npz, table, feasible, trend, merged = make_inputs(
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
        cfg=physical_cfg(
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
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=np.linspace(50.0, 1200.0, 12, dtype=np.float32),
        with_geometry=False,
    )
    trend = with_invalid_trend_centers(trend)
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
        cfg=physical_cfg(),
    )

    assert np.all(
        result.physical_model_status
        == np.uint8(PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP)
    )
    assert np.all(result.fine_center_t_sec >= feasible.feasible_lo_sec)
    assert np.all(result.fine_center_t_sec <= feasible.feasible_hi_sec)


def test_fallback_status_reports_robust_when_trend_and_feasible_clip_are_unusable() -> None:
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=np.linspace(50.0, 1200.0, 12, dtype=np.float32),
        with_geometry=False,
    )
    trend = with_invalid_trend_centers(trend)
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
        cfg=physical_cfg(),
    )

    assert np.all(
        result.physical_model_status
        == np.uint8(PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST)
    )
    assert np.all(result.fine_center_i == table.n_samples_orig - 1)


def test_insufficient_observations_falls_back_inside_sample_range() -> None:
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=np.linspace(50.0, 500.0, 6, dtype=np.float32),
    )
    cfg = physical_cfg({'two_piece_ransac': {'min_pts': 4}})

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
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=np.linspace(50.0, 1200.0, 12, dtype=np.float32),
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg({'two_piece_ransac': {'min_pts': 3}}),
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
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=np.linspace(50.0, 1200.0, 12, dtype=np.float32),
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg({'two_piece_ransac': {'min_pts': 3}}),
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
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=offsets,
        pick_i=pick_i,
    )

    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg({'two_piece_ransac': {'min_pts': 2}}),
    )

    assert np.all(
        result.physical_model_status
        == np.uint8(PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT)
    )
    assert np.all(result.physical_model_failure_reason == PHYSICAL_MODEL_FAILURE_NONE)
