from __future__ import annotations

import inspect
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from seisai_engine.pipelines.fbpick.common import (
    COARSE_GEOMETRY_EXTRA_OPTIONAL_KEYS,
    COARSE_GEOMETRY_OPTIONAL_KEYS,
    REASON_MASK_FILLED_FROM_TREND,
    REASON_MASK_INFEASIBLE,
    REASON_MASK_LOW_SCORE,
    ROBUST_CENTER_OPTIONAL_KEYS,
    ROBUST_OPTIONAL_KEYS,
    ROBUST_PHYSICAL_DIAGNOSTIC_OPTIONAL_KEYS,
    ROBUST_PHYSICAL_OPTIONAL_KEYS,
    ROBUST_REQUIRED_KEYS,
    ROBUST_RUNTIME_DIAGNOSTIC_OPTIONAL_KEYS,
    ROBUST_SOURCE_COARSE_OBSERVED,
    ROBUST_SOURCE_TREND_FILL,
    build_lineage_payload,
    load_coarse_npz,
    load_robust_npz,
    read_git_sha,
    save_coarse_npz,
    save_robust_npz,
)
from seisai_engine.pipelines.fbpick.physics.confidence import compute_confidence_terms
from seisai_engine.pipelines.fbpick.physics.config import (
    load_physics_lite_config,
    physics_lite_config_to_dict,
)
from seisai_engine.pipelines.fbpick.physics.feasible import (
    compute_feasible_band,
    compute_velocity_t0_band_from_arrays,
)
from seisai_engine.pipelines.fbpick.physics.merge import apply_keep_reject_fill
from seisai_engine.pipelines.fbpick.physics.physical_center import (
    PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
    PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED,
    PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND,
    PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED,
    PHYSICAL_MODEL_STATUS_TWO_PIECE_OK,
    PHYSICAL_OFFSET_SOURCE_HEADER,
)
from seisai_engine.pipelines.fbpick.physics.pick_table import (
    normalize_coarse_pick_table,
)
from seisai_engine.pipelines.fbpick.physics.run import (
    build_robust_payload_from_coarse,
    derive_physics_runtime_summary_path,
    run_physics_lite,
)
from seisai_engine.pipelines.fbpick.physics.runtime_diagnostics import (
    PhysicalRuntimeDiagnostics,
    runtime_summary_from_npz_fields,
)
from seisai_engine.pipelines.fbpick.physics.trend import TrendResult, build_trend_result

_ANCHOR_OPTIONAL_KEYS = {
    'physical_anchor_group_id',
    'physical_anchor_is_anchor',
    'physical_anchor_nearest_anchor_group_id',
    'physical_anchor_source_distance_m',
    'n_anchor_groups',
    'anchor_stride_source_groups',
    'anchor_selection_mode',
    'anchor_source_distance_p50_m',
    'anchor_source_distance_p90_m',
    'anchor_source_distance_max_m',
}


def _make_coarse_payload(
    *,
    coarse_pick_i: np.ndarray,
    coarse_pmax: np.ndarray,
    offsets_m: np.ndarray,
    dt_sec: float = 0.004,
    ffid_values: np.ndarray | None = None,
    chno_values: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    pick_i = np.asarray(coarse_pick_i, dtype=np.int32)
    pmax = np.asarray(coarse_pmax, dtype=np.float32)
    offsets = np.asarray(offsets_m, dtype=np.float32)
    n_traces = int(pick_i.shape[0])
    if pick_i.ndim != 1 or pmax.shape != (n_traces,) or offsets.shape != (n_traces,):
        msg = 'coarse arrays must be 1D and same length'
        raise ValueError(msg)
    ffid = (
        np.ones((n_traces,), dtype=np.int32)
        if ffid_values is None
        else np.asarray(ffid_values, dtype=np.int32)
    )
    chno = (
        np.arange(1, n_traces + 1, dtype=np.int32)
        if chno_values is None
        else np.asarray(chno_values, dtype=np.int32)
    )
    return {
        'dt_sec': np.asarray(dt_sec, dtype=np.float32),
        'n_samples_orig': np.asarray(512, dtype=np.int32),
        'n_traces': np.asarray(n_traces, dtype=np.int32),
        'ffid_values': ffid,
        'chno_values': chno,
        'offsets_m': offsets,
        'trace_indices': np.arange(n_traces, dtype=np.int64),
        'coarse_pick_i': pick_i,
        'coarse_pick_t_sec': pick_i.astype(np.float32) * np.float32(dt_sec),
        'coarse_pmax': pmax,
        'coarse_prob_summary': pmax.copy(),
        'lineage': np.asarray('{"stage":"coarse-test"}'),
    }


def _make_physical_geometry(offsets_m: np.ndarray) -> dict[str, np.ndarray]:
    offsets = np.asarray(offsets_m, dtype=np.float32)
    n_traces = int(offsets.shape[0])
    return {
        'source_x_m': np.zeros((n_traces,), dtype=np.float32),
        'source_y_m': np.zeros((n_traces,), dtype=np.float32),
        'receiver_x_m': offsets.astype(np.float32, copy=True),
        'receiver_y_m': np.zeros((n_traces,), dtype=np.float32),
        'offset_abs_geom_m': np.abs(offsets).astype(np.float32, copy=False),
        'geometry_valid_mask': np.ones((n_traces,), dtype=np.bool_),
    }


def _make_robust_payload() -> dict[str, np.ndarray]:
    return {
        'dt_sec': np.asarray(0.004, dtype=np.float32),
        'n_samples_orig': np.asarray(512, dtype=np.int32),
        'n_traces': np.asarray(3, dtype=np.int32),
        'ffid_values': np.array([1, 1, 1], dtype=np.int32),
        'chno_values': np.array([1, 2, 3], dtype=np.int32),
        'offsets_m': np.array([100.0, 200.0, 300.0], dtype=np.float32),
        'trace_indices': np.array([0, 1, 2], dtype=np.int64),
        'robust_pick_i': np.array([100, 101, 102], dtype=np.int32),
        'robust_pick_t_sec': np.array([0.4, 0.404, 0.408], dtype=np.float32),
        'robust_conf': np.array([0.9, 0.8, 0.7], dtype=np.float32),
        'robust_source': np.array([0, 2, 0], dtype=np.uint8),
        'used_theoretical_mask': np.array([False, False, False], dtype=np.bool_),
        'reason_mask': np.array(
            [
                0,
                REASON_MASK_INFEASIBLE | REASON_MASK_FILLED_FROM_TREND,
                0,
            ],
            dtype=np.uint8,
        ),
        'conf_prob1': np.array([0.9, 0.2, 0.8], dtype=np.float32),
        'conf_trend1': np.array([0.9, 0.6, 0.8], dtype=np.float32),
        'conf_rs1': np.array([1.0, 1.0, 1.0], dtype=np.float32),
        'lineage': np.asarray(
            '{"iter_id":"","source_model_id":"x","cfg_hash":"y","git_sha":"z"}'
        ),
    }


def _make_robust_optional_payload() -> dict[str, np.ndarray]:
    center_i = np.array([100, 105, 110], dtype=np.int32)
    center_t_sec = center_i.astype(np.float32) * np.float32(0.004)
    return {
        'trend_center_i': center_i,
        'trend_center_t_sec': center_t_sec,
        'physical_center_i': center_i,
        'physical_center_t_sec': center_t_sec,
        'fine_center_i': center_i,
        'fine_center_t_sec': center_t_sec,
        'physical_model_status': np.array([0, 1, 2], dtype=np.uint8),
        'physical_model_failure_reason': np.array([0, 2, 3], dtype=np.uint8),
        'physical_offset_source': np.array([1, 2, 0], dtype=np.uint8),
        'physical_model_break_offset_m': np.array(
            [500.0, np.nan, 600.0],
            dtype=np.float32,
        ),
        'physical_model_slope_near_s_per_m': np.array(
            [0.001, np.nan, 0.0012],
            dtype=np.float32,
        ),
        'physical_model_slope_far_s_per_m': np.array(
            [0.0004, np.nan, 0.0005],
            dtype=np.float32,
        ),
        'physical_model_velocity_near_m_s': np.array(
            [1000.0, np.nan, 833.0],
            dtype=np.float32,
        ),
        'physical_model_velocity_far_m_s': np.array(
            [2500.0, np.nan, 2000.0],
            dtype=np.float32,
        ),
        'physical_model_neighbor_count': np.array([3, 3, 3], dtype=np.int32),
        'physical_prefilter_valid_count': np.array([8, 8, 8], dtype=np.int32),
        'physical_model_segment_id': np.array([0, -1, 0], dtype=np.int32),
        'physical_model_side': np.array([1, 0, 1], dtype=np.int8),
        'physical_model_resid_p50_ms': np.array(
            [1.0, np.nan, 2.0],
            dtype=np.float32,
        ),
        'physical_model_resid_p90_ms': np.array(
            [3.0, np.nan, 4.0],
            dtype=np.float32,
        ),
        'physical_anchor_group_id': np.array([0, 1, 2], dtype=np.int32),
        'physical_anchor_is_anchor': np.array([True, False, True], dtype=np.bool_),
        'physical_anchor_nearest_anchor_group_id': np.array(
            [0, 0, 2],
            dtype=np.int32,
        ),
        'physical_anchor_source_distance_m': np.array(
            [0.0, 10.0, 0.0],
            dtype=np.float32,
        ),
        'physical_runtime_t0_shift_ms': np.array(
            [10.0, np.nan, 20.0],
            dtype=np.float32,
        ),
        'physical_runtime_reuse_resid_p50_ms': np.array(
            [1.0, np.nan, 2.0],
            dtype=np.float32,
        ),
        'physical_runtime_reuse_resid_p90_ms': np.array(
            [2.0, np.nan, 3.0],
            dtype=np.float32,
        ),
        'physical_runtime_reuse_valid_count': np.array([3, 0, 4], dtype=np.int32),
        'physical_runtime_refit_mask': np.array([False, True, False], dtype=np.bool_),
        'physical_runtime_fit_source': np.array([1, 2, 3], dtype=np.uint8),
        'physics_total_sec': np.asarray(1.0, dtype=np.float64),
        'physical_center_total_sec': np.asarray(0.7, dtype=np.float64),
        'ransac_fit_total_sec': np.asarray(0.3, dtype=np.float64),
        'n_fit_calls': np.asarray(4, dtype=np.int64),
        'n_anchor_fit_calls': np.asarray(2, dtype=np.int64),
        'n_cache_hits': np.asarray(2, dtype=np.int64),
        'n_cache_misses': np.asarray(4, dtype=np.int64),
        'cache_hit_rate': np.asarray(2.0 / 6.0, dtype=np.float64),
        'n_source_groups': np.asarray(1, dtype=np.int64),
        'n_non_anchor_groups': np.asarray(1, dtype=np.int64),
        'n_reused_predictions': np.asarray(3, dtype=np.int64),
        'n_t0_shifted_groups': np.asarray(1, dtype=np.int64),
        'n_t0_shifted_predictions': np.asarray(3, dtype=np.int64),
        't0_shift_ms_p50': np.asarray(10.0, dtype=np.float64),
        't0_shift_ms_p90': np.asarray(10.0, dtype=np.float64),
        't0_shift_ms_p99': np.asarray(10.0, dtype=np.float64),
        'reuse_resid_p90_ms_p50': np.asarray(2.0, dtype=np.float64),
        'reuse_resid_p90_ms_p90': np.asarray(2.0, dtype=np.float64),
        'n_adaptive_refit_calls': np.asarray(1, dtype=np.int64),
        'adaptive_refit_rate': np.asarray(1.0, dtype=np.float64),
        'n_adaptive_refit_success': np.asarray(1, dtype=np.int64),
        'n_adaptive_refit_failed': np.asarray(0, dtype=np.int64),
        'n_fallback_full_fit_no_compatible_anchor': np.asarray(0, dtype=np.int64),
        'n_unique_fit_contexts': np.asarray(4, dtype=np.int64),
        'fit_call_reduction_rate_vs_full': np.asarray(0.5, dtype=np.float64),
        'ransac_fit_time_p50_sec': np.asarray(0.05, dtype=np.float64),
        'ransac_fit_time_p90_sec': np.asarray(0.09, dtype=np.float64),
        'ransac_fit_time_p99_sec': np.asarray(0.099, dtype=np.float64),
        'observation_sampling_enabled': np.asarray(1, dtype=np.int64),
        'observation_sampling_method': np.asarray('offset_bin'),
        'max_obs_per_fit': np.asarray(256, dtype=np.int64),
        'n_offset_bins': np.asarray(64, dtype=np.int64),
        'obs_count_before_p50': np.asarray(12.0, dtype=np.float64),
        'obs_count_before_p90': np.asarray(16.0, dtype=np.float64),
        'obs_count_before_p99': np.asarray(19.0, dtype=np.float64),
        'obs_count_after_p50': np.asarray(8.0, dtype=np.float64),
        'obs_count_after_p90': np.asarray(10.0, dtype=np.float64),
        'obs_count_after_p99': np.asarray(11.0, dtype=np.float64),
        'obs_downsample_rate_p50': np.asarray(0.25, dtype=np.float64),
        'obs_downsample_rate_p90': np.asarray(0.5, dtype=np.float64),
        'obs_count_for_fit_p50': np.asarray(8.0, dtype=np.float64),
        'obs_count_for_fit_p90': np.asarray(10.0, dtype=np.float64),
        'obs_count_for_fit_p99': np.asarray(11.0, dtype=np.float64),
        'n_anchor_groups': np.asarray(2, dtype=np.int64),
        'anchor_stride_source_groups': np.asarray(2, dtype=np.int64),
        'anchor_selection_mode': np.asarray('source_xy_stride'),
        'anchor_source_distance_p50_m': np.asarray(0.0, dtype=np.float64),
        'anchor_source_distance_p90_m': np.asarray(8.0, dtype=np.float64),
        'anchor_source_distance_max_m': np.asarray(10.0, dtype=np.float64),
    }


def _write_git_repo(repo_root: Path, *, sha: str) -> None:
    ref_path = repo_root / '.git' / 'refs' / 'heads' / 'main'
    ref_path.parent.mkdir(parents=True)
    (repo_root / '.git' / 'HEAD').write_text('ref: refs/heads/main\n', encoding='utf-8')
    ref_path.write_text(f'{sha}\n', encoding='utf-8')


def _physical_trend_blocks() -> dict[str, object]:
    return {
        'physical_trend': {
            'enabled': True,
            'fit_kind': 'two_piece_ransac_autobreak',
            'use_geometry_offset': True,
            'min_offset_spread_m': 12.5,
            'coord_group_tol_m': 2.0,
            'segment_by_offset_sign': False,
            'split_by_offset_gap': True,
            'gap_ratio': 6.0,
            'min_gap_m': 25.0,
        },
        'neighbor_context': {
            'enabled': True,
            'mode': 'nearest_source_xy',
            'k_neighbors': 7,
            'max_source_distance_m': 1000.0,
            'include_self': False,
        },
        'physical_prefilter': {
            'enabled': True,
            'vmin_m_s': 400.0,
            'vmax_m_s': 5500.0,
            't0_lo_ms': -10.0,
            't0_hi_ms': 180.0,
            'pmax_min': 0.25,
            'use_existing_feasible_mask': True,
        },
        'two_piece_ransac': {
            'n_iter': 300,
            'inlier_th_ms': 30.0,
            'min_pts': 10,
            'n_break_cand': 32,
            'q_lo': 0.2,
            'q_hi': 0.8,
            'seed': 11,
            'slope_eps': 1.0e-5,
            'sort_offsets': False,
        },
        'physical_projection': {
            'mode': 'model',
        },
    }


def test_physical_runtime_diagnostics_initializes_with_zero_counts() -> None:
    diagnostics = PhysicalRuntimeDiagnostics()
    summary = diagnostics.to_summary()

    assert summary['n_fit_calls'] == 0
    assert summary['n_cache_hits'] == 0
    assert summary['n_cache_misses'] == 0
    assert summary['cache_hit_rate'] == 0.0
    assert summary['observation_sampling_enabled'] == 0
    assert summary['obs_count_before_p50'] == 0.0
    assert summary['obs_count_after_p50'] == 0.0
    assert summary['fit_executor_enabled'] == 0
    assert summary['fit_executor_backend'] == 'serial'
    assert summary['fit_executor_max_workers'] == 0
    assert summary['fit_executor_wall_sec'] == 0.0
    assert summary['fit_executor_tasks'] == 0
    assert summary['n_source_groups'] == 0
    assert summary['n_side_contexts_built'] == 0
    assert summary['n_gap_contexts_built'] == 0
    assert summary['n_gap_fast_path_calls'] == 0
    assert summary['n_gap_fallback_calls'] == 0
    assert summary['side_obs_count_p50'] == 0.0
    assert summary['gap_segment_obs_count_p50'] == 0.0
    assert summary['n_unique_fit_contexts'] == 0
    assert summary['n_prediction_batches'] == 0
    assert summary['n_t0_shifted_groups'] == 0
    assert summary['n_adaptive_refit_calls'] == 0


def test_physical_runtime_diagnostics_fit_timer_increments_counts() -> None:
    diagnostics = PhysicalRuntimeDiagnostics()

    with diagnostics.time_ransac_fit(obs_count=8, obs_count_before=12):
        pass

    summary = diagnostics.to_summary()
    assert summary['n_fit_calls'] == 1
    assert summary['ransac_fit_total_sec'] >= 0.0
    assert summary['obs_count_for_fit_p50'] == 8.0
    assert summary['obs_count_before_p50'] == 12.0
    assert summary['obs_count_after_p50'] == 8.0
    assert summary['obs_downsample_rate_p50'] == pytest.approx(1.0 / 3.0)


def test_physical_runtime_diagnostics_detailed_timer_and_derived_fields() -> None:
    diagnostics = PhysicalRuntimeDiagnostics(detailed_timing=True)
    diagnostics.physical_center_total_sec = 10.0
    diagnostics.ransac_fit_total_sec = 3.0

    with diagnostics.time_block('neighbor_plan_sec'):
        pass
    diagnostics.inc('n_prediction_calls', 2)
    diagnostics.inc('n_prediction_batches')
    diagnostics.set_fit_executor(enabled=True, backend='thread', max_workers=2)
    diagnostics.record_fit_executor_run(wall_sec=0.25, tasks=3)

    summary = diagnostics.to_summary()
    assert summary['neighbor_plan_sec'] >= 0.0
    assert summary['non_ransac_total_sec'] == pytest.approx(7.0)
    assert summary['n_prediction_calls'] == 2
    assert summary['n_prediction_batches'] == 1
    assert summary['fit_executor_enabled'] == 1
    assert summary['fit_executor_backend'] == 'thread'
    assert summary['fit_executor_max_workers'] == 2
    assert summary['fit_executor_wall_sec'] == pytest.approx(0.25)
    assert summary['fit_executor_tasks'] == 3


def test_load_physics_lite_config_defaults_include_physical_trend_blocks() -> None:
    cfg = load_physics_lite_config({})

    assert cfg.physical_trend.enabled is False
    assert cfg.physical_trend.fit_kind == 'two_piece_ransac_autobreak'
    assert cfg.physical_trend.min_offset_spread_m == 1.0
    assert cfg.physical_trend.min_gap_m is None
    assert cfg.neighbor_context.mode == 'nearest_source_xy'
    assert cfg.neighbor_context.k_neighbors == 5
    assert cfg.neighbor_context.max_source_distance_m is None
    assert cfg.physical_prefilter.vmin_m_s == 300.0
    assert cfg.physical_prefilter.vmax_m_s == 6000.0
    assert cfg.two_piece_ransac.q_lo == 0.15
    assert cfg.two_piece_ransac.q_hi == 0.85
    assert cfg.physical_projection.mode == 'model'
    assert cfg.physical_runtime.fit_policy == 'full'
    assert cfg.physical_runtime.diagnostics_enabled is True
    assert cfg.physical_runtime.write_runtime_summary is True
    assert cfg.physical_runtime.diagnostics.enabled is True
    assert cfg.physical_runtime.diagnostics.detailed_timing is False
    assert cfg.physical_runtime.anchor_selection.enabled is False
    assert cfg.physical_runtime.anchor_selection.mode == 'source_xy_stride'
    assert cfg.physical_runtime.anchor_selection.anchor_stride_source_groups == 5
    assert cfg.physical_runtime.anchor_selection.anchor_spacing_m is None
    assert cfg.physical_runtime.anchor_selection.include_first is True
    assert cfg.physical_runtime.anchor_selection.include_last is True
    assert cfg.physical_runtime.anchor_reuse.enabled is True
    assert cfg.physical_runtime.anchor_reuse.non_anchor_mode == 'nearest_anchor'
    assert cfg.physical_runtime.anchor_reuse.max_anchor_distance_m is None
    assert cfg.physical_runtime.anchor_reuse.reuse_segment_policy == (
        'same_side_and_gap'
    )
    assert cfg.physical_runtime.anchor_reuse.fallback_if_no_compatible_segment == (
        'full_fit'
    )
    assert cfg.physical_runtime.t0_shift.enabled is True
    assert cfg.physical_runtime.t0_shift.estimator == 'median'
    assert cfg.physical_runtime.t0_shift.min_valid_for_t0_shift == 8
    assert cfg.physical_runtime.t0_shift.t0_shift_clip_ms == 60.0
    assert cfg.physical_runtime.t0_shift.use_physical_prefilter_mask is True
    assert cfg.physical_runtime.t0_shift.use_pmax_min is True
    assert cfg.physical_runtime.adaptive_refit.enabled is False
    assert cfg.physical_runtime.adaptive_refit.resid_p90_ms_gt == 50.0
    assert cfg.physical_runtime.adaptive_refit.median_abs_shift_ms_gt == 40.0
    assert cfg.physical_runtime.adaptive_refit.min_valid_for_resid_check == 8
    assert cfg.physical_runtime.adaptive_refit.fallback_if_refit_fails == (
        'nearest_anchor_plus_t0_shift'
    )
    assert cfg.physical_runtime.observation_sampling.enabled is False
    assert cfg.physical_runtime.observation_sampling.method == 'offset_bin'
    assert cfg.physical_runtime.observation_sampling.max_obs_per_fit == 256
    assert cfg.physical_runtime.observation_sampling.n_offset_bins == 64
    assert cfg.physical_runtime.observation_sampling.bin_pick == 'pmax_max'
    assert (
        cfg.physical_runtime.observation_sampling.min_obs_per_fit_after_sampling == 8
    )
    assert cfg.physical_runtime.observation_sampling.preserve_edge_bins is True
    assert cfg.physical_runtime.fit_executor.enabled is False
    assert cfg.physical_runtime.fit_executor.backend == 'process'
    assert cfg.physical_runtime.fit_executor.max_workers is None
    assert cfg.physical_runtime.fit_executor.torch_num_threads_per_worker == 1
    assert cfg.physical_runtime.fit_executor.chunksize == 1


def test_load_physics_lite_config_accepts_nested_diagnostics_block() -> None:
    cfg = load_physics_lite_config(
        {
            'physical_runtime': {
                'diagnostics': {
                    'enabled': True,
                    'detailed_timing': True,
                    'save_json': False,
                    'save_npz_scalars': True,
                    'save_per_trace_context': False,
                }
            }
        }
    )

    assert cfg.physical_runtime.diagnostics_enabled is True
    assert cfg.physical_runtime.write_runtime_summary is False
    assert cfg.physical_runtime.diagnostics.detailed_timing is True


def test_load_physics_lite_config_accepts_physical_trend_blocks() -> None:
    cfg = load_physics_lite_config(_physical_trend_blocks())

    assert cfg.physical_trend.enabled is True
    assert cfg.physical_trend.min_offset_spread_m == 12.5
    assert cfg.physical_trend.coord_group_tol_m == 2.0
    assert cfg.physical_trend.segment_by_offset_sign is False
    assert cfg.physical_trend.min_gap_m == 25.0
    assert cfg.neighbor_context.k_neighbors == 7
    assert cfg.neighbor_context.max_source_distance_m == 1000.0
    assert cfg.neighbor_context.include_self is False
    assert cfg.physical_prefilter.pmax_min == 0.25
    assert cfg.physical_prefilter.use_existing_feasible_mask is True
    assert cfg.two_piece_ransac.n_iter == 300
    assert cfg.two_piece_ransac.sort_offsets is False
    assert cfg.physical_runtime.fit_policy == 'full'
    assert cfg.physical_runtime.anchor_selection.enabled is False


def test_load_physics_lite_config_accepts_anchor_selection_runtime_block() -> None:
    cfg = load_physics_lite_config(
        {
            'physical_runtime': {
                'fit_policy': 'anchor_source_xy',
                'anchor_selection': {
                    'enabled': True,
                    'mode': 'source_xy_stride',
                    'anchor_stride_source_groups': 3,
                    'anchor_spacing_m': None,
                    'include_first': False,
                    'include_last': True,
                },
                'anchor_reuse': {
                    'enabled': True,
                    'non_anchor_mode': 'nearest_anchor_plus_t0_shift',
                    'max_anchor_distance_m': 250.0,
                    'reuse_segment_policy': 'same_side_and_gap',
                    'fallback_if_no_compatible_segment': 'existing_trend',
                },
                't0_shift': {
                    'enabled': True,
                    'estimator': 'median',
                    'min_valid_for_t0_shift': 6,
                    't0_shift_clip_ms': 45.0,
                    'use_physical_prefilter_mask': False,
                    'use_pmax_min': False,
                },
                'adaptive_refit': {
                    'enabled': True,
                    'resid_p90_ms_gt': 35.0,
                    'median_abs_shift_ms_gt': 25.0,
                    'min_valid_for_resid_check': 6,
                    'fallback_if_refit_fails': 'nearest_anchor',
                },
                'observation_sampling': {
                    'enabled': True,
                    'method': 'offset_bin',
                    'max_obs_per_fit': 100,
                    'n_offset_bins': 20,
                    'bin_pick': 'median_time',
                    'min_obs_per_fit_after_sampling': 6,
                    'preserve_edge_bins': False,
                },
                'fit_executor': {
                    'enabled': True,
                    'backend': 'thread',
                    'max_workers': 2,
                    'torch_num_threads_per_worker': 1,
                    'chunksize': 2,
                },
            },
        }
    )

    assert cfg.physical_runtime.fit_policy == 'anchor_source_xy'
    assert cfg.physical_runtime.anchor_selection.enabled is True
    assert cfg.physical_runtime.anchor_selection.anchor_stride_source_groups == 3
    assert cfg.physical_runtime.anchor_selection.include_first is False
    assert cfg.physical_runtime.anchor_selection.include_last is True
    assert cfg.physical_runtime.anchor_reuse.non_anchor_mode == (
        'nearest_anchor_plus_t0_shift'
    )
    assert cfg.physical_runtime.anchor_reuse.max_anchor_distance_m == 250.0
    assert cfg.physical_runtime.anchor_reuse.fallback_if_no_compatible_segment == (
        'existing_trend'
    )
    assert cfg.physical_runtime.t0_shift.min_valid_for_t0_shift == 6
    assert cfg.physical_runtime.t0_shift.t0_shift_clip_ms == 45.0
    assert cfg.physical_runtime.t0_shift.use_physical_prefilter_mask is False
    assert cfg.physical_runtime.t0_shift.use_pmax_min is False
    assert cfg.physical_runtime.adaptive_refit.enabled is True
    assert cfg.physical_runtime.adaptive_refit.resid_p90_ms_gt == 35.0
    assert cfg.physical_runtime.adaptive_refit.fallback_if_refit_fails == (
        'nearest_anchor'
    )
    assert cfg.physical_runtime.observation_sampling.enabled is True
    assert cfg.physical_runtime.observation_sampling.max_obs_per_fit == 100
    assert cfg.physical_runtime.observation_sampling.n_offset_bins == 20
    assert cfg.physical_runtime.observation_sampling.bin_pick == 'median_time'
    assert cfg.physical_runtime.observation_sampling.preserve_edge_bins is False
    assert cfg.physical_runtime.fit_executor.enabled is True
    assert cfg.physical_runtime.fit_executor.backend == 'thread'
    assert cfg.physical_runtime.fit_executor.max_workers == 2
    assert cfg.physical_runtime.fit_executor.chunksize == 2


def test_physical_center_example_config_enables_physical_trend() -> None:
    import yaml

    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / 'examples/config_run_fbpick_physics_physical_center.yaml'
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))

    cfg = load_physics_lite_config(raw)

    assert cfg.physical_trend.enabled is True
    assert cfg.physical_trend.fit_kind == 'two_piece_ransac_autobreak'
    assert cfg.physical_trend.use_geometry_offset is True
    assert cfg.physical_trend.min_offset_spread_m == pytest.approx(1.0)
    assert cfg.physical_prefilter.enabled is True
    assert cfg.physical_prefilter.vmin_m_s == pytest.approx(300.0)
    assert cfg.physical_prefilter.vmax_m_s == pytest.approx(6000.0)
    assert cfg.two_piece_ransac.n_iter == 200
    assert cfg.two_piece_ransac.min_pts == 8
    assert cfg.two_piece_ransac.sort_offsets is True


@pytest.mark.parametrize(
    ('cfg', 'match'),
    [
        (
            {'physical_prefilter': {'vmin_m_s': 6000.0, 'vmax_m_s': 300.0}},
            'physical_prefilter.vmax_m_s',
        ),
        (
            {'physical_prefilter': {'vmin_m_s': math.nan}},
            'physical_prefilter.vmin_m_s',
        ),
        (
            {'physical_prefilter': {'vmax_m_s': math.inf}},
            'physical_prefilter.vmax_m_s',
        ),
        (
            {'two_piece_ransac': {'q_lo': 0.8, 'q_hi': 0.2}},
            'q_lo < q_hi',
        ),
        (
            {'neighbor_context': {'k_neighbors': 0}},
            'neighbor_context.k_neighbors',
        ),
        (
            {'neighbor_context': {'max_source_distance_m': math.nan}},
            'neighbor_context.max_source_distance_m',
        ),
        (
            {'physical_trend': {'fit_kind': 'linear'}},
            'physical_trend.fit_kind',
        ),
        (
            {'physical_trend': {'coord_group_tol_m': math.nan}},
            'physical_trend.coord_group_tol_m',
        ),
        (
            {'physical_trend': {'min_offset_spread_m': -1.0}},
            'physical_trend.min_offset_spread_m',
        ),
        (
            {'physical_trend': {'min_offset_spread_m': math.inf}},
            'physical_trend.min_offset_spread_m',
        ),
        (
            {'physical_trend': {'gap_ratio': math.inf}},
            'physical_trend.gap_ratio',
        ),
        (
            {'physical_trend': {'min_gap_m': math.inf}},
            'physical_trend.min_gap_m',
        ),
        (
            {'physical_projection': {'mode': 'observed'}},
            'physical_projection.mode',
        ),
        (
            {'physical_runtime': {'fit_policy': 'cached'}},
            'physical_runtime.fit_policy',
        ),
        (
            {
                'physical_runtime': {
                    'anchor_selection': {'mode': 'source_distance'}
                }
            },
            'physical_runtime.anchor_selection.mode',
        ),
        (
            {
                'physical_runtime': {
                    'anchor_selection': {'anchor_stride_source_groups': 0}
                }
            },
            'physical_runtime.anchor_selection.anchor_stride_source_groups',
        ),
        (
            {
                'physical_runtime': {
                    'anchor_selection': {'anchor_spacing_m': 100.0}
                }
            },
            'physical_runtime.anchor_selection.anchor_spacing_m',
        ),
        (
            {
                'physical_runtime': {
                    'anchor_reuse': {'non_anchor_mode': 'all_anchors'}
                }
            },
            'physical_runtime.anchor_reuse.non_anchor_mode',
        ),
        (
            {
                'physical_runtime': {
                    'anchor_reuse': {'max_anchor_distance_m': -1.0}
                }
            },
            'physical_runtime.anchor_reuse.max_anchor_distance_m',
        ),
        (
            {
                'physical_runtime': {
                    'anchor_reuse': {'reuse_segment_policy': 'any_segment'}
                }
            },
            'physical_runtime.anchor_reuse.reuse_segment_policy',
        ),
        (
            {
                'physical_runtime': {
                    'anchor_reuse': {
                        'fallback_if_no_compatible_segment': 'skip',
                    }
                }
            },
            'physical_runtime.anchor_reuse.fallback_if_no_compatible_segment',
        ),
        (
            {'physical_runtime': {'t0_shift': {'estimator': 'mean'}}},
            'physical_runtime.t0_shift.estimator',
        ),
        (
            {'physical_runtime': {'t0_shift': {'min_valid_for_t0_shift': 0}}},
            'physical_runtime.t0_shift.min_valid_for_t0_shift',
        ),
        (
            {'physical_runtime': {'t0_shift': {'t0_shift_clip_ms': -1.0}}},
            'physical_runtime.t0_shift.t0_shift_clip_ms',
        ),
        (
            {'physical_runtime': {'adaptive_refit': {'resid_p90_ms_gt': -1.0}}},
            'physical_runtime.adaptive_refit.resid_p90_ms_gt',
        ),
        (
            {
                'physical_runtime': {
                    'adaptive_refit': {'median_abs_shift_ms_gt': math.inf}
                }
            },
            'physical_runtime.adaptive_refit.median_abs_shift_ms_gt',
        ),
        (
            {
                'physical_runtime': {
                    'adaptive_refit': {'min_valid_for_resid_check': 0}
                }
            },
            'physical_runtime.adaptive_refit.min_valid_for_resid_check',
        ),
        (
            {
                'physical_runtime': {
                    'adaptive_refit': {'fallback_if_refit_fails': 'full_fit'}
                }
            },
            'physical_runtime.adaptive_refit.fallback_if_refit_fails',
        ),
        (
            {
                'physical_runtime': {
                    'observation_sampling': {'max_obs_per_fit': 0}
                }
            },
            'physical_runtime.observation_sampling.max_obs_per_fit',
        ),
        (
            {
                'physical_runtime': {
                    'observation_sampling': {'method': 'trace_stride'}
                }
            },
            'physical_runtime.observation_sampling.method',
        ),
        (
            {
                'physical_runtime': {
                    'observation_sampling': {'bin_pick': 'first'}
                }
            },
            'physical_runtime.observation_sampling.bin_pick',
        ),
        (
            {
                'physical_runtime': {
                    'observation_sampling': {
                        'max_obs_per_fit': 8,
                        'min_obs_per_fit_after_sampling': 9,
                    }
                }
            },
            'min_obs_per_fit_after_sampling',
        ),
        (
            {'physical_runtime': {'fit_executor': {'backend': 'fork'}}},
            'physical_runtime.fit_executor.backend',
        ),
        (
            {'physical_runtime': {'fit_executor': {'max_workers': 0}}},
            'physical_runtime.fit_executor.max_workers',
        ),
        (
            {
                'physical_runtime': {
                    'fit_executor': {'torch_num_threads_per_worker': 0}
                }
            },
            'physical_runtime.fit_executor.torch_num_threads_per_worker',
        ),
        (
            {'physical_runtime': {'fit_executor': {'chunksize': 0}}},
            'physical_runtime.fit_executor.chunksize',
        ),
    ],
)
def test_load_physics_lite_config_rejects_invalid_physical_trend_blocks(
    cfg: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        load_physics_lite_config(cfg)


def test_physics_lite_config_to_dict_includes_physical_trend_blocks() -> None:
    cfg = load_physics_lite_config(_physical_trend_blocks())
    out = physics_lite_config_to_dict(cfg)

    assert set(out).issuperset(
        {
            'physical_trend',
            'neighbor_context',
            'physical_prefilter',
            'two_piece_ransac',
            'physical_projection',
            'physical_runtime',
        }
    )
    assert out['physical_trend']['enabled'] is True
    assert out['neighbor_context']['max_source_distance_m'] == 1000.0
    assert out['physical_projection']['mode'] == 'model'
    assert out['physical_runtime']['diagnostics_enabled'] is True
    assert out['physical_runtime']['fit_policy'] == 'full'
    assert out['physical_runtime']['anchor_selection']['enabled'] is False
    assert out['physical_runtime']['anchor_reuse']['enabled'] is True
    assert out['physical_runtime']['fit_executor']['enabled'] is False


def test_normalize_coarse_pick_table_preserves_contract(tmp_path: Path) -> None:
    coarse = _make_coarse_payload(
        coarse_pick_i=np.array([10, 20, 30, 40], dtype=np.int32),
        coarse_pmax=np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        offsets_m=np.array([100.0, 200.0, -300.0, 400.0], dtype=np.float32),
        ffid_values=np.array([11, 11, 11, 11], dtype=np.int32),
        chno_values=np.array([1, 2, 3, 4], dtype=np.int32),
    )
    coarse_path = save_coarse_npz(tmp_path / 'norm.coarse.npz', **coarse)

    table = normalize_coarse_pick_table(load_coarse_npz(coarse_path))

    assert table.n_traces == 4
    assert table.n_samples_orig == 512
    assert table.dt_scalar_sec == np.float32(0.004)
    assert table.shot_id.dtype == np.int32
    assert table.trace_id.dtype == np.int64
    assert table.ffid.dtype == np.int32
    assert table.chno.dtype == np.int32
    assert table.offset_m.dtype == np.float32
    assert table.dt_sec.dtype == np.float32
    assert table.coarse_pick_i.dtype == np.int32
    assert table.coarse_pick_t_sec.dtype == np.float32
    assert table.coarse_pmax.dtype == np.float32
    np.testing.assert_array_equal(table.shot_id, coarse['ffid_values'])
    np.testing.assert_array_equal(table.trace_id, coarse['trace_indices'])


def test_save_and_load_coarse_npz_preserve_optional_geometry(tmp_path: Path) -> None:
    n_traces = 3
    geometry = {
        'source_x_m': np.array([10.0, 10.0, np.nan], dtype=np.float32),
        'source_y_m': np.array([20.0, 20.0, np.nan], dtype=np.float32),
        'receiver_x_m': np.array([13.0, 16.0, np.nan], dtype=np.float32),
        'receiver_y_m': np.array([24.0, 28.0, np.nan], dtype=np.float32),
        'offset_abs_geom_m': np.array([5.0, 10.0, np.nan], dtype=np.float32),
        'geometry_valid_mask': np.array([True, True, False], dtype=np.bool_),
    }
    payload = _make_coarse_payload(
        coarse_pick_i=np.array([10, 20, 30], dtype=np.int32),
        coarse_pmax=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        offsets_m=np.array([100.0, 200.0, 300.0], dtype=np.float32),
    )
    assert int(np.asarray(payload['n_traces']).item()) == n_traces

    out_path = save_coarse_npz(
        tmp_path / 'geometry.coarse.npz',
        **payload,
        **geometry,
    )
    loaded = load_coarse_npz(out_path)

    assert set(COARSE_GEOMETRY_OPTIONAL_KEYS).issubset(loaded.keys())
    for key in COARSE_GEOMETRY_OPTIONAL_KEYS:
        assert loaded[key].shape == (n_traces,)
        expected_dtype = np.bool_ if key == 'geometry_valid_mask' else np.float32
        assert loaded[key].dtype == np.dtype(expected_dtype)
        np.testing.assert_array_equal(loaded[key], geometry[key])


def test_save_and_load_coarse_npz_preserve_extra_signed_geometry(
    tmp_path: Path,
) -> None:
    geometry = {
        'source_x_m': np.array([0.0, 0.0, np.nan], dtype=np.float32),
        'source_y_m': np.array([0.0, 0.0, np.nan], dtype=np.float32),
        'receiver_x_m': np.array([-10.0, 10.0, np.nan], dtype=np.float32),
        'receiver_y_m': np.array([0.0, 0.0, np.nan], dtype=np.float32),
        'offset_abs_geom_m': np.array([10.0, 10.0, np.nan], dtype=np.float32),
        'geometry_valid_mask': np.array([True, True, False], dtype=np.bool_),
        'offset_signed_geom_m': np.array([-10.0, 10.0, np.nan], dtype=np.float32),
    }
    payload = _make_coarse_payload(
        coarse_pick_i=np.array([10, 20, 30], dtype=np.int32),
        coarse_pmax=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        offsets_m=np.array([100.0, 200.0, 300.0], dtype=np.float32),
    )

    out_path = save_coarse_npz(
        tmp_path / 'signed_geometry.coarse.npz',
        **payload,
        **geometry,
    )
    loaded = load_coarse_npz(out_path)

    assert set(COARSE_GEOMETRY_EXTRA_OPTIONAL_KEYS).issubset(loaded.keys())
    assert loaded['offset_signed_geom_m'].dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        loaded['offset_signed_geom_m'],
        geometry['offset_signed_geom_m'],
    )


def test_save_coarse_npz_rejects_partial_geometry(tmp_path: Path) -> None:
    payload = _make_coarse_payload(
        coarse_pick_i=np.array([10, 20], dtype=np.int32),
        coarse_pmax=np.array([0.9, 0.8], dtype=np.float32),
        offsets_m=np.array([100.0, 200.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match='provided together'):
        save_coarse_npz(
            tmp_path / 'partial_geometry.coarse.npz',
            **payload,
            source_x_m=np.array([10.0, 20.0], dtype=np.float32),
        )


def test_save_coarse_npz_rejects_nan_geometry_on_valid_trace(tmp_path: Path) -> None:
    payload = _make_coarse_payload(
        coarse_pick_i=np.array([10, 20], dtype=np.int32),
        coarse_pmax=np.array([0.9, 0.8], dtype=np.float32),
        offsets_m=np.array([100.0, 200.0], dtype=np.float32),
    )
    geometry = {
        'source_x_m': np.array([10.0, np.nan], dtype=np.float32),
        'source_y_m': np.array([20.0, 20.0], dtype=np.float32),
        'receiver_x_m': np.array([13.0, 16.0], dtype=np.float32),
        'receiver_y_m': np.array([24.0, 28.0], dtype=np.float32),
        'offset_abs_geom_m': np.array([5.0, 10.0], dtype=np.float32),
        'geometry_valid_mask': np.array([True, True], dtype=np.bool_),
    }

    with pytest.raises(ValueError, match='source_x_m must be finite'):
        save_coarse_npz(tmp_path / 'bad_geometry.coarse.npz', **payload, **geometry)


def test_save_coarse_npz_rejects_negative_valid_geometry_offset(
    tmp_path: Path,
) -> None:
    payload = _make_coarse_payload(
        coarse_pick_i=np.array([10, 20], dtype=np.int32),
        coarse_pmax=np.array([0.9, 0.8], dtype=np.float32),
        offsets_m=np.array([100.0, 200.0], dtype=np.float32),
    )
    geometry = {
        'source_x_m': np.array([10.0, 10.0], dtype=np.float32),
        'source_y_m': np.array([20.0, 20.0], dtype=np.float32),
        'receiver_x_m': np.array([13.0, 16.0], dtype=np.float32),
        'receiver_y_m': np.array([24.0, 28.0], dtype=np.float32),
        'offset_abs_geom_m': np.array([5.0, -10.0], dtype=np.float32),
        'geometry_valid_mask': np.array([True, True], dtype=np.bool_),
    }

    with pytest.raises(ValueError, match='offset_abs_geom_m must be >= 0'):
        save_coarse_npz(
            tmp_path / 'negative_offset_geometry.coarse.npz',
            **payload,
            **geometry,
        )


def test_save_coarse_npz_rejects_nan_signed_geometry_on_valid_trace(
    tmp_path: Path,
) -> None:
    payload = _make_coarse_payload(
        coarse_pick_i=np.array([10, 20], dtype=np.int32),
        coarse_pmax=np.array([0.9, 0.8], dtype=np.float32),
        offsets_m=np.array([100.0, 200.0], dtype=np.float32),
    )
    geometry = {
        'source_x_m': np.array([0.0, 0.0], dtype=np.float32),
        'source_y_m': np.array([0.0, 0.0], dtype=np.float32),
        'receiver_x_m': np.array([-10.0, 10.0], dtype=np.float32),
        'receiver_y_m': np.array([0.0, 0.0], dtype=np.float32),
        'offset_abs_geom_m': np.array([10.0, 10.0], dtype=np.float32),
        'geometry_valid_mask': np.array([True, True], dtype=np.bool_),
        'offset_signed_geom_m': np.array([-10.0, np.nan], dtype=np.float32),
    }

    with pytest.raises(ValueError, match='offset_signed_geom_m must be finite'):
        save_coarse_npz(tmp_path / 'bad_signed.coarse.npz', **payload, **geometry)


def test_load_coarse_npz_accepts_legacy_payload_without_geometry(tmp_path: Path) -> None:
    payload = _make_coarse_payload(
        coarse_pick_i=np.array([10, 20], dtype=np.int32),
        coarse_pmax=np.array([0.9, 0.8], dtype=np.float32),
        offsets_m=np.array([100.0, 200.0], dtype=np.float32),
    )

    out_path = save_coarse_npz(tmp_path / 'legacy.coarse.npz', **payload)
    loaded = load_coarse_npz(out_path)

    for key in COARSE_GEOMETRY_OPTIONAL_KEYS:
        assert key not in loaded


def test_velocity_t0_band_from_arrays_accepts_synthetic_trend_in_range() -> None:
    offset_m = np.array([0.0, 100.0, 200.0], dtype=np.float32)
    pick_t_sec = np.float32(0.02) + offset_m / np.float32(1000.0)

    feasible = compute_velocity_t0_band_from_arrays(
        offset_m=offset_m,
        pick_t_sec=pick_t_sec,
        vmin_m_s=800.0,
        vmax_m_s=1200.0,
        t0_lo_ms=10.0,
        t0_hi_ms=30.0,
    )

    assert feasible.feasible_mask.tolist() == [True, True, True]
    assert feasible.feasible_lo_sec.dtype == np.float32
    assert feasible.feasible_hi_sec.dtype == np.float32


def test_velocity_t0_band_from_arrays_rejects_t0_outside_range() -> None:
    feasible = compute_velocity_t0_band_from_arrays(
        offset_m=np.array([0.0, 0.0], dtype=np.float32),
        pick_t_sec=np.array([-0.02, 0.06], dtype=np.float32),
        vmin_m_s=800.0,
        vmax_m_s=1200.0,
        t0_lo_ms=0.0,
        t0_hi_ms=50.0,
    )

    assert feasible.feasible_mask.tolist() == [False, False]


def test_velocity_t0_band_from_arrays_rejects_velocity_outside_range() -> None:
    offset_m = np.array([1000.0, 1000.0], dtype=np.float32)
    pick_t_sec = np.array(
        [
            1000.0 / 6000.0,
            1000.0 / 500.0,
        ],
        dtype=np.float32,
    )

    feasible = compute_velocity_t0_band_from_arrays(
        offset_m=offset_m,
        pick_t_sec=pick_t_sec,
        vmin_m_s=1000.0,
        vmax_m_s=5000.0,
        t0_lo_ms=0.0,
        t0_hi_ms=0.0,
    )

    assert feasible.feasible_mask.tolist() == [False, False]


def test_velocity_t0_band_from_arrays_uses_abs_for_negative_offsets() -> None:
    feasible = compute_velocity_t0_band_from_arrays(
        offset_m=np.array([-100.0, 100.0], dtype=np.float32),
        pick_t_sec=np.array([0.12, 0.12], dtype=np.float32),
        vmin_m_s=800.0,
        vmax_m_s=1200.0,
        t0_lo_ms=10.0,
        t0_hi_ms=30.0,
    )

    assert feasible.feasible_mask.tolist() == [True, True]
    np.testing.assert_array_equal(
        feasible.feasible_lo_sec,
        np.array(
            [100.0 / 1200.0 + 0.01, 100.0 / 1200.0 + 0.01],
            dtype=np.float32,
        ),
    )


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        (
            {
                'offset_m': np.array([0.0, 1.0], dtype=np.float32),
                'pick_t_sec': np.array([0.0], dtype=np.float32),
            },
            'same shape',
        ),
        (
            {
                'offset_m': np.array([[0.0]], dtype=np.float32),
                'pick_t_sec': np.array([0.0], dtype=np.float32),
            },
            'offset_m must be a 1D array',
        ),
        (
            {
                'offset_m': np.array([0.0], dtype=np.float32),
                'pick_t_sec': np.array([[0.0]], dtype=np.float32),
            },
            'pick_t_sec must be a 1D array',
        ),
        (
            {
                'offset_m': np.array([np.nan], dtype=np.float32),
                'pick_t_sec': np.array([0.0], dtype=np.float32),
            },
            'offset_m must be finite',
        ),
        (
            {
                'offset_m': np.array([0.0], dtype=np.float32),
                'pick_t_sec': np.array([np.nan], dtype=np.float32),
            },
            'pick_t_sec must be finite',
        ),
        (
            {
                'offset_m': np.array([0.0], dtype=np.float32),
                'pick_t_sec': np.array([0.0], dtype=np.float32),
                'vmin_m_s': 0.0,
            },
            'vmin_m_s',
        ),
        (
            {
                'offset_m': np.array([0.0], dtype=np.float32),
                'pick_t_sec': np.array([0.0], dtype=np.float32),
                'vmax_m_s': 500.0,
            },
            'vmax_m_s',
        ),
        (
            {
                'offset_m': np.array([0.0], dtype=np.float32),
                'pick_t_sec': np.array([0.0], dtype=np.float32),
                't0_lo_ms': 10.0,
                't0_hi_ms': 0.0,
            },
            't0_lo_ms',
        ),
    ],
)
def test_velocity_t0_band_from_arrays_rejects_invalid_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    params: dict[str, object] = {
        'offset_m': np.array([0.0], dtype=np.float32),
        'pick_t_sec': np.array([0.0], dtype=np.float32),
        'vmin_m_s': 1000.0,
        'vmax_m_s': 5000.0,
        't0_lo_ms': 0.0,
        't0_hi_ms': 100.0,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        compute_velocity_t0_band_from_arrays(**params)


def test_feasible_band_rejects_early_and_late_picks() -> None:
    coarse = _make_coarse_payload(
        coarse_pick_i=np.array([0, 125, 300], dtype=np.int32),
        coarse_pmax=np.array([0.9, 0.9, 0.9], dtype=np.float32),
        offsets_m=np.array([100.0, 100.0, 100.0], dtype=np.float32),
    )
    table = normalize_coarse_pick_table(coarse)
    feasible = compute_feasible_band(
        table,
        load_physics_lite_config({}).feasible_band,
    )

    assert feasible.feasible_mask.tolist() == [False, True, False]
    assert feasible.feasible_lo_sec.dtype == np.float32
    assert feasible.feasible_hi_sec.dtype == np.float32


def test_feasible_band_preserves_table_n_traces_validation() -> None:
    coarse = _make_coarse_payload(
        coarse_pick_i=np.array([10], dtype=np.int32),
        coarse_pmax=np.array([0.9], dtype=np.float32),
        offsets_m=np.array([100.0], dtype=np.float32),
    )
    table = normalize_coarse_pick_table(coarse)
    cfg = load_physics_lite_config({}).feasible_band

    with pytest.raises(ValueError, match='table.n_traces must be positive'):
        compute_feasible_band(replace(table, n_traces=0), cfg)

    with pytest.raises(ValueError, match='coarse_pick_t_sec must have shape'):
        compute_feasible_band(replace(table, n_traces=2), cfg)


def test_conf_trend1_drops_with_trend_deviation() -> None:
    cfg = load_physics_lite_config(
        {
            'trend': {
                'trend_var_half_win_traces': 0,
                'trend_var_min_count': 1,
            }
        }
    )
    coarse = _make_coarse_payload(
        coarse_pick_i=np.array([100, 150], dtype=np.int32),
        coarse_pmax=np.array([0.9, 0.9], dtype=np.float32),
        offsets_m=np.array([100.0, 200.0], dtype=np.float32),
    )
    table = normalize_coarse_pick_table(coarse)
    feasible = compute_feasible_band(table, cfg.feasible_band)
    trend = TrendResult(
        seed_mask=np.array([True, True], dtype=np.bool_),
        seed_threshold=np.float32(0.0),
        local_center_sec=np.array([0.4, 0.4], dtype=np.float32),
        local_center_valid=np.array([True, True], dtype=np.bool_),
        local_discard_mask=np.array([False, False], dtype=np.bool_),
        global_center_sec=np.array([0.4, 0.4], dtype=np.float32),
        trend_center_sec=np.array([0.4, 0.4], dtype=np.float32),
        trend_center_i=np.array([100, 100], dtype=np.int32),
        filled_mask=np.array([False, False], dtype=np.bool_),
    )

    confidence = compute_confidence_terms(table, feasible, trend, cfg)

    assert confidence.conf_trend1[0] > confidence.conf_trend1[1]


def test_keep_reject_fill_uses_trend_center_and_source_flags() -> None:
    cfg = load_physics_lite_config({})
    coarse = _make_coarse_payload(
        coarse_pick_i=np.array([100, 101, 102, 250, 104, 105], dtype=np.int32),
        coarse_pmax=np.array([0.95, 0.95, 0.9, 0.01, 0.95, 0.95], dtype=np.float32),
        offsets_m=np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0], dtype=np.float32),
    )
    table = normalize_coarse_pick_table(coarse)
    feasible = compute_feasible_band(table, cfg.feasible_band)
    trend = build_trend_result(table, feasible, cfg)
    confidence = compute_confidence_terms(table, feasible, trend, cfg)
    merged = apply_keep_reject_fill(table, feasible, trend, confidence, cfg)

    assert merged.keep_mask.dtype == np.bool_
    assert merged.reject_mask[3]
    assert merged.robust_pick_i[3] == trend.trend_center_i[3]
    assert merged.robust_source[0] == ROBUST_SOURCE_COARSE_OBSERVED
    assert merged.robust_source[3] == ROBUST_SOURCE_TREND_FILL
    assert not bool(np.any(merged.used_theoretical_mask))
    assert bool(merged.reason_mask[3] & np.uint8(REASON_MASK_LOW_SCORE))
    assert bool(merged.reason_mask[3] & np.uint8(REASON_MASK_FILLED_FROM_TREND))


def test_save_and_load_robust_npz_preserve_contract(tmp_path: Path) -> None:
    payload = {
        'dt_sec': np.asarray(0.004, dtype=np.float32),
        'n_samples_orig': np.asarray(512, dtype=np.int32),
        'n_traces': np.asarray(3, dtype=np.int32),
        'ffid_values': np.array([1, 1, 1], dtype=np.int32),
        'chno_values': np.array([1, 2, 3], dtype=np.int32),
        'offsets_m': np.array([100.0, 200.0, 300.0], dtype=np.float32),
        'trace_indices': np.array([0, 1, 2], dtype=np.int64),
        'robust_pick_i': np.array([100, 101, 102], dtype=np.int32),
        'robust_pick_t_sec': np.array([0.4, 0.404, 0.408], dtype=np.float32),
        'robust_conf': np.array([0.9, 0.8, 0.7], dtype=np.float32),
        'robust_source': np.array([0, 2, 0], dtype=np.uint8),
        'used_theoretical_mask': np.array([False, False, False], dtype=np.bool_),
        'reason_mask': np.array(
            [
                0,
                REASON_MASK_INFEASIBLE | REASON_MASK_FILLED_FROM_TREND,
                0,
            ],
            dtype=np.uint8,
        ),
        'conf_prob1': np.array([0.9, 0.2, 0.8], dtype=np.float32),
        'conf_trend1': np.array([0.9, 0.6, 0.8], dtype=np.float32),
        'conf_rs1': np.array([1.0, 1.0, 1.0], dtype=np.float32),
        'lineage': np.asarray('{"iter_id":"","source_model_id":"x","cfg_hash":"y","git_sha":"z"}'),
    }
    out_path = save_robust_npz(tmp_path / 'roundtrip.robust.npz', **payload)
    loaded = load_robust_npz(out_path)

    assert set(ROBUST_REQUIRED_KEYS).issubset(loaded.keys())
    for key, value in payload.items():
        if key == 'lineage':
            assert np.asarray(loaded[key]).ndim == 0
            assert np.asarray(loaded[key]).item() == np.asarray(value).item()
        else:
            np.testing.assert_array_equal(loaded[key], value)


def test_robust_optional_schema_constants_match_io_fields() -> None:
    handled_optional = {
        name
        for name, parameter in inspect.signature(save_robust_npz).parameters.items()
        if parameter.default is None
    }

    assert 'fine_center_i' in ROBUST_CENTER_OPTIONAL_KEYS
    assert 'fine_center_t_sec' in ROBUST_CENTER_OPTIONAL_KEYS
    assert (
        *ROBUST_CENTER_OPTIONAL_KEYS,
        *ROBUST_PHYSICAL_DIAGNOSTIC_OPTIONAL_KEYS,
        *ROBUST_RUNTIME_DIAGNOSTIC_OPTIONAL_KEYS,
    ) == ROBUST_OPTIONAL_KEYS
    assert set(ROBUST_OPTIONAL_KEYS) == handled_optional
    assert ROBUST_PHYSICAL_OPTIONAL_KEYS == ROBUST_OPTIONAL_KEYS


def test_save_and_load_robust_npz_preserve_all_optional_fields(tmp_path: Path) -> None:
    payload = {
        **_make_robust_payload(),
        **_make_robust_optional_payload(),
    }

    out_path = save_robust_npz(tmp_path / 'all_optional.robust.npz', **payload)
    loaded = load_robust_npz(out_path)

    assert set(ROBUST_OPTIONAL_KEYS).issubset(loaded.keys())
    for key in ROBUST_OPTIONAL_KEYS:
        np.testing.assert_array_equal(loaded[key], payload[key])


def test_save_and_load_robust_npz_preserve_prefixed_runtime_scalars(
    tmp_path: Path,
) -> None:
    payload = {
        **_make_robust_payload(),
        'physical_runtime_n_fit_calls': np.asarray(7, dtype=np.int64),
        'physical_runtime_ransac_fit_total_sec': np.asarray(1.25, dtype=np.float64),
        'physical_runtime_non_ransac_total_sec': np.asarray(4.5, dtype=np.float64),
    }

    out_path = save_robust_npz(tmp_path / 'runtime_scalars.robust.npz', **payload)
    loaded = load_robust_npz(out_path)

    assert int(np.asarray(loaded['physical_runtime_n_fit_calls']).item()) == 7
    assert float(
        np.asarray(loaded['physical_runtime_ransac_fit_total_sec']).item()
    ) == pytest.approx(1.25)
    assert float(
        np.asarray(loaded['physical_runtime_non_ransac_total_sec']).item()
    ) == pytest.approx(4.5)


def test_save_and_load_robust_npz_preserve_optional_center_fields(tmp_path: Path) -> None:
    center_i = np.array([100, 105, 110], dtype=np.int32)
    center_t_sec = center_i.astype(np.float32) * np.float32(0.004)
    payload = {
        'dt_sec': np.asarray(0.004, dtype=np.float32),
        'n_samples_orig': np.asarray(512, dtype=np.int32),
        'n_traces': np.asarray(3, dtype=np.int32),
        'ffid_values': np.array([1, 1, 1], dtype=np.int32),
        'chno_values': np.array([1, 2, 3], dtype=np.int32),
        'offsets_m': np.array([100.0, 200.0, 300.0], dtype=np.float32),
        'trace_indices': np.array([0, 1, 2], dtype=np.int64),
        'robust_pick_i': np.array([101, 106, 111], dtype=np.int32),
        'robust_pick_t_sec': np.array([0.404, 0.424, 0.444], dtype=np.float32),
        'robust_conf': np.array([0.9, 0.8, 0.7], dtype=np.float32),
        'robust_source': np.array([0, 2, 0], dtype=np.uint8),
        'used_theoretical_mask': np.array([False, False, False], dtype=np.bool_),
        'reason_mask': np.zeros((3,), dtype=np.uint8),
        'conf_prob1': np.array([0.9, 0.2, 0.8], dtype=np.float32),
        'conf_trend1': np.array([0.9, 0.6, 0.8], dtype=np.float32),
        'conf_rs1': np.array([1.0, 1.0, 1.0], dtype=np.float32),
        'lineage': np.asarray('{"iter_id":"","source_model_id":"x","cfg_hash":"y","git_sha":"z"}'),
        'trend_center_i': center_i,
        'trend_center_t_sec': center_t_sec,
        'fine_center_i': center_i,
        'fine_center_t_sec': center_t_sec,
    }

    out_path = save_robust_npz(tmp_path / 'centers.robust.npz', **payload)
    loaded = load_robust_npz(out_path)

    np.testing.assert_array_equal(loaded['trend_center_i'], center_i)
    np.testing.assert_array_equal(loaded['trend_center_t_sec'], center_t_sec)
    np.testing.assert_array_equal(loaded['fine_center_i'], center_i)
    np.testing.assert_array_equal(loaded['fine_center_t_sec'], center_t_sec)


def test_save_and_load_robust_npz_preserve_optional_physical_diagnostics(
    tmp_path: Path,
) -> None:
    center_i = np.array([100, 105, 110], dtype=np.int32)
    center_t_sec = center_i.astype(np.float32) * np.float32(0.004)
    payload = {
        **_make_robust_payload(),
        'physical_center_i': center_i,
        'physical_center_t_sec': center_t_sec,
        'fine_center_i': center_i,
        'fine_center_t_sec': center_t_sec,
        'physical_model_status': np.array([0, 1, 2], dtype=np.uint8),
        'physical_model_failure_reason': np.array([0, 2, 3], dtype=np.uint8),
        'physical_offset_source': np.array([1, 2, 0], dtype=np.uint8),
        'physical_model_break_offset_m': np.array(
            [500.0, np.nan, 600.0],
            dtype=np.float32,
        ),
        'physical_model_slope_near_s_per_m': np.array(
            [0.001, np.nan, 0.0012],
            dtype=np.float32,
        ),
        'physical_model_slope_far_s_per_m': np.array(
            [0.0004, np.nan, 0.0005],
            dtype=np.float32,
        ),
        'physical_model_velocity_near_m_s': np.array(
            [1000.0, np.nan, 833.0],
            dtype=np.float32,
        ),
        'physical_model_velocity_far_m_s': np.array(
            [2500.0, np.nan, 2000.0],
            dtype=np.float32,
        ),
        'physical_model_neighbor_count': np.array([3, 3, 3], dtype=np.int32),
        'physical_prefilter_valid_count': np.array([8, 8, 8], dtype=np.int32),
        'physical_model_segment_id': np.array([0, -1, 0], dtype=np.int32),
        'physical_model_side': np.array([1, 0, 1], dtype=np.int8),
        'physical_model_resid_p50_ms': np.array([1.0, np.nan, 2.0], dtype=np.float32),
        'physical_model_resid_p90_ms': np.array([3.0, np.nan, 4.0], dtype=np.float32),
        'physical_anchor_group_id': np.array([0, 1, 2], dtype=np.int32),
        'physical_anchor_is_anchor': np.array([True, False, True], dtype=np.bool_),
        'physical_anchor_nearest_anchor_group_id': np.array(
            [0, 0, 2],
            dtype=np.int32,
        ),
        'physical_anchor_source_distance_m': np.array(
            [0.0, 10.0, 0.0],
            dtype=np.float32,
        ),
        'physical_runtime_t0_shift_ms': np.array(
            [10.0, np.nan, 20.0],
            dtype=np.float32,
        ),
        'physical_runtime_reuse_resid_p50_ms': np.array(
            [1.0, np.nan, 2.0],
            dtype=np.float32,
        ),
        'physical_runtime_reuse_resid_p90_ms': np.array(
            [2.0, np.nan, 3.0],
            dtype=np.float32,
        ),
        'physical_runtime_reuse_valid_count': np.array([3, 0, 4], dtype=np.int32),
        'physical_runtime_refit_mask': np.array([False, True, False], dtype=np.bool_),
        'physical_runtime_fit_source': np.array([1, 2, 3], dtype=np.uint8),
    }

    out_path = save_robust_npz(tmp_path / 'physical.robust.npz', **payload)
    loaded = load_robust_npz(out_path)

    assert set(ROBUST_PHYSICAL_DIAGNOSTIC_OPTIONAL_KEYS).issubset(loaded.keys())
    for key in ROBUST_PHYSICAL_DIAGNOSTIC_OPTIONAL_KEYS:
        np.testing.assert_array_equal(loaded[key], payload[key])


def test_load_robust_npz_rejects_bad_physical_diagnostic_dtype_and_shape(
    tmp_path: Path,
) -> None:
    payload = {
        **_make_robust_payload(),
        'physical_model_status': np.array([0, 1, 2], dtype=np.uint8),
        'physical_model_neighbor_count': np.array([3, 3, 3], dtype=np.int32),
    }

    bad_dtype_path = tmp_path / 'bad_dtype.robust.npz'
    np.savez_compressed(
        bad_dtype_path,
        **{
            **payload,
            'physical_model_status': np.array([0, 1, 2], dtype=np.int32),
        },
    )
    with pytest.raises(ValueError, match='physical_model_status dtype'):
        load_robust_npz(bad_dtype_path)

    bad_shape_path = tmp_path / 'bad_shape.robust.npz'
    np.savez_compressed(
        bad_shape_path,
        **{
            **payload,
            'physical_model_neighbor_count': np.array([3, 3], dtype=np.int32),
        },
    )
    with pytest.raises(ValueError, match='physical_model_neighbor_count must be 1D'):
        load_robust_npz(bad_shape_path)


def test_save_robust_npz_rejects_bad_physical_diagnostic_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='physical_model_status length'):
        save_robust_npz(
            tmp_path / 'bad_physical_shape.robust.npz',
            **_make_robust_payload(),
            physical_model_status=np.array([0, 1], dtype=np.uint8),
        )


def test_run_physics_lite_end_to_end_outputs_full_covering_robust_picks(
    tmp_path: Path,
) -> None:
    n_traces = 48
    base_pick = np.linspace(90, 150, n_traces).round().astype(np.int32)
    coarse_pick_i = base_pick.copy()
    coarse_pick_i[18:24] = np.array([20, 18, 16, 260, 280, 300], dtype=np.int32)
    coarse_pmax = np.full((n_traces,), 0.95, dtype=np.float32)
    coarse_pmax[18:24] = np.array([0.02, 0.01, 0.03, 0.02, 0.01, 0.02], dtype=np.float32)
    offsets_m = np.linspace(50.0, 2000.0, n_traces, dtype=np.float32)
    coarse_path = save_coarse_npz(
        tmp_path / 'synthetic.coarse.npz',
        **_make_coarse_payload(
            coarse_pick_i=coarse_pick_i,
            coarse_pmax=coarse_pmax,
            offsets_m=offsets_m,
        ),
    )

    out_path = run_physics_lite(
        coarse_path,
        cfg={},
        source_model_id='coarse-model',
        iter_id='',
        repo_root=tmp_path,
    )
    robust = load_robust_npz(out_path)
    lineage = json.loads(np.asarray(robust['lineage']).item())
    summary_path = derive_physics_runtime_summary_path(out_path)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))

    assert lineage['git_sha'] is None

    assert out_path.name == 'synthetic.robust.npz'
    assert summary_path.name == 'synthetic.physics_runtime_summary.json'
    assert 'physics_total_sec' in robust
    assert 'physical_center_total_sec' in robust
    assert 'ransac_fit_total_sec' in robust
    assert 'n_fit_calls' in robust
    assert 'n_cache_hits' in robust
    assert 'n_cache_misses' in robust
    assert 'cache_hit_rate' in robust
    assert summary['physics_total_sec'] == pytest.approx(
        float(np.asarray(robust['physics_total_sec']).item())
    )
    assert robust['robust_pick_i'].shape == (n_traces,)
    assert robust['robust_pick_t_sec'].shape == (n_traces,)
    assert robust['trend_center_i'].shape == (n_traces,)
    assert robust['trend_center_t_sec'].shape == (n_traces,)
    assert robust['physical_center_i'].shape == (n_traces,)
    assert robust['physical_center_t_sec'].shape == (n_traces,)
    assert robust['fine_center_i'].shape == (n_traces,)
    assert robust['fine_center_t_sec'].shape == (n_traces,)
    assert robust['physical_model_status'].shape == (n_traces,)
    assert robust['physical_model_failure_reason'].shape == (n_traces,)
    assert robust['trend_center_i'].dtype == np.int32
    assert robust['trend_center_t_sec'].dtype == np.float32
    assert robust['physical_center_i'].dtype == np.int32
    assert robust['physical_center_t_sec'].dtype == np.float32
    assert robust['fine_center_i'].dtype == np.int32
    assert robust['fine_center_t_sec'].dtype == np.float32
    assert robust['physical_model_status'].dtype == np.uint8
    assert robust['physical_model_failure_reason'].dtype == np.uint8
    np.testing.assert_array_equal(robust['physical_center_i'], robust['trend_center_i'])
    np.testing.assert_array_equal(
        robust['physical_center_t_sec'],
        robust['trend_center_t_sec'],
    )
    np.testing.assert_array_equal(robust['fine_center_i'], robust['trend_center_i'])
    np.testing.assert_array_equal(
        robust['fine_center_t_sec'],
        robust['trend_center_t_sec'],
    )
    assert np.all(
        robust['physical_model_status']
        == np.uint8(PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED)
    )
    assert np.all(
        robust['physical_model_failure_reason']
        == np.uint8(PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED)
    )
    np.testing.assert_array_equal(
        robust['fine_center_t_sec'],
        robust['fine_center_i'].astype(np.float32)
        * np.asarray(robust['dt_sec'], dtype=np.float32),
    )
    assert np.all(robust['robust_pick_i'] >= 0)
    assert np.all(robust['robust_pick_i'] < int(np.asarray(robust['n_samples_orig']).item()))
    assert not bool(np.any(robust['used_theoretical_mask']))
    assert np.any(robust['robust_source'] == np.uint8(ROBUST_SOURCE_COARSE_OBSERVED))
    assert np.any(robust['robust_source'][18:24] == np.uint8(ROBUST_SOURCE_TREND_FILL))
    assert np.any(robust['reason_mask'][18:24] & np.uint8(REASON_MASK_FILLED_FROM_TREND))
    assert np.all(np.isfinite(robust['robust_pick_t_sec']))


def test_run_physics_lite_allows_disabling_runtime_diagnostics(
    tmp_path: Path,
) -> None:
    n_traces = 8
    coarse_path = save_coarse_npz(
        tmp_path / 'disabled_runtime.coarse.npz',
        **_make_coarse_payload(
            coarse_pick_i=np.arange(100, 108, dtype=np.int32),
            coarse_pmax=np.full((n_traces,), 0.95, dtype=np.float32),
            offsets_m=np.linspace(50.0, 400.0, n_traces, dtype=np.float32),
        ),
    )

    out_path = run_physics_lite(
        coarse_path,
        cfg={'physical_runtime': {'diagnostics_enabled': False}},
        source_model_id='coarse-model',
        iter_id='',
        repo_root=tmp_path,
    )
    robust = load_robust_npz(out_path)

    assert 'physics_total_sec' not in robust
    assert not derive_physics_runtime_summary_path(out_path).exists()
    assert robust['physical_center_i'].shape == (n_traces,)


def test_build_robust_payload_from_coarse_returns_required_keys(tmp_path: Path) -> None:
    payload = build_robust_payload_from_coarse(
        _make_coarse_payload(
            coarse_pick_i=np.array([100, 102, 104, 106], dtype=np.int32),
            coarse_pmax=np.array([0.9, 0.9, 0.8, 0.85], dtype=np.float32),
            offsets_m=np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
        ),
        cfg={},
        source_model_id='coarse-model',
        iter_id='',
        repo_root=tmp_path,
    )
    lineage = json.loads(np.asarray(payload['lineage']).item())

    assert set(ROBUST_REQUIRED_KEYS).issubset(payload.keys())
    assert payload['robust_pick_i'].dtype == np.int32
    assert payload['robust_pick_t_sec'].dtype == np.float32
    assert payload['robust_source'].dtype == np.uint8
    assert payload['trend_center_i'].dtype == np.int32
    assert payload['trend_center_t_sec'].dtype == np.float32
    assert payload['physical_center_i'].dtype == np.int32
    assert payload['physical_center_t_sec'].dtype == np.float32
    assert payload['fine_center_i'].dtype == np.int32
    assert payload['fine_center_t_sec'].dtype == np.float32
    assert payload['physical_model_status'].dtype == np.uint8
    assert payload['physical_model_failure_reason'].dtype == np.uint8
    np.testing.assert_array_equal(
        payload['physical_center_i'],
        payload['trend_center_i'],
    )
    np.testing.assert_array_equal(
        payload['physical_center_t_sec'],
        payload['trend_center_t_sec'],
    )
    np.testing.assert_array_equal(payload['fine_center_i'], payload['trend_center_i'])
    np.testing.assert_array_equal(
        payload['fine_center_t_sec'],
        payload['trend_center_t_sec'],
    )
    assert np.all(
        payload['physical_model_status']
        == np.uint8(PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED)
    )
    assert np.all(
        payload['physical_model_failure_reason']
        == np.uint8(PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED)
    )
    assert lineage['source_model_id'] == 'coarse-model'
    assert lineage['git_sha'] is None


def test_build_robust_payload_from_coarse_uses_physical_center_with_geometry(
    tmp_path: Path,
) -> None:
    offsets_m = np.linspace(50.0, 1200.0, 24, dtype=np.float32)
    pick_t_sec = np.float32(0.02) + offsets_m / np.float32(3000.0)
    coarse_pick_i = np.rint(pick_t_sec / np.float32(0.004)).astype(np.int32)
    coarse_npz = {
        **_make_coarse_payload(
            coarse_pick_i=coarse_pick_i,
            coarse_pmax=np.full((24,), 0.95, dtype=np.float32),
            offsets_m=offsets_m,
        ),
        **_make_physical_geometry(offsets_m),
    }

    payload = build_robust_payload_from_coarse(
        coarse_npz,
        cfg={
            'physical_trend': {
                'enabled': True,
                'segment_by_offset_sign': False,
                'split_by_offset_gap': False,
            },
            'neighbor_context': {'enabled': False},
            'physical_prefilter': {'enabled': False},
            'two_piece_ransac': {'min_pts': 3, 'seed': 7},
        },
        source_model_id='coarse-model',
        iter_id='',
        repo_root=tmp_path,
    )

    assert (set(ROBUST_PHYSICAL_OPTIONAL_KEYS) - _ANCHOR_OPTIONAL_KEYS).issubset(
        payload.keys()
    )
    assert np.any(
        payload['physical_model_status']
        == np.uint8(PHYSICAL_MODEL_STATUS_TWO_PIECE_OK)
    )
    np.testing.assert_array_equal(
        payload['fine_center_i'],
        payload['physical_center_i'],
    )
    np.testing.assert_array_equal(
        payload['fine_center_t_sec'],
        payload['physical_center_t_sec'],
    )


def test_runtime_diagnostics_do_not_change_physical_center_outputs(
    tmp_path: Path,
) -> None:
    offsets_m = np.linspace(50.0, 1200.0, 24, dtype=np.float32)
    pick_t_sec = np.float32(0.02) + offsets_m / np.float32(3000.0)
    coarse_pick_i = np.rint(pick_t_sec / np.float32(0.004)).astype(np.int32)
    coarse_npz = {
        **_make_coarse_payload(
            coarse_pick_i=coarse_pick_i,
            coarse_pmax=np.full((24,), 0.95, dtype=np.float32),
            offsets_m=offsets_m,
        ),
        **_make_physical_geometry(offsets_m),
    }
    base_cfg = {
        'physical_trend': {
            'enabled': True,
            'segment_by_offset_sign': False,
            'split_by_offset_gap': False,
        },
        'neighbor_context': {'enabled': False},
        'physical_prefilter': {'enabled': False},
        'two_piece_ransac': {'min_pts': 3, 'seed': 7},
    }

    baseline = build_robust_payload_from_coarse(
        coarse_npz,
        cfg={**base_cfg, 'physical_runtime': {'diagnostics_enabled': False}},
        source_model_id='coarse-model',
        iter_id='',
        repo_root=tmp_path,
    )
    diagnostic = build_robust_payload_from_coarse(
        coarse_npz,
        cfg={**base_cfg, 'physical_runtime': {'diagnostics_enabled': True}},
        source_model_id='coarse-model',
        iter_id='',
        repo_root=tmp_path,
    )

    for key in (
        'physical_center_i',
        'fine_center_i',
        'physical_model_status',
        'physical_model_failure_reason',
    ):
        np.testing.assert_array_equal(diagnostic[key], baseline[key])
    assert 'physics_total_sec' in diagnostic
    assert 'physics_total_sec' not in baseline


def test_anchor_selection_dry_run_does_not_change_physical_outputs(
    tmp_path: Path,
) -> None:
    n_groups = 11
    traces_per_group = 6
    group_source_x = np.arange(n_groups, dtype=np.float32) * np.float32(100.0)
    group_offsets = np.linspace(50.0, 550.0, traces_per_group, dtype=np.float32)
    source_x_m = np.repeat(group_source_x, traces_per_group).astype(np.float32)
    source_y_m = np.zeros((n_groups * traces_per_group,), dtype=np.float32)
    offsets_m = np.tile(group_offsets, n_groups).astype(np.float32)
    receiver_x_m = source_x_m + offsets_m
    pick_t_sec = np.float32(0.02) + offsets_m / np.float32(2500.0)
    coarse_pick_i = np.rint(pick_t_sec / np.float32(0.004)).astype(np.int32)
    coarse_npz = {
        **_make_coarse_payload(
            coarse_pick_i=coarse_pick_i,
            coarse_pmax=np.full_like(offsets_m, 0.95, dtype=np.float32),
            offsets_m=offsets_m,
            ffid_values=np.repeat(
                np.arange(n_groups, dtype=np.int32),
                traces_per_group,
            ),
        ),
        'source_x_m': source_x_m,
        'source_y_m': source_y_m,
        'receiver_x_m': receiver_x_m.astype(np.float32),
        'receiver_y_m': np.zeros_like(receiver_x_m, dtype=np.float32),
        'offset_abs_geom_m': offsets_m.astype(np.float32),
        'geometry_valid_mask': np.ones_like(offsets_m, dtype=np.bool_),
    }
    base_cfg = {
        'physical_trend': {
            'enabled': True,
            'segment_by_offset_sign': False,
            'split_by_offset_gap': False,
        },
        'neighbor_context': {'enabled': False},
        'physical_prefilter': {'enabled': False},
        'two_piece_ransac': {'min_pts': 2, 'seed': 7},
    }

    baseline = build_robust_payload_from_coarse(
        coarse_npz,
        cfg={**base_cfg, 'physical_runtime': {'diagnostics_enabled': True}},
        source_model_id='coarse-model',
        iter_id='',
        repo_root=tmp_path,
    )
    dry_run = build_robust_payload_from_coarse(
        coarse_npz,
        cfg={
            **base_cfg,
            'physical_runtime': {
                'fit_policy': 'full',
                'diagnostics_enabled': True,
                'anchor_selection': {
                    'enabled': True,
                    'mode': 'source_xy_stride',
                    'anchor_stride_source_groups': 5,
                    'include_first': True,
                    'include_last': True,
                },
            },
        },
        source_model_id='coarse-model',
        iter_id='',
        repo_root=tmp_path,
    )

    for key in (
        'physical_center_i',
        'fine_center_i',
        'physical_model_status',
        'physical_model_failure_reason',
    ):
        np.testing.assert_array_equal(dry_run[key], baseline[key])
    assert int(np.asarray(dry_run['n_fit_calls']).item()) == int(
        np.asarray(baseline['n_fit_calls']).item()
    )
    assert int(np.asarray(dry_run['n_source_groups']).item()) == n_groups
    assert int(np.asarray(dry_run['n_anchor_groups']).item()) == 3
    assert int(np.asarray(dry_run['anchor_stride_source_groups']).item()) == 5
    assert np.asarray(dry_run['anchor_selection_mode']).item() == 'source_xy_stride'
    summary = runtime_summary_from_npz_fields(dry_run)
    assert summary is not None
    assert summary['n_anchor_groups'] == 3
    assert summary['anchor_selection_mode'] == 'source_xy_stride'
    assert float(np.asarray(dry_run['anchor_source_distance_p50_m']).item()) == 100.0
    assert float(np.asarray(dry_run['anchor_source_distance_p90_m']).item()) == 200.0
    assert float(np.asarray(dry_run['anchor_source_distance_max_m']).item()) == 200.0
    assert 'physical_anchor_group_id' in dry_run
    assert 'physical_anchor_group_id' not in baseline
    np.testing.assert_array_equal(
        dry_run['physical_anchor_is_anchor'][::traces_per_group],
        np.array(
            [True, False, False, False, False, True, False, False, False, False, True],
            dtype=np.bool_,
        ),
    )


def test_run_physics_lite_with_enabled_physical_and_no_geometry_saves_fallback_status(
    tmp_path: Path,
) -> None:
    n_traces = 16
    offsets_m = np.linspace(50.0, 800.0, n_traces, dtype=np.float32)
    coarse_pick_i = np.rint(
        (np.float32(0.02) + offsets_m / np.float32(3000.0)) / np.float32(0.004)
    ).astype(np.int32)
    coarse_path = save_coarse_npz(
        tmp_path / 'legacy_geometry.coarse.npz',
        **_make_coarse_payload(
            coarse_pick_i=coarse_pick_i,
            coarse_pmax=np.full((n_traces,), 0.95, dtype=np.float32),
            offsets_m=offsets_m,
        ),
    )

    out_path = run_physics_lite(
        coarse_path,
        cfg={
            'physical_trend': {'enabled': True},
            'two_piece_ransac': {'min_pts': 3},
        },
        source_model_id='coarse-model',
        iter_id='',
        repo_root=tmp_path,
    )
    robust = load_robust_npz(out_path)

    assert (set(ROBUST_PHYSICAL_OPTIONAL_KEYS) - _ANCHOR_OPTIONAL_KEYS).issubset(
        robust.keys()
    )
    assert np.all(
        robust['physical_model_status']
        == np.uint8(PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND)
    )
    assert np.all(
        robust['physical_model_failure_reason']
        == np.uint8(PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID)
    )
    np.testing.assert_array_equal(robust['fine_center_i'], robust['trend_center_i'])


def test_run_physics_lite_uses_header_offsets_when_geometry_offset_disabled(
    tmp_path: Path,
) -> None:
    n_traces = 24
    offsets_m = np.linspace(50.0, 1200.0, n_traces, dtype=np.float32)
    coarse_pick_i = np.rint(
        (np.float32(0.02) + offsets_m / np.float32(3000.0)) / np.float32(0.004)
    ).astype(np.int32)
    coarse_path = save_coarse_npz(
        tmp_path / 'header_offset.coarse.npz',
        **_make_coarse_payload(
            coarse_pick_i=coarse_pick_i,
            coarse_pmax=np.full((n_traces,), 0.95, dtype=np.float32),
            offsets_m=offsets_m,
        ),
    )

    out_path = run_physics_lite(
        coarse_path,
        cfg={
            'physical_trend': {
                'enabled': True,
                'use_geometry_offset': False,
                'segment_by_offset_sign': False,
                'split_by_offset_gap': False,
            },
            'physical_prefilter': {'enabled': False},
            'two_piece_ransac': {'min_pts': 3, 'seed': 7},
        },
        source_model_id='coarse-model',
        iter_id='',
        repo_root=tmp_path,
    )
    robust = load_robust_npz(out_path)

    assert np.any(
        robust['physical_model_status'] == np.uint8(PHYSICAL_MODEL_STATUS_TWO_PIECE_OK)
    )
    assert not np.all(
        robust['physical_model_failure_reason']
        == np.uint8(PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID)
    )
    assert np.all(
        robust['physical_offset_source'] == np.uint8(PHYSICAL_OFFSET_SOURCE_HEADER)
    )


def test_build_lineage_payload_reads_git_sha_from_ancestor_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / 'repo'
    nested_root = repo_root / 'packages' / 'seisai-engine'
    sha = 'deadbeef1234567890abcdef1234567890abcdef'

    nested_root.mkdir(parents=True)
    _write_git_repo(repo_root, sha=sha)

    assert read_git_sha(nested_root) == sha

    lineage = json.loads(
        np.asarray(
            build_lineage_payload({}, repo_root=nested_root, source_model_id='coarse-model', iter_id='i0')
        ).item()
    )

    assert lineage['source_model_id'] == 'coarse-model'
    assert lineage['iter_id'] == 'i0'
    assert lineage['git_sha'] == sha
