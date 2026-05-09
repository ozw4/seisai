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
    PHYSICAL_OFFSET_SOURCE_HEADER,
    PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
    PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED,
    PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND,
    PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED,
    PHYSICAL_MODEL_STATUS_TWO_PIECE_OK,
)
from seisai_engine.pipelines.fbpick.physics.pick_table import (
    normalize_coarse_pick_table,
)
from seisai_engine.pipelines.fbpick.physics.run import (
    build_robust_payload_from_coarse,
    run_physics_lite,
)
from seisai_engine.pipelines.fbpick.physics.trend import TrendResult, build_trend_result


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
        }
    )
    assert out['physical_trend']['enabled'] is True
    assert out['neighbor_context']['max_source_distance_m'] == 1000.0
    assert out['physical_projection']['mode'] == 'model'


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
    assert ROBUST_OPTIONAL_KEYS == (
        *ROBUST_CENTER_OPTIONAL_KEYS,
        *ROBUST_PHYSICAL_DIAGNOSTIC_OPTIONAL_KEYS,
    )
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

    assert lineage['git_sha'] is None

    assert out_path.name == 'synthetic.robust.npz'
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

    assert set(ROBUST_PHYSICAL_OPTIONAL_KEYS).issubset(payload.keys())
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

    assert set(ROBUST_PHYSICAL_OPTIONAL_KEYS).issubset(robust.keys())
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
