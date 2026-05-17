from __future__ import annotations

import numpy as np
import torch
from fbpick_physical_center_helpers import (
    ConstantTrendModel,
    fit_linear_model,
    make_inputs,
    physical_cfg,
)
from seisai_engine.pipelines.fbpick.physics.physical_center import (
    PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT,
    PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT,
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_FULL_FIT_NO_COMPATIBLE_ANCHOR,
    PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE,
    build_geometry_two_piece_physical_center,
)
from seisai_engine.pipelines.fbpick.physics.runtime_diagnostics import (
    PhysicalRuntimeDiagnostics,
)
from seisai_pick.trend.trend_fit_strategy import TwoPieceRansacAutoBreakStrategy


def test_anchor_source_xy_reuses_nearest_anchor_without_non_anchor_fit(
    monkeypatch,
) -> None:
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        x_np = x_abs.detach().cpu().numpy().copy()
        y_np = y_sec.detach().cpu().numpy().copy()
        calls.append((x_np, y_np))
        return ConstantTrendModel(float(y_np[0]))

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
    coarse_npz, table, feasible, trend, merged = make_inputs(
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
        cfg=physical_cfg(
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
        return fit_linear_model(x_abs, y_sec)

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)

    traces_per_group = 4
    offsets = np.tile(
        np.asarray([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
        2,
    )
    anchor_pick_i = np.asarray([100, 200, 300, 400], dtype=np.int32)
    pick_i = np.concatenate([anchor_pick_i, anchor_pick_i + 10]).astype(np.int32)
    source_x = np.repeat(np.asarray([0.0, 100.0], dtype=np.float32), traces_per_group)
    coarse_npz, table, feasible, trend, merged = make_inputs(
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
        cfg=physical_cfg(
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
        return fit_linear_model(x_abs, y_sec)

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)

    traces_per_group = 4
    offsets = np.tile(
        np.asarray([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
        2,
    )
    anchor_pick_i = np.asarray([100, 200, 300, 400], dtype=np.int32)
    pick_i = np.concatenate([anchor_pick_i, anchor_pick_i + 100]).astype(np.int32)
    source_x = np.repeat(np.asarray([0.0, 100.0], dtype=np.float32), traces_per_group)
    coarse_npz, table, feasible, trend, merged = make_inputs(
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
        cfg=physical_cfg(
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
        return fit_linear_model(x_abs, y_sec)

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
    coarse_npz, table, feasible, trend, merged = make_inputs(
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
        cfg=physical_cfg(
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
            return fit_linear_model(x_abs, y_sec)
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
    coarse_npz, table, feasible, trend, merged = make_inputs(
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
        cfg=physical_cfg(
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
        return ConstantTrendModel(float(y_np[0]))

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)

    traces_per_group = 4
    offsets = np.tile(
        np.asarray([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
        2,
    )
    pick_i = np.asarray([100, 101, 102, 103, 300, 301, 302, 303], dtype=np.int32)
    source_x = np.repeat(np.asarray([0.0, 100.0], dtype=np.float32), traces_per_group)
    signed_offset = np.asarray([-1.0] * traces_per_group + [1.0] * traces_per_group)
    coarse_npz, table, feasible, trend, merged = make_inputs(
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
        cfg=physical_cfg(
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
