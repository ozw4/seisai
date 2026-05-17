from __future__ import annotations

import numpy as np
import pytest
import torch
from fbpick_physical_center_helpers import (
    fake_piecewise_model,
    fit_linear_model,
    make_inputs,
    physical_cfg,
    two_piece_time_sec,
)
from seisai_engine.pipelines.fbpick.physics import (
    physical_center_context_fit as context_fit_mod,
)
from seisai_engine.pipelines.fbpick.physics import (
    physical_center_geometry as physical_center_geometry_mod,
)
from seisai_engine.pipelines.fbpick.physics import (
    physical_center_observation as observation_mod,
)
from seisai_engine.pipelines.fbpick.physics.geometry import (
    CoarseGeometry,
    SourceGroup,
    signed_offset_side_from_geometry,
    split_offset_gap_segments,
)
from seisai_engine.pipelines.fbpick.physics.physical_center import (
    build_geometry_two_piece_physical_center,
)
from seisai_engine.pipelines.fbpick.physics.runtime_diagnostics import (
    PhysicalRuntimeDiagnostics,
)
from seisai_pick.trend.trend_fit_strategy import TwoPieceRansacAutoBreakStrategy


def _source_groups_for_context_tests() -> tuple[
    SourceGroup,
    ...,
]:
    return (
        SourceGroup(
            group_id=0,
            source_key_x=0,
            source_key_y=0,
            source_x_m=0.0,
            source_y_m=0.0,
            trace_indices=np.asarray([0, 1], dtype=np.int64),
        ),
        SourceGroup(
            group_id=1,
            source_key_x=100,
            source_key_y=0,
            source_x_m=100.0,
            source_y_m=0.0,
            trace_indices=np.asarray([2, 3], dtype=np.int64),
        ),
        SourceGroup(
            group_id=2,
            source_key_x=250,
            source_key_y=0,
            source_x_m=250.0,
            source_y_m=0.0,
            trace_indices=np.asarray([4, 5], dtype=np.int64),
        ),
        SourceGroup(
            group_id=3,
            source_key_x=600,
            source_key_y=0,
            source_x_m=600.0,
            source_y_m=0.0,
            trace_indices=np.asarray([6, 7], dtype=np.int64),
        ),
    )


def _single_group_context(
    obs_indices: np.ndarray,
    *,
    side_context: observation_mod._SideObservationContext | None = None,
) -> dict[int, observation_mod._GroupObservationContext]:
    obs = np.asarray(obs_indices, dtype=np.int64)
    return {
        0: observation_mod._GroupObservationContext(
            group_id=0,
            neighbor_group_ids=np.asarray([0], dtype=np.int64),
            neighbor_indices=obs,
            valid_obs_indices=obs,
            valid_obs_key=tuple(obs.tolist()),
            neighbor_count=1,
            prefilter_valid_count=int(obs.size),
            side_context=side_context,
        )
    }


def test_group_observation_contexts_match_existing_selection_and_filtering() -> None:
    groups = _source_groups_for_context_tests()
    groups_by_id = {int(group.group_id): group for group in groups}
    valid_for_fit = np.asarray(
        [True, False, True, True, False, True, True, False],
        dtype=np.bool_,
    )
    cfg = physical_cfg(
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

    contexts = observation_mod._build_group_observation_contexts(
        groups=groups,
        groups_by_id=groups_by_id,
        valid_for_fit=valid_for_fit,
        cfg=cfg,
        use_neighbor_context=True,
    )

    assert set(contexts) == {0, 1, 2, 3}
    for group in groups:
        group_id = int(group.group_id)
        expected_group_ids = observation_mod._select_group_ids(
            groups=groups,
            target_group_id=group_id,
            cfg=cfg,
            use_neighbor_context=True,
        )
        expected_neighbor_indices = observation_mod._concat_group_traces(
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

    plan = context_fit_mod._build_observation_plan(
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
    cfg = physical_cfg({'neighbor_context': {'enabled': False}})

    contexts = observation_mod._build_group_observation_contexts(
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
    cfg = physical_cfg(
        {
            'physical_trend': {
                'segment_by_offset_sign': True,
                'split_by_offset_gap': False,
            },
            'neighbor_context': {'enabled': False},
            'two_piece_ransac': {'min_pts': 2},
        }
    )
    labels = physical_center_geometry_mod._signed_offset_side_labels(
        np.asarray([-3.0, 2.0, -1.0, 0.0, 4.0, 5.0, -6.0, 7.0], dtype=np.float32)
    )
    cache = observation_mod._ObservationPlanCache(offset_signed_labels=labels)
    diagnostics = PhysicalRuntimeDiagnostics(detailed_timing=True)

    contexts = observation_mod._build_group_observation_contexts(
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


def test_observation_plan_propagates_precomputed_obs_keys() -> None:
    obs = np.arange(12, dtype=np.int64)
    offsets = np.asarray(
        [
            100.0,
            110.0,
            120.0,
            130.0,
            1000.0,
            1010.0,
            1020.0,
            1030.0,
            2000.0,
            2010.0,
            2020.0,
            2030.0,
        ],
        dtype=np.float32,
    )

    disabled_cfg = physical_cfg(
        {
            'physical_trend': {
                'segment_by_offset_sign': False,
                'split_by_offset_gap': False,
            },
            'neighbor_context': {'enabled': False},
            'two_piece_ransac': {'min_pts': 2},
        }
    )
    disabled_plan = context_fit_mod._build_observation_plan(
        trace_idx=4,
        target_group_id=0,
        group_context_by_id=_single_group_context(obs),
        geometry=None,
        offset_abs_m=offsets,
        offset_signed_m=None,
        cfg=disabled_cfg,
    )
    assert disabled_plan is not None
    assert disabled_plan.obs_key == tuple(obs.tolist())

    gap_cfg = physical_cfg(
        {
            'physical_trend': {
                'segment_by_offset_sign': False,
                'split_by_offset_gap': True,
            },
            'neighbor_context': {'enabled': False},
            'two_piece_ransac': {'min_pts': 2},
        }
    )
    gap_plan = context_fit_mod._build_observation_plan(
        trace_idx=8,
        target_group_id=0,
        group_context_by_id=_single_group_context(obs),
        geometry=None,
        offset_abs_m=offsets,
        offset_signed_m=None,
        cfg=gap_cfg,
    )
    assert gap_plan is not None
    np.testing.assert_array_equal(gap_plan.obs_indices, np.asarray([8, 9, 10, 11]))
    assert gap_plan.obs_key == (8, 9, 10, 11)

    labels = physical_center_geometry_mod._signed_offset_side_labels(
        np.asarray([-1.0] * 4 + [1.0] * 8, dtype=np.float32)
    )
    cache = observation_mod._ObservationPlanCache(offset_signed_labels=labels)
    side_context = observation_mod._build_side_observation_context(
        labels=labels,
        obs_indices=obs,
        obs_key=tuple(obs.tolist()),
        cache=cache,
    )
    side_cfg = physical_cfg(
        {
            'physical_trend': {
                'segment_by_offset_sign': True,
                'split_by_offset_gap': False,
            },
            'neighbor_context': {'enabled': False},
            'two_piece_ransac': {'min_pts': 2},
        }
    )
    side_plan = context_fit_mod._build_observation_plan(
        trace_idx=4,
        target_group_id=0,
        group_context_by_id=_single_group_context(obs, side_context=side_context),
        geometry=None,
        offset_abs_m=offsets,
        offset_signed_m=labels.side.astype(np.float32),
        cfg=side_cfg,
        plan_cache=cache,
    )
    assert side_plan is not None
    np.testing.assert_array_equal(side_plan.obs_indices, np.arange(4, 12))
    assert side_plan.obs_key == tuple(range(4, 12))

    side_gap_cfg = physical_cfg(
        {
            'physical_trend': {
                'segment_by_offset_sign': True,
                'split_by_offset_gap': True,
            },
            'neighbor_context': {'enabled': False},
            'two_piece_ransac': {'min_pts': 2},
        }
    )
    side_gap_plan = context_fit_mod._build_observation_plan(
        trace_idx=8,
        target_group_id=0,
        group_context_by_id=_single_group_context(obs, side_context=side_context),
        geometry=None,
        offset_abs_m=offsets,
        offset_signed_m=labels.side.astype(np.float32),
        cfg=side_gap_cfg,
        plan_cache=cache,
    )
    assert side_gap_plan is not None
    np.testing.assert_array_equal(
        side_gap_plan.obs_indices,
        np.asarray([8, 9, 10, 11]),
    )
    assert side_gap_plan.obs_key == (8, 9, 10, 11)

    fallback_plan = context_fit_mod._build_observation_plan(
        trace_idx=10,
        target_group_id=0,
        group_context_by_id=_single_group_context(obs[:8]),
        geometry=None,
        offset_abs_m=np.asarray(
            [
                100.0,
                110.0,
                120.0,
                130.0,
                1000.0,
                1010.0,
                1020.0,
                1030.0,
                2000.0,
                2010.0,
                1025.0,
            ],
            dtype=np.float32,
        ),
        offset_signed_m=None,
        cfg=gap_cfg,
    )
    assert fallback_plan is not None
    assert fallback_plan.obs_key == tuple(fallback_plan.obs_indices.tolist())


def test_side_context_lookup_diagnostics_are_separate_from_cache_hits() -> None:
    obs = np.arange(4, dtype=np.int64)
    obs_key = tuple(obs.tolist())
    labels = physical_center_geometry_mod._signed_offset_side_labels(
        np.asarray([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)
    )
    cache = observation_mod._ObservationPlanCache(offset_signed_labels=labels)
    diagnostics = PhysicalRuntimeDiagnostics(detailed_timing=True)

    context = observation_mod._build_side_observation_context(
        labels=labels,
        obs_indices=obs,
        obs_key=obs_key,
        cache=cache,
        runtime_diagnostics=diagnostics,
    )
    cached_context = observation_mod._build_side_observation_context(
        labels=labels,
        obs_indices=obs,
        obs_key=obs_key,
        cache=cache,
        runtime_diagnostics=diagnostics,
    )
    assert cached_context is context

    observation_mod._obs_with_target_signed_offset_side(
        trace_idx=2,
        obs_indices=obs,
        signed_offset_m=labels.side.astype(np.float32),
        obs_key=obs_key,
        cache=cache,
        labels=labels,
        side_context=context,
        runtime_diagnostics=diagnostics,
    )

    summary = diagnostics.to_summary()
    assert summary['n_side_contexts_built'] == 1
    assert summary['n_side_context_cache_hits'] == 1
    assert summary['n_side_context_lookup_calls'] == 1

def _saved_signed_geometry(
    *,
    offset_abs_m: np.ndarray,
    offset_signed_m: np.ndarray,
    geometry_valid_mask: np.ndarray,
) -> CoarseGeometry:
    offsets = np.asarray(offset_abs_m, dtype=np.float32)
    n_traces = int(offsets.size)
    return CoarseGeometry(
        source_x_m=np.zeros((n_traces,), dtype=np.float32),
        source_y_m=np.zeros((n_traces,), dtype=np.float32),
        receiver_x_m=offsets.astype(np.float32, copy=True),
        receiver_y_m=np.zeros((n_traces,), dtype=np.float32),
        offset_abs_geom_m=offsets,
        geometry_valid_mask=np.asarray(geometry_valid_mask, dtype=np.bool_),
        offset_signed_geom_m=np.asarray(offset_signed_m, dtype=np.float32),
    )


def _saved_geometry_plan_pair(
    *,
    trace_idx: int,
    offset_abs_m: np.ndarray,
    offset_signed_m: np.ndarray,
    geometry_valid_mask: np.ndarray,
    cfg: object,
) -> tuple[
    observation_mod._ObservationPlan,
    observation_mod._ObservationPlan,
]:
    obs = np.arange(int(np.asarray(offset_abs_m).size), dtype=np.int64)
    geometry = _saved_signed_geometry(
        offset_abs_m=offset_abs_m,
        offset_signed_m=offset_signed_m,
        geometry_valid_mask=geometry_valid_mask,
    )
    labels = physical_center_geometry_mod._signed_offset_side_labels(
        offset_signed_m,
        finite_mask=geometry.geometry_valid_mask,
    )
    group = SourceGroup(
        group_id=0,
        source_key_x=0,
        source_key_y=0,
        source_x_m=0.0,
        source_y_m=0.0,
        trace_indices=obs,
    )
    cache = observation_mod._ObservationPlanCache(offset_signed_labels=labels)
    contexts = observation_mod._build_group_observation_contexts(
        groups=(group,),
        groups_by_id={0: group},
        valid_for_fit=np.ones((obs.size,), dtype=np.bool_),
        cfg=cfg,
        use_neighbor_context=False,
        offset_signed_labels=labels,
        plan_cache=cache,
    )
    fast = context_fit_mod._build_observation_plan(
        trace_idx=trace_idx,
        target_group_id=0,
        group_context_by_id=contexts,
        geometry=geometry,
        offset_abs_m=offset_abs_m,
        offset_signed_m=offset_signed_m,
        cfg=cfg,
        plan_cache=cache,
    )
    legacy = _legacy_build_observation_plan(
        trace_idx=trace_idx,
        target_group_id=0,
        group_context_by_id=contexts,
        geometry=geometry,
        offset_abs_m=offset_abs_m,
        offset_signed_m=None,
        cfg=cfg,
    )
    assert fast is not None
    assert legacy is not None
    return fast, legacy


@pytest.mark.parametrize('split_by_offset_gap', [False, True])
def test_saved_geometry_signed_offset_fast_path_matches_legacy_edge_cases(
    split_by_offset_gap: bool,
) -> None:
    cfg = physical_cfg(
        {
            'physical_trend': {
                'use_geometry_offset': True,
                'segment_by_offset_sign': True,
                'split_by_offset_gap': split_by_offset_gap,
            },
            'neighbor_context': {'enabled': False},
            'two_piece_ransac': {'min_pts': 2},
        }
    )
    offset_abs_m = np.asarray(
        [
            100.0,
            110.0,
            120.0,
            130.0,
            500.0,
            510.0,
            520.0,
            530.0,
            1000.0,
            1010.0,
            1020.0,
            1030.0,
        ],
        dtype=np.float32,
    )
    offset_signed_m = np.asarray(
        [
            -100.0,
            -110.0,
            -120.0,
            -130.0,
            0.0,
            np.nan,
            np.nan,
            0.0,
            1000.0,
            1010.0,
            1020.0,
            1030.0,
        ],
        dtype=np.float32,
    )
    geometry_valid_mask = np.asarray(
        [True, True, True, True, True, False, True, True, True, True, True, True]
    )

    for trace_idx in (0, 5, 8):
        fast, legacy = _saved_geometry_plan_pair(
            trace_idx=trace_idx,
            offset_abs_m=offset_abs_m,
            offset_signed_m=offset_signed_m,
            geometry_valid_mask=geometry_valid_mask,
            cfg=cfg,
        )
        np.testing.assert_array_equal(fast.obs_indices, legacy.obs_indices)
        assert fast.obs_key == tuple(fast.obs_indices.tolist())
        assert fast.side == legacy.side
        assert fast.relaxed == legacy.relaxed
        assert fast.segment_id == legacy.segment_id

    sparse_fast, sparse_legacy = _saved_geometry_plan_pair(
        trace_idx=3,
        offset_abs_m=np.asarray([100.0, 110.0, 120.0, 130.0], dtype=np.float32),
        offset_signed_m=np.asarray([np.nan, np.nan, np.nan, 130.0], dtype=np.float32),
        geometry_valid_mask=np.asarray([False, False, False, True]),
        cfg=cfg,
    )
    np.testing.assert_array_equal(sparse_fast.obs_indices, sparse_legacy.obs_indices)
    assert sparse_fast.side == sparse_legacy.side
    assert sparse_fast.relaxed == sparse_legacy.relaxed
    assert sparse_fast.segment_id == sparse_legacy.segment_id


def test_physical_center_uses_saved_signed_offset_for_side_segmentation(
    monkeypatch,
) -> None:
    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    offsets = np.linspace(50.0, 600.0, 12, dtype=np.float32)
    coarse_npz, table, feasible, trend, merged = make_inputs(offsets_m=offsets)
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
        cfg=physical_cfg(
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
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    offsets = np.linspace(50.0, 600.0, 12, dtype=np.float32)
    coarse_npz, table, feasible, trend, merged = make_inputs(offsets_m=offsets)
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
        cfg=physical_cfg(
            {
                'physical_trend': {'split_by_offset_gap': False},
                'two_piece_ransac': {'min_pts': 3},
            }
        ),
    )

    np.testing.assert_array_equal(result.physical_model_side, expected_side)


def _legacy_append_index(indices: np.ndarray, trace_idx: int) -> np.ndarray:
    return observation_mod._stable_unique(
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
    signed = signed_offset_side_from_geometry(
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
    segment_id = split_offset_gap_segments(
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
    min_fit_obs=None,
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
        return observation_mod._ObservationPlan(
            obs_indices=valid_obs,
            obs_key=tuple(np.asarray(valid_obs, dtype=np.int64).tolist()),
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
        return observation_mod._ObservationPlan(
            obs_indices=segment_obs,
            obs_key=tuple(np.asarray(segment_obs, dtype=np.int64).tolist()),
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
        return observation_mod._ObservationPlan(
            obs_indices=side_obs,
            obs_key=tuple(np.asarray(side_obs, dtype=np.int64).tolist()),
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=0,
            side=side,
            relaxed=True,
        )

    if side_reliable and int(valid_obs.size) >= min_fit_obs:
        return observation_mod._ObservationPlan(
            obs_indices=valid_obs,
            obs_key=tuple(np.asarray(valid_obs, dtype=np.int64).tolist()),
            neighbor_count=neighbor_count,
            prefilter_valid_count=prefilter_valid_count,
            segment_id=0,
            side=0,
            relaxed=True,
        )

    return observation_mod._ObservationPlan(
        obs_indices=segment_obs,
        obs_key=tuple(np.asarray(segment_obs, dtype=np.int64).tolist()),
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
        return fit_linear_model(x_abs, y_sec)

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    side = np.asarray([-1.0] * 8 + [1.0] * 8, dtype=np.float32)
    offsets = np.tile(
        np.asarray(
            [100.0, 110.0, 120.0, 130.0, 1000.0, 1010.0, 1020.0, 1030.0],
            dtype=np.float32,
        ),
        2,
    )
    pick_i = np.rint(two_piece_time_sec(offsets) / np.float32(0.001)).astype(
        np.int32
    )
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=offsets,
        pick_i=pick_i,
    )
    coarse_npz['receiver_x_m'] = (side * offsets).astype(np.float32)
    coarse_npz['offset_abs_geom_m'] = offsets.astype(np.float32)
    coarse_npz['offset_signed_geom_m'] = (side * offsets).astype(np.float32)

    cfg = physical_cfg(
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
        context_fit_mod,
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
    cfg = physical_cfg(
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
    cache = observation_mod._ObservationPlanCache()
    diagnostics = PhysicalRuntimeDiagnostics(detailed_timing=True)

    fast_obs, fast_key, fast_segment_id = (
        observation_mod._obs_with_target_gap_segment(
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
        observation_mod._obs_with_target_gap_segment(
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
