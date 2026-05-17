from __future__ import annotations

from dataclasses import fields

import numpy as np
import torch
from fbpick_physical_center_helpers import (
    PHYSICAL_CENTER_RESULT_DTYPE_CONTRACT,
    RecordingProgressReporter,
    fake_piecewise_model,
    make_inputs,
    physical_cfg,
)
from seisai_engine.pipelines.fbpick.physics import (
    physical_center as physical_center_mod,
)
from seisai_engine.pipelines.fbpick.physics.physical_center import (
    PhysicalCenterResult,
    build_geometry_two_piece_physical_center,
)
from seisai_engine.pipelines.fbpick.physics.runtime_diagnostics import (
    PHYSICS_RUNTIME_BASE_DIAGNOSTIC_KEYS,
    PhysicalRuntimeDiagnostics,
)
from seisai_pick.trend.trend_fit_strategy import TwoPieceRansacAutoBreakStrategy


def test_physical_center_public_import_contract_is_stable() -> None:
    expected_exports = (
        'PHYSICAL_MODEL_FAILURE_FIT_FAILED',
        'PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID',
        'PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS',
        'PHYSICAL_MODEL_FAILURE_LABELS',
        'PHYSICAL_MODEL_FAILURE_NONE',
        'PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED',
        'PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID',
        'PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND',
        'PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP',
        'PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT',
        'PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST',
        'PHYSICAL_MODEL_STATUS_FIT_FAILED',
        'PHYSICAL_MODEL_STATUS_GEOMETRY_INVALID',
        'PHYSICAL_MODEL_STATUS_INSUFFICIENT_OBSERVATIONS',
        'PHYSICAL_MODEL_STATUS_LABELS',
        'PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED',
        'PHYSICAL_MODEL_STATUS_TWO_PIECE_OK',
        'PHYSICAL_OFFSET_SOURCE_GEOMETRY',
        'PHYSICAL_OFFSET_SOURCE_HEADER',
        'PHYSICAL_OFFSET_SOURCE_LABELS',
        'PHYSICAL_OFFSET_SOURCE_NONE',
        'PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT',
        'PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT',
        'PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND',
        'PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_FULL_FIT_NO_COMPATIBLE_ANCHOR',
        'PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST',
        'PHYSICAL_RUNTIME_FIT_SOURCE_FULL_FIT',
        'PHYSICAL_RUNTIME_FIT_SOURCE_LABELS',
        'PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE',
        'PhysicalCenterFallbackPreflight',
        'PhysicalCenterResult',
        'build_geometry_two_piece_physical_center',
        'preflight_geometry_two_piece_fallback',
    )
    expected_constant_values = {
        'PHYSICAL_MODEL_STATUS_TWO_PIECE_OK': 0,
        'PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT': 1,
        'PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND': 2,
        'PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP': 3,
        'PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST': 4,
        'PHYSICAL_MODEL_STATUS_GEOMETRY_INVALID': 5,
        'PHYSICAL_MODEL_STATUS_INSUFFICIENT_OBSERVATIONS': 6,
        'PHYSICAL_MODEL_STATUS_FIT_FAILED': 7,
        'PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED': 8,
        'PHYSICAL_MODEL_FAILURE_NONE': 0,
        'PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED': 1,
        'PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID': 2,
        'PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS': 3,
        'PHYSICAL_MODEL_FAILURE_FIT_FAILED': 4,
        'PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID': 5,
        'PHYSICAL_OFFSET_SOURCE_NONE': 0,
        'PHYSICAL_OFFSET_SOURCE_GEOMETRY': 1,
        'PHYSICAL_OFFSET_SOURCE_HEADER': 2,
        'PHYSICAL_RUNTIME_FIT_SOURCE_FULL_FIT': 0,
        'PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT': 1,
        'PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE': 2,
        'PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_FULL_FIT_NO_COMPATIBLE_ANCHOR': 3,
        'PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND': 4,
        'PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST': 5,
        'PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT': 6,
    }

    assert tuple(physical_center_mod.__all__) == expected_exports
    for name in expected_exports:
        assert hasattr(physical_center_mod, name)
    for name, value in expected_constant_values.items():
        assert getattr(physical_center_mod, name) == value
    assert physical_center_mod.PHYSICAL_MODEL_STATUS_LABELS == {
        0: 'two_piece_ok',
        1: 'relaxed_segment_ok',
        2: 'fallback_existing_trend',
        3: 'fallback_feasible_clip',
        4: 'fallback_robust',
        5: 'geometry_invalid',
        6: 'insufficient_observations',
        7: 'fit_failed',
        8: 'physical_disabled',
    }
    assert physical_center_mod.PHYSICAL_MODEL_FAILURE_LABELS == {
        0: 'none',
        1: 'physical_disabled',
        2: 'geometry_invalid',
        3: 'insufficient_observations',
        4: 'fit_failed',
        5: 'prediction_invalid',
    }
    assert physical_center_mod.PHYSICAL_OFFSET_SOURCE_LABELS == {
        0: 'none',
        1: 'geometry_offset',
        2: 'header_offset',
    }
    assert physical_center_mod.PHYSICAL_RUNTIME_FIT_SOURCE_LABELS == {
        0: 'full_fit',
        1: 'anchor_fit',
        2: 'nearest_anchor_reuse',
        3: 'fallback_full_fit_no_compatible_anchor',
        4: 'fallback_existing_trend',
        5: 'fallback_robust',
        6: 'adaptive_refit',
    }

def test_physical_center_diagnostic_arrays_are_save_friendly() -> None:
    offsets = np.linspace(50.0, 2000.0, 20, dtype=np.float32)
    coarse_npz, table, feasible, trend, merged = make_inputs(offsets_m=offsets)
    result = build_geometry_two_piece_physical_center(
        coarse_npz=coarse_npz,
        table=table,
        feasible=feasible,
        trend=trend,
        merged=merged,
        cfg=physical_cfg({'two_piece_ransac': {'min_pts': 3}}),
    )

    assert tuple(PHYSICAL_CENTER_RESULT_DTYPE_CONTRACT) == tuple(
        field.name for field in fields(PhysicalCenterResult)
    )
    for field_name, dtype in PHYSICAL_CENTER_RESULT_DTYPE_CONTRACT.items():
        arr = getattr(result, field_name)
        assert arr.shape == (table.n_traces,)
        assert arr.dtype == np.dtype(dtype)
        assert arr.ndim == 1
        assert not arr.dtype.hasobject


def test_physical_center_runtime_and_progress_keys_are_stable(
    monkeypatch,
) -> None:
    def fake_fit(self, x_abs: torch.Tensor, y_sec: torch.Tensor):
        return fake_piecewise_model()

    monkeypatch.setattr(TwoPieceRansacAutoBreakStrategy, 'fit', fake_fit)
    coarse_npz, table, feasible, trend, merged = make_inputs(
        offsets_m=np.linspace(50.0, 1600.0, 12, dtype=np.float32),
    )
    progress = RecordingProgressReporter()
    diagnostics = PhysicalRuntimeDiagnostics()

    build_geometry_two_piece_physical_center(
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
        runtime_diagnostics=diagnostics,
        progress=progress,
        progress_context={'run_id': 'guardrail'},
    )

    events_by_name: dict[str, list[dict[str, object]]] = {}
    for event, event_fields in progress.events:
        events_by_name.setdefault(event, []).append(event_fields)
    assert {
        'physical-center.start',
        'physical-center.stage_start',
        'physical-center.stage_done',
        'physical-center.contexts_built',
        'physical-center.fit_start',
        'fit.progress',
        'physical-center.done',
    }.issubset(events_by_name)
    assert {
        'run_id',
        'fit_kind',
        'fit_policy',
        'groups',
        'sampling',
        'executor',
    }.issubset(events_by_name['physical-center.start'][0])
    assert {'run_id', 'stage'}.issubset(
        events_by_name['physical-center.stage_start'][0]
    )
    assert {'run_id', 'stage', 'elapsed'}.issubset(
        events_by_name['physical-center.stage_done'][0]
    )
    assert {'run_id', 'work_items', 'unique_keys'}.issubset(
        events_by_name['physical-center.contexts_built'][-1]
    )
    assert {'run_id', 'work_items', 'executor'}.issubset(
        events_by_name['physical-center.fit_start'][0]
    )
    assert {
        'run_id',
        'done',
        'total',
        'elapsed',
        'rate',
        'eta',
        'force',
        'cache_hit',
        'cache_miss',
        'n_fit_calls',
        'fit_total_sec',
    }.issubset(events_by_name['fit.progress'][-1])
    assert {'run_id', 'status', 'n_traces'}.issubset(
        events_by_name['physical-center.done'][-1]
    )

    summary = diagnostics.to_summary()
    assert set(PHYSICS_RUNTIME_BASE_DIAGNOSTIC_KEYS).issubset(summary)
    for key in (
        'physical_center_total_sec',
        'n_traces',
        'n_fit_calls',
        'n_source_groups',
        'n_prediction_calls',
        'observation_sampling_enabled',
    ):
        assert key in summary
