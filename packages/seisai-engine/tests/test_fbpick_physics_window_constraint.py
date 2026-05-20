from __future__ import annotations

import numpy as np
from seisai_engine.pipelines.fbpick.common import (
    FINE_WINDOW_REJECT_BAND_TOO_NARROW_FOR_256,
    FINE_WINDOW_REJECT_CENTER_OUTSIDE_PREFILTER_BAND,
    FINE_WINDOW_REJECT_FALLBACK_ROBUST_NOT_ALLOWED,
    FINE_WINDOW_REJECT_OK,
    FINE_WINDOW_REJECT_WINDOW_OUTSIDE_PREFILTER_BAND,
    ROBUST_SOURCE_COARSE_OBSERVED,
    load_robust_npz,
    save_robust_npz,
)
from seisai_engine.pipelines.fbpick.physics.config import (
    PhysicalFineWindowConstraintCfg,
    PhysicalPrefilterCfg,
)
from seisai_engine.pipelines.fbpick.physics.physical_center_types import (
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST,
)
from seisai_engine.pipelines.fbpick.physics.window_constraint import (
    evaluate_fine_window_constraint,
)


def _prefilter_band(lo_i: int, hi_i: int) -> PhysicalPrefilterCfg:
    return PhysicalPrefilterCfg(
        vmin_m_s=300.0,
        vmax_m_s=6000.0,
        t0_lo_ms=float(lo_i),
        t0_hi_ms=float(hi_i),
    )


def _evaluate(
    *,
    lo_i: int,
    hi_i: int,
    center_i: int,
    constraint: PhysicalFineWindowConstraintCfg | None = None,
    physical_runtime_fit_source: np.ndarray | None = None,
):
    return evaluate_fine_window_constraint(
        offsets_m=np.asarray([0.0], dtype=np.float32),
        dt_sec=0.001,
        n_samples_orig=512,
        fine_center_i=np.asarray([center_i], dtype=np.int32),
        physical_prefilter=_prefilter_band(lo_i, hi_i),
        constraint=(
            PhysicalFineWindowConstraintCfg()
            if constraint is None
            else constraint
        ),
        physical_runtime_fit_source=physical_runtime_fit_source,
    )


def test_fine_window_constraint_accepts_center_and_window_inside_band() -> None:
    result = _evaluate(lo_i=0, hi_i=255, center_i=128)

    np.testing.assert_array_equal(result.fine_window_physical_lo_i, [0])
    np.testing.assert_array_equal(result.fine_window_physical_hi_i, [255])
    np.testing.assert_array_equal(result.fine_center_valid_mask, [True])
    np.testing.assert_array_equal(result.fine_window_valid_mask, [True])
    np.testing.assert_array_equal(
        result.fine_window_reject_reason,
        [FINE_WINDOW_REJECT_OK],
    )


def test_fine_window_constraint_rejects_center_outside_band() -> None:
    result = _evaluate(lo_i=0, hi_i=255, center_i=300)

    np.testing.assert_array_equal(result.fine_center_valid_mask, [False])
    np.testing.assert_array_equal(result.fine_window_valid_mask, [False])
    np.testing.assert_array_equal(
        result.fine_window_reject_reason,
        [FINE_WINDOW_REJECT_CENTER_OUTSIDE_PREFILTER_BAND],
    )


def test_fine_window_constraint_rejects_window_outside_band() -> None:
    result = _evaluate(lo_i=100, hi_i=400, center_i=128)

    np.testing.assert_array_equal(result.fine_center_valid_mask, [True])
    np.testing.assert_array_equal(result.fine_window_valid_mask, [False])
    np.testing.assert_array_equal(
        result.fine_window_reject_reason,
        [FINE_WINDOW_REJECT_WINDOW_OUTSIDE_PREFILTER_BAND],
    )


def test_fine_window_constraint_rejects_band_too_narrow_for_256() -> None:
    result = _evaluate(lo_i=100, hi_i=200, center_i=128)

    np.testing.assert_array_equal(result.fine_center_valid_mask, [True])
    np.testing.assert_array_equal(result.fine_window_valid_mask, [False])
    np.testing.assert_array_equal(
        result.fine_window_reject_reason,
        [FINE_WINDOW_REJECT_BAND_TOO_NARROW_FOR_256],
    )


def test_fine_window_constraint_rejects_disallowed_robust_fallback_source() -> None:
    result = _evaluate(
        lo_i=0,
        hi_i=511,
        center_i=128,
        constraint=PhysicalFineWindowConstraintCfg(
            allow_robust_fallback_as_fine_center=False,
        ),
        physical_runtime_fit_source=np.asarray(
            [PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST],
            dtype=np.uint8,
        ),
    )

    np.testing.assert_array_equal(result.fine_center_valid_mask, [False])
    np.testing.assert_array_equal(result.fine_window_valid_mask, [False])
    np.testing.assert_array_equal(
        result.fine_window_reject_reason,
        [FINE_WINDOW_REJECT_FALLBACK_ROBUST_NOT_ALLOWED],
    )


def test_save_robust_npz_roundtrips_fine_window_constraint_fields(tmp_path) -> None:
    n_traces = 2
    dt_sec = np.float32(0.001)
    robust_pick_i = np.asarray([128, 300], dtype=np.int32)
    payload = {
        'dt_sec': dt_sec,
        'n_samples_orig': np.asarray(512, dtype=np.int32),
        'n_traces': np.asarray(n_traces, dtype=np.int32),
        'ffid_values': np.asarray([1, 1], dtype=np.int32),
        'chno_values': np.asarray([1, 2], dtype=np.int32),
        'offsets_m': np.asarray([0.0, 0.0], dtype=np.float32),
        'trace_indices': np.arange(n_traces, dtype=np.int64),
        'robust_pick_i': robust_pick_i,
        'robust_pick_t_sec': robust_pick_i.astype(np.float32) * dt_sec,
        'robust_conf': np.asarray([1.0, 0.5], dtype=np.float32),
        'robust_source': np.full(
            (n_traces,),
            ROBUST_SOURCE_COARSE_OBSERVED,
            dtype=np.uint8,
        ),
        'used_theoretical_mask': np.zeros((n_traces,), dtype=np.bool_),
        'reason_mask': np.zeros((n_traces,), dtype=np.uint8),
        'conf_prob1': np.asarray([1.0, 0.5], dtype=np.float32),
        'conf_trend1': np.asarray([1.0, 0.5], dtype=np.float32),
        'conf_rs1': np.asarray([1.0, 0.5], dtype=np.float32),
        'lineage': np.asarray('{"stage":"window-constraint-test"}'),
        'fine_center_valid_mask': np.asarray([True, False], dtype=np.bool_),
        'fine_window_valid_mask': np.asarray([True, False], dtype=np.bool_),
        'fine_window_physical_lo_i': np.asarray([0, 0], dtype=np.int32),
        'fine_window_physical_hi_i': np.asarray([255, 255], dtype=np.int32),
        'fine_window_reject_reason': np.asarray(
            [
                FINE_WINDOW_REJECT_OK,
                FINE_WINDOW_REJECT_CENTER_OUTSIDE_PREFILTER_BAND,
            ],
            dtype=np.uint8,
        ),
    }

    out_path = save_robust_npz(tmp_path / 'window.robust.npz', **payload)
    loaded = load_robust_npz(out_path)

    for key in (
        'fine_center_valid_mask',
        'fine_window_valid_mask',
        'fine_window_physical_lo_i',
        'fine_window_physical_hi_i',
        'fine_window_reject_reason',
    ):
        np.testing.assert_array_equal(loaded[key], payload[key])
