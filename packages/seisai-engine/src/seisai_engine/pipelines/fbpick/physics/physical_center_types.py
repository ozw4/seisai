"""Physical center public constants and result types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # noqa: TC002

PHYSICAL_MODEL_STATUS_TWO_PIECE_OK = 0
PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT = 1
PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND = 2
PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP = 3
PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST = 4
PHYSICAL_MODEL_STATUS_GEOMETRY_INVALID = 5
PHYSICAL_MODEL_STATUS_INSUFFICIENT_OBSERVATIONS = 6
PHYSICAL_MODEL_STATUS_FIT_FAILED = 7
PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED = 8

PHYSICAL_MODEL_STATUS_LABELS = {
    PHYSICAL_MODEL_STATUS_TWO_PIECE_OK: 'two_piece_ok',
    PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT: 'relaxed_segment_ok',
    PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND: 'fallback_existing_trend',
    PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP: 'fallback_feasible_clip',
    PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST: 'fallback_robust',
    PHYSICAL_MODEL_STATUS_GEOMETRY_INVALID: 'geometry_invalid',
    PHYSICAL_MODEL_STATUS_INSUFFICIENT_OBSERVATIONS: 'insufficient_observations',
    PHYSICAL_MODEL_STATUS_FIT_FAILED: 'fit_failed',
    PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED: 'physical_disabled',
}

PHYSICAL_MODEL_FAILURE_NONE = 0
PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED = 1
PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID = 2
PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS = 3
PHYSICAL_MODEL_FAILURE_FIT_FAILED = 4
PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID = 5

PHYSICAL_MODEL_FAILURE_LABELS = {
    PHYSICAL_MODEL_FAILURE_NONE: 'none',
    PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED: 'physical_disabled',
    PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID: 'geometry_invalid',
    PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS: 'insufficient_observations',
    PHYSICAL_MODEL_FAILURE_FIT_FAILED: 'fit_failed',
    PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID: 'prediction_invalid',
}

PHYSICAL_OFFSET_SOURCE_NONE = 0
PHYSICAL_OFFSET_SOURCE_GEOMETRY = 1
PHYSICAL_OFFSET_SOURCE_HEADER = 2

PHYSICAL_OFFSET_SOURCE_LABELS = {
    PHYSICAL_OFFSET_SOURCE_NONE: 'none',
    PHYSICAL_OFFSET_SOURCE_GEOMETRY: 'geometry_offset',
    PHYSICAL_OFFSET_SOURCE_HEADER: 'header_offset',
}

PHYSICAL_RUNTIME_FIT_SOURCE_FULL_FIT = 0
PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT = 1
PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE = 2
PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_FULL_FIT_NO_COMPATIBLE_ANCHOR = 3
PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND = 4
PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST = 5
PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT = 6

PHYSICAL_RUNTIME_FIT_SOURCE_LABELS = {
    PHYSICAL_RUNTIME_FIT_SOURCE_FULL_FIT: 'full_fit',
    PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT: 'anchor_fit',
    PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE: 'nearest_anchor_reuse',
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_FULL_FIT_NO_COMPATIBLE_ANCHOR: (
        'fallback_full_fit_no_compatible_anchor'
    ),
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND: 'fallback_existing_trend',
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST: 'fallback_robust',
    PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT: 'adaptive_refit',
}

__all__ = [
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
]


@dataclass(frozen=True)
class PhysicalCenterFallbackPreflight:
    """Preflight summary for configured physical-center fallback behavior."""

    status: str | None
    reason: str | None
    fallback_mode: str | None
    geometry_loaded: bool
    groups: int | None


@dataclass(frozen=True)
class PhysicalCenterResult:
    """Arrays produced by physical-center construction."""

    physical_center_i: np.ndarray
    physical_center_t_sec: np.ndarray
    fine_center_i: np.ndarray
    fine_center_t_sec: np.ndarray
    physical_model_status: np.ndarray
    physical_model_failure_reason: np.ndarray
    physical_offset_source: np.ndarray
    physical_model_break_offset_m: np.ndarray
    physical_model_slope_near_s_per_m: np.ndarray
    physical_model_slope_far_s_per_m: np.ndarray
    physical_model_velocity_near_m_s: np.ndarray
    physical_model_velocity_far_m_s: np.ndarray
    physical_model_neighbor_count: np.ndarray
    physical_prefilter_valid_count: np.ndarray
    physical_model_segment_id: np.ndarray
    physical_model_side: np.ndarray
    physical_model_resid_p50_ms: np.ndarray
    physical_model_resid_p90_ms: np.ndarray
    physical_anchor_group_id: np.ndarray
    physical_anchor_is_anchor: np.ndarray
    physical_anchor_nearest_anchor_group_id: np.ndarray
    physical_anchor_source_distance_m: np.ndarray
    physical_runtime_t0_shift_ms: np.ndarray
    physical_runtime_reuse_resid_p50_ms: np.ndarray
    physical_runtime_reuse_resid_p90_ms: np.ndarray
    physical_runtime_reuse_valid_count: np.ndarray
    physical_runtime_refit_mask: np.ndarray
    physical_runtime_fit_source: np.ndarray
