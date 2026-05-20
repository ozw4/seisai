"""Fine-window validity checks against the physical prefilter band."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from seisai_engine.pipelines.fbpick.common import (
    FINE_WINDOW_REJECT_BAND_INVALID,
    FINE_WINDOW_REJECT_BAND_TOO_NARROW_FOR_256,
    FINE_WINDOW_REJECT_CENTER_OUTSIDE_PREFILTER_BAND,
    FINE_WINDOW_REJECT_FALLBACK_FEASIBLE_CLIP_NOT_ALLOWED,
    FINE_WINDOW_REJECT_FALLBACK_ROBUST_NOT_ALLOWED,
    FINE_WINDOW_REJECT_OK,
    FINE_WINDOW_REJECT_WINDOW_OUTSIDE_PREFILTER_BAND,
)

from .physical_center_types import (
    PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP,
    PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST,
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST,
)

if TYPE_CHECKING:
    from .config import PhysicalFineWindowConstraintCfg, PhysicalPrefilterCfg

__all__ = [
    'FineWindowConstraintResult',
    'compute_physical_prefilter_sample_band',
    'evaluate_fine_window_constraint',
]


@dataclass(frozen=True)
class FineWindowConstraintResult:
    """Per-trace masks and reasons for fine-window physical-band validity."""

    fine_center_valid_mask: np.ndarray
    fine_window_valid_mask: np.ndarray
    fine_window_physical_lo_i: np.ndarray
    fine_window_physical_hi_i: np.ndarray
    fine_window_reject_reason: np.ndarray


def _as_vector(
    name: str,
    value: np.ndarray,
    *,
    length: int,
    dtype: object,
) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 1 or int(arr.shape[0]) != int(length):
        msg = f'{name} must be 1D with length n_traces'
        raise ValueError(msg)
    return arr


def compute_physical_prefilter_sample_band(
    *,
    offsets_m: np.ndarray,
    dt_sec: float,
    n_samples_orig: int,
    physical_prefilter: PhysicalPrefilterCfg,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the configured physical prefilter band to sample intervals."""
    offsets = np.asarray(offsets_m, dtype=np.float64)
    dt = float(dt_sec)
    n_samples = int(n_samples_orig)
    if offsets.ndim != 1:
        msg = 'offsets_m must be 1D'
        raise ValueError(msg)
    if (not np.isfinite(dt)) or dt <= 0.0:
        msg = 'dt_sec must be finite and > 0'
        raise ValueError(msg)
    if n_samples <= 0:
        msg = 'n_samples_orig must be positive'
        raise ValueError(msg)

    abs_offset = np.abs(offsets)
    lo_t = (
        abs_offset / float(physical_prefilter.vmax_m_s)
        + float(physical_prefilter.t0_lo_ms) / 1000.0
    )
    hi_t = (
        abs_offset / float(physical_prefilter.vmin_m_s)
        + float(physical_prefilter.t0_hi_ms) / 1000.0
    )
    lo_i = np.ceil(lo_t / dt).astype(np.int64)
    hi_i = np.floor(hi_t / dt).astype(np.int64)
    lo_i = np.clip(lo_i, 0, n_samples - 1).astype(np.int32, copy=False)
    hi_i = np.clip(hi_i, 0, n_samples - 1).astype(np.int32, copy=False)
    return lo_i, hi_i


def evaluate_fine_window_constraint(  # noqa: PLR0913, PLR0915
    *,
    offsets_m: np.ndarray,
    dt_sec: float,
    n_samples_orig: int,
    fine_center_i: np.ndarray,
    physical_prefilter: PhysicalPrefilterCfg,
    constraint: PhysicalFineWindowConstraintCfg,
    physical_model_status: np.ndarray | None = None,
    physical_runtime_fit_source: np.ndarray | None = None,
) -> FineWindowConstraintResult:
    """Evaluate center and full-window containment against the prefilter band."""
    centers = np.asarray(fine_center_i, dtype=np.int64)
    if centers.ndim != 1:
        msg = 'fine_center_i must be 1D'
        raise ValueError(msg)
    n_traces = int(centers.shape[0])
    offsets = _as_vector(
        'offsets_m',
        np.asarray(offsets_m),
        length=n_traces,
        dtype=np.float64,
    )
    lo_i, hi_i = compute_physical_prefilter_sample_band(
        offsets_m=offsets,
        dt_sec=dt_sec,
        n_samples_orig=n_samples_orig,
        physical_prefilter=physical_prefilter,
    )

    if not bool(constraint.enabled):
        return FineWindowConstraintResult(
            fine_center_valid_mask=np.ones((n_traces,), dtype=np.bool_),
            fine_window_valid_mask=np.ones((n_traces,), dtype=np.bool_),
            fine_window_physical_lo_i=lo_i,
            fine_window_physical_hi_i=hi_i,
            fine_window_reject_reason=np.full(
                (n_traces,),
                FINE_WINDOW_REJECT_OK,
                dtype=np.uint8,
            ),
        )

    band_valid = lo_i <= hi_i
    center_inside_band = (
        band_valid & (lo_i.astype(np.int64) <= centers) & (centers <= hi_i)
    )
    time_len = int(constraint.time_len)
    center_index = int(constraint.center_index)
    window_start_i = centers - center_index
    window_end_i = window_start_i + time_len - 1
    window_inside_band = (
        band_valid
        & (lo_i.astype(np.int64) <= window_start_i)
        & (window_end_i <= hi_i.astype(np.int64))
    )
    band_width = hi_i.astype(np.int64) - lo_i.astype(np.int64) + 1
    band_too_narrow = band_valid & (band_width < time_len)

    center_valid = center_inside_band.copy()
    window_valid = band_valid.copy()
    if bool(constraint.require_center_inside_band):
        window_valid &= center_inside_band
    if bool(constraint.require_window_inside_band):
        window_valid &= window_inside_band

    reason = np.full((n_traces,), FINE_WINDOW_REJECT_OK, dtype=np.uint8)
    reason[~band_valid] = np.uint8(FINE_WINDOW_REJECT_BAND_INVALID)

    disallowed_source = np.zeros((n_traces,), dtype=np.bool_)
    if (
        physical_runtime_fit_source is not None
        and not bool(constraint.allow_robust_fallback_as_fine_center)
    ):
        fit_source = _as_vector(
            'physical_runtime_fit_source',
            np.asarray(physical_runtime_fit_source),
            length=n_traces,
            dtype=np.uint8,
        )
        disallowed_source |= (
            fit_source == np.uint8(PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST)
        )
    if physical_model_status is not None:
        status = _as_vector(
            'physical_model_status',
            np.asarray(physical_model_status),
            length=n_traces,
            dtype=np.uint8,
        )
        if not bool(constraint.allow_robust_fallback_as_fine_center):
            disallowed_source |= (
                status == np.uint8(PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST)
            )
        if not bool(constraint.allow_feasible_clip_as_fine_center):
            feasible_clip = (
                status == np.uint8(PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP)
            )
            apply = band_valid & feasible_clip & (reason == FINE_WINDOW_REJECT_OK)
            reason[apply] = np.uint8(
                FINE_WINDOW_REJECT_FALLBACK_FEASIBLE_CLIP_NOT_ALLOWED
            )
            center_valid[feasible_clip] = False
            window_valid[feasible_clip] = False

    apply = band_valid & disallowed_source & (reason == FINE_WINDOW_REJECT_OK)
    reason[apply] = np.uint8(FINE_WINDOW_REJECT_FALLBACK_ROBUST_NOT_ALLOWED)
    center_valid[disallowed_source] = False
    window_valid[disallowed_source] = False

    apply = (
        band_valid
        & bool(constraint.require_center_inside_band)
        & (~center_inside_band)
        & (reason == FINE_WINDOW_REJECT_OK)
    )
    reason[apply] = np.uint8(FINE_WINDOW_REJECT_CENTER_OUTSIDE_PREFILTER_BAND)
    apply = band_too_narrow & (reason == FINE_WINDOW_REJECT_OK)
    reason[apply] = np.uint8(FINE_WINDOW_REJECT_BAND_TOO_NARROW_FOR_256)
    apply = (
        band_valid
        & bool(constraint.require_window_inside_band)
        & (~window_inside_band)
        & (reason == FINE_WINDOW_REJECT_OK)
    )
    reason[apply] = np.uint8(FINE_WINDOW_REJECT_WINDOW_OUTSIDE_PREFILTER_BAND)
    window_valid[reason != FINE_WINDOW_REJECT_OK] = False

    return FineWindowConstraintResult(
        fine_center_valid_mask=center_valid.astype(np.bool_, copy=False),
        fine_window_valid_mask=window_valid.astype(np.bool_, copy=False),
        fine_window_physical_lo_i=lo_i.astype(np.int32, copy=False),
        fine_window_physical_hi_i=hi_i.astype(np.int32, copy=False),
        fine_window_reject_reason=reason,
    )
