from __future__ import annotations

# ruff: noqa: D100,D101,D102,D103
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    'PHYSICS_RUNTIME_ANCHOR_DIAGNOSTIC_KEYS',
    'PHYSICS_RUNTIME_BASE_DIAGNOSTIC_KEYS',
    'PHYSICS_RUNTIME_DIAGNOSTIC_KEYS',
    'PHYSICS_RUNTIME_STRING_DIAGNOSTIC_KEYS',
    'PhysicalRuntimeDiagnostics',
    'derive_physics_runtime_summary_path',
    'runtime_summary_from_npz_fields',
    'write_physics_runtime_summary',
]

PHYSICS_RUNTIME_BASE_DIAGNOSTIC_KEYS = (
    'physics_total_sec',
    'physical_center_total_sec',
    'ransac_fit_total_sec',
    'n_fit_calls',
    'n_anchor_fit_calls',
    'n_cache_hits',
    'n_cache_misses',
    'cache_hit_rate',
    'n_source_groups',
    'n_non_anchor_groups',
    'n_reused_predictions',
    'n_t0_shifted_groups',
    'n_t0_shifted_predictions',
    't0_shift_ms_p50',
    't0_shift_ms_p90',
    't0_shift_ms_p99',
    'reuse_resid_p90_ms_p50',
    'reuse_resid_p90_ms_p90',
    'n_adaptive_refit_calls',
    'adaptive_refit_rate',
    'n_adaptive_refit_success',
    'n_adaptive_refit_failed',
    'n_fallback_full_fit_no_compatible_anchor',
    'n_unique_fit_contexts',
    'fit_call_reduction_rate_vs_full',
    'ransac_fit_time_p50_sec',
    'ransac_fit_time_p90_sec',
    'ransac_fit_time_p99_sec',
    'obs_count_for_fit_p50',
    'obs_count_for_fit_p90',
    'obs_count_for_fit_p99',
)
PHYSICS_RUNTIME_ANCHOR_DIAGNOSTIC_KEYS = (
    'n_anchor_groups',
    'anchor_stride_source_groups',
    'anchor_selection_mode',
    'anchor_source_distance_p50_m',
    'anchor_source_distance_p90_m',
    'anchor_source_distance_max_m',
)
PHYSICS_RUNTIME_DIAGNOSTIC_KEYS = (
    *PHYSICS_RUNTIME_BASE_DIAGNOSTIC_KEYS,
    *PHYSICS_RUNTIME_ANCHOR_DIAGNOSTIC_KEYS,
)
PHYSICS_RUNTIME_STRING_DIAGNOSTIC_KEYS = frozenset({'anchor_selection_mode'})


def _percentile(values: list[float] | list[int], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


@dataclass
class PhysicalRuntimeDiagnostics:
    physics_total_sec: float = 0.0
    physical_center_total_sec: float = 0.0
    ransac_fit_total_sec: float = 0.0
    n_fit_calls: int = 0
    n_anchor_fit_calls: int = 0
    n_cache_hits: int = 0
    n_cache_misses: int = 0
    n_source_groups: int = 0
    n_non_anchor_groups: int = 0
    n_reused_predictions: int = 0
    n_t0_shifted_groups: int = 0
    n_t0_shifted_predictions: int = 0
    n_adaptive_refit_calls: int = 0
    n_adaptive_refit_success: int = 0
    n_adaptive_refit_failed: int = 0
    n_fallback_full_fit_no_compatible_anchor: int = 0
    n_unique_fit_contexts: int = 0
    fit_call_reduction_rate_vs_full: float = 0.0
    _anchor_summary: dict[str, float | int | str] | None = field(
        default=None,
        repr=False,
    )
    _fit_times_sec: list[float] = field(default_factory=list, repr=False)
    _fit_obs_counts: list[int] = field(default_factory=list, repr=False)
    _t0_shift_abs_ms: list[float] = field(default_factory=list, repr=False)
    _reuse_resid_p90_ms: list[float] = field(default_factory=list, repr=False)

    @contextmanager
    def time_physics(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.physics_total_sec += time.perf_counter() - start

    @contextmanager
    def time_physical_center(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.physical_center_total_sec += time.perf_counter() - start

    @contextmanager
    def time_ransac_fit(self, *, obs_count: int) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.n_fit_calls += 1
            self.ransac_fit_total_sec += elapsed
            self._fit_times_sec.append(float(elapsed))
            self._fit_obs_counts.append(int(obs_count))

    @property
    def cache_hit_rate(self) -> float:
        total = int(self.n_cache_hits) + int(self.n_cache_misses)
        if total == 0:
            return 0.0
        return float(self.n_cache_hits) / float(total)

    @property
    def adaptive_refit_rate(self) -> float:
        if int(self.n_non_anchor_groups) <= 0:
            return 0.0
        return float(self.n_adaptive_refit_calls) / float(self.n_non_anchor_groups)

    def record_cache_hit(self) -> None:
        self.n_cache_hits += 1

    def record_cache_miss(self) -> None:
        self.n_cache_misses += 1

    def record_anchor_fit_calls(self, value: int) -> None:
        self.n_anchor_fit_calls += int(value)

    def record_reused_predictions(self, value: int) -> None:
        self.n_reused_predictions += int(value)

    def record_t0_shifted_group(
        self,
        *,
        t0_shift_ms: float,
        prediction_count: int,
        reuse_resid_p90_ms: float,
    ) -> None:
        self.n_t0_shifted_groups += 1
        self.n_t0_shifted_predictions += int(prediction_count)
        shift_abs = abs(float(t0_shift_ms))
        if np.isfinite(shift_abs):
            self._t0_shift_abs_ms.append(shift_abs)
        resid_p90 = float(reuse_resid_p90_ms)
        if np.isfinite(resid_p90):
            self._reuse_resid_p90_ms.append(resid_p90)

    def record_adaptive_refit(self, *, success: bool) -> None:
        self.n_adaptive_refit_calls += 1
        if bool(success):
            self.n_adaptive_refit_success += 1
        else:
            self.n_adaptive_refit_failed += 1

    def record_fallback_full_fit_no_compatible_anchor(self, value: int) -> None:
        self.n_fallback_full_fit_no_compatible_anchor += int(value)

    def set_source_groups(self, value: int) -> None:
        self.n_source_groups = int(value)

    def set_anchor_reuse_groups(self, *, n_non_anchor_groups: int) -> None:
        self.n_non_anchor_groups = int(n_non_anchor_groups)

    def set_unique_fit_contexts(self, value: int) -> None:
        self.n_unique_fit_contexts = int(value)

    def set_fit_call_reduction_rate_vs_full(
        self,
        *,
        full_fit_call_count_estimate: int,
    ) -> None:
        full_count = int(full_fit_call_count_estimate)
        if full_count <= 0:
            self.fit_call_reduction_rate_vs_full = 0.0
            return
        reduction = (float(full_count) - float(self.n_fit_calls)) / float(full_count)
        self.fit_call_reduction_rate_vs_full = float(max(0.0, reduction))

    def set_anchor_selection(
        self,
        *,
        n_anchor_groups: int,
        anchor_stride_source_groups: int,
        anchor_selection_mode: str,
        source_distance_m: np.ndarray,
    ) -> None:
        distances = np.asarray(source_distance_m, dtype=np.float64)
        distances = distances[np.isfinite(distances)]
        self._anchor_summary = {
            'n_anchor_groups': int(n_anchor_groups),
            'anchor_stride_source_groups': int(anchor_stride_source_groups),
            'anchor_selection_mode': str(anchor_selection_mode),
            'anchor_source_distance_p50_m': _percentile(distances.tolist(), 50.0),
            'anchor_source_distance_p90_m': _percentile(distances.tolist(), 90.0),
            'anchor_source_distance_max_m': (
                0.0 if distances.size == 0 else float(np.max(distances))
            ),
        }

    def to_summary(self) -> dict[str, float | int | str]:
        summary: dict[str, float | int | str] = {
            'physics_total_sec': float(self.physics_total_sec),
            'physical_center_total_sec': float(self.physical_center_total_sec),
            'ransac_fit_total_sec': float(self.ransac_fit_total_sec),
            'n_fit_calls': int(self.n_fit_calls),
            'n_anchor_fit_calls': int(self.n_anchor_fit_calls),
            'n_cache_hits': int(self.n_cache_hits),
            'n_cache_misses': int(self.n_cache_misses),
            'cache_hit_rate': float(self.cache_hit_rate),
            'n_source_groups': int(self.n_source_groups),
            'n_non_anchor_groups': int(self.n_non_anchor_groups),
            'n_reused_predictions': int(self.n_reused_predictions),
            'n_t0_shifted_groups': int(self.n_t0_shifted_groups),
            'n_t0_shifted_predictions': int(self.n_t0_shifted_predictions),
            't0_shift_ms_p50': _percentile(self._t0_shift_abs_ms, 50.0),
            't0_shift_ms_p90': _percentile(self._t0_shift_abs_ms, 90.0),
            't0_shift_ms_p99': _percentile(self._t0_shift_abs_ms, 99.0),
            'reuse_resid_p90_ms_p50': _percentile(
                self._reuse_resid_p90_ms,
                50.0,
            ),
            'reuse_resid_p90_ms_p90': _percentile(
                self._reuse_resid_p90_ms,
                90.0,
            ),
            'n_adaptive_refit_calls': int(self.n_adaptive_refit_calls),
            'adaptive_refit_rate': float(self.adaptive_refit_rate),
            'n_adaptive_refit_success': int(self.n_adaptive_refit_success),
            'n_adaptive_refit_failed': int(self.n_adaptive_refit_failed),
            'n_fallback_full_fit_no_compatible_anchor': int(
                self.n_fallback_full_fit_no_compatible_anchor
            ),
            'n_unique_fit_contexts': int(self.n_unique_fit_contexts),
            'fit_call_reduction_rate_vs_full': float(
                self.fit_call_reduction_rate_vs_full
            ),
            'ransac_fit_time_p50_sec': _percentile(self._fit_times_sec, 50.0),
            'ransac_fit_time_p90_sec': _percentile(self._fit_times_sec, 90.0),
            'ransac_fit_time_p99_sec': _percentile(self._fit_times_sec, 99.0),
            'obs_count_for_fit_p50': _percentile(self._fit_obs_counts, 50.0),
            'obs_count_for_fit_p90': _percentile(self._fit_obs_counts, 90.0),
            'obs_count_for_fit_p99': _percentile(self._fit_obs_counts, 99.0),
        }
        if self._anchor_summary is not None:
            summary.update(self._anchor_summary)
        return summary

    def to_npz_fields(self) -> dict[str, np.ndarray]:
        summary = self.to_summary()
        int_keys = {
            'n_fit_calls',
            'n_anchor_fit_calls',
            'n_cache_hits',
            'n_cache_misses',
            'n_source_groups',
            'n_non_anchor_groups',
            'n_reused_predictions',
            'n_t0_shifted_groups',
            'n_t0_shifted_predictions',
            'n_adaptive_refit_calls',
            'n_adaptive_refit_success',
            'n_adaptive_refit_failed',
            'n_fallback_full_fit_no_compatible_anchor',
            'n_unique_fit_contexts',
            'n_anchor_groups',
            'anchor_stride_source_groups',
        }
        out: dict[str, np.ndarray] = {}
        for key, value in summary.items():
            if key in PHYSICS_RUNTIME_STRING_DIAGNOSTIC_KEYS:
                out[key] = np.asarray(str(value))
            else:
                out[key] = np.asarray(
                    value,
                    dtype=np.int64 if key in int_keys else np.float64,
                )
        return out


def runtime_summary_from_npz_fields(
    payload: dict[str, np.ndarray],
) -> dict[str, float | int | str] | None:
    if 'physics_total_sec' not in payload:
        return None
    summary: dict[str, float | int | str] = {}
    int_keys = {
        'n_fit_calls',
        'n_anchor_fit_calls',
        'n_cache_hits',
        'n_cache_misses',
        'n_source_groups',
        'n_non_anchor_groups',
        'n_reused_predictions',
        'n_t0_shifted_groups',
        'n_t0_shifted_predictions',
        'n_adaptive_refit_calls',
        'n_adaptive_refit_success',
        'n_adaptive_refit_failed',
        'n_fallback_full_fit_no_compatible_anchor',
        'n_unique_fit_contexts',
        'n_anchor_groups',
        'anchor_stride_source_groups',
    }
    for key in PHYSICS_RUNTIME_BASE_DIAGNOSTIC_KEYS:
        if key not in payload:
            return None
        value = np.asarray(payload[key]).item()
        summary[key] = int(value) if key in int_keys else float(value)
    anchor_present = [
        key for key in PHYSICS_RUNTIME_ANCHOR_DIAGNOSTIC_KEYS if key in payload
    ]
    if anchor_present:
        missing = [
            key for key in PHYSICS_RUNTIME_ANCHOR_DIAGNOSTIC_KEYS if key not in payload
        ]
        if missing:
            return None
        for key in PHYSICS_RUNTIME_ANCHOR_DIAGNOSTIC_KEYS:
            value = np.asarray(payload[key]).item()
            if key in PHYSICS_RUNTIME_STRING_DIAGNOSTIC_KEYS:
                summary[key] = str(value)
            else:
                summary[key] = int(value) if key in int_keys else float(value)
    return summary


def derive_physics_runtime_summary_path(robust_npz_path: str | Path) -> Path:
    path = Path(robust_npz_path).expanduser().resolve()
    suffix = '.robust.npz'
    tag = path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem
    return path.with_name(f'{tag}.physics_runtime_summary.json')


def write_physics_runtime_summary(
    robust_npz_path: str | Path,
    summary: dict[str, float | int | str],
) -> Path:
    out_path = derive_physics_runtime_summary_path(robust_npz_path)
    out_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    return out_path
