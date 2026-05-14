from __future__ import annotations

# ruff: noqa: D100,D101,D102,D103
import json
import time
from collections import defaultdict
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
    'load_coarse_npz_sec',
    'normalize_table_sec',
    'feasible_band_sec',
    'trend_result_sec',
    'confidence_sec',
    'merge_sec',
    'physical_center_total_sec',
    'save_robust_npz_sec',
    'ransac_fit_total_sec',
    'non_ransac_total_sec',
    'geometry_load_sec',
    'source_grouping_sec',
    'source_group_ordering_sec',
    'neighbor_plan_sec',
    'valid_mask_build_sec',
    'velocity_prefilter_sec',
    'side_segment_build_sec',
    'side_filter_precompute_sec',
    'side_filter_lookup_sec',
    'gap_segment_precompute_sec',
    'gap_segment_lookup_sec',
    'gap_segment_fallback_sec',
    'side_segment_key_build_sec',
    'anchor_selection_sec',
    'anchor_lookup_sec',
    'compatible_anchor_search_sec',
    'observation_sampling_sec',
    't0_shift_sec',
    'adaptive_refit_decision_sec',
    'prediction_sec',
    'assignment_sec',
    'fallback_sec',
    'diagnostics_aggregate_sec',
    'anchor_overhead_total_sec',
    'fit_overhead_total_sec',
    'n_traces',
    'n_fit_contexts',
    'n_fit_calls',
    'n_anchor_fit_calls',
    'n_adaptive_refit_calls',
    'n_fallback_full_fit_no_compatible_anchor',
    'n_reuse_contexts',
    'n_cache_hits',
    'n_cache_misses',
    'cache_hit_rate',
    'n_source_groups',
    'n_side_contexts_built',
    'n_side_context_cache_hits',
    'n_side_context_cache_misses',
    'n_side_context_lookup_calls',
    'n_gap_contexts_built',
    'n_gap_context_cache_hits',
    'n_gap_context_cache_misses',
    'n_gap_fast_path_calls',
    'n_gap_fallback_calls',
    'n_gap_trace_in_obs',
    'n_gap_trace_not_in_obs',
    'n_side_gap_precomputed_fit_keys',
    'n_precomputed_fit_key_used',
    'n_fit_key_built_from_indices',
    'n_fit_key_built_after_sampling',
    'n_fit_key_missing_precomputed',
    'n_non_anchor_groups',
    'n_reused_predictions',
    'n_t0_shifted_groups',
    'n_t0_shifted_predictions',
    't0_shift_ms_p50',
    't0_shift_ms_p90',
    't0_shift_ms_p99',
    'reuse_resid_p90_ms_p50',
    'reuse_resid_p90_ms_p90',
    'adaptive_refit_rate',
    'n_adaptive_refit_success',
    'n_adaptive_refit_failed',
    'n_unique_fit_contexts',
    'n_prediction_calls',
    'n_prediction_batches',
    'anchor_reuse_rate',
    'fit_call_reduction_rate_vs_full',
    'ransac_fit_time_p50_sec',
    'ransac_fit_time_p90_sec',
    'ransac_fit_time_p99_sec',
    'fit_executor_enabled',
    'fit_executor_backend',
    'fit_executor_max_workers',
    'fit_executor_wall_sec',
    'fit_executor_tasks',
    'observation_sampling_enabled',
    'observation_sampling_method',
    'max_obs_per_fit',
    'n_offset_bins',
    'obs_count_before_p50',
    'obs_count_before_p90',
    'obs_count_before_p99',
    'obs_count_before_downsample_p50',
    'obs_count_before_downsample_p90',
    'obs_count_before_downsample_p99',
    'obs_count_after_p50',
    'obs_count_after_p90',
    'obs_count_after_p99',
    'obs_count_after_downsample_p50',
    'obs_count_after_downsample_p90',
    'obs_count_after_downsample_p99',
    'obs_downsample_rate_p50',
    'obs_downsample_rate_p90',
    'n_downsampled_fit_contexts',
    'obs_count_for_fit_p50',
    'obs_count_for_fit_p90',
    'obs_count_for_fit_p99',
    'side_obs_count_p50',
    'side_obs_count_p90',
    'side_obs_count_p99',
    'gap_segment_obs_count_p50',
    'gap_segment_obs_count_p90',
    'gap_segment_obs_count_p99',
    'compatible_anchor_search_candidates_p50',
    'compatible_anchor_search_candidates_p90',
    'compatible_anchor_search_candidates_max',
    'n_no_compatible_anchor_contexts',
    'nearest_anchor_distance_m_p50',
    'nearest_anchor_distance_m_p90',
    'nearest_anchor_distance_m_p99',
    'nearest_anchor_distance_m_max',
    't0_shift_ms_max',
    'reuse_resid_p50_ms_p50',
    'reuse_resid_p50_ms_p90',
    'reuse_resid_p90_ms_p99',
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
PHYSICS_RUNTIME_STRING_DIAGNOSTIC_KEYS = frozenset(
    {
        'anchor_selection_mode',
        'fit_executor_backend',
        'observation_sampling_method',
    }
)
PHYSICS_RUNTIME_PREFIXED_NPZ_KEYS = frozenset(
    {
        'physics_total_sec',
        'physical_center_total_sec',
        'ransac_fit_total_sec',
        'non_ransac_total_sec',
        'n_fit_calls',
        'n_cache_hits',
        'n_cache_misses',
        'cache_hit_rate',
        'n_source_groups',
        'n_anchor_groups',
        'n_non_anchor_groups',
        'n_adaptive_refit_calls',
        'n_fallback_full_fit_no_compatible_anchor',
    }
)

_SIDE_GAP_COUNTER_KEYS = frozenset(
    {
        'n_side_contexts_built',
        'n_side_context_cache_hits',
        'n_side_context_cache_misses',
        'n_side_context_lookup_calls',
        'n_gap_contexts_built',
        'n_gap_context_cache_hits',
        'n_gap_context_cache_misses',
        'n_gap_fast_path_calls',
        'n_gap_fallback_calls',
        'n_gap_trace_in_obs',
        'n_gap_trace_not_in_obs',
        'n_side_gap_precomputed_fit_keys',
        'n_precomputed_fit_key_used',
        'n_fit_key_built_from_indices',
        'n_fit_key_built_after_sampling',
        'n_fit_key_missing_precomputed',
    }
)


def _percentile(values: list[float] | list[int], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


@dataclass
class PhysicalRuntimeDiagnostics:
    detailed_timing: bool = False
    physics_total_sec: float = 0.0
    physical_center_total_sec: float = 0.0
    ransac_fit_total_sec: float = 0.0
    n_traces: int = 0
    n_fit_calls: int = 0
    n_anchor_fit_calls: int = 0
    n_cache_hits: int = 0
    n_cache_misses: int = 0
    n_source_groups: int = 0
    n_side_contexts_built: int = 0
    n_side_context_cache_hits: int = 0
    n_side_context_cache_misses: int = 0
    n_side_context_lookup_calls: int = 0
    n_gap_contexts_built: int = 0
    n_gap_context_cache_hits: int = 0
    n_gap_context_cache_misses: int = 0
    n_gap_fast_path_calls: int = 0
    n_gap_fallback_calls: int = 0
    n_gap_trace_in_obs: int = 0
    n_gap_trace_not_in_obs: int = 0
    n_side_gap_precomputed_fit_keys: int = 0
    n_precomputed_fit_key_used: int = 0
    n_fit_key_built_from_indices: int = 0
    n_fit_key_built_after_sampling: int = 0
    n_fit_key_missing_precomputed: int = 0
    n_non_anchor_groups: int = 0
    n_reused_predictions: int = 0
    n_t0_shifted_groups: int = 0
    n_t0_shifted_predictions: int = 0
    n_adaptive_refit_calls: int = 0
    n_adaptive_refit_success: int = 0
    n_adaptive_refit_failed: int = 0
    n_fallback_full_fit_no_compatible_anchor: int = 0
    n_no_compatible_anchor_contexts: int = 0
    n_reuse_contexts: int = 0
    n_unique_fit_contexts: int = 0
    n_prediction_calls: int = 0
    n_prediction_batches: int = 0
    fit_call_reduction_rate_vs_full: float = 0.0
    observation_sampling_enabled: int = 0
    observation_sampling_method: str = 'offset_bin'
    max_obs_per_fit: int = 0
    n_offset_bins: int = 0
    fit_executor_enabled: int = 0
    fit_executor_backend: str = 'serial'
    fit_executor_max_workers: int = 0
    fit_executor_wall_sec: float = 0.0
    fit_executor_tasks: int = 0
    _anchor_summary: dict[str, float | int | str] | None = field(
        default=None,
        repr=False,
    )
    _fit_times_sec: list[float] = field(default_factory=list, repr=False)
    _fit_obs_counts: list[int] = field(default_factory=list, repr=False)
    _fit_obs_counts_before: list[int] = field(default_factory=list, repr=False)
    _fit_obs_counts_after: list[int] = field(default_factory=list, repr=False)
    _fit_obs_downsample_rates: list[float] = field(default_factory=list, repr=False)
    _side_obs_counts: list[int] = field(default_factory=list, repr=False)
    _gap_segment_obs_counts: list[int] = field(default_factory=list, repr=False)
    _t0_shift_abs_ms: list[float] = field(default_factory=list, repr=False)
    _reuse_resid_p50_ms: list[float] = field(default_factory=list, repr=False)
    _reuse_resid_p90_ms: list[float] = field(default_factory=list, repr=False)
    _compatible_anchor_search_candidates: list[int] = field(
        default_factory=list,
        repr=False,
    )
    _nearest_anchor_distance_m: list[float] = field(default_factory=list, repr=False)
    _adaptive_refit_reason_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int),
        repr=False,
    )
    _timings_sec: dict[str, float] = field(
        default_factory=lambda: defaultdict(float),
        repr=False,
    )

    @contextmanager
    def time_block(self, key: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            if bool(self.detailed_timing):
                self._timings_sec[str(key)] += time.perf_counter() - start

    def add_timing(self, key: str, elapsed_sec: float) -> None:
        if bool(self.detailed_timing):
            self._timings_sec[str(key)] += float(elapsed_sec)

    def inc(self, key: str, n: int = 1) -> None:
        value = int(n)
        if key == 'n_prediction_calls':
            self.n_prediction_calls += value
        elif key == 'n_prediction_batches':
            self.n_prediction_batches += value
        elif key == 'n_reuse_contexts':
            self.n_reuse_contexts += value
        elif key in _SIDE_GAP_COUNTER_KEYS:
            setattr(self, key, int(getattr(self, key)) + value)

    def record_side_obs_count(self, value: int) -> None:
        self._side_obs_counts.append(int(value))

    def record_gap_segment_obs_count(self, value: int) -> None:
        self._gap_segment_obs_counts.append(int(value))

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
    def time_ransac_fit(
        self,
        *,
        obs_count: int,
        obs_count_before: int | None = None,
    ) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.record_ransac_fit(
                elapsed_sec=elapsed,
                obs_count=obs_count,
                obs_count_before=obs_count_before,
            )

    def record_ransac_fit(
        self,
        *,
        elapsed_sec: float,
        obs_count: int,
        obs_count_before: int | None = None,
    ) -> None:
        before_count = (
            int(obs_count) if obs_count_before is None else int(obs_count_before)
        )
        after_count = int(obs_count)
        self.n_fit_calls += 1
        self.ransac_fit_total_sec += float(elapsed_sec)
        self._fit_times_sec.append(float(elapsed_sec))
        self._fit_obs_counts.append(after_count)
        self._fit_obs_counts_before.append(before_count)
        self._fit_obs_counts_after.append(after_count)
        rate = 0.0
        if before_count > 0:
            rate = max(0.0, 1.0 - (float(after_count) / float(before_count)))
        self._fit_obs_downsample_rates.append(float(rate))

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

    @property
    def anchor_reuse_rate(self) -> float:
        if int(self.n_traces) <= 0:
            return 0.0
        return float(self.n_reused_predictions) / float(self.n_traces)

    def record_cache_hit(self, value: int = 1) -> None:
        self.n_cache_hits += int(value)

    def record_cache_miss(self, value: int = 1) -> None:
        self.n_cache_misses += int(value)

    def record_anchor_fit_calls(self, value: int) -> None:
        self.n_anchor_fit_calls += int(value)

    def record_reused_predictions(self, value: int) -> None:
        self.n_reused_predictions += int(value)

    def record_reuse_contexts(self, value: int) -> None:
        self.n_reuse_contexts += int(value)

    def record_no_compatible_anchor_context(self) -> None:
        self.n_no_compatible_anchor_contexts += 1

    def record_compatible_anchor_search_candidates(self, value: int) -> None:
        self._compatible_anchor_search_candidates.append(int(value))

    def record_nearest_anchor_distance(self, value: float) -> None:
        distance = float(value)
        if np.isfinite(distance):
            self._nearest_anchor_distance_m.append(distance)

    def record_t0_shifted_group(
        self,
        *,
        t0_shift_ms: float,
        prediction_count: int,
        reuse_resid_p50_ms: float = np.nan,
        reuse_resid_p90_ms: float,
    ) -> None:
        self.n_t0_shifted_groups += 1
        self.n_t0_shifted_predictions += int(prediction_count)
        shift_abs = abs(float(t0_shift_ms))
        if np.isfinite(shift_abs):
            self._t0_shift_abs_ms.append(shift_abs)
        resid_p90 = float(reuse_resid_p90_ms)
        resid_p50 = float(reuse_resid_p50_ms)
        if np.isfinite(resid_p50):
            self._reuse_resid_p50_ms.append(resid_p50)
        if np.isfinite(resid_p90):
            self._reuse_resid_p90_ms.append(resid_p90)

    def record_adaptive_refit(self, *, success: bool) -> None:
        self.n_adaptive_refit_calls += 1
        if bool(success):
            self.n_adaptive_refit_success += 1
        else:
            self.n_adaptive_refit_failed += 1

    def record_adaptive_refit_decision(self, *, triggered: bool) -> None:
        key = 'triggered' if bool(triggered) else 'not_triggered'
        self._adaptive_refit_reason_counts[key] += 1

    def record_fallback_full_fit_no_compatible_anchor(self, value: int) -> None:
        self.n_fallback_full_fit_no_compatible_anchor += int(value)

    def set_traces(self, value: int) -> None:
        self.n_traces = int(value)

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

    def set_observation_sampling(
        self,
        *,
        enabled: bool,
        method: str,
        max_obs_per_fit: int,
        n_offset_bins: int,
    ) -> None:
        self.observation_sampling_enabled = int(bool(enabled))
        self.observation_sampling_method = str(method)
        self.max_obs_per_fit = int(max_obs_per_fit)
        self.n_offset_bins = int(n_offset_bins)

    def set_fit_executor(
        self,
        *,
        enabled: bool,
        backend: str,
        max_workers: int | None,
    ) -> None:
        self.fit_executor_enabled = int(bool(enabled))
        self.fit_executor_backend = str(backend) if bool(enabled) else 'serial'
        self.fit_executor_max_workers = (
            0 if max_workers is None else int(max_workers)
        )

    def record_fit_executor_run(self, *, wall_sec: float, tasks: int) -> None:
        self.fit_executor_wall_sec += float(wall_sec)
        self.fit_executor_tasks += int(tasks)

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
            'nearest_anchor_distance_m_p50': _percentile(distances.tolist(), 50.0),
            'nearest_anchor_distance_m_p90': _percentile(distances.tolist(), 90.0),
            'nearest_anchor_distance_m_p99': _percentile(distances.tolist(), 99.0),
            'nearest_anchor_distance_m_max': (
                0.0 if distances.size == 0 else float(np.max(distances))
            ),
        }

    def to_summary(self) -> dict[str, object]:
        non_ransac_total_sec = max(
            0.0,
            float(self.physical_center_total_sec) - float(self.ransac_fit_total_sec),
        )
        timing = self._timings_sec
        obs_before_p50 = _percentile(self._fit_obs_counts_before, 50.0)
        obs_before_p90 = _percentile(self._fit_obs_counts_before, 90.0)
        obs_before_p99 = _percentile(self._fit_obs_counts_before, 99.0)
        obs_after_p50 = _percentile(self._fit_obs_counts_after, 50.0)
        obs_after_p90 = _percentile(self._fit_obs_counts_after, 90.0)
        obs_after_p99 = _percentile(self._fit_obs_counts_after, 99.0)
        summary: dict[str, object] = {
            'physics_total_sec': float(self.physics_total_sec),
            'load_coarse_npz_sec': float(timing.get('load_coarse_npz_sec', 0.0)),
            'normalize_table_sec': float(timing.get('normalize_table_sec', 0.0)),
            'feasible_band_sec': float(timing.get('feasible_band_sec', 0.0)),
            'trend_result_sec': float(timing.get('trend_result_sec', 0.0)),
            'confidence_sec': float(timing.get('confidence_sec', 0.0)),
            'merge_sec': float(timing.get('merge_sec', 0.0)),
            'physical_center_total_sec': float(self.physical_center_total_sec),
            'save_robust_npz_sec': float(timing.get('save_robust_npz_sec', 0.0)),
            'ransac_fit_total_sec': float(self.ransac_fit_total_sec),
            'non_ransac_total_sec': float(non_ransac_total_sec),
            'geometry_load_sec': float(timing.get('geometry_load_sec', 0.0)),
            'source_grouping_sec': float(timing.get('source_grouping_sec', 0.0)),
            'source_group_ordering_sec': float(
                timing.get('source_group_ordering_sec', 0.0)
            ),
            'neighbor_plan_sec': float(timing.get('neighbor_plan_sec', 0.0)),
            'valid_mask_build_sec': float(timing.get('valid_mask_build_sec', 0.0)),
            'velocity_prefilter_sec': float(
                timing.get('velocity_prefilter_sec', 0.0)
            ),
            'side_segment_build_sec': float(
                timing.get('side_segment_build_sec', 0.0)
            ),
            'side_filter_precompute_sec': float(
                timing.get('side_filter_precompute_sec', 0.0)
            ),
            'side_filter_lookup_sec': float(
                timing.get('side_filter_lookup_sec', 0.0)
            ),
            'gap_segment_precompute_sec': float(
                timing.get('gap_segment_precompute_sec', 0.0)
            ),
            'gap_segment_lookup_sec': float(
                timing.get('gap_segment_lookup_sec', 0.0)
            ),
            'gap_segment_fallback_sec': float(
                timing.get('gap_segment_fallback_sec', 0.0)
            ),
            'side_segment_key_build_sec': float(
                timing.get('side_segment_key_build_sec', 0.0)
            ),
            'anchor_selection_sec': float(timing.get('anchor_selection_sec', 0.0)),
            'anchor_lookup_sec': float(timing.get('anchor_lookup_sec', 0.0)),
            'compatible_anchor_search_sec': float(
                timing.get('compatible_anchor_search_sec', 0.0)
            ),
            'observation_sampling_sec': float(
                timing.get('observation_sampling_sec', 0.0)
            ),
            't0_shift_sec': float(timing.get('t0_shift_sec', 0.0)),
            'adaptive_refit_decision_sec': float(
                timing.get('adaptive_refit_decision_sec', 0.0)
            ),
            'prediction_sec': float(timing.get('prediction_sec', 0.0)),
            'assignment_sec': float(timing.get('assignment_sec', 0.0)),
            'fallback_sec': float(timing.get('fallback_sec', 0.0)),
            'diagnostics_aggregate_sec': float(
                timing.get('diagnostics_aggregate_sec', 0.0)
            ),
            'anchor_overhead_total_sec': float(
                timing.get('anchor_lookup_sec', 0.0)
                + timing.get('compatible_anchor_search_sec', 0.0)
            ),
            'fit_overhead_total_sec': float(
                timing.get('valid_mask_build_sec', 0.0)
                + timing.get('velocity_prefilter_sec', 0.0)
                + timing.get('side_segment_build_sec', 0.0)
                + timing.get('observation_sampling_sec', 0.0)
            ),
            'n_traces': int(self.n_traces),
            'n_fit_contexts': int(self.n_cache_hits + self.n_cache_misses),
            'n_fit_calls': int(self.n_fit_calls),
            'n_anchor_fit_calls': int(self.n_anchor_fit_calls),
            'n_reuse_contexts': int(self.n_reuse_contexts),
            'n_cache_hits': int(self.n_cache_hits),
            'n_cache_misses': int(self.n_cache_misses),
            'cache_hit_rate': float(self.cache_hit_rate),
            'n_source_groups': int(self.n_source_groups),
            'n_side_contexts_built': int(self.n_side_contexts_built),
            'n_side_context_cache_hits': int(self.n_side_context_cache_hits),
            'n_side_context_cache_misses': int(self.n_side_context_cache_misses),
            'n_side_context_lookup_calls': int(self.n_side_context_lookup_calls),
            'n_gap_contexts_built': int(self.n_gap_contexts_built),
            'n_gap_context_cache_hits': int(self.n_gap_context_cache_hits),
            'n_gap_context_cache_misses': int(self.n_gap_context_cache_misses),
            'n_gap_fast_path_calls': int(self.n_gap_fast_path_calls),
            'n_gap_fallback_calls': int(self.n_gap_fallback_calls),
            'n_gap_trace_in_obs': int(self.n_gap_trace_in_obs),
            'n_gap_trace_not_in_obs': int(self.n_gap_trace_not_in_obs),
            'n_side_gap_precomputed_fit_keys': int(
                self.n_side_gap_precomputed_fit_keys
            ),
            'n_precomputed_fit_key_used': int(self.n_precomputed_fit_key_used),
            'n_fit_key_built_from_indices': int(
                self.n_fit_key_built_from_indices
            ),
            'n_fit_key_built_after_sampling': int(
                self.n_fit_key_built_after_sampling
            ),
            'n_fit_key_missing_precomputed': int(
                self.n_fit_key_missing_precomputed
            ),
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
            'n_prediction_calls': int(self.n_prediction_calls),
            'n_prediction_batches': int(self.n_prediction_batches),
            'anchor_reuse_rate': float(self.anchor_reuse_rate),
            'fit_call_reduction_rate_vs_full': float(
                self.fit_call_reduction_rate_vs_full
            ),
            'ransac_fit_time_p50_sec': _percentile(self._fit_times_sec, 50.0),
            'ransac_fit_time_p90_sec': _percentile(self._fit_times_sec, 90.0),
            'ransac_fit_time_p99_sec': _percentile(self._fit_times_sec, 99.0),
            'fit_executor_enabled': int(self.fit_executor_enabled),
            'fit_executor_backend': str(self.fit_executor_backend),
            'fit_executor_max_workers': int(self.fit_executor_max_workers),
            'fit_executor_wall_sec': float(self.fit_executor_wall_sec),
            'fit_executor_tasks': int(self.fit_executor_tasks),
            'observation_sampling_enabled': int(
                self.observation_sampling_enabled
            ),
            'observation_sampling_method': str(self.observation_sampling_method),
            'max_obs_per_fit': int(self.max_obs_per_fit),
            'n_offset_bins': int(self.n_offset_bins),
            'obs_count_before_p50': obs_before_p50,
            'obs_count_before_p90': obs_before_p90,
            'obs_count_before_p99': obs_before_p99,
            'obs_count_before_downsample_p50': obs_before_p50,
            'obs_count_before_downsample_p90': obs_before_p90,
            'obs_count_before_downsample_p99': obs_before_p99,
            'obs_count_after_p50': obs_after_p50,
            'obs_count_after_p90': obs_after_p90,
            'obs_count_after_p99': obs_after_p99,
            'obs_count_after_downsample_p50': obs_after_p50,
            'obs_count_after_downsample_p90': obs_after_p90,
            'obs_count_after_downsample_p99': obs_after_p99,
            'obs_downsample_rate_p50': _percentile(
                self._fit_obs_downsample_rates,
                50.0,
            ),
            'obs_downsample_rate_p90': _percentile(
                self._fit_obs_downsample_rates,
                90.0,
            ),
            'n_downsampled_fit_contexts': int(
                sum(1 for value in self._fit_obs_downsample_rates if value > 0.0)
            ),
            'obs_count_for_fit_p50': _percentile(self._fit_obs_counts, 50.0),
            'obs_count_for_fit_p90': _percentile(self._fit_obs_counts, 90.0),
            'obs_count_for_fit_p99': _percentile(self._fit_obs_counts, 99.0),
            'side_obs_count_p50': _percentile(self._side_obs_counts, 50.0),
            'side_obs_count_p90': _percentile(self._side_obs_counts, 90.0),
            'side_obs_count_p99': _percentile(self._side_obs_counts, 99.0),
            'gap_segment_obs_count_p50': _percentile(
                self._gap_segment_obs_counts,
                50.0,
            ),
            'gap_segment_obs_count_p90': _percentile(
                self._gap_segment_obs_counts,
                90.0,
            ),
            'gap_segment_obs_count_p99': _percentile(
                self._gap_segment_obs_counts,
                99.0,
            ),
            'compatible_anchor_search_candidates_p50': _percentile(
                self._compatible_anchor_search_candidates,
                50.0,
            ),
            'compatible_anchor_search_candidates_p90': _percentile(
                self._compatible_anchor_search_candidates,
                90.0,
            ),
            'compatible_anchor_search_candidates_max': (
                0
                if not self._compatible_anchor_search_candidates
                else int(max(self._compatible_anchor_search_candidates))
            ),
            'n_no_compatible_anchor_contexts': int(
                self.n_no_compatible_anchor_contexts
            ),
            'nearest_anchor_distance_m_p50': _percentile(
                self._nearest_anchor_distance_m,
                50.0,
            ),
            'nearest_anchor_distance_m_p90': _percentile(
                self._nearest_anchor_distance_m,
                90.0,
            ),
            'nearest_anchor_distance_m_p99': _percentile(
                self._nearest_anchor_distance_m,
                99.0,
            ),
            'nearest_anchor_distance_m_max': (
                0.0
                if not self._nearest_anchor_distance_m
                else float(max(self._nearest_anchor_distance_m))
            ),
            't0_shift_ms_max': (
                0.0
                if not self._t0_shift_abs_ms
                else float(max(self._t0_shift_abs_ms))
            ),
            'reuse_resid_p50_ms_p50': _percentile(self._reuse_resid_p50_ms, 50.0),
            'reuse_resid_p50_ms_p90': _percentile(self._reuse_resid_p50_ms, 90.0),
            'reuse_resid_p90_ms_p99': _percentile(self._reuse_resid_p90_ms, 99.0),
        }
        summary['adaptive_refit_reason_counts'] = dict(
            self._adaptive_refit_reason_counts
        )
        if self._anchor_summary is not None:
            summary.update(self._anchor_summary)
        return summary

    def to_npz_fields(self) -> dict[str, np.ndarray]:
        summary = self.to_summary()
        int_keys = {
            'n_traces',
            'n_fit_contexts',
            'n_fit_calls',
            'n_anchor_fit_calls',
            'n_reuse_contexts',
            'n_cache_hits',
            'n_cache_misses',
            'n_source_groups',
            'n_side_contexts_built',
            'n_side_context_cache_hits',
            'n_side_context_cache_misses',
            'n_side_context_lookup_calls',
            'n_gap_contexts_built',
            'n_gap_context_cache_hits',
            'n_gap_context_cache_misses',
            'n_gap_fast_path_calls',
            'n_gap_fallback_calls',
            'n_gap_trace_in_obs',
            'n_gap_trace_not_in_obs',
            'n_side_gap_precomputed_fit_keys',
            'n_precomputed_fit_key_used',
            'n_fit_key_built_from_indices',
            'n_fit_key_built_after_sampling',
            'n_fit_key_missing_precomputed',
            'n_non_anchor_groups',
            'n_reused_predictions',
            'n_t0_shifted_groups',
            'n_t0_shifted_predictions',
            'n_adaptive_refit_calls',
            'n_adaptive_refit_success',
            'n_adaptive_refit_failed',
            'n_fallback_full_fit_no_compatible_anchor',
            'n_unique_fit_contexts',
            'n_prediction_calls',
            'n_prediction_batches',
            'n_downsampled_fit_contexts',
            'compatible_anchor_search_candidates_max',
            'n_no_compatible_anchor_contexts',
            'observation_sampling_enabled',
            'fit_executor_enabled',
            'fit_executor_max_workers',
            'fit_executor_tasks',
            'max_obs_per_fit',
            'n_offset_bins',
            'n_anchor_groups',
            'anchor_stride_source_groups',
        }
        out: dict[str, np.ndarray] = {}
        for key, value in summary.items():
            if isinstance(value, dict):
                continue
            if key in PHYSICS_RUNTIME_STRING_DIAGNOSTIC_KEYS:
                out[key] = np.asarray(str(value))
            else:
                out[key] = np.asarray(
                    value,
                    dtype=np.int64 if key in int_keys else np.float64,
                )
            if key in PHYSICS_RUNTIME_PREFIXED_NPZ_KEYS:
                out[f'physical_runtime_{key}'] = out[key].copy()
        return out


def runtime_summary_from_npz_fields(
    payload: dict[str, np.ndarray],
) -> dict[str, float | int | str] | None:
    if (
        'physics_total_sec' not in payload
        and 'physical_runtime_physics_total_sec' not in payload
    ):
        return None
    summary: dict[str, float | int | str] = {}
    int_keys = {
        'n_traces',
        'n_fit_contexts',
        'n_fit_calls',
        'n_anchor_fit_calls',
        'n_reuse_contexts',
        'n_cache_hits',
        'n_cache_misses',
        'n_source_groups',
        'n_side_contexts_built',
        'n_side_context_cache_hits',
        'n_side_context_cache_misses',
        'n_side_context_lookup_calls',
        'n_gap_contexts_built',
        'n_gap_context_cache_hits',
        'n_gap_context_cache_misses',
        'n_gap_fast_path_calls',
        'n_gap_fallback_calls',
        'n_gap_trace_in_obs',
        'n_gap_trace_not_in_obs',
        'n_side_gap_precomputed_fit_keys',
        'n_precomputed_fit_key_used',
        'n_fit_key_built_from_indices',
        'n_fit_key_built_after_sampling',
        'n_fit_key_missing_precomputed',
        'n_non_anchor_groups',
        'n_reused_predictions',
        'n_t0_shifted_groups',
        'n_t0_shifted_predictions',
        'n_adaptive_refit_calls',
        'n_adaptive_refit_success',
        'n_adaptive_refit_failed',
        'n_fallback_full_fit_no_compatible_anchor',
        'n_unique_fit_contexts',
        'n_prediction_calls',
        'n_prediction_batches',
        'n_downsampled_fit_contexts',
        'compatible_anchor_search_candidates_max',
        'n_no_compatible_anchor_contexts',
        'observation_sampling_enabled',
        'fit_executor_enabled',
        'fit_executor_max_workers',
        'fit_executor_tasks',
        'max_obs_per_fit',
        'n_offset_bins',
        'n_anchor_groups',
        'anchor_stride_source_groups',
    }
    for key in PHYSICS_RUNTIME_BASE_DIAGNOSTIC_KEYS:
        payload_key = key if key in payload else f'physical_runtime_{key}'
        if payload_key not in payload:
            continue
        value = np.asarray(payload[payload_key]).item()
        if key in PHYSICS_RUNTIME_STRING_DIAGNOSTIC_KEYS:
            summary[key] = str(value)
        else:
            summary[key] = int(value) if key in int_keys else float(value)
    anchor_present = [
        key
        for key in PHYSICS_RUNTIME_ANCHOR_DIAGNOSTIC_KEYS
        if key in payload or f'physical_runtime_{key}' in payload
    ]
    if anchor_present:
        missing = [
            key
            for key in PHYSICS_RUNTIME_ANCHOR_DIAGNOSTIC_KEYS
            if key not in payload and f'physical_runtime_{key}' not in payload
        ]
        if missing:
            return summary
        for key in PHYSICS_RUNTIME_ANCHOR_DIAGNOSTIC_KEYS:
            payload_key = key if key in payload else f'physical_runtime_{key}'
            value = np.asarray(payload[payload_key]).item()
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
    summary: dict[str, object],
) -> Path:
    out_path = derive_physics_runtime_summary_path(robust_npz_path)
    out_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    return out_path
