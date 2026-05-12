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
    'PHYSICS_RUNTIME_DIAGNOSTIC_KEYS',
    'PhysicalRuntimeDiagnostics',
    'derive_physics_runtime_summary_path',
    'runtime_summary_from_npz_fields',
    'write_physics_runtime_summary',
]

PHYSICS_RUNTIME_DIAGNOSTIC_KEYS = (
    'physics_total_sec',
    'physical_center_total_sec',
    'ransac_fit_total_sec',
    'n_fit_calls',
    'n_cache_hits',
    'n_cache_misses',
    'cache_hit_rate',
    'n_source_groups',
    'n_unique_fit_contexts',
    'ransac_fit_time_p50_sec',
    'ransac_fit_time_p90_sec',
    'ransac_fit_time_p99_sec',
    'obs_count_for_fit_p50',
    'obs_count_for_fit_p90',
    'obs_count_for_fit_p99',
)


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
    n_cache_hits: int = 0
    n_cache_misses: int = 0
    n_source_groups: int = 0
    n_unique_fit_contexts: int = 0
    _fit_times_sec: list[float] = field(default_factory=list, repr=False)
    _fit_obs_counts: list[int] = field(default_factory=list, repr=False)

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

    def record_cache_hit(self) -> None:
        self.n_cache_hits += 1

    def record_cache_miss(self) -> None:
        self.n_cache_misses += 1

    def set_source_groups(self, value: int) -> None:
        self.n_source_groups = int(value)

    def set_unique_fit_contexts(self, value: int) -> None:
        self.n_unique_fit_contexts = int(value)

    def to_summary(self) -> dict[str, float | int]:
        return {
            'physics_total_sec': float(self.physics_total_sec),
            'physical_center_total_sec': float(self.physical_center_total_sec),
            'ransac_fit_total_sec': float(self.ransac_fit_total_sec),
            'n_fit_calls': int(self.n_fit_calls),
            'n_cache_hits': int(self.n_cache_hits),
            'n_cache_misses': int(self.n_cache_misses),
            'cache_hit_rate': float(self.cache_hit_rate),
            'n_source_groups': int(self.n_source_groups),
            'n_unique_fit_contexts': int(self.n_unique_fit_contexts),
            'ransac_fit_time_p50_sec': _percentile(self._fit_times_sec, 50.0),
            'ransac_fit_time_p90_sec': _percentile(self._fit_times_sec, 90.0),
            'ransac_fit_time_p99_sec': _percentile(self._fit_times_sec, 99.0),
            'obs_count_for_fit_p50': _percentile(self._fit_obs_counts, 50.0),
            'obs_count_for_fit_p90': _percentile(self._fit_obs_counts, 90.0),
            'obs_count_for_fit_p99': _percentile(self._fit_obs_counts, 99.0),
        }

    def to_npz_fields(self) -> dict[str, np.ndarray]:
        summary = self.to_summary()
        int_keys = {
            'n_fit_calls',
            'n_cache_hits',
            'n_cache_misses',
            'n_source_groups',
            'n_unique_fit_contexts',
        }
        return {
            key: np.asarray(value, dtype=np.int64 if key in int_keys else np.float64)
            for key, value in summary.items()
        }


def runtime_summary_from_npz_fields(
    payload: dict[str, np.ndarray],
) -> dict[str, float | int] | None:
    if 'physics_total_sec' not in payload:
        return None
    summary: dict[str, float | int] = {}
    int_keys = {
        'n_fit_calls',
        'n_cache_hits',
        'n_cache_misses',
        'n_source_groups',
        'n_unique_fit_contexts',
    }
    for key in PHYSICS_RUNTIME_DIAGNOSTIC_KEYS:
        if key not in payload:
            return None
        value = np.asarray(payload[key]).item()
        summary[key] = int(value) if key in int_keys else float(value)
    return summary


def derive_physics_runtime_summary_path(robust_npz_path: str | Path) -> Path:
    path = Path(robust_npz_path).expanduser().resolve()
    suffix = '.robust.npz'
    tag = path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem
    return path.with_name(f'{tag}.physics_runtime_summary.json')


def write_physics_runtime_summary(
    robust_npz_path: str | Path,
    summary: dict[str, float | int],
) -> Path:
    out_path = derive_physics_runtime_summary_path(robust_npz_path)
    out_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    return out_path
