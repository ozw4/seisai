from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from seisai_engine.pipelines.fbpick.common import (
    ROBUST_SOURCE_COARSE_OBSERVED,
    build_lineage_payload,
    load_coarse_npz,
    save_robust_npz,
)

from .confidence import ConfidenceResult, compute_confidence_terms
from .config import load_physics_lite_config, physics_lite_config_to_dict
from .feasible import compute_physical_band_result
from .merge import MergeResult, apply_keep_reject_fill
from .physical_center import (
    build_geometry_two_piece_physical_center,
    preflight_geometry_two_piece_fallback,
)
from .physical_center_fallback_policy import apply_physics_fallback_policy
from .pick_table import normalize_coarse_pick_table
from .progress import build_progress_reporter
from .runtime_diagnostics import (
    PhysicalRuntimeDiagnostics,
    derive_physics_runtime_summary_path,
    runtime_summary_from_npz_fields,
    write_physics_runtime_summary,
)
from .trend import (
    PartialTrendFallbackResult,
    TrendResult,
    build_partial_trend_fallback,
    build_trend_result,
)
from .window_constraint import (
    evaluate_fine_window_constraint,
    resolve_physical_prefilter_offsets_m,
)

__all__ = [
    'build_robust_payload_from_coarse',
    'derive_physics_runtime_summary_path',
    'derive_robust_npz_path',
    'run_physics_lite',
]


def derive_robust_npz_path(coarse_npz_path: str | Path) -> Path:
    path = Path(coarse_npz_path).expanduser().resolve()
    if not path.name.endswith('.coarse.npz'):
        msg = f'expected *.coarse.npz input, got {path.name}'
        raise ValueError(msg)
    stem = path.name[: -len('.coarse.npz')]
    return path.with_name(f'{stem}.robust.npz')


def _status_counts_line(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        return ''
    unique, counts = np.unique(arr, return_counts=True)
    return ','.join(
        f'{int(status)}:{int(count)}'
        for status, count in zip(unique.tolist(), counts.tolist(), strict=True)
    )


def _payload_scalar(payload: Mapping[str, np.ndarray], key: str) -> object | None:
    if key not in payload:
        return None
    return np.asarray(payload[key]).item()


def _add_trend_runtime_summary_fields(
    summary: dict[str, object],
    payload: Mapping[str, np.ndarray],
) -> None:
    materialized = _payload_scalar(
        payload,
        'physical_runtime_trend_result_materialized',
    )
    computed = _payload_scalar(payload, 'physical_runtime_trend_result_computed')
    elapsed = _payload_scalar(
        payload,
        'physical_runtime_trend_result_elapsed_sec',
    )
    started_before = _payload_scalar(
        payload,
        'physical_runtime_physical_center_started_before_trend_result',
    )
    mode = _payload_scalar(payload, 'physical_runtime_trend_result_mode')
    reason = _payload_scalar(payload, 'physical_runtime_trend_result_reason')
    legacy_output = _payload_scalar(payload, 'physical_runtime_legacy_trend_output')
    fallback_existing_trend_mode = _payload_scalar(
        payload,
        'physical_runtime_fallback_existing_trend_mode',
    )
    partial_enabled = _payload_scalar(
        payload,
        'physical_runtime_partial_trend_fallback_enabled',
    )
    partial_n_targets = _payload_scalar(
        payload,
        'physical_runtime_partial_trend_fallback_n_targets',
    )
    partial_n_valid = _payload_scalar(
        payload,
        'physical_runtime_partial_trend_fallback_n_valid',
    )
    partial_n_robust = _payload_scalar(
        payload,
        'physical_runtime_partial_trend_fallback_n_robust',
    )
    partial_elapsed = _payload_scalar(
        payload,
        'physical_runtime_partial_trend_fallback_elapsed_sec',
    )
    partial_too_many = _payload_scalar(
        payload,
        'physical_runtime_partial_trend_fallback_too_many',
    )

    if materialized is not None:
        summary['trend_result_materialized'] = bool(int(materialized))
    if computed is not None:
        summary['trend_result_computed'] = bool(int(computed))
    if elapsed is not None:
        summary['trend_result_elapsed_sec'] = float(elapsed)
    if started_before is not None:
        summary['physical_center_started_before_trend_result'] = bool(
            int(started_before)
        )
    if mode is not None:
        summary['trend_result_mode'] = str(mode)
    if reason is not None:
        reason_str = str(reason)
        summary['trend_result_reason'] = reason_str if reason_str else None
    if legacy_output is not None:
        summary['legacy_trend_output'] = str(legacy_output)
    if fallback_existing_trend_mode is not None:
        summary['fallback_existing_trend_mode'] = str(
            fallback_existing_trend_mode
        )
    if partial_enabled is not None:
        summary['partial_trend_fallback_enabled'] = bool(int(partial_enabled))
    if partial_n_targets is not None:
        summary['partial_trend_fallback_n_targets'] = int(partial_n_targets)
    if partial_n_valid is not None:
        summary['partial_trend_fallback_n_valid'] = int(partial_n_valid)
    if partial_n_robust is not None:
        summary['partial_trend_fallback_n_robust'] = int(partial_n_robust)
    if partial_elapsed is not None:
        summary['partial_trend_fallback_elapsed_sec'] = float(partial_elapsed)
    if partial_too_many is not None:
        summary['partial_trend_fallback_too_many'] = bool(int(partial_too_many))


def _build_unmaterialized_trend(table) -> TrendResult:
    n = int(table.n_traces)
    return TrendResult(
        seed_mask=np.zeros((n,), dtype=np.bool_),
        seed_threshold=np.float32(np.nan),
        local_center_sec=np.full((n,), np.nan, dtype=np.float32),
        local_center_valid=np.zeros((n,), dtype=np.bool_),
        local_discard_mask=np.zeros((n,), dtype=np.bool_),
        global_center_sec=np.full((n,), np.nan, dtype=np.float32),
        trend_center_sec=np.full((n,), np.nan, dtype=np.float32),
        trend_center_i=np.full((n,), -1, dtype=np.int32),
        filled_mask=np.zeros((n,), dtype=np.bool_),
    )


class LazyTrendResultProvider:
    def __init__(
        self,
        *,
        mode: str,
        build_fn: Callable[[], TrendResult],
        build_partial_fn: Callable[[np.ndarray], PartialTrendFallbackResult | None]
        | None = None,
        fallback_existing_trend_mode: str = 'full',
        partial_cfg: Any | None = None,
        n_traces: int | None = None,
        progress: Any,
        progress_context: Mapping[str, object],
    ) -> None:
        self.mode = str(mode)
        self._build_fn = build_fn
        self._build_partial_fn = build_partial_fn
        self.fallback_existing_trend_mode = str(fallback_existing_trend_mode)
        self._partial_cfg = partial_cfg
        self._n_traces = None if n_traces is None else int(n_traces)
        self._progress = progress
        self._progress_context = dict(progress_context)
        self._trend_result: TrendResult | None = None
        self._partial_cache: dict[int, tuple[int, np.float32, bool]] = {}
        self.computed = False
        self.reason: str | None = None
        self.elapsed_sec = 0.0
        self.partial_elapsed_sec = 0.0
        self.partial_n_targets = 0
        self.partial_n_valid = 0
        self.partial_n_robust = 0
        self.partial_too_many = False

    @property
    def trend_result(self) -> TrendResult | None:
        return self._trend_result

    def get(self, reason: str) -> TrendResult:
        if self.mode == 'disabled':
            msg = 'trend_result is disabled but was requested'
            raise RuntimeError(msg)
        if self._trend_result is None:
            self.reason = str(reason)
            lazy = self.mode != 'eager'
            self._progress.emit(
                'physics.stage_start',
                **self._progress_context,
                stage='trend_result',
                lazy=lazy,
                reason=self.reason,
            )
            stage_start = perf_counter()
            self._trend_result = self._build_fn()
            self.elapsed_sec = perf_counter() - stage_start
            self.computed = True
            self._progress.emit(
                'physics.stage_done',
                **self._progress_context,
                stage='trend_result',
                lazy=lazy,
                reason=self.reason,
                elapsed=self.elapsed_sec,
            )
        return self._trend_result

    def _partial_too_many(self, n_targets: int) -> bool:
        partial_cfg = self._partial_cfg
        if partial_cfg is None:
            return False
        if int(n_targets) > int(partial_cfg.max_traces):
            return True
        if self._n_traces is None or self._n_traces <= 0:
            return False
        return (
            float(n_targets) / float(self._n_traces)
            > float(partial_cfg.max_fraction)
        )

    @staticmethod
    def _partial_result_from_cache(
        indices: np.ndarray,
        cache: Mapping[int, tuple[int, np.float32, bool]],
    ) -> PartialTrendFallbackResult:
        requested = np.asarray(indices, dtype=np.int64).reshape(-1)
        center_i = np.full((requested.size,), -1, dtype=np.int32)
        center_t_sec = np.full((requested.size,), np.nan, dtype=np.float32)
        valid = np.zeros((requested.size,), dtype=np.bool_)
        for pos, trace_idx in enumerate(requested.tolist()):
            cached_i, cached_t, cached_valid = cache[int(trace_idx)]
            if bool(cached_valid):
                center_i[pos] = np.int32(cached_i)
                center_t_sec[pos] = np.float32(cached_t)
                valid[pos] = True
        fallback_to_robust = ~valid
        return PartialTrendFallbackResult(
            indices=requested.copy(),
            center_i=center_i,
            center_t_sec=center_t_sec,
            valid_mask=valid,
            fallback_to_robust_mask=fallback_to_robust.astype(
                np.bool_,
                copy=False,
            ),
            elapsed_sec=0.0,
            n_targets=int(requested.size),
            n_valid=int(np.count_nonzero(valid)),
            n_robust=int(np.count_nonzero(fallback_to_robust)),
        )

    def _cache_robust_partial(self, indices: np.ndarray) -> None:
        for trace_idx in np.asarray(indices, dtype=np.int64).reshape(-1).tolist():
            self._partial_cache[int(trace_idx)] = (-1, np.float32(np.nan), False)

    def _record_partial_robust_targets(self, n_targets: int) -> None:
        self.partial_n_targets += int(n_targets)
        self.partial_n_robust += int(n_targets)

    def record_partial_too_many_robust_fallback(self, n_targets: int) -> None:
        self.partial_too_many = True
        self._record_partial_robust_targets(n_targets)

    def get_partial(
        self,
        indices: np.ndarray,
        *,
        reason: str,
    ) -> PartialTrendFallbackResult | None:
        requested = np.asarray(indices, dtype=np.int64).reshape(-1)
        if requested.size == 0:
            return self._partial_result_from_cache(requested, self._partial_cache)

        mode = str(self.fallback_existing_trend_mode)
        if mode == 'full':
            return None
        missing = np.asarray(
            [
                int(trace_idx)
                for trace_idx in np.unique(requested)
                if int(trace_idx) not in self._partial_cache
            ],
            dtype=np.int64,
        )
        if mode == 'robust':
            self._cache_robust_partial(missing)
            return self._partial_result_from_cache(requested, self._partial_cache)
        if mode != 'partial':
            return None

        if missing.size > 0:
            partial_cfg = self._partial_cfg
            if partial_cfg is not None and not bool(partial_cfg.enabled):
                return None
            n_effective_targets = len(self._partial_cache) + int(missing.size)
            if self._partial_too_many(n_effective_targets):
                fallback = (
                    'robust'
                    if partial_cfg is None
                    else str(partial_cfg.fallback_if_too_many)
                )
                if fallback == 'full':
                    self.partial_too_many = True
                    return None
                if fallback == 'error':
                    msg = (
                        'partial trend fallback target count exceeds '
                        f'configured limits: n_targets={n_effective_targets}'
                    )
                    raise RuntimeError(msg)
                self._cache_robust_partial(missing)
                self.record_partial_too_many_robust_fallback(int(missing.size))
                return self._partial_result_from_cache(
                    requested,
                    self._partial_cache,
                )

            if self._build_partial_fn is None:
                return None
            emit_progress = (
                True if partial_cfg is None else bool(partial_cfg.emit_progress)
            )
            if emit_progress:
                self._progress.emit(
                    'physics.stage_start',
                    **self._progress_context,
                    stage='partial_trend_fallback',
                    lazy=True,
                    reason=str(reason),
                    n_targets=int(missing.size),
                )
            stage_start = perf_counter()
            result = self._build_partial_fn(missing)
            elapsed = perf_counter() - stage_start
            if result is None:
                if emit_progress:
                    self._progress.emit(
                        'physics.stage_done',
                        **self._progress_context,
                        stage='partial_trend_fallback',
                        lazy=True,
                        reason=str(reason),
                        elapsed=elapsed,
                        n_targets=int(missing.size),
                        fallback='full',
                    )
                return None
            self.partial_elapsed_sec += float(elapsed)
            self.partial_n_targets += int(result.n_targets)
            self.partial_n_valid += int(result.n_valid)
            self.partial_n_robust += int(result.n_robust)
            for pos, trace_idx in enumerate(
                np.asarray(result.indices, dtype=np.int64).reshape(-1).tolist()
            ):
                valid = bool(np.asarray(result.valid_mask, dtype=np.bool_)[pos])
                center_i = int(np.asarray(result.center_i, dtype=np.int64)[pos])
                center_t = np.float32(
                    np.asarray(result.center_t_sec, dtype=np.float32)[pos]
                )
                self._partial_cache[int(trace_idx)] = (center_i, center_t, valid)
            if emit_progress:
                self._progress.emit(
                    'physics.stage_done',
                    **self._progress_context,
                    stage='partial_trend_fallback',
                    lazy=True,
                    reason=str(reason),
                    elapsed=elapsed,
                    n_targets=int(result.n_targets),
                    n_valid=int(result.n_valid),
                    n_robust=int(result.n_robust),
                )
        return self._partial_result_from_cache(requested, self._partial_cache)


def _partial_trend_fallback_runtime_fields(
    *,
    typed_cfg,
    trend_provider: LazyTrendResultProvider,
) -> dict[str, np.ndarray]:
    partial_cfg = typed_cfg.physical_runtime.partial_trend_fallback
    return {
        'physical_runtime_fallback_existing_trend_mode': np.asarray(
            str(typed_cfg.physical_runtime.fallback_existing_trend_mode)
        ),
        'physical_runtime_partial_trend_fallback_enabled': np.asarray(
            int(bool(partial_cfg.enabled)),
            dtype=np.int64,
        ),
        'physical_runtime_partial_trend_fallback_max_fraction': np.asarray(
            float(partial_cfg.max_fraction),
            dtype=np.float64,
        ),
        'physical_runtime_partial_trend_fallback_max_traces': np.asarray(
            int(partial_cfg.max_traces),
            dtype=np.int64,
        ),
        'physical_runtime_partial_trend_fallback_fallback_if_too_many': (
            np.asarray(str(partial_cfg.fallback_if_too_many))
        ),
        'physical_runtime_partial_trend_fallback_n_targets': np.asarray(
            int(trend_provider.partial_n_targets),
            dtype=np.int64,
        ),
        'physical_runtime_partial_trend_fallback_n_valid': np.asarray(
            int(trend_provider.partial_n_valid),
            dtype=np.int64,
        ),
        'physical_runtime_partial_trend_fallback_n_robust': np.asarray(
            int(trend_provider.partial_n_robust),
            dtype=np.int64,
        ),
        'physical_runtime_partial_trend_fallback_elapsed_sec': np.asarray(
            float(trend_provider.partial_elapsed_sec),
            dtype=np.float64,
        ),
        'physical_runtime_partial_trend_fallback_too_many': np.asarray(
            int(bool(trend_provider.partial_too_many)),
            dtype=np.int64,
        ),
    }


def _build_coarse_observed_confidence(table) -> ConfidenceResult:
    n = int(table.n_traces)
    conf_prob1 = np.clip(
        np.asarray(table.coarse_pmax, dtype=np.float32),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    conf_trend1 = np.ones((n,), dtype=np.float32)
    conf_rs1 = np.ones((n,), dtype=np.float32)
    return ConfidenceResult(
        conf_prob1=conf_prob1,
        conf_trend1=conf_trend1,
        conf_rs1=conf_rs1,
        total_score=conf_prob1.copy(),
    )


def _build_coarse_observed_merge(table, confidence: ConfidenceResult) -> MergeResult:
    n = int(table.n_traces)
    return MergeResult(
        keep_mask=np.ones((n,), dtype=np.bool_),
        reject_mask=np.zeros((n,), dtype=np.bool_),
        score_threshold=np.float32(0.0),
        robust_pick_i=np.asarray(table.coarse_pick_i, dtype=np.int32).copy(),
        robust_pick_t_sec=np.asarray(table.coarse_pick_t_sec, dtype=np.float32).copy(),
        robust_conf=np.asarray(confidence.total_score, dtype=np.float32).copy(),
        robust_source=np.full(
            (n,),
            np.uint8(ROBUST_SOURCE_COARSE_OBSERVED),
            dtype=np.uint8,
        ),
        used_theoretical_mask=np.zeros((n,), dtype=np.bool_),
        reason_mask=np.zeros((n,), dtype=np.uint8),
    )


def _physical_center_first_enabled(typed_cfg) -> bool:
    return bool(typed_cfg.physical_trend.enabled) and str(
        typed_cfg.physical_runtime.trend_result_mode
    ) in {'lazy', 'disabled'}


def _should_use_lazy_robust_fallback_path(
    coarse_npz: Mapping[str, np.ndarray],
    *,
    table,
    typed_cfg,
    reporter: Any,
    progress_fields: Mapping[str, object],
) -> bool:
    runtime = typed_cfg.physical_runtime
    if str(runtime.trend_result_mode) not in {'lazy', 'disabled'}:
        return False
    if not bool(typed_cfg.physical_trend.enabled):
        return False

    reporter.emit(
        'physics.stage_start',
        **progress_fields,
        stage='geometry_preflight',
    )
    stage_start = perf_counter()
    preflight = preflight_geometry_two_piece_fallback(
        coarse_npz=coarse_npz,
        table=table,
        cfg=typed_cfg,
    )
    use_lazy_robust = (
        preflight.status is not None and str(preflight.fallback_mode) == 'robust'
    )
    reporter.emit(
        'physics.stage_done',
        **progress_fields,
        stage='geometry_preflight',
        elapsed=perf_counter() - stage_start,
        geometry_loaded=preflight.geometry_loaded,
        groups=preflight.groups,
        fallback_detected=preflight.status is not None,
        fallback_reason=preflight.reason,
        fallback=preflight.fallback_mode,
        lazy_robust_fallback=use_lazy_robust,
    )
    return use_lazy_robust


def _raise_if_trend_materialization_disabled(typed_cfg) -> None:
    if str(typed_cfg.physical_runtime.trend_result_mode) == 'disabled':
        msg = (
            "physical_runtime.trend_result_mode='disabled' cannot materialize "
            'trend_result for this input'
        )
        raise ValueError(msg)


def build_robust_payload_from_coarse(
    coarse_npz: Mapping[str, np.ndarray],
    *,
    cfg: dict[str, Any] | None,
    source_model_id: str | None = None,
    iter_id: int | str | None = '',
    repo_root: Path | None = None,
    runtime_diagnostics: PhysicalRuntimeDiagnostics | None = None,
    progress: Any | None = None,
    progress_context: Mapping[str, object] | None = None,
) -> dict[str, np.ndarray]:
    typed_cfg = load_physics_lite_config(cfg)
    canonical_cfg = physics_lite_config_to_dict(typed_cfg)
    reporter = (
        progress
        if progress is not None
        else build_progress_reporter(typed_cfg.physical_runtime.progress)
    )
    progress_fields = dict(progress_context or {})
    if runtime_diagnostics is None and bool(
        typed_cfg.physical_runtime.diagnostics_enabled
    ):
        runtime_diagnostics = PhysicalRuntimeDiagnostics(
            detailed_timing=bool(
                typed_cfg.physical_runtime.diagnostics.detailed_timing
            )
        )

    if runtime_diagnostics is None:
        reporter.emit('physics.stage_start', **progress_fields, stage='normalize_table')
        stage_start = perf_counter()
        table = normalize_coarse_pick_table(coarse_npz)
        reporter.emit(
            'physics.stage_done',
            **progress_fields,
            stage='normalize_table',
            elapsed=perf_counter() - stage_start,
            n_traces=int(table.n_traces),
        )
        reporter.emit('physics.stage_start', **progress_fields, stage='physical_band')
        stage_start = perf_counter()
        feasible = compute_physical_band_result(table, typed_cfg.physical_band)
        reporter.emit(
            'physics.stage_done',
            **progress_fields,
            stage='physical_band',
            elapsed=perf_counter() - stage_start,
        )
        trend_provider = LazyTrendResultProvider(
            mode=str(typed_cfg.physical_runtime.trend_result_mode),
            build_fn=lambda: build_trend_result(table, feasible, typed_cfg),
            build_partial_fn=lambda indices: build_partial_trend_fallback(
                table,
                feasible,
                typed_cfg,
                indices,
            ),
            fallback_existing_trend_mode=str(
                typed_cfg.physical_runtime.fallback_existing_trend_mode
            ),
            partial_cfg=typed_cfg.physical_runtime.partial_trend_fallback,
            n_traces=int(table.n_traces),
            progress=reporter,
            progress_context=progress_fields,
        )
        physical_center_first = _physical_center_first_enabled(typed_cfg)
        if physical_center_first:
            _should_use_lazy_robust_fallback_path(
                coarse_npz,
                table=table,
                typed_cfg=typed_cfg,
                reporter=reporter,
                progress_fields=progress_fields,
            )
            trend = _build_unmaterialized_trend(table)
            confidence = _build_coarse_observed_confidence(table)
            merged = _build_coarse_observed_merge(table, confidence)
        else:
            _raise_if_trend_materialization_disabled(typed_cfg)
            trend = trend_provider.get(reason='eager')
            reporter.emit('physics.stage_start', **progress_fields, stage='confidence')
            stage_start = perf_counter()
            confidence = compute_confidence_terms(table, feasible, trend, typed_cfg)
            reporter.emit(
                'physics.stage_done',
                **progress_fields,
                stage='confidence',
                elapsed=perf_counter() - stage_start,
            )
            reporter.emit('physics.stage_start', **progress_fields, stage='merge')
            stage_start = perf_counter()
            merged = apply_keep_reject_fill(
                table,
                feasible,
                trend,
                confidence,
                typed_cfg,
            )
            reporter.emit(
                'physics.stage_done',
                **progress_fields,
                stage='merge',
                elapsed=perf_counter() - stage_start,
            )
        reporter.emit('physics.stage_start', **progress_fields, stage='physical_center')
        stage_start = perf_counter()
        physical = build_geometry_two_piece_physical_center(
            coarse_npz=coarse_npz,
            table=table,
            feasible=feasible,
            trend=trend,
            merged=merged,
            cfg=typed_cfg,
            trend_provider=trend_provider if physical_center_first else None,
            progress=reporter,
            progress_context=progress_fields,
        )
        reporter.emit(
            'physics.stage_done',
            **progress_fields,
            stage='physical_center',
            elapsed=perf_counter() - stage_start,
        )
        if (
            physical_center_first
            and str(typed_cfg.physical_runtime.legacy_trend_output) == 'always'
        ):
            trend = trend_provider.get(reason='legacy_trend_output')
        elif physical_center_first and trend_provider.trend_result is not None:
            trend = trend_provider.trend_result
        trend_materialized = bool(trend_provider.computed)
    else:
        with runtime_diagnostics.time_physics():
            reporter.emit(
                'physics.stage_start',
                **progress_fields,
                stage='normalize_table',
            )
            stage_start = perf_counter()
            with runtime_diagnostics.time_block('normalize_table_sec'):
                table = normalize_coarse_pick_table(coarse_npz)
            runtime_diagnostics.set_traces(int(table.n_traces))
            reporter.emit(
                'physics.stage_done',
                **progress_fields,
                stage='normalize_table',
                elapsed=perf_counter() - stage_start,
                n_traces=int(table.n_traces),
            )
            reporter.emit(
                'physics.stage_start',
                **progress_fields,
                stage='physical_band',
            )
            stage_start = perf_counter()
            with runtime_diagnostics.time_block('physical_band_sec'):
                feasible = compute_physical_band_result(table, typed_cfg.physical_band)
            reporter.emit(
                'physics.stage_done',
                **progress_fields,
                stage='physical_band',
                elapsed=perf_counter() - stage_start,
            )
            def _build_trend_result_for_provider() -> TrendResult:
                with runtime_diagnostics.time_block('trend_result_sec'):
                    return build_trend_result(table, feasible, typed_cfg)

            def _build_partial_trend_fallback_for_provider(
                indices: np.ndarray,
            ) -> PartialTrendFallbackResult | None:
                with runtime_diagnostics.time_block('partial_trend_fallback_sec'):
                    return build_partial_trend_fallback(
                        table,
                        feasible,
                        typed_cfg,
                        indices,
                    )

            trend_provider = LazyTrendResultProvider(
                mode=str(typed_cfg.physical_runtime.trend_result_mode),
                build_fn=_build_trend_result_for_provider,
                build_partial_fn=_build_partial_trend_fallback_for_provider,
                fallback_existing_trend_mode=str(
                    typed_cfg.physical_runtime.fallback_existing_trend_mode
                ),
                partial_cfg=typed_cfg.physical_runtime.partial_trend_fallback,
                n_traces=int(table.n_traces),
                progress=reporter,
                progress_context=progress_fields,
            )
            physical_center_first = _physical_center_first_enabled(typed_cfg)
            if physical_center_first:
                _should_use_lazy_robust_fallback_path(
                    coarse_npz,
                    table=table,
                    typed_cfg=typed_cfg,
                    reporter=reporter,
                    progress_fields=progress_fields,
                )
                trend = _build_unmaterialized_trend(table)
                confidence = _build_coarse_observed_confidence(table)
                merged = _build_coarse_observed_merge(table, confidence)
            else:
                _raise_if_trend_materialization_disabled(typed_cfg)
                trend = trend_provider.get(reason='eager')
                reporter.emit(
                    'physics.stage_start',
                    **progress_fields,
                    stage='confidence',
                )
                stage_start = perf_counter()
                with runtime_diagnostics.time_block('confidence_sec'):
                    confidence = compute_confidence_terms(
                        table,
                        feasible,
                        trend,
                        typed_cfg,
                    )
                reporter.emit(
                    'physics.stage_done',
                    **progress_fields,
                    stage='confidence',
                    elapsed=perf_counter() - stage_start,
                )
                reporter.emit('physics.stage_start', **progress_fields, stage='merge')
                stage_start = perf_counter()
                with runtime_diagnostics.time_block('merge_sec'):
                    merged = apply_keep_reject_fill(
                        table,
                        feasible,
                        trend,
                        confidence,
                        typed_cfg,
                    )
                reporter.emit(
                    'physics.stage_done',
                    **progress_fields,
                    stage='merge',
                    elapsed=perf_counter() - stage_start,
                )
            reporter.emit(
                'physics.stage_start',
                **progress_fields,
                stage='physical_center',
            )
            stage_start = perf_counter()
            with runtime_diagnostics.time_physical_center():
                physical = build_geometry_two_piece_physical_center(
                    coarse_npz=coarse_npz,
                    table=table,
                    feasible=feasible,
                    trend=trend,
                    merged=merged,
                    cfg=typed_cfg,
                    trend_provider=trend_provider if physical_center_first else None,
                    runtime_diagnostics=runtime_diagnostics,
                    progress=reporter,
                    progress_context=progress_fields,
                )
            reporter.emit(
                'physics.stage_done',
                **progress_fields,
                stage='physical_center',
                elapsed=perf_counter() - stage_start,
            )
            if (
                physical_center_first
                and str(typed_cfg.physical_runtime.legacy_trend_output) == 'always'
            ):
                trend = trend_provider.get(reason='legacy_trend_output')
            elif physical_center_first and trend_provider.trend_result is not None:
                trend = trend_provider.trend_result
            trend_materialized = bool(trend_provider.computed)

    legacy_trend_output = str(typed_cfg.physical_runtime.legacy_trend_output)
    include_trend_output = legacy_trend_output != 'omit' or bool(trend_materialized)
    window_offsets_m = resolve_physical_prefilter_offsets_m(
        coarse_npz=coarse_npz,
        table=table,
        cfg=typed_cfg,
    )
    window_constraint = evaluate_fine_window_constraint(
        offsets_m=window_offsets_m,
        dt_sec=float(table.dt_scalar_sec),
        n_samples_orig=int(table.n_samples_orig),
        fine_center_i=np.asarray(physical.fine_center_i, dtype=np.int32),
        physical_prefilter=typed_cfg.physical_prefilter,
        constraint=typed_cfg.physical_runtime.fine_window_constraint,
        physical_model_status=np.asarray(
            physical.physical_model_status,
            dtype=np.uint8,
        ),
        physical_runtime_fit_source=np.asarray(
            physical.physical_runtime_fit_source,
            dtype=np.uint8,
        ),
    )
    physical, window_constraint, fallback_policy_diagnostics = (
        apply_physics_fallback_policy(
            physical=physical,
            initial_window_constraint=window_constraint,
            coarse_npz=coarse_npz,
            table=table,
            cfg=typed_cfg,
        )
    )
    payload = {
        'dt_sec': np.asarray(table.dt_scalar_sec, dtype=np.float32),
        'n_samples_orig': np.asarray(table.n_samples_orig, dtype=np.int32),
        'n_traces': np.asarray(table.n_traces, dtype=np.int32),
        'ffid_values': np.asarray(table.ffid, dtype=np.int32),
        'chno_values': np.asarray(table.chno, dtype=np.int32),
        'offsets_m': np.asarray(table.offset_m, dtype=np.float32),
        'trace_indices': np.asarray(table.trace_id, dtype=np.int64),
        'robust_pick_i': np.asarray(merged.robust_pick_i, dtype=np.int32),
        'robust_pick_t_sec': np.asarray(merged.robust_pick_t_sec, dtype=np.float32),
        'robust_conf': np.asarray(merged.robust_conf, dtype=np.float32),
        'robust_source': np.asarray(merged.robust_source, dtype=np.uint8),
        'used_theoretical_mask': np.asarray(
            merged.used_theoretical_mask,
            dtype=np.bool_,
        ),
        'reason_mask': np.asarray(merged.reason_mask, dtype=np.uint8),
        'conf_prob1': np.asarray(confidence.conf_prob1, dtype=np.float32),
        'conf_trend1': np.asarray(confidence.conf_trend1, dtype=np.float32),
        'conf_rs1': np.asarray(confidence.conf_rs1, dtype=np.float32),
        'physical_center_i': np.asarray(physical.physical_center_i, dtype=np.int32),
        'physical_center_t_sec': np.asarray(
            physical.physical_center_t_sec,
            dtype=np.float32,
        ),
        'fine_center_i': np.asarray(physical.fine_center_i, dtype=np.int32),
        'fine_center_t_sec': np.asarray(physical.fine_center_t_sec, dtype=np.float32),
        'physical_model_status': np.asarray(
            physical.physical_model_status,
            dtype=np.uint8,
        ),
        'physical_model_failure_reason': np.asarray(
            physical.physical_model_failure_reason,
            dtype=np.uint8,
        ),
        'physical_offset_source': np.asarray(
            physical.physical_offset_source,
            dtype=np.uint8,
        ),
        'physical_model_break_offset_m': np.asarray(
            physical.physical_model_break_offset_m,
            dtype=np.float32,
        ),
        'physical_model_slope_near_s_per_m': np.asarray(
            physical.physical_model_slope_near_s_per_m,
            dtype=np.float32,
        ),
        'physical_model_slope_far_s_per_m': np.asarray(
            physical.physical_model_slope_far_s_per_m,
            dtype=np.float32,
        ),
        'physical_model_velocity_near_m_s': np.asarray(
            physical.physical_model_velocity_near_m_s,
            dtype=np.float32,
        ),
        'physical_model_velocity_far_m_s': np.asarray(
            physical.physical_model_velocity_far_m_s,
            dtype=np.float32,
        ),
        'physical_model_neighbor_count': np.asarray(
            physical.physical_model_neighbor_count,
            dtype=np.int32,
        ),
        'physical_prefilter_valid_count': np.asarray(
            physical.physical_prefilter_valid_count,
            dtype=np.int32,
        ),
        'physical_model_segment_id': np.asarray(
            physical.physical_model_segment_id,
            dtype=np.int32,
        ),
        'physical_model_side': np.asarray(physical.physical_model_side, dtype=np.int8),
        'physical_model_resid_p50_ms': np.asarray(
            physical.physical_model_resid_p50_ms,
            dtype=np.float32,
        ),
        'physical_model_resid_p90_ms': np.asarray(
            physical.physical_model_resid_p90_ms,
            dtype=np.float32,
        ),
        'physical_runtime_t0_shift_ms': np.asarray(
            physical.physical_runtime_t0_shift_ms,
            dtype=np.float32,
        ),
        'physical_runtime_reuse_resid_p50_ms': np.asarray(
            physical.physical_runtime_reuse_resid_p50_ms,
            dtype=np.float32,
        ),
        'physical_runtime_reuse_resid_p90_ms': np.asarray(
            physical.physical_runtime_reuse_resid_p90_ms,
            dtype=np.float32,
        ),
        'physical_runtime_reuse_valid_count': np.asarray(
            physical.physical_runtime_reuse_valid_count,
            dtype=np.int32,
        ),
        'physical_runtime_refit_mask': np.asarray(
            physical.physical_runtime_refit_mask,
            dtype=np.bool_,
        ),
        'physical_runtime_fit_source': np.asarray(
            physical.physical_runtime_fit_source,
            dtype=np.uint8,
        ),
        'physical_fit_model_type': np.asarray(physical.physical_fit_model_type),
        'physical_fit_selected_model': np.asarray(
            physical.physical_fit_selected_model
        ),
        'physical_fit_relative_improvement': np.asarray(
            physical.physical_fit_relative_improvement,
            dtype=np.float32,
        ),
        'physical_fit_single_line_cost': np.asarray(
            physical.physical_fit_single_line_cost,
            dtype=np.float32,
        ),
        'physical_fit_two_piece_cost': np.asarray(
            physical.physical_fit_two_piece_cost,
            dtype=np.float32,
        ),
        'physical_fit_single_line_slope': np.asarray(
            physical.physical_fit_single_line_slope,
            dtype=np.float32,
        ),
        'physical_fit_single_line_t0_sec': np.asarray(
            physical.physical_fit_single_line_t0_sec,
            dtype=np.float32,
        ),
        'physical_fit_two_piece_slope_near': np.asarray(
            physical.physical_fit_two_piece_slope_near,
            dtype=np.float32,
        ),
        'physical_fit_two_piece_slope_far': np.asarray(
            physical.physical_fit_two_piece_slope_far,
            dtype=np.float32,
        ),
        'physical_fit_two_piece_break_offset_m': np.asarray(
            physical.physical_fit_two_piece_break_offset_m,
            dtype=np.float32,
        ),
        'fine_center_valid_mask': np.asarray(
            window_constraint.fine_center_valid_mask,
            dtype=np.bool_,
        ),
        'fine_window_valid_mask': np.asarray(
            window_constraint.fine_window_valid_mask,
            dtype=np.bool_,
        ),
        'fine_window_physical_lo_i': np.asarray(
            window_constraint.fine_window_physical_lo_i,
            dtype=np.int32,
        ),
        'fine_window_physical_hi_i': np.asarray(
            window_constraint.fine_window_physical_hi_i,
            dtype=np.int32,
        ),
        'fine_window_reject_reason': np.asarray(
            window_constraint.fine_window_reject_reason,
            dtype=np.uint8,
        ),
        'physical_center_source': np.asarray(
            fallback_policy_diagnostics['physical_center_source']
        ),
        'physical_fallback_source': np.asarray(
            fallback_policy_diagnostics['physical_fallback_source']
        ),
        'physical_neighbor_source_index': np.asarray(
            fallback_policy_diagnostics['physical_neighbor_source_index'],
            dtype=np.int32,
        ),
        'physical_neighbor_source_distance': np.asarray(
            fallback_policy_diagnostics['physical_neighbor_source_distance'],
            dtype=np.float32,
        ),
        'coarse_in_band_fallback_mask': np.asarray(
            fallback_policy_diagnostics['coarse_in_band_fallback_mask'],
            dtype=np.bool_,
        ),
        'reject_physics_mask': np.asarray(
            fallback_policy_diagnostics['reject_physics_mask'],
            dtype=np.bool_,
        ),
        'reject_physics_reason': np.asarray(
            fallback_policy_diagnostics['reject_physics_reason']
        ),
        'physical_runtime_trend_result_materialized': np.asarray(
            int(bool(trend_materialized)),
            dtype=np.int64,
        ),
        'physical_runtime_trend_result_computed': np.asarray(
            int(bool(trend_materialized)),
            dtype=np.int64,
        ),
        'physical_runtime_trend_result_elapsed_sec': np.asarray(
            float(trend_provider.elapsed_sec),
            dtype=np.float64,
        ),
        'physical_runtime_physical_center_started_before_trend_result': np.asarray(
            int(bool(physical_center_first)),
            dtype=np.int64,
        ),
        'physical_runtime_trend_result_mode': np.asarray(
            str(typed_cfg.physical_runtime.trend_result_mode)
        ),
        'physical_runtime_trend_result_reason': np.asarray(
            '' if trend_provider.reason is None else str(trend_provider.reason)
        ),
        'physical_runtime_legacy_trend_output': np.asarray(legacy_trend_output),
        'lineage': build_lineage_payload(
            canonical_cfg,
            repo_root=repo_root,
            source_model_id=source_model_id,
            iter_id=iter_id,
        ),
    }
    payload.update(
        _partial_trend_fallback_runtime_fields(
            typed_cfg=typed_cfg,
            trend_provider=trend_provider,
        )
    )
    if include_trend_output:
        payload.update(
            {
                'trend_center_i': np.asarray(trend.trend_center_i, dtype=np.int32),
                'trend_center_t_sec': np.asarray(
                    trend.trend_center_sec,
                    dtype=np.float32,
                ),
            }
        )
    if (
        bool(typed_cfg.physical_runtime.anchor_selection.enabled)
        or typed_cfg.physical_runtime.fit_policy == 'anchor_source_xy'
    ):
        payload.update(
            {
                'physical_anchor_group_id': np.asarray(
                    physical.physical_anchor_group_id,
                    dtype=np.int32,
                ),
                'physical_anchor_is_anchor': np.asarray(
                    physical.physical_anchor_is_anchor,
                    dtype=np.bool_,
                ),
                'physical_anchor_nearest_anchor_group_id': np.asarray(
                    physical.physical_anchor_nearest_anchor_group_id,
                    dtype=np.int32,
                ),
                'physical_anchor_source_distance_m': np.asarray(
                    physical.physical_anchor_source_distance_m,
                    dtype=np.float32,
                ),
            }
        )
    if (
        runtime_diagnostics is not None
        and bool(typed_cfg.physical_runtime.diagnostics.save_npz_scalars)
    ):
        start = perf_counter()
        runtime_diagnostics.to_npz_fields()
        runtime_diagnostics.add_timing(
            'diagnostics_aggregate_sec',
            perf_counter() - start,
        )
        payload.update(runtime_diagnostics.to_npz_fields())
    return payload


def run_physics_lite(
    coarse_npz_path: str | Path,
    *,
    cfg: dict[str, Any] | None,
    out_path: str | Path | None = None,
    source_model_id: str | None = None,
    iter_id: int | str | None = '',
    repo_root: Path | None = None,
    progress: Any | None = None,
    progress_context: Mapping[str, object] | None = None,
) -> Path:
    coarse_path = Path(coarse_npz_path).expanduser().resolve()
    typed_cfg = load_physics_lite_config(cfg)
    reporter = (
        progress
        if progress is not None
        else build_progress_reporter(typed_cfg.physical_runtime.progress)
    )
    progress_fields = dict(progress_context or {})
    target_path = (
        derive_robust_npz_path(coarse_path) if out_path is None else Path(out_path)
    )
    run_start = perf_counter()
    reporter.emit(
        'physics.start',
        **progress_fields,
        coarse=coarse_path,
        out=target_path,
        fit_kind=str(typed_cfg.physical_trend.fit_kind),
        fit_policy=str(typed_cfg.physical_runtime.fit_policy),
    )
    runtime_diagnostics = (
        PhysicalRuntimeDiagnostics(
            detailed_timing=bool(
                typed_cfg.physical_runtime.diagnostics.detailed_timing
            )
        )
        if bool(typed_cfg.physical_runtime.diagnostics_enabled)
        else None
    )
    reporter.emit(
        'physics.stage_start',
        **progress_fields,
        stage='load_coarse_npz',
    )
    stage_start = perf_counter()
    if runtime_diagnostics is None:
        coarse = load_coarse_npz(coarse_path)
    else:
        with runtime_diagnostics.time_block('load_coarse_npz_sec'):
            coarse = load_coarse_npz(coarse_path)
    reporter.emit(
        'physics.stage_done',
        **progress_fields,
        stage='load_coarse_npz',
        elapsed=perf_counter() - stage_start,
    )
    payload = build_robust_payload_from_coarse(
        coarse,
        cfg=cfg,
        source_model_id=source_model_id,
        iter_id=iter_id,
        repo_root=repo_root,
        runtime_diagnostics=runtime_diagnostics,
        progress=reporter,
        progress_context=progress_fields,
    )
    reporter.emit('physics.stage_start', **progress_fields, stage='save_robust_npz')
    stage_start = perf_counter()
    if runtime_diagnostics is None:
        saved_path = save_robust_npz(target_path, **payload)
    else:
        with runtime_diagnostics.time_block('save_robust_npz_sec'):
            saved_path = save_robust_npz(target_path, **payload)
    reporter.emit(
        'physics.stage_done',
        **progress_fields,
        stage='save_robust_npz',
        elapsed=perf_counter() - stage_start,
        out=saved_path,
    )
    summary = (
        runtime_diagnostics.to_summary()
        if runtime_diagnostics is not None
        else runtime_summary_from_npz_fields(payload)
    )
    if summary is not None:
        _add_trend_runtime_summary_fields(summary, payload)
    summary_path = None
    if summary is not None and bool(typed_cfg.physical_runtime.write_runtime_summary):
        reporter.emit('physics.stage_start', **progress_fields, stage='runtime_summary')
        stage_start = perf_counter()
        summary_path = write_physics_runtime_summary(saved_path, summary)
        reporter.emit(
            'physics.stage_done',
            **progress_fields,
            stage='runtime_summary',
            elapsed=perf_counter() - stage_start,
            summary=summary_path,
        )
    summary_fields: dict[str, object] = {}
    if summary is not None:
        for key in (
            'n_traces',
            'n_source_groups',
            'n_unique_fit_contexts',
            'n_fit_calls',
            'cache_hit_rate',
            'physics_total_sec',
            'physical_center_total_sec',
            'ransac_fit_total_sec',
        ):
            if key in summary:
                summary_fields[key] = summary[key]
    if 'physical_model_status' in payload:
        summary_fields['status_counts'] = _status_counts_line(
            payload['physical_model_status']
        )
    reporter.emit(
        'physics.done',
        **progress_fields,
        elapsed=perf_counter() - run_start,
        out=saved_path,
        summary=summary_path,
        **summary_fields,
    )
    return saved_path
