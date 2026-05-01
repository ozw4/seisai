from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .trace_anchor import TraceSegment, split_trace_segments_by_offset_gap

__all__ = [
    'CoarseCoverageEvalConfig',
    'CoarsePrediction',
    'TraceErrors',
    'build_gap_neighborhood_mask',
    'compute_confidence_bin_metrics',
    'compute_gap_neighborhood_metrics',
    'compute_summary_metrics',
    'compute_trace_errors',
    'load_coarse_prediction_npz',
    'load_fb_labels',
    'run_coarse_coverage_eval',
    'run_eval_from_config',
    'validate_eval_pair',
    'write_eval_reports',
]

COARSE_EVAL_REQUIRED_KEYS = (
    'coarse_pick_i',
    'coarse_pick_t_sec',
    'coarse_pmax',
    'dt_sec',
    'n_samples_orig',
    'n_traces',
    'trace_indices',
    'offsets_m',
    'ffid_values',
    'chno_values',
)
COARSE_PICK_T_SEC_ATOL = 1.0e-5


@dataclass(frozen=True)
class CoarsePrediction:
    path: Path
    coarse_pick_i: np.ndarray
    coarse_pick_t_sec: np.ndarray
    coarse_pmax: np.ndarray
    dt_sec: float
    n_samples_orig: int
    n_traces: int
    trace_indices: np.ndarray
    offsets_m: np.ndarray
    ffid_values: np.ndarray
    chno_values: np.ndarray
    payload: dict[str, np.ndarray]


@dataclass(frozen=True)
class TraceErrors:
    error_samples: np.ndarray
    abs_error_samples: np.ndarray
    error_ms: np.ndarray
    abs_error_ms: np.ndarray
    confidence: np.ndarray
    valid_indices: np.ndarray
    n_traces: int
    n_valid: int
    n_invalid: int


@dataclass(frozen=True)
class CoarseCoverageEvalConfig:
    fine_window_half_samples: int
    coverage_thresholds_samples: tuple[int, ...] = (32, 64, 128, 256)
    coverage_thresholds_ms: tuple[float, ...] = (10.0, 20.0, 50.0, 100.0)
    gap_neighborhood_traces: int = 10
    confidence_bins: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    make_figures: bool = False
    gap_ratio: float = 5.0
    min_gap_m: float | None = None


def _require_scalar(payload: dict[str, np.ndarray], key: str) -> np.ndarray:
    arr = np.asarray(payload[key])
    if arr.ndim != 0:
        msg = f'{key} must be scalar'
        raise ValueError(msg)
    return arr


def _require_1d(payload: dict[str, np.ndarray], key: str, *, length: int) -> np.ndarray:
    arr = np.asarray(payload[key])
    if arr.ndim != 1 or int(arr.shape[0]) != int(length):
        msg = f'{key} must be 1D with length n_traces'
        raise ValueError(msg)
    return arr


def _require_numeric_1d(
    payload: dict[str, np.ndarray],
    key: str,
    *,
    length: int,
) -> np.ndarray:
    arr = _require_1d(payload, key, length=length)
    if not np.issubdtype(arr.dtype, np.number):
        msg = f'{key} must be numeric'
        raise TypeError(msg)
    return arr


def _coerce_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        msg = f'{name} must be positive int'
        raise TypeError(msg)
    out = int(value)
    if out <= 0:
        msg = f'{name} must be positive'
        raise ValueError(msg)
    return out


def _coerce_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        msg = f'{name} must be non-negative int'
        raise TypeError(msg)
    out = int(value)
    if out < 0:
        msg = f'{name} must be >= 0'
        raise ValueError(msg)
    return out


def _coerce_threshold_samples(values: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        msg = f'{name} must be list[int]'
        raise TypeError(msg)
    out: list[int] = []
    for value in values:
        out.append(_coerce_nonnegative_int(name, value))
    if not out:
        msg = f'{name} must not be empty'
        raise ValueError(msg)
    if len(set(out)) != len(out):
        msg = f'{name} must not contain duplicates'
        raise ValueError(msg)
    return tuple(out)


def _coerce_threshold_floats(values: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        msg = f'{name} must be list[float]'
        raise TypeError(msg)
    out: list[float] = []
    for value in values:
        if isinstance(value, bool):
            msg = f'{name} must contain numbers'
            raise TypeError(msg)
        item = float(value)
        if not np.isfinite(item) or item < 0.0:
            msg = f'{name} must contain finite values >= 0'
            raise ValueError(msg)
        out.append(item)
    if not out:
        msg = f'{name} must not be empty'
        raise ValueError(msg)
    if len(set(out)) != len(out):
        msg = f'{name} must not contain duplicates'
        raise ValueError(msg)
    return tuple(out)


def _coerce_confidence_bins(values: object) -> tuple[float, ...]:
    bins = _coerce_threshold_floats(values, name='eval.confidence_bins')
    if len(bins) < 2:
        msg = 'eval.confidence_bins must contain at least two edges'
        raise ValueError(msg)
    if any(hi <= lo for lo, hi in zip(bins[:-1], bins[1:], strict=True)):
        msg = 'eval.confidence_bins must be strictly increasing'
        raise ValueError(msg)
    return bins


def _threshold_token(value: int | float) -> str:
    as_float = float(value)
    if as_float.is_integer():
        return str(int(as_float))
    return f'{as_float:g}'.replace('.', 'p').replace('-', 'm')


def _coverage_key_samples(threshold: int) -> str:
    return f'coverage_{_threshold_token(threshold)}'


def _coverage_key_ms(threshold_ms: float) -> str:
    return f'coverage_ms_{_threshold_token(threshold_ms)}'


def load_coarse_prediction_npz(path: str | Path) -> CoarsePrediction:
    npz_path = Path(path).expanduser().resolve()
    if not npz_path.is_file():
        msg = f'coarse npz not found: {npz_path}'
        raise FileNotFoundError(msg)

    with np.load(npz_path, allow_pickle=False) as z:
        missing = [key for key in COARSE_EVAL_REQUIRED_KEYS if key not in z.files]
        if missing:
            msg = f'coarse npz missing keys: {missing}'
            raise KeyError(msg)
        payload = {key: z[key] for key in z.files}

    dt_sec = float(_require_scalar(payload, 'dt_sec').item())
    if not np.isfinite(dt_sec) or dt_sec <= 0.0:
        msg = 'dt_sec must be finite and > 0'
        raise ValueError(msg)
    n_samples_orig = _coerce_positive_int(
        'n_samples_orig',
        _require_scalar(payload, 'n_samples_orig').item(),
    )
    n_traces = _coerce_positive_int(
        'n_traces',
        _require_scalar(payload, 'n_traces').item(),
    )

    coarse_pick_i_raw = _require_1d(payload, 'coarse_pick_i', length=n_traces)
    if not np.issubdtype(coarse_pick_i_raw.dtype, np.integer):
        msg = 'coarse_pick_i dtype must be integer'
        raise TypeError(msg)
    coarse_pick_i = coarse_pick_i_raw.astype(np.int64, copy=False)

    coarse_pick_t_sec_raw = _require_1d(
        payload,
        'coarse_pick_t_sec',
        length=n_traces,
    )
    if not np.issubdtype(coarse_pick_t_sec_raw.dtype, np.floating):
        msg = 'coarse_pick_t_sec dtype must be float'
        raise TypeError(msg)
    coarse_pick_t_sec = coarse_pick_t_sec_raw.astype(np.float64, copy=False)

    coarse_pmax_raw = _require_1d(payload, 'coarse_pmax', length=n_traces)
    if not np.issubdtype(coarse_pmax_raw.dtype, np.floating):
        msg = 'coarse_pmax dtype must be float'
        raise TypeError(msg)
    coarse_pmax = coarse_pmax_raw.astype(np.float64, copy=False)

    if np.any(coarse_pick_i < 0) or np.any(coarse_pick_i >= n_samples_orig):
        msg = 'coarse_pick_i must lie in [0, n_samples_orig - 1]'
        raise ValueError(msg)
    if not np.all(np.isfinite(coarse_pick_t_sec)):
        msg = 'coarse_pick_t_sec must be finite'
        raise ValueError(msg)
    if not np.all(np.isfinite(coarse_pmax)):
        msg = 'coarse_pmax must be finite'
        raise ValueError(msg)
    expected_t_sec = coarse_pick_i.astype(np.float64) * float(dt_sec)
    if not np.allclose(
        coarse_pick_t_sec,
        expected_t_sec,
        rtol=0.0,
        atol=COARSE_PICK_T_SEC_ATOL,
    ):
        msg = 'coarse_pick_t_sec must match coarse_pick_i * dt_sec'
        raise ValueError(msg)

    trace_indices_raw = _require_1d(payload, 'trace_indices', length=n_traces)
    if not np.issubdtype(trace_indices_raw.dtype, np.integer):
        msg = 'trace_indices dtype must be integer'
        raise TypeError(msg)
    trace_indices = trace_indices_raw.astype(np.int64, copy=False)
    if not np.array_equal(trace_indices, np.arange(n_traces, dtype=np.int64)):
        msg = 'trace_indices must equal np.arange(n_traces) for FB array alignment'
        raise ValueError(msg)

    offsets_m = _require_numeric_1d(payload, 'offsets_m', length=n_traces).astype(
        np.float64,
        copy=False,
    )
    if not np.all(np.isfinite(offsets_m)):
        msg = 'offsets_m must be finite'
        raise ValueError(msg)
    ffid_values = _require_numeric_1d(payload, 'ffid_values', length=n_traces).astype(
        np.int64,
        copy=False,
    )
    chno_values = _require_numeric_1d(payload, 'chno_values', length=n_traces).astype(
        np.int64,
        copy=False,
    )

    return CoarsePrediction(
        path=npz_path,
        coarse_pick_i=coarse_pick_i,
        coarse_pick_t_sec=coarse_pick_t_sec,
        coarse_pmax=coarse_pmax,
        dt_sec=float(dt_sec),
        n_samples_orig=int(n_samples_orig),
        n_traces=int(n_traces),
        trace_indices=trace_indices,
        offsets_m=offsets_m,
        ffid_values=ffid_values,
        chno_values=chno_values,
        payload=payload,
    )


def load_fb_labels(path: str | Path) -> np.ndarray:
    fb_path = Path(path).expanduser().resolve()
    if not fb_path.is_file():
        msg = f'fb labels not found: {fb_path}'
        raise FileNotFoundError(msg)

    loaded = np.load(fb_path, allow_pickle=False)
    try:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if 'fb_i' not in loaded.files:
                msg = 'fb npz must contain key "fb_i"'
                raise KeyError(msg)
            arr = np.asarray(loaded['fb_i'])
        else:
            arr = np.asarray(loaded)
    finally:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            loaded.close()

    if arr.ndim != 1:
        msg = f'fb labels must be 1D, got shape={arr.shape}'
        raise ValueError(msg)
    if not np.issubdtype(arr.dtype, np.number):
        msg = 'fb labels must be numeric'
        raise TypeError(msg)
    return arr.astype(np.float64, copy=False)


def validate_eval_pair(
    *,
    coarse_pick_i: np.ndarray,
    fb_i: np.ndarray,
    n_samples_orig: int,
) -> np.ndarray:
    pick = np.asarray(coarse_pick_i)
    fb = np.asarray(fb_i, dtype=np.float64)
    if pick.ndim != 1:
        msg = f'coarse_pick_i must be 1D, got shape={pick.shape}'
        raise ValueError(msg)
    if fb.ndim != 1:
        msg = f'fb_i must be 1D, got shape={fb.shape}'
        raise ValueError(msg)
    if pick.shape != fb.shape:
        msg = f'fb_i shape {fb.shape} != coarse_pick_i shape {pick.shape}'
        raise ValueError(msg)
    n_samples = _coerce_positive_int('n_samples_orig', n_samples_orig)
    return np.isfinite(fb) & (fb >= 0.0) & (fb < float(n_samples))


def compute_trace_errors(
    *,
    coarse_pick_i: np.ndarray,
    fb_i: np.ndarray,
    dt_sec: float,
    n_samples_orig: int,
    coarse_pmax: np.ndarray | None = None,
) -> TraceErrors:
    dt = float(dt_sec)
    if not np.isfinite(dt) or dt <= 0.0:
        msg = 'dt_sec must be finite and > 0'
        raise ValueError(msg)

    valid = validate_eval_pair(
        coarse_pick_i=coarse_pick_i,
        fb_i=fb_i,
        n_samples_orig=n_samples_orig,
    )
    pick = np.asarray(coarse_pick_i, dtype=np.float64)
    fb = np.asarray(fb_i, dtype=np.float64)
    n_samples = _coerce_positive_int('n_samples_orig', n_samples_orig)
    if np.any(pick < 0.0) or np.any(pick >= float(n_samples)):
        msg = 'coarse_pick_i must lie in [0, n_samples_orig - 1]'
        raise ValueError(msg)
    valid_indices = np.flatnonzero(valid).astype(np.int64, copy=False)
    error_samples = pick[valid] - fb[valid]
    abs_error_samples = np.abs(error_samples)
    error_ms = error_samples * dt * 1000.0
    abs_error_ms = np.abs(error_ms)

    if coarse_pmax is None:
        confidence = np.full((valid_indices.size,), np.nan, dtype=np.float64)
    else:
        pmax = np.asarray(coarse_pmax, dtype=np.float64)
        if pmax.shape != pick.shape:
            msg = f'coarse_pmax shape {pmax.shape} != coarse_pick_i shape {pick.shape}'
            raise ValueError(msg)
        if not np.all(np.isfinite(pmax)):
            msg = 'coarse_pmax must be finite'
            raise ValueError(msg)
        confidence = pmax[valid]

    n_traces = int(pick.shape[0])
    n_valid = int(valid_indices.size)
    return TraceErrors(
        error_samples=error_samples.astype(np.float64, copy=False),
        abs_error_samples=abs_error_samples.astype(np.float64, copy=False),
        error_ms=error_ms.astype(np.float64, copy=False),
        abs_error_ms=abs_error_ms.astype(np.float64, copy=False),
        confidence=confidence.astype(np.float64, copy=False),
        valid_indices=valid_indices,
        n_traces=n_traces,
        n_valid=n_valid,
        n_invalid=n_traces - n_valid,
    )


def _empty_errors(n_traces: int) -> TraceErrors:
    empty = np.empty((0,), dtype=np.float64)
    return TraceErrors(
        error_samples=empty,
        abs_error_samples=empty,
        error_ms=empty,
        abs_error_ms=empty,
        confidence=empty,
        valid_indices=np.empty((0,), dtype=np.int64),
        n_traces=int(n_traces),
        n_valid=0,
        n_invalid=int(n_traces),
    )


def _subset_errors_by_trace_mask(errors: TraceErrors, trace_mask: np.ndarray) -> TraceErrors:
    mask = np.asarray(trace_mask, dtype=np.bool_)
    if mask.ndim != 1 or int(mask.shape[0]) != int(errors.n_traces):
        msg = 'trace_mask must be 1D with length n_traces'
        raise ValueError(msg)
    valid_selector = mask[errors.valid_indices]
    n_traces = int(np.count_nonzero(mask))
    n_valid = int(np.count_nonzero(valid_selector))
    return TraceErrors(
        error_samples=errors.error_samples[valid_selector],
        abs_error_samples=errors.abs_error_samples[valid_selector],
        error_ms=errors.error_ms[valid_selector],
        abs_error_ms=errors.abs_error_ms[valid_selector],
        confidence=errors.confidence[valid_selector],
        valid_indices=np.arange(n_valid, dtype=np.int64),
        n_traces=n_traces,
        n_valid=n_valid,
        n_invalid=n_traces - n_valid,
    )


def _merge_errors(items: list[TraceErrors]) -> TraceErrors:
    if not items:
        return _empty_errors(0)
    n_traces = int(sum(item.n_traces for item in items))
    n_invalid = int(sum(item.n_invalid for item in items))
    n_valid = int(sum(item.n_valid for item in items))
    if n_valid == 0:
        empty = np.empty((0,), dtype=np.float64)
        return TraceErrors(
            error_samples=empty,
            abs_error_samples=empty,
            error_ms=empty,
            abs_error_ms=empty,
            confidence=empty,
            valid_indices=np.empty((0,), dtype=np.int64),
            n_traces=n_traces,
            n_valid=0,
            n_invalid=n_invalid,
        )
    return TraceErrors(
        error_samples=np.concatenate([item.error_samples for item in items], axis=0),
        abs_error_samples=np.concatenate(
            [item.abs_error_samples for item in items],
            axis=0,
        ),
        error_ms=np.concatenate([item.error_ms for item in items], axis=0),
        abs_error_ms=np.concatenate([item.abs_error_ms for item in items], axis=0),
        confidence=np.concatenate([item.confidence for item in items], axis=0),
        valid_indices=np.arange(n_valid, dtype=np.int64),
        n_traces=n_traces,
        n_valid=n_valid,
        n_invalid=n_invalid,
    )


def _mean(values: np.ndarray) -> float:
    if int(values.size) == 0:
        return float('nan')
    return float(np.mean(values.astype(np.float64, copy=False)))


def _percentile(values: np.ndarray, q: float) -> float:
    if int(values.size) == 0:
        return float('nan')
    return float(np.percentile(values.astype(np.float64, copy=False), float(q)))


def _max(values: np.ndarray) -> float:
    if int(values.size) == 0:
        return float('nan')
    return float(np.max(values.astype(np.float64, copy=False)))


def _rate(mask: np.ndarray) -> float:
    if int(mask.size) == 0:
        return float('nan')
    return float(np.mean(mask.astype(np.float64, copy=False)))


def _metrics_from_arrays(
    *,
    error_samples: np.ndarray,
    abs_error_samples: np.ndarray,
    error_ms: np.ndarray,
    abs_error_ms: np.ndarray,
    confidence: np.ndarray,
    n_valid: int,
    n_invalid: int,
    coverage_thresholds_samples: tuple[int, ...],
    coverage_thresholds_ms: tuple[float, ...],
    fine_window_half_samples: int,
) -> dict[str, Any]:
    fine_half = _coerce_nonnegative_int(
        'fine_window_half_samples',
        fine_window_half_samples,
    )
    out: dict[str, Any] = {
        'n_valid': int(n_valid),
        'n_invalid': int(n_invalid),
        'mae_samples': _mean(abs_error_samples),
        'median_abs_samples': _percentile(abs_error_samples, 50.0),
        'p50_abs_samples': _percentile(abs_error_samples, 50.0),
        'p90_abs_samples': _percentile(abs_error_samples, 90.0),
        'p95_abs_samples': _percentile(abs_error_samples, 95.0),
        'p99_abs_samples': _percentile(abs_error_samples, 99.0),
        'max_abs_samples': _max(abs_error_samples),
        'bias_samples': _mean(error_samples),
        'mae_ms': _mean(abs_error_ms),
        'median_abs_ms': _percentile(abs_error_ms, 50.0),
        'p50_abs_ms': _percentile(abs_error_ms, 50.0),
        'p90_abs_ms': _percentile(abs_error_ms, 90.0),
        'p95_abs_ms': _percentile(abs_error_ms, 95.0),
        'p99_abs_ms': _percentile(abs_error_ms, 99.0),
        'max_abs_ms': _max(abs_error_ms),
        'bias_ms': _mean(error_ms),
        'mean_confidence': _mean(confidence[np.isfinite(confidence)]),
        'median_confidence': _percentile(confidence[np.isfinite(confidence)], 50.0),
    }
    for threshold in coverage_thresholds_samples:
        out[_coverage_key_samples(threshold)] = _rate(abs_error_samples <= threshold)
    for threshold_ms in coverage_thresholds_ms:
        out[_coverage_key_ms(threshold_ms)] = _rate(abs_error_ms <= threshold_ms)

    fine_coverage = _rate(abs_error_samples <= fine_half)
    out['coverage_fine_window'] = fine_coverage
    out['n_fail_fine_window'] = int(np.count_nonzero(abs_error_samples > fine_half))
    out['failure_rate_fine_window'] = (
        float('nan') if np.isnan(fine_coverage) else float(1.0 - fine_coverage)
    )
    return out


def compute_summary_metrics(
    errors: TraceErrors,
    *,
    coverage_thresholds_samples: tuple[int, ...] = (32, 64, 128, 256),
    coverage_thresholds_ms: tuple[float, ...] = (10.0, 20.0, 50.0, 100.0),
    fine_window_half_samples: int = 128,
) -> dict[str, Any]:
    return _metrics_from_arrays(
        error_samples=errors.error_samples,
        abs_error_samples=errors.abs_error_samples,
        error_ms=errors.error_ms,
        abs_error_ms=errors.abs_error_ms,
        confidence=errors.confidence,
        n_valid=errors.n_valid,
        n_invalid=errors.n_invalid,
        coverage_thresholds_samples=coverage_thresholds_samples,
        coverage_thresholds_ms=coverage_thresholds_ms,
        fine_window_half_samples=fine_window_half_samples,
    )


def compute_confidence_bin_metrics(
    errors: TraceErrors,
    *,
    confidence_bins: tuple[float, ...],
    coverage_thresholds_samples: tuple[int, ...] = (32, 64, 128, 256),
    coverage_thresholds_ms: tuple[float, ...] = (10.0, 20.0, 50.0, 100.0),
    fine_window_half_samples: int = 128,
) -> list[dict[str, Any]]:
    bins = _coerce_confidence_bins(list(confidence_bins))
    rows: list[dict[str, Any]] = []
    conf = errors.confidence
    for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:], strict=True)):
        if idx == len(bins) - 2:
            selector = (conf >= lo) & (conf <= hi)
        else:
            selector = (conf >= lo) & (conf < hi)
        n_valid = int(np.count_nonzero(selector))
        metrics = _metrics_from_arrays(
            error_samples=errors.error_samples[selector],
            abs_error_samples=errors.abs_error_samples[selector],
            error_ms=errors.error_ms[selector],
            abs_error_ms=errors.abs_error_ms[selector],
            confidence=conf[selector],
            n_valid=n_valid,
            n_invalid=0,
            coverage_thresholds_samples=coverage_thresholds_samples,
            coverage_thresholds_ms=coverage_thresholds_ms,
            fine_window_half_samples=fine_window_half_samples,
        )
        rows.append(
            {
                'bin_lo': float(lo),
                'bin_hi': float(hi),
                **metrics,
            }
        )
    return rows


def _coerce_segment_spans_from_payload(
    coarse: CoarsePrediction,
) -> tuple[TraceSegment, ...] | None:
    payload = coarse.payload
    start_key = None
    stop_key = None
    for candidate_start, candidate_stop in (
        ('segment_start_pos', 'segment_stop_pos'),
        ('segment_start', 'segment_stop'),
        ('segment_starts', 'segment_stops'),
    ):
        if candidate_start in payload and candidate_stop in payload:
            start_key = candidate_start
            stop_key = candidate_stop
            break
    if start_key is None or stop_key is None:
        return None

    starts = np.asarray(payload[start_key], dtype=np.int64)
    stops = np.asarray(payload[stop_key], dtype=np.int64)
    if starts.ndim != 1 or stops.ndim != 1 or starts.shape != stops.shape:
        msg = 'segment start/stop metadata must be same-length 1D arrays'
        raise ValueError(msg)

    if 'segment_ids' in payload:
        segment_ids = np.asarray(payload['segment_ids'], dtype=np.int64)
        if segment_ids.shape != starts.shape:
            msg = 'segment_ids must match segment start/stop shape'
            raise ValueError(msg)
    else:
        segment_ids = np.arange(int(starts.size), dtype=np.int64)

    segments: list[TraceSegment] = []
    covered = np.zeros((coarse.n_traces,), dtype=np.bool_)
    for sid, start, stop in zip(segment_ids, starts, stops, strict=True):
        start_i = int(start)
        stop_i = int(stop)
        if start_i < 0 or stop_i <= start_i or stop_i > coarse.n_traces:
            msg = f'invalid segment span: [{start_i}, {stop_i})'
            raise ValueError(msg)
        if np.any(covered[start_i:stop_i]):
            msg = f'overlapping segment span: [{start_i}, {stop_i})'
            raise ValueError(msg)
        covered[start_i:stop_i] = True
        segments.append(
            TraceSegment(
                segment_id=int(sid),
                start_pos=start_i,
                stop_pos=stop_i,
                n_traces=stop_i - start_i,
            )
        )
    if not segments:
        return None
    if not np.all(covered):
        msg = 'segment metadata must cover every trace exactly once'
        raise ValueError(msg)
    return tuple(segments)


def build_gap_neighborhood_mask(
    *,
    n_traces: int,
    segments: tuple[TraceSegment, ...],
    gap_neighborhood_traces: int,
) -> np.ndarray:
    n_trace = _coerce_positive_int('n_traces', n_traces)
    half = _coerce_nonnegative_int(
        'gap_neighborhood_traces',
        gap_neighborhood_traces,
    )
    mask = np.zeros((n_trace,), dtype=np.bool_)
    ordered = sorted(segments, key=lambda segment: int(segment.start_pos))
    for segment in ordered:
        if segment.start_pos < 0 or segment.stop_pos <= segment.start_pos:
            msg = 'segments must have valid [start_pos, stop_pos) spans'
            raise ValueError(msg)
        if segment.stop_pos > n_trace:
            msg = 'segment stop_pos exceeds n_traces'
            raise ValueError(msg)
    for segment in ordered[1:]:
        boundary = int(segment.start_pos)
        start = max(0, boundary - half - 1)
        stop = min(n_trace, boundary + half + 1)
        mask[start:stop] = True
    return mask


def _detect_gap_segments(
    coarse: CoarsePrediction,
    *,
    gap_ratio: float,
    min_gap_m: float | None,
) -> tuple[TraceSegment, ...]:
    return split_trace_segments_by_offset_gap(
        coarse.offsets_m,
        gap_ratio=float(gap_ratio),
        min_gap_m=None if min_gap_m is None else float(min_gap_m),
    )


def compute_gap_neighborhood_metrics(
    errors: TraceErrors,
    *,
    gap_mask: np.ndarray,
    coverage_thresholds_samples: tuple[int, ...] = (32, 64, 128, 256),
    coverage_thresholds_ms: tuple[float, ...] = (10.0, 20.0, 50.0, 100.0),
    fine_window_half_samples: int = 128,
) -> list[dict[str, Any]]:
    gap_errors = _subset_errors_by_trace_mask(errors, gap_mask)
    non_gap_errors = _subset_errors_by_trace_mask(errors, ~np.asarray(gap_mask))
    rows: list[dict[str, Any]] = []
    for group, group_errors in (
        ('gap_neighborhood', gap_errors),
        ('non_gap', non_gap_errors),
    ):
        rows.append(
            {
                'group': group,
                **compute_summary_metrics(
                    group_errors,
                    coverage_thresholds_samples=coverage_thresholds_samples,
                    coverage_thresholds_ms=coverage_thresholds_ms,
                    fine_window_half_samples=fine_window_half_samples,
                ),
            }
        )
    return rows


def _unique_scalar_or_empty(values: np.ndarray) -> int | str:
    arr = np.asarray(values)
    if arr.size == 0:
        return ''
    unique = np.unique(arr)
    if int(unique.size) == 1:
        return int(unique[0])
    return ''


def _source_file_from_payload(coarse: CoarsePrediction) -> str:
    for key in ('source_file', 'segy_file', 'segy_path'):
        if key not in coarse.payload:
            continue
        arr = np.asarray(coarse.payload[key])
        if arr.ndim == 0:
            return str(arr.item())
    return ''


def _metric_columns(
    *,
    coverage_thresholds_samples: tuple[int, ...],
    coverage_thresholds_ms: tuple[float, ...],
) -> list[str]:
    return [
        'n_valid',
        'n_invalid',
        'mae_samples',
        'median_abs_samples',
        'p50_abs_samples',
        'p90_abs_samples',
        'p95_abs_samples',
        'p99_abs_samples',
        'max_abs_samples',
        'bias_samples',
        'mae_ms',
        'median_abs_ms',
        'p50_abs_ms',
        'p90_abs_ms',
        'p95_abs_ms',
        'p99_abs_ms',
        'max_abs_ms',
        'bias_ms',
        *[_coverage_key_samples(v) for v in coverage_thresholds_samples],
        'coverage_fine_window',
        *[_coverage_key_ms(v) for v in coverage_thresholds_ms],
        'mean_confidence',
        'median_confidence',
        'n_fail_fine_window',
        'failure_rate_fine_window',
    ]


def _format_csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return ''
        return f'{value:.10g}'
    return value


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_csv_value(row.get(key, '')) for key in columns})
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=True, indent=2, sort_keys=True)
        + '\n',
        encoding='utf-8',
    )
    return path


def _plot_eval_figures(
    *,
    out_dir: Path,
    global_errors: TraceErrors,
    per_gather_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    coverage_thresholds_samples: tuple[int, ...],
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def _save(name: str) -> None:
        path = figures_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        paths[name] = path

    plt.figure(figsize=(8.0, 4.8))
    plt.hist(global_errors.abs_error_samples, bins=50, color='#3572a5')
    plt.xlabel('Absolute Error (samples)')
    plt.ylabel('Trace Count')
    _save('error_hist_samples.png')

    plt.figure(figsize=(8.0, 4.8))
    plt.hist(global_errors.abs_error_ms, bins=50, color='#4f8f58')
    plt.xlabel('Absolute Error (ms)')
    plt.ylabel('Trace Count')
    _save('error_hist_ms.png')

    coverage_values = [
        _rate(global_errors.abs_error_samples <= threshold)
        for threshold in coverage_thresholds_samples
    ]
    plt.figure(figsize=(8.0, 4.8))
    plt.plot(coverage_thresholds_samples, coverage_values, marker='o')
    plt.ylim(0.0, 1.02)
    plt.xlabel('Threshold (samples)')
    plt.ylabel('Coverage')
    _save('coverage_by_threshold.png')

    conf = global_errors.confidence
    err = global_errors.abs_error_samples
    if int(conf.size) > 10000:
        idx = np.linspace(0, int(conf.size) - 1, 10000).round().astype(np.int64)
        conf = conf[idx]
        err = err[idx]
    plt.figure(figsize=(8.0, 4.8))
    plt.scatter(conf, err, s=5.0, alpha=0.35)
    plt.xlabel('coarse_pmax')
    plt.ylabel('Absolute Error (samples)')
    _save('confidence_vs_abs_error.png')

    gather_x = np.arange(len(per_gather_rows), dtype=np.int64)
    plt.figure(figsize=(10.0, 4.8))
    plt.bar(
        gather_x,
        [
            float(row.get('coverage_fine_window', float('nan')))
            for row in per_gather_rows
        ],
    )
    plt.ylim(0.0, 1.02)
    plt.xlabel('Gather')
    plt.ylabel('Coverage@Fine Window')
    _save('per_gather_coverage.png')

    plt.figure(figsize=(10.0, 4.8))
    plt.bar(
        gather_x,
        [float(row.get('p95_abs_samples', float('nan'))) for row in per_gather_rows],
    )
    plt.xlabel('Gather')
    plt.ylabel('P95 Absolute Error (samples)')
    _save('per_gather_p95_error.png')

    plt.figure(figsize=(7.0, 4.8))
    plt.bar(
        [str(row['group']) for row in gap_rows],
        [float(row.get('coverage_fine_window', float('nan'))) for row in gap_rows],
    )
    plt.ylim(0.0, 1.02)
    plt.ylabel('Coverage@Fine Window')
    _save('gap_neighborhood_comparison.png')
    return paths


def write_eval_reports(
    *,
    out_dir: str | Path,
    per_gather_rows: list[dict[str, Any]],
    summary_row: dict[str, Any],
    confidence_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    per_segment_rows: list[dict[str, Any]],
    coverage_thresholds_samples: tuple[int, ...],
    coverage_thresholds_ms: tuple[float, ...],
    make_figures: bool,
    global_errors: TraceErrors,
) -> dict[str, Path]:
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    metric_cols = _metric_columns(
        coverage_thresholds_samples=coverage_thresholds_samples,
        coverage_thresholds_ms=coverage_thresholds_ms,
    )
    report_paths: dict[str, Path] = {}
    report_paths['per_gather_csv'] = _write_csv(
        out_path / 'per_gather.csv',
        [
            'coarse_file',
            'fb_file',
            'source_file',
            'ffid',
            'n_traces',
            'n_samples_orig',
            'dt_sec',
            *metric_cols,
        ],
        per_gather_rows,
    )
    summary_columns = [
        'n_gathers',
        'n_traces_total',
        'n_valid_total',
        'n_invalid_total',
        'fine_window_half_samples',
        *metric_cols,
        'per_segment_status',
    ]
    report_paths['summary_csv'] = _write_csv(
        out_path / 'summary.csv',
        summary_columns,
        [summary_row],
    )
    report_paths['summary_json'] = _write_json(out_path / 'summary.json', summary_row)

    compact_cols = [
        'n_valid',
        'mae_samples',
        'mae_ms',
        'p90_abs_samples',
        'p95_abs_samples',
        'coverage_fine_window',
        'failure_rate_fine_window',
    ]
    report_paths['confidence_bins_csv'] = _write_csv(
        out_path / 'confidence_bins.csv',
        ['bin_lo', 'bin_hi', *compact_cols],
        confidence_rows,
    )
    report_paths['gap_neighborhood_csv'] = _write_csv(
        out_path / 'gap_neighborhood.csv',
        ['group', *compact_cols],
        gap_rows,
    )
    if per_segment_rows:
        report_paths['per_segment_csv'] = _write_csv(
            out_path / 'per_segment.csv',
            [
                'coarse_file',
                'source_file',
                'ffid',
                'segment_id',
                'segment_start',
                'segment_stop',
                'n_traces',
                'n_valid',
                'mae_samples',
                'mae_ms',
                'p90_abs_samples',
                'p95_abs_samples',
                'coverage_fine_window',
                'mean_confidence',
            ],
            per_segment_rows,
        )
    if make_figures:
        figures = _plot_eval_figures(
            out_dir=out_path,
            global_errors=global_errors,
            per_gather_rows=per_gather_rows,
            gap_rows=gap_rows,
            coverage_thresholds_samples=coverage_thresholds_samples,
        )
        report_paths.update({f'figure_{name}': path for name, path in figures.items()})
    return report_paths


def _load_eval_config_from_dict(cfg: dict[str, Any]) -> CoarseCoverageEvalConfig:
    eval_cfg = cfg.get('eval', {})
    if eval_cfg is None:
        eval_cfg = {}
    if not isinstance(eval_cfg, dict):
        msg = 'eval must be dict'
        raise TypeError(msg)
    fine_window_half = _coerce_nonnegative_int(
        'eval.fine_window_half_samples',
        eval_cfg.get('fine_window_half_samples', 128),
    )
    coverage_thresholds_samples = _coerce_threshold_samples(
        eval_cfg.get('coverage_thresholds_samples', [32, 64, 128, 256]),
        name='eval.coverage_thresholds_samples',
    )
    coverage_thresholds_ms = _coerce_threshold_floats(
        eval_cfg.get('coverage_thresholds_ms', [10, 20, 50, 100]),
        name='eval.coverage_thresholds_ms',
    )
    confidence_bins = _coerce_confidence_bins(
        eval_cfg.get('confidence_bins', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    )
    make_figures = eval_cfg.get('make_figures', False)
    if not isinstance(make_figures, bool):
        msg = 'eval.make_figures must be bool'
        raise TypeError(msg)
    gap_ratio = float(eval_cfg.get('gap_ratio', 5.0))
    if not np.isfinite(gap_ratio) or gap_ratio <= 1.0:
        msg = 'eval.gap_ratio must be finite and > 1.0'
        raise ValueError(msg)
    min_gap_raw = eval_cfg.get('min_gap_m', None)
    min_gap_m = None if min_gap_raw is None else float(min_gap_raw)
    if min_gap_m is not None and (not np.isfinite(min_gap_m) or min_gap_m <= 0.0):
        msg = 'eval.min_gap_m must be null or > 0'
        raise ValueError(msg)
    return CoarseCoverageEvalConfig(
        fine_window_half_samples=fine_window_half,
        coverage_thresholds_samples=coverage_thresholds_samples,
        coverage_thresholds_ms=coverage_thresholds_ms,
        gap_neighborhood_traces=_coerce_nonnegative_int(
            'eval.gap_neighborhood_traces',
            eval_cfg.get('gap_neighborhood_traces', 10),
        ),
        confidence_bins=confidence_bins,
        make_figures=make_figures,
        gap_ratio=gap_ratio,
        min_gap_m=min_gap_m,
    )


def _validate_paths_cfg(cfg: dict[str, Any]) -> tuple[list[str], list[str], Path]:
    paths = cfg.get('paths')
    if not isinstance(paths, dict):
        msg = 'paths must be dict'
        raise TypeError(msg)
    coarse_files = paths.get('coarse_files')
    fb_files = paths.get('fb_files')
    out_dir = paths.get('out_dir')
    if (
        not isinstance(coarse_files, list)
        or not coarse_files
        or not all(isinstance(item, str) for item in coarse_files)
    ):
        msg = 'paths.coarse_files must be non-empty list[str]'
        raise TypeError(msg)
    if (
        not isinstance(fb_files, list)
        or not fb_files
        or not all(isinstance(item, str) for item in fb_files)
    ):
        msg = 'paths.fb_files must be non-empty list[str]'
        raise TypeError(msg)
    if len(coarse_files) != len(fb_files):
        msg = 'paths.coarse_files and paths.fb_files must have the same length'
        raise ValueError(msg)
    if not isinstance(out_dir, str) or not out_dir:
        msg = 'paths.out_dir must be non-empty str'
        raise TypeError(msg)
    return list(coarse_files), list(fb_files), Path(out_dir)


def _segment_rows_for_gather(
    *,
    coarse: CoarsePrediction,
    errors: TraceErrors,
    segments: tuple[TraceSegment, ...],
    coverage_thresholds_samples: tuple[int, ...],
    coverage_thresholds_ms: tuple[float, ...],
    fine_window_half_samples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_file = _source_file_from_payload(coarse)
    ffid = _unique_scalar_or_empty(coarse.ffid_values)
    for segment in segments:
        mask = np.zeros((coarse.n_traces,), dtype=np.bool_)
        mask[int(segment.start_pos) : int(segment.stop_pos)] = True
        segment_errors = _subset_errors_by_trace_mask(errors, mask)
        metrics = compute_summary_metrics(
            segment_errors,
            coverage_thresholds_samples=coverage_thresholds_samples,
            coverage_thresholds_ms=coverage_thresholds_ms,
            fine_window_half_samples=fine_window_half_samples,
        )
        rows.append(
            {
                'coarse_file': str(coarse.path),
                'source_file': source_file,
                'ffid': ffid,
                'segment_id': int(segment.segment_id),
                'segment_start': int(segment.start_pos),
                'segment_stop': int(segment.stop_pos),
                'n_traces': int(segment.n_traces),
                **metrics,
            }
        )
    return rows


def run_coarse_coverage_eval(
    *,
    coarse_files: list[str] | tuple[str, ...],
    fb_files: list[str] | tuple[str, ...],
    out_dir: str | Path,
    eval_config: CoarseCoverageEvalConfig,
) -> dict[str, Path]:
    if len(coarse_files) != len(fb_files):
        msg = 'coarse_files and fb_files must have the same length'
        raise ValueError(msg)
    if len(coarse_files) == 0:
        msg = 'coarse_files must not be empty'
        raise ValueError(msg)

    per_gather_rows: list[dict[str, Any]] = []
    per_segment_rows: list[dict[str, Any]] = []
    all_errors: list[TraceErrors] = []
    gap_error_groups = {'gap_neighborhood': [], 'non_gap': []}
    segment_skipped = 0

    for coarse_path, fb_path in zip(coarse_files, fb_files, strict=True):
        coarse = load_coarse_prediction_npz(coarse_path)
        fb_i = load_fb_labels(fb_path)
        errors = compute_trace_errors(
            coarse_pick_i=coarse.coarse_pick_i,
            fb_i=fb_i,
            dt_sec=coarse.dt_sec,
            n_samples_orig=coarse.n_samples_orig,
            coarse_pmax=coarse.coarse_pmax,
        )
        all_errors.append(errors)

        metrics = compute_summary_metrics(
            errors,
            coverage_thresholds_samples=eval_config.coverage_thresholds_samples,
            coverage_thresholds_ms=eval_config.coverage_thresholds_ms,
            fine_window_half_samples=eval_config.fine_window_half_samples,
        )
        per_gather_rows.append(
            {
                'coarse_file': str(coarse.path),
                'fb_file': str(Path(fb_path).expanduser().resolve()),
                'source_file': _source_file_from_payload(coarse),
                'ffid': _unique_scalar_or_empty(coarse.ffid_values),
                'n_traces': int(coarse.n_traces),
                'n_samples_orig': int(coarse.n_samples_orig),
                'dt_sec': float(coarse.dt_sec),
                **metrics,
            }
        )

        payload_segments = _coerce_segment_spans_from_payload(coarse)
        if payload_segments is None:
            segment_skipped += 1
            gap_segments = _detect_gap_segments(
                coarse,
                gap_ratio=eval_config.gap_ratio,
                min_gap_m=eval_config.min_gap_m,
            )
        else:
            gap_segments = payload_segments
            per_segment_rows.extend(
                _segment_rows_for_gather(
                    coarse=coarse,
                    errors=errors,
                    segments=payload_segments,
                    coverage_thresholds_samples=(
                        eval_config.coverage_thresholds_samples
                    ),
                    coverage_thresholds_ms=eval_config.coverage_thresholds_ms,
                    fine_window_half_samples=eval_config.fine_window_half_samples,
                )
            )

        gap_mask = build_gap_neighborhood_mask(
            n_traces=coarse.n_traces,
            segments=gap_segments,
            gap_neighborhood_traces=eval_config.gap_neighborhood_traces,
        )
        gap_error_groups['gap_neighborhood'].append(
            _subset_errors_by_trace_mask(errors, gap_mask)
        )
        gap_error_groups['non_gap'].append(_subset_errors_by_trace_mask(errors, ~gap_mask))

    global_errors = _merge_errors(all_errors)
    global_metrics = compute_summary_metrics(
        global_errors,
        coverage_thresholds_samples=eval_config.coverage_thresholds_samples,
        coverage_thresholds_ms=eval_config.coverage_thresholds_ms,
        fine_window_half_samples=eval_config.fine_window_half_samples,
    )
    per_segment_status = (
        'written'
        if per_segment_rows
        else f'skipped:no_segment_metadata ({segment_skipped} gathers)'
    )
    summary_row = {
        'n_gathers': int(len(coarse_files)),
        'n_traces_total': int(global_errors.n_traces),
        'n_valid_total': int(global_errors.n_valid),
        'n_invalid_total': int(global_errors.n_invalid),
        'fine_window_half_samples': int(eval_config.fine_window_half_samples),
        **global_metrics,
        'per_segment_status': per_segment_status,
    }

    confidence_rows = compute_confidence_bin_metrics(
        global_errors,
        confidence_bins=eval_config.confidence_bins,
        coverage_thresholds_samples=eval_config.coverage_thresholds_samples,
        coverage_thresholds_ms=eval_config.coverage_thresholds_ms,
        fine_window_half_samples=eval_config.fine_window_half_samples,
    )
    gap_rows: list[dict[str, Any]] = []
    for group, items in gap_error_groups.items():
        group_errors = _merge_errors(items)
        gap_rows.append(
            {
                'group': group,
                **compute_summary_metrics(
                    group_errors,
                    coverage_thresholds_samples=(
                        eval_config.coverage_thresholds_samples
                    ),
                    coverage_thresholds_ms=eval_config.coverage_thresholds_ms,
                    fine_window_half_samples=eval_config.fine_window_half_samples,
                ),
            }
        )

    return write_eval_reports(
        out_dir=out_dir,
        per_gather_rows=per_gather_rows,
        summary_row=summary_row,
        confidence_rows=confidence_rows,
        gap_rows=gap_rows,
        per_segment_rows=per_segment_rows,
        coverage_thresholds_samples=eval_config.coverage_thresholds_samples,
        coverage_thresholds_ms=eval_config.coverage_thresholds_ms,
        make_figures=eval_config.make_figures,
        global_errors=global_errors,
    )


def run_eval_from_config(config_path: str | Path) -> dict[str, Path]:
    from seisai_engine.pipelines.common import load_cfg_with_base_dir, resolve_cfg_paths
    from seisai_utils.listfiles import expand_cfg_listfiles

    cfg, base_dir = load_cfg_with_base_dir(Path(config_path))
    resolve_cfg_paths(
        cfg,
        base_dir,
        keys=['paths.coarse_files', 'paths.fb_files', 'paths.out_dir'],
    )
    expand_cfg_listfiles(cfg, keys=['paths.coarse_files', 'paths.fb_files'])
    coarse_files, fb_files, out_dir = _validate_paths_cfg(cfg)
    eval_config = _load_eval_config_from_dict(cfg)
    return run_coarse_coverage_eval(
        coarse_files=coarse_files,
        fb_files=fb_files,
        out_dir=out_dir,
        eval_config=eval_config,
    )
