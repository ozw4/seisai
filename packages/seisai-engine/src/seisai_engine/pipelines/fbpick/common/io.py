from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import (
    COARSE_REQUIRED_KEYS,
    FINAL_REQUIRED_KEYS,
    FINE_RESULT_REQUIRED_KEYS,
    ROBUST_REQUIRED_KEYS,
    ROBUST_SOURCE_COARSE_OBSERVED,
    ROBUST_SOURCE_THEORETICAL,
    ROBUST_SOURCE_TREND_FILL,
)

__all__ = [
    'COARSE_REQUIRED_KEYS',
    'FINAL_REQUIRED_KEYS',
    'FINE_RESULT_REQUIRED_KEYS',
    'ROBUST_REQUIRED_KEYS',
    'build_fbpick_final_payload',
    'load_coarse_npz',
    'load_fbpick_final_npz',
    'load_robust_npz',
    'save_coarse_npz',
    'save_fbpick_final_npz',
    'save_robust_npz',
    'validate_fbpick_final_payload',
    'validate_fine_result_payload',
]


def _coerce_scalar(name: str, value, *, dtype) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 0:
        msg = f'{name} must be scalar'
        raise ValueError(msg)
    return arr


def _coerce_vector(name: str, value, *, dtype, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 1:
        msg = f'{name} must be 1D'
        raise ValueError(msg)
    if int(arr.shape[0]) != int(length):
        msg = f'{name} length {int(arr.shape[0])} != n_traces {int(length)}'
        raise ValueError(msg)
    return arr


def _coerce_bool_vector(name: str, value, *, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.bool_)
    if arr.ndim != 1:
        msg = f'{name} must be 1D'
        raise ValueError(msg)
    if int(arr.shape[0]) != int(length):
        msg = f'{name} length {int(arr.shape[0])} != n_traces {int(length)}'
        raise ValueError(msg)
    return arr


def _coerce_lineage(lineage) -> np.ndarray:
    if isinstance(lineage, np.ndarray):
        arr = lineage
    else:
        arr = np.asarray(lineage)
    if arr.ndim != 0:
        msg = 'lineage must be scalar'
        raise ValueError(msg)
    return arr


def _validate_unit_interval(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        msg = f'{name} must be finite'
        raise ValueError(msg)
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        msg = f'{name} must lie in [0, 1]'
        raise ValueError(msg)


def _validate_pick_time_pair(
    *,
    pick_key: str,
    time_key: str,
    pick_i: np.ndarray,
    pick_t_sec: np.ndarray,
    dt_sec: float,
    n_samples_orig: int,
) -> None:
    if np.any(pick_i < 0) or np.any(pick_i >= int(n_samples_orig)):
        msg = f'{pick_key} must lie in [0, n_samples_orig)'
        raise ValueError(msg)
    if not np.all(np.isfinite(pick_t_sec)):
        msg = f'{time_key} must be finite'
        raise ValueError(msg)
    expected_t_sec = pick_i.astype(np.float32) * np.float32(dt_sec)
    if not np.array_equal(pick_t_sec, expected_t_sec):
        msg = f'{time_key} must equal {pick_key} * dt_sec'
        raise ValueError(msg)


def _require_exact_dtype(name: str, arr: np.ndarray, *, dtype) -> None:
    if arr.dtype != np.dtype(dtype):
        msg = f'{name} dtype must be {np.dtype(dtype)}, got {arr.dtype}'
        raise ValueError(msg)


def _require_payload_scalar(payload: dict[str, np.ndarray], name: str, *, dtype) -> np.ndarray:
    if name not in payload:
        msg = f'payload missing key: {name}'
        raise KeyError(msg)
    arr = np.asarray(payload[name])
    if arr.ndim != 0:
        msg = f'{name} must be scalar'
        raise ValueError(msg)
    _require_exact_dtype(name, arr, dtype=dtype)
    return arr


def _require_payload_vector(
    payload: dict[str, np.ndarray],
    name: str,
    *,
    dtype,
    length: int,
) -> np.ndarray:
    if name not in payload:
        msg = f'payload missing key: {name}'
        raise KeyError(msg)
    arr = np.asarray(payload[name])
    if arr.ndim != 1 or int(arr.shape[0]) != int(length):
        msg = f'{name} must be 1D with length n_traces'
        raise ValueError(msg)
    _require_exact_dtype(name, arr, dtype=dtype)
    return arr


def _validate_high_conf_threshold(value: float) -> np.float32:
    threshold = float(value)
    if threshold < 0.0 or threshold > 1.0:
        msg = 'infer.high_conf_threshold must lie in [0, 1]'
        raise ValueError(msg)
    return np.float32(threshold)


def save_coarse_npz(
    path: str | Path,
    *,
    dt_sec: float,
    n_samples_orig: int,
    n_traces: int,
    ffid_values,
    chno_values,
    offsets_m,
    trace_indices,
    coarse_pick_i,
    coarse_pick_t_sec,
    coarse_pmax,
    coarse_prob_summary,
    lineage,
) -> Path:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_traces_int = int(n_traces)
    n_samples_orig_int = int(n_samples_orig)
    if n_traces_int <= 0:
        msg = 'n_traces must be positive'
        raise ValueError(msg)
    if n_samples_orig_int <= 0:
        msg = 'n_samples_orig must be positive'
        raise ValueError(msg)

    arrays = {
        'dt_sec': _coerce_scalar('dt_sec', dt_sec, dtype=np.float32),
        'n_samples_orig': _coerce_scalar(
            'n_samples_orig',
            n_samples_orig_int,
            dtype=np.int32,
        ),
        'n_traces': _coerce_scalar('n_traces', n_traces_int, dtype=np.int32),
        'ffid_values': _coerce_vector(
            'ffid_values',
            ffid_values,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'chno_values': _coerce_vector(
            'chno_values',
            chno_values,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'offsets_m': _coerce_vector(
            'offsets_m',
            offsets_m,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'trace_indices': _coerce_vector(
            'trace_indices',
            trace_indices,
            dtype=np.int64,
            length=n_traces_int,
        ),
        'coarse_pick_i': _coerce_vector(
            'coarse_pick_i',
            coarse_pick_i,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'coarse_pick_t_sec': _coerce_vector(
            'coarse_pick_t_sec',
            coarse_pick_t_sec,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'coarse_pmax': _coerce_vector(
            'coarse_pmax',
            coarse_pmax,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'coarse_prob_summary': _coerce_vector(
            'coarse_prob_summary',
            coarse_prob_summary,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'lineage': _coerce_lineage(lineage),
    }

    coarse_pick_i_arr = arrays['coarse_pick_i']
    if np.any(coarse_pick_i_arr < 0) or np.any(coarse_pick_i_arr >= n_samples_orig_int):
        msg = 'coarse_pick_i must lie in [0, n_samples_orig)'
        raise ValueError(msg)

    np.savez_compressed(out_path, **arrays)
    return out_path


def load_coarse_npz(path: str | Path) -> dict[str, np.ndarray]:
    npz_path = Path(path).expanduser().resolve()
    if not npz_path.is_file():
        msg = f'coarse npz not found: {npz_path}'
        raise FileNotFoundError(msg)

    with np.load(npz_path, allow_pickle=False) as z:
        missing = [key for key in COARSE_REQUIRED_KEYS if key not in z.files]
        if missing:
            msg = f'coarse npz missing keys: {missing}'
            raise KeyError(msg)
        out = {key: z[key] for key in z.files}

    n_traces = int(np.asarray(out['n_traces']).item())
    n_samples_orig = int(np.asarray(out['n_samples_orig']).item())
    if n_traces <= 0:
        msg = 'n_traces must be positive'
        raise ValueError(msg)
    if n_samples_orig <= 0:
        msg = 'n_samples_orig must be positive'
        raise ValueError(msg)
    for key in (
        'ffid_values',
        'chno_values',
        'offsets_m',
        'trace_indices',
        'coarse_pick_i',
        'coarse_pick_t_sec',
        'coarse_pmax',
        'coarse_prob_summary',
    ):
        arr = np.asarray(out[key])
        if arr.ndim != 1 or int(arr.shape[0]) != n_traces:
            msg = f'{key} must be 1D with length n_traces'
            raise ValueError(msg)
    lineage = np.asarray(out['lineage'])
    if lineage.ndim != 0:
        msg = 'lineage must be scalar'
        raise ValueError(msg)
    return out


def save_robust_npz(
    path: str | Path,
    *,
    dt_sec: float,
    n_samples_orig: int,
    n_traces: int,
    ffid_values,
    chno_values,
    offsets_m,
    trace_indices,
    robust_pick_i,
    robust_pick_t_sec,
    robust_conf,
    robust_source,
    used_theoretical_mask,
    reason_mask,
    conf_prob1,
    conf_trend1,
    conf_rs1,
    lineage,
    trend_center_i=None,
    trend_center_t_sec=None,
    fine_center_i=None,
    fine_center_t_sec=None,
) -> Path:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_traces_int = int(n_traces)
    n_samples_orig_int = int(n_samples_orig)
    if n_traces_int <= 0:
        msg = 'n_traces must be positive'
        raise ValueError(msg)
    if n_samples_orig_int <= 0:
        msg = 'n_samples_orig must be positive'
        raise ValueError(msg)

    arrays = {
        'dt_sec': _coerce_scalar('dt_sec', dt_sec, dtype=np.float32),
        'n_samples_orig': _coerce_scalar(
            'n_samples_orig',
            n_samples_orig_int,
            dtype=np.int32,
        ),
        'n_traces': _coerce_scalar('n_traces', n_traces_int, dtype=np.int32),
        'ffid_values': _coerce_vector(
            'ffid_values',
            ffid_values,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'chno_values': _coerce_vector(
            'chno_values',
            chno_values,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'offsets_m': _coerce_vector(
            'offsets_m',
            offsets_m,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'trace_indices': _coerce_vector(
            'trace_indices',
            trace_indices,
            dtype=np.int64,
            length=n_traces_int,
        ),
        'robust_pick_i': _coerce_vector(
            'robust_pick_i',
            robust_pick_i,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'robust_pick_t_sec': _coerce_vector(
            'robust_pick_t_sec',
            robust_pick_t_sec,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'robust_conf': _coerce_vector(
            'robust_conf',
            robust_conf,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'robust_source': _coerce_vector(
            'robust_source',
            robust_source,
            dtype=np.uint8,
            length=n_traces_int,
        ),
        'used_theoretical_mask': _coerce_bool_vector(
            'used_theoretical_mask',
            used_theoretical_mask,
            length=n_traces_int,
        ),
        'reason_mask': _coerce_vector(
            'reason_mask',
            reason_mask,
            dtype=np.uint8,
            length=n_traces_int,
        ),
        'conf_prob1': _coerce_vector(
            'conf_prob1',
            conf_prob1,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'conf_trend1': _coerce_vector(
            'conf_trend1',
            conf_trend1,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'conf_rs1': _coerce_vector(
            'conf_rs1',
            conf_rs1,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'lineage': _coerce_lineage(lineage),
    }

    optional_center_values = {
        'trend_center_i': trend_center_i,
        'trend_center_t_sec': trend_center_t_sec,
        'fine_center_i': fine_center_i,
        'fine_center_t_sec': fine_center_t_sec,
    }
    for pick_key, time_key in (
        ('trend_center_i', 'trend_center_t_sec'),
        ('fine_center_i', 'fine_center_t_sec'),
    ):
        pick_value = optional_center_values[pick_key]
        time_value = optional_center_values[time_key]
        if pick_value is None and time_value is None:
            continue
        if pick_value is None or time_value is None:
            msg = f'{pick_key} and {time_key} must be provided together'
            raise ValueError(msg)
        arrays[pick_key] = _coerce_vector(
            pick_key,
            pick_value,
            dtype=np.int32,
            length=n_traces_int,
        )
        arrays[time_key] = _coerce_vector(
            time_key,
            time_value,
            dtype=np.float32,
            length=n_traces_int,
        )
        _validate_pick_time_pair(
            pick_key=pick_key,
            time_key=time_key,
            pick_i=arrays[pick_key],
            pick_t_sec=arrays[time_key],
            dt_sec=float(arrays['dt_sec'].item()),
            n_samples_orig=n_samples_orig_int,
        )

    robust_pick_i_arr = arrays['robust_pick_i']
    if np.any(robust_pick_i_arr < 0) or np.any(robust_pick_i_arr >= n_samples_orig_int):
        msg = 'robust_pick_i must lie in [0, n_samples_orig)'
        raise ValueError(msg)

    offsets_arr = arrays['offsets_m']
    if not np.all(np.isfinite(offsets_arr)):
        msg = 'offsets_m must be finite'
        raise ValueError(msg)

    if not np.all(np.isfinite(arrays['robust_pick_t_sec'])):
        msg = 'robust_pick_t_sec must be finite'
        raise ValueError(msg)
    for key in ('robust_conf', 'conf_prob1', 'conf_trend1', 'conf_rs1'):
        _validate_unit_interval(key, arrays[key])

    valid_sources = {
        ROBUST_SOURCE_COARSE_OBSERVED,
        ROBUST_SOURCE_THEORETICAL,
        ROBUST_SOURCE_TREND_FILL,
    }
    if not set(np.unique(arrays['robust_source']).tolist()).issubset(valid_sources):
        msg = 'robust_source contains unsupported values'
        raise ValueError(msg)
    if np.any(
        arrays['used_theoretical_mask']
        & (arrays['robust_source'] != ROBUST_SOURCE_THEORETICAL)
    ):
        msg = 'used_theoretical_mask requires robust_source == 1'
        raise ValueError(msg)

    np.savez_compressed(out_path, **arrays)
    return out_path


def load_robust_npz(path: str | Path) -> dict[str, np.ndarray]:
    npz_path = Path(path).expanduser().resolve()
    if not npz_path.is_file():
        msg = f'robust npz not found: {npz_path}'
        raise FileNotFoundError(msg)

    with np.load(npz_path, allow_pickle=False) as z:
        missing = [key for key in ROBUST_REQUIRED_KEYS if key not in z.files]
        if missing:
            msg = f'robust npz missing keys: {missing}'
            raise KeyError(msg)
        out = {key: z[key] for key in z.files}

    n_traces = int(np.asarray(out['n_traces']).item())
    n_samples_orig = int(np.asarray(out['n_samples_orig']).item())
    if n_traces <= 0:
        msg = 'n_traces must be positive'
        raise ValueError(msg)
    if n_samples_orig <= 0:
        msg = 'n_samples_orig must be positive'
        raise ValueError(msg)

    _require_exact_dtype('dt_sec', np.asarray(out['dt_sec']), dtype=np.float32)
    _require_exact_dtype('n_samples_orig', np.asarray(out['n_samples_orig']), dtype=np.int32)
    _require_exact_dtype('n_traces', np.asarray(out['n_traces']), dtype=np.int32)

    vector_specs = (
        ('ffid_values', np.int32),
        ('chno_values', np.int32),
        ('offsets_m', np.float32),
        ('trace_indices', np.int64),
        ('robust_pick_i', np.int32),
        ('robust_pick_t_sec', np.float32),
        ('robust_conf', np.float32),
        ('robust_source', np.uint8),
        ('used_theoretical_mask', np.bool_),
        ('reason_mask', np.uint8),
        ('conf_prob1', np.float32),
        ('conf_trend1', np.float32),
        ('conf_rs1', np.float32),
    )
    for key, dtype in vector_specs:
        arr = np.asarray(out[key])
        if arr.ndim != 1 or int(arr.shape[0]) != n_traces:
            msg = f'{key} must be 1D with length n_traces'
            raise ValueError(msg)
        _require_exact_dtype(key, arr, dtype=dtype)

    for pick_key, time_key in (
        ('trend_center_i', 'trend_center_t_sec'),
        ('fine_center_i', 'fine_center_t_sec'),
    ):
        if pick_key not in out and time_key not in out:
            continue
        if pick_key not in out or time_key not in out:
            msg = f'robust npz must contain both {pick_key} and {time_key}'
            raise KeyError(msg)
        pick_i = np.asarray(out[pick_key])
        pick_t_sec = np.asarray(out[time_key])
        if pick_i.ndim != 1 or int(pick_i.shape[0]) != n_traces:
            msg = f'{pick_key} must be 1D with length n_traces'
            raise ValueError(msg)
        if pick_t_sec.ndim != 1 or int(pick_t_sec.shape[0]) != n_traces:
            msg = f'{time_key} must be 1D with length n_traces'
            raise ValueError(msg)
        _require_exact_dtype(pick_key, pick_i, dtype=np.int32)
        _require_exact_dtype(time_key, pick_t_sec, dtype=np.float32)
        _validate_pick_time_pair(
            pick_key=pick_key,
            time_key=time_key,
            pick_i=pick_i,
            pick_t_sec=pick_t_sec,
            dt_sec=float(np.asarray(out['dt_sec']).item()),
            n_samples_orig=n_samples_orig,
        )

    if np.any(np.asarray(out['robust_pick_i']) < 0) or np.any(
        np.asarray(out['robust_pick_i']) >= n_samples_orig
    ):
        msg = 'robust_pick_i must lie in [0, n_samples_orig)'
        raise ValueError(msg)

    for key in ('robust_conf', 'conf_prob1', 'conf_trend1', 'conf_rs1'):
        _validate_unit_interval(key, np.asarray(out[key]))

    lineage = np.asarray(out['lineage'])
    if lineage.ndim != 0:
        msg = 'lineage must be scalar'
        raise ValueError(msg)
    return out


def validate_fine_result_payload(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not isinstance(payload, dict):
        msg = 'payload must be dict'
        raise TypeError(msg)

    missing = [key for key in FINE_RESULT_REQUIRED_KEYS if key not in payload]
    if missing:
        msg = f'fine result payload missing keys: {missing}'
        raise KeyError(msg)

    dt_sec = np.float32(_require_payload_scalar(payload, 'dt_sec', dtype=np.float32).item())
    n_samples_orig = int(
        _require_payload_scalar(payload, 'n_samples_orig', dtype=np.int32).item()
    )
    n_traces = int(_require_payload_scalar(payload, 'n_traces', dtype=np.int32).item())
    if n_traces <= 0:
        msg = 'n_traces must be positive'
        raise ValueError(msg)
    if n_samples_orig <= 0:
        msg = 'n_samples_orig must be positive'
        raise ValueError(msg)

    vector_specs = (
        ('trace_indices', np.int64),
        ('fine_pick_local_i', np.int32),
        ('fine_pick_local_f', np.float32),
        ('fine_pmax', np.float32),
        ('final_pick_i', np.int32),
        ('final_pick_f', np.float32),
        ('final_pick_t_sec', np.float32),
        ('final_conf', np.float32),
        ('window_start_i', np.int32),
        ('window_end_i', np.int32),
    )
    arrays = {
        key: _require_payload_vector(payload, key, dtype=dtype, length=n_traces)
        for key, dtype in vector_specs
    }

    trace_indices = arrays['trace_indices']
    if not np.array_equal(trace_indices, np.arange(n_traces, dtype=np.int64)):
        msg = 'trace_indices must equal np.arange(n_traces)'
        raise ValueError(msg)

    fine_pick_local_i = arrays['fine_pick_local_i']
    if np.any(fine_pick_local_i < 0) or np.any(fine_pick_local_i >= 256):
        msg = 'fine_pick_local_i must lie in [0, 256)'
        raise ValueError(msg)

    fine_pick_local_f = arrays['fine_pick_local_f']
    expected_local_f = fine_pick_local_i.astype(np.float32)
    if not np.array_equal(fine_pick_local_f, expected_local_f):
        msg = 'fine_pick_local_f must equal float32(fine_pick_local_i)'
        raise ValueError(msg)

    window_start_i = arrays['window_start_i']
    window_end_i = arrays['window_end_i']
    if np.any((window_end_i - window_start_i) != 256):
        msg = 'window_end_i - window_start_i must equal 256 for every trace'
        raise ValueError(msg)

    expected_final_pick_f = window_start_i.astype(np.float32) + fine_pick_local_f
    if not np.allclose(
        arrays['final_pick_f'],
        expected_final_pick_f,
        atol=1.0e-6,
        rtol=0.0,
    ):
        msg = 'final_pick_f must equal window_start_i + fine_pick_local_f'
        raise ValueError(msg)

    expected_final_pick_i = (
        window_start_i.astype(np.int64) + fine_pick_local_i.astype(np.int64)
    ).astype(np.int32)
    if not np.array_equal(arrays['final_pick_i'], expected_final_pick_i):
        msg = 'final_pick_i must equal window_start_i + fine_pick_local_i'
        raise ValueError(msg)

    expected_final_pick_t_sec = expected_final_pick_f * dt_sec
    if not np.allclose(
        arrays['final_pick_t_sec'],
        expected_final_pick_t_sec,
        atol=1.0e-6,
        rtol=0.0,
    ):
        msg = 'final_pick_t_sec must equal final_pick_f * dt_sec'
        raise ValueError(msg)

    _validate_unit_interval('fine_pmax', arrays['fine_pmax'])
    _validate_unit_interval('final_conf', arrays['final_conf'])
    if not np.allclose(
        arrays['final_conf'],
        arrays['fine_pmax'],
        atol=1.0e-6,
        rtol=0.0,
    ):
        msg = 'final_conf must equal fine_pmax for fine local infer payload'
        raise ValueError(msg)

    return payload


def validate_fbpick_final_payload(
    payload: dict[str, np.ndarray],
    *,
    high_conf_threshold: float | None = None,
) -> dict[str, np.ndarray]:
    if not isinstance(payload, dict):
        msg = 'payload must be dict'
        raise TypeError(msg)

    missing = [key for key in FINAL_REQUIRED_KEYS if key not in payload]
    if missing:
        msg = f'fbpick final payload missing keys: {missing}'
        raise KeyError(msg)

    dt_sec = np.float32(_require_payload_scalar(payload, 'dt_sec', dtype=np.float32).item())
    n_samples_orig = int(
        _require_payload_scalar(payload, 'n_samples_orig', dtype=np.int32).item()
    )
    n_traces = int(_require_payload_scalar(payload, 'n_traces', dtype=np.int32).item())
    if n_traces <= 0:
        msg = 'n_traces must be positive'
        raise ValueError(msg)
    if n_samples_orig <= 0:
        msg = 'n_samples_orig must be positive'
        raise ValueError(msg)

    vector_specs = (
        ('ffid_values', np.int32),
        ('chno_values', np.int32),
        ('offsets_m', np.float32),
        ('trace_indices', np.int64),
        ('coarse_pick_i', np.int32),
        ('coarse_pmax', np.float32),
        ('robust_pick_i', np.int32),
        ('robust_conf', np.float32),
        ('robust_source', np.uint8),
        ('used_theoretical_mask', np.bool_),
        ('reason_mask', np.uint8),
        ('window_start_i', np.int32),
        ('window_end_i', np.int32),
        ('fine_pick_local_f', np.float32),
        ('fine_pick_local_i', np.int32),
        ('fine_pmax', np.float32),
        ('final_pick_f', np.float32),
        ('final_pick_i', np.int32),
        ('final_pick_t_sec', np.float32),
        ('final_conf', np.float32),
        ('high_conf_mask', np.bool_),
        ('reject_mask', np.bool_),
    )
    arrays = {
        key: _require_payload_vector(payload, key, dtype=dtype, length=n_traces)
        for key, dtype in vector_specs
    }

    lineage = np.asarray(payload['lineage'])
    if lineage.ndim != 0:
        msg = 'lineage must be scalar'
        raise ValueError(msg)

    if not np.all(np.isfinite(arrays['offsets_m'])):
        msg = 'offsets_m must be finite'
        raise ValueError(msg)
    if not np.all(np.isfinite(arrays['final_pick_t_sec'])):
        msg = 'final_pick_t_sec must be finite'
        raise ValueError(msg)

    coarse_pick_i = arrays['coarse_pick_i']
    if np.any(coarse_pick_i < 0) or np.any(coarse_pick_i >= n_samples_orig):
        msg = 'coarse_pick_i must lie in [0, n_samples_orig)'
        raise ValueError(msg)

    robust_pick_i = arrays['robust_pick_i']
    if np.any(robust_pick_i < 0) or np.any(robust_pick_i >= n_samples_orig):
        msg = 'robust_pick_i must lie in [0, n_samples_orig)'
        raise ValueError(msg)

    for key in ('coarse_pmax', 'robust_conf', 'fine_pmax', 'final_conf'):
        _validate_unit_interval(key, arrays[key])

    valid_sources = {
        ROBUST_SOURCE_COARSE_OBSERVED,
        ROBUST_SOURCE_THEORETICAL,
        ROBUST_SOURCE_TREND_FILL,
    }
    robust_source = arrays['robust_source']
    if not set(np.unique(robust_source).tolist()).issubset(valid_sources):
        msg = 'robust_source contains unsupported values'
        raise ValueError(msg)
    if np.any(
        arrays['used_theoretical_mask']
        & (robust_source != np.uint8(ROBUST_SOURCE_THEORETICAL))
    ):
        msg = 'used_theoretical_mask requires robust_source == 1'
        raise ValueError(msg)

    fine_pick_local_i = arrays['fine_pick_local_i']
    if np.any(fine_pick_local_i < 0) or np.any(fine_pick_local_i >= 256):
        msg = 'fine_pick_local_i must lie in [0, 256)'
        raise ValueError(msg)

    fine_pick_local_f = arrays['fine_pick_local_f']
    expected_local_f = fine_pick_local_i.astype(np.float32)
    if not np.array_equal(fine_pick_local_f, expected_local_f):
        msg = 'fine_pick_local_f must equal float32(fine_pick_local_i)'
        raise ValueError(msg)

    window_start_i = arrays['window_start_i']
    window_end_i = arrays['window_end_i']
    if np.any((window_end_i - window_start_i) != 255):
        msg = 'window_end_i must equal window_start_i + 255 for every trace'
        raise ValueError(msg)

    expected_final_pick_f = window_start_i.astype(np.float32) + fine_pick_local_f
    if not np.allclose(
        arrays['final_pick_f'],
        expected_final_pick_f,
        atol=1.0e-6,
        rtol=0.0,
    ):
        msg = 'final_pick_f must equal window_start_i + fine_pick_local_f'
        raise ValueError(msg)

    expected_final_pick_i = (
        window_start_i.astype(np.int64) + fine_pick_local_i.astype(np.int64)
    ).astype(np.int32)
    if not np.array_equal(arrays['final_pick_i'], expected_final_pick_i):
        msg = 'final_pick_i must equal window_start_i + fine_pick_local_i'
        raise ValueError(msg)

    expected_final_pick_t_sec = expected_final_pick_f * dt_sec
    if not np.allclose(
        arrays['final_pick_t_sec'],
        expected_final_pick_t_sec,
        atol=1.0e-6,
        rtol=0.0,
    ):
        msg = 'final_pick_t_sec must equal final_pick_f * dt_sec'
        raise ValueError(msg)

    expected_final_conf = np.clip(
        np.minimum(arrays['fine_pmax'], arrays['robust_conf']),
        0.0,
        1.0,
    ).astype(np.float32)
    if not np.allclose(
        arrays['final_conf'],
        expected_final_conf,
        atol=1.0e-6,
        rtol=0.0,
    ):
        msg = 'final_conf must equal clip(min(fine_pmax, robust_conf), 0, 1)'
        raise ValueError(msg)

    expected_reject_mask = (
        (robust_source != np.uint8(ROBUST_SOURCE_COARSE_OBSERVED))
        | (arrays['reason_mask'] != 0)
    )
    if not np.array_equal(arrays['reject_mask'], expected_reject_mask):
        msg = 'reject_mask must equal (robust_source != 0) OR (reason_mask != 0)'
        raise ValueError(msg)

    if high_conf_threshold is not None:
        threshold = _validate_high_conf_threshold(high_conf_threshold)
        expected_high_conf_mask = (
            (~expected_reject_mask) & (arrays['final_conf'] >= threshold)
        )
        if not np.array_equal(arrays['high_conf_mask'], expected_high_conf_mask):
            msg = (
                'high_conf_mask must equal (~reject_mask) AND '
                '(final_conf >= infer.high_conf_threshold)'
            )
            raise ValueError(msg)

    return payload


def save_fbpick_final_npz(
    path: str | Path,
    *,
    dt_sec: float,
    n_samples_orig: int,
    n_traces: int,
    ffid_values,
    chno_values,
    offsets_m,
    trace_indices,
    coarse_pick_i,
    coarse_pmax,
    robust_pick_i,
    robust_conf,
    robust_source,
    used_theoretical_mask,
    reason_mask,
    window_start_i,
    window_end_i,
    fine_pick_local_f,
    fine_pick_local_i,
    fine_pmax,
    final_pick_f,
    final_pick_i,
    final_pick_t_sec,
    final_conf,
    high_conf_mask,
    reject_mask,
    lineage,
) -> Path:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_traces_int = int(n_traces)
    n_samples_orig_int = int(n_samples_orig)
    if n_traces_int <= 0:
        msg = 'n_traces must be positive'
        raise ValueError(msg)
    if n_samples_orig_int <= 0:
        msg = 'n_samples_orig must be positive'
        raise ValueError(msg)

    arrays = {
        'dt_sec': _coerce_scalar('dt_sec', dt_sec, dtype=np.float32),
        'n_samples_orig': _coerce_scalar(
            'n_samples_orig',
            n_samples_orig_int,
            dtype=np.int32,
        ),
        'n_traces': _coerce_scalar('n_traces', n_traces_int, dtype=np.int32),
        'ffid_values': _coerce_vector(
            'ffid_values',
            ffid_values,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'chno_values': _coerce_vector(
            'chno_values',
            chno_values,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'offsets_m': _coerce_vector(
            'offsets_m',
            offsets_m,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'trace_indices': _coerce_vector(
            'trace_indices',
            trace_indices,
            dtype=np.int64,
            length=n_traces_int,
        ),
        'coarse_pick_i': _coerce_vector(
            'coarse_pick_i',
            coarse_pick_i,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'coarse_pmax': _coerce_vector(
            'coarse_pmax',
            coarse_pmax,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'robust_pick_i': _coerce_vector(
            'robust_pick_i',
            robust_pick_i,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'robust_conf': _coerce_vector(
            'robust_conf',
            robust_conf,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'robust_source': _coerce_vector(
            'robust_source',
            robust_source,
            dtype=np.uint8,
            length=n_traces_int,
        ),
        'used_theoretical_mask': _coerce_bool_vector(
            'used_theoretical_mask',
            used_theoretical_mask,
            length=n_traces_int,
        ),
        'reason_mask': _coerce_vector(
            'reason_mask',
            reason_mask,
            dtype=np.uint8,
            length=n_traces_int,
        ),
        'window_start_i': _coerce_vector(
            'window_start_i',
            window_start_i,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'window_end_i': _coerce_vector(
            'window_end_i',
            window_end_i,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'fine_pick_local_f': _coerce_vector(
            'fine_pick_local_f',
            fine_pick_local_f,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'fine_pick_local_i': _coerce_vector(
            'fine_pick_local_i',
            fine_pick_local_i,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'fine_pmax': _coerce_vector(
            'fine_pmax',
            fine_pmax,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'final_pick_f': _coerce_vector(
            'final_pick_f',
            final_pick_f,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'final_pick_i': _coerce_vector(
            'final_pick_i',
            final_pick_i,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'final_pick_t_sec': _coerce_vector(
            'final_pick_t_sec',
            final_pick_t_sec,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'final_conf': _coerce_vector(
            'final_conf',
            final_conf,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'high_conf_mask': _coerce_bool_vector(
            'high_conf_mask',
            high_conf_mask,
            length=n_traces_int,
        ),
        'reject_mask': _coerce_bool_vector(
            'reject_mask',
            reject_mask,
            length=n_traces_int,
        ),
        'lineage': _coerce_lineage(lineage),
    }

    validate_fbpick_final_payload(arrays)
    np.savez_compressed(out_path, **arrays)
    return out_path


def load_fbpick_final_npz(path: str | Path) -> dict[str, np.ndarray]:
    npz_path = Path(path).expanduser().resolve()
    if not npz_path.is_file():
        msg = f'fbpick final npz not found: {npz_path}'
        raise FileNotFoundError(msg)

    with np.load(npz_path, allow_pickle=False) as z:
        missing = [key for key in FINAL_REQUIRED_KEYS if key not in z.files]
        if missing:
            msg = f'fbpick final npz missing keys: {missing}'
            raise KeyError(msg)
        out = {key: z[key] for key in z.files}

    validate_fbpick_final_payload(out)
    return out


def _extract_alignment_scalars(
    payload: dict[str, np.ndarray],
    *,
    payload_name: str,
) -> tuple[int, np.float32, int]:
    if not isinstance(payload, dict):
        msg = f'{payload_name} must be dict'
        raise TypeError(msg)
    n_traces = int(_require_payload_scalar(payload, 'n_traces', dtype=np.int32).item())
    dt_sec = np.float32(_require_payload_scalar(payload, 'dt_sec', dtype=np.float32).item())
    n_samples_orig = int(
        _require_payload_scalar(payload, 'n_samples_orig', dtype=np.int32).item()
    )
    if n_traces <= 0:
        msg = f'{payload_name} n_traces must be positive'
        raise ValueError(msg)
    if n_samples_orig <= 0:
        msg = f'{payload_name} n_samples_orig must be positive'
        raise ValueError(msg)
    return n_traces, dt_sec, n_samples_orig


def _validate_coarse_payload_for_final(payload: dict[str, np.ndarray]) -> None:
    if not isinstance(payload, dict):
        msg = 'coarse_payload must be dict'
        raise TypeError(msg)
    missing = [key for key in COARSE_REQUIRED_KEYS if key not in payload]
    if missing:
        msg = f'coarse payload missing keys: {missing}'
        raise KeyError(msg)

    n_traces = int(_require_payload_scalar(payload, 'n_traces', dtype=np.int32).item())
    n_samples_orig = int(
        _require_payload_scalar(payload, 'n_samples_orig', dtype=np.int32).item()
    )
    if n_traces <= 0:
        msg = 'coarse payload n_traces must be positive'
        raise ValueError(msg)
    if n_samples_orig <= 0:
        msg = 'coarse payload n_samples_orig must be positive'
        raise ValueError(msg)

    _require_payload_scalar(payload, 'dt_sec', dtype=np.float32)
    arrays = {
        'ffid_values': _require_payload_vector(
            payload,
            'ffid_values',
            dtype=np.int32,
            length=n_traces,
        ),
        'chno_values': _require_payload_vector(
            payload,
            'chno_values',
            dtype=np.int32,
            length=n_traces,
        ),
        'offsets_m': _require_payload_vector(
            payload,
            'offsets_m',
            dtype=np.float32,
            length=n_traces,
        ),
        'trace_indices': _require_payload_vector(
            payload,
            'trace_indices',
            dtype=np.int64,
            length=n_traces,
        ),
        'coarse_pick_i': _require_payload_vector(
            payload,
            'coarse_pick_i',
            dtype=np.int32,
            length=n_traces,
        ),
        'coarse_pick_t_sec': _require_payload_vector(
            payload,
            'coarse_pick_t_sec',
            dtype=np.float32,
            length=n_traces,
        ),
        'coarse_pmax': _require_payload_vector(
            payload,
            'coarse_pmax',
            dtype=np.float32,
            length=n_traces,
        ),
        'coarse_prob_summary': _require_payload_vector(
            payload,
            'coarse_prob_summary',
            dtype=np.float32,
            length=n_traces,
        ),
    }
    if np.any(arrays['coarse_pick_i'] < 0) or np.any(arrays['coarse_pick_i'] >= n_samples_orig):
        msg = 'coarse payload coarse_pick_i must lie in [0, n_samples_orig)'
        raise ValueError(msg)
    _validate_unit_interval('coarse_pmax', arrays['coarse_pmax'])
    _validate_unit_interval('coarse_prob_summary', arrays['coarse_prob_summary'])
    if not np.all(np.isfinite(arrays['offsets_m'])):
        msg = 'coarse payload offsets_m must be finite'
        raise ValueError(msg)
    if not np.all(np.isfinite(arrays['coarse_pick_t_sec'])):
        msg = 'coarse payload coarse_pick_t_sec must be finite'
        raise ValueError(msg)
    if np.asarray(payload['lineage']).ndim != 0:
        msg = 'coarse payload lineage must be scalar'
        raise ValueError(msg)


def _validate_robust_payload_for_final(payload: dict[str, np.ndarray]) -> None:
    if not isinstance(payload, dict):
        msg = 'robust_payload must be dict'
        raise TypeError(msg)
    missing = [key for key in ROBUST_REQUIRED_KEYS if key not in payload]
    if missing:
        msg = f'robust payload missing keys: {missing}'
        raise KeyError(msg)

    n_traces = int(_require_payload_scalar(payload, 'n_traces', dtype=np.int32).item())
    n_samples_orig = int(
        _require_payload_scalar(payload, 'n_samples_orig', dtype=np.int32).item()
    )
    if n_traces <= 0:
        msg = 'robust payload n_traces must be positive'
        raise ValueError(msg)
    if n_samples_orig <= 0:
        msg = 'robust payload n_samples_orig must be positive'
        raise ValueError(msg)

    _require_payload_scalar(payload, 'dt_sec', dtype=np.float32)
    arrays = {
        'ffid_values': _require_payload_vector(
            payload,
            'ffid_values',
            dtype=np.int32,
            length=n_traces,
        ),
        'chno_values': _require_payload_vector(
            payload,
            'chno_values',
            dtype=np.int32,
            length=n_traces,
        ),
        'offsets_m': _require_payload_vector(
            payload,
            'offsets_m',
            dtype=np.float32,
            length=n_traces,
        ),
        'trace_indices': _require_payload_vector(
            payload,
            'trace_indices',
            dtype=np.int64,
            length=n_traces,
        ),
        'robust_pick_i': _require_payload_vector(
            payload,
            'robust_pick_i',
            dtype=np.int32,
            length=n_traces,
        ),
        'robust_pick_t_sec': _require_payload_vector(
            payload,
            'robust_pick_t_sec',
            dtype=np.float32,
            length=n_traces,
        ),
        'robust_conf': _require_payload_vector(
            payload,
            'robust_conf',
            dtype=np.float32,
            length=n_traces,
        ),
        'robust_source': _require_payload_vector(
            payload,
            'robust_source',
            dtype=np.uint8,
            length=n_traces,
        ),
        'used_theoretical_mask': _require_payload_vector(
            payload,
            'used_theoretical_mask',
            dtype=np.bool_,
            length=n_traces,
        ),
        'reason_mask': _require_payload_vector(
            payload,
            'reason_mask',
            dtype=np.uint8,
            length=n_traces,
        ),
        'conf_prob1': _require_payload_vector(
            payload,
            'conf_prob1',
            dtype=np.float32,
            length=n_traces,
        ),
        'conf_trend1': _require_payload_vector(
            payload,
            'conf_trend1',
            dtype=np.float32,
            length=n_traces,
        ),
        'conf_rs1': _require_payload_vector(
            payload,
            'conf_rs1',
            dtype=np.float32,
            length=n_traces,
        ),
    }
    if np.any(arrays['robust_pick_i'] < 0) or np.any(arrays['robust_pick_i'] >= n_samples_orig):
        msg = 'robust payload robust_pick_i must lie in [0, n_samples_orig)'
        raise ValueError(msg)
    if not np.all(np.isfinite(arrays['offsets_m'])):
        msg = 'robust payload offsets_m must be finite'
        raise ValueError(msg)
    if not np.all(np.isfinite(arrays['robust_pick_t_sec'])):
        msg = 'robust payload robust_pick_t_sec must be finite'
        raise ValueError(msg)
    for key in ('robust_conf', 'conf_prob1', 'conf_trend1', 'conf_rs1'):
        _validate_unit_interval(key, arrays[key])
    valid_sources = {
        ROBUST_SOURCE_COARSE_OBSERVED,
        ROBUST_SOURCE_THEORETICAL,
        ROBUST_SOURCE_TREND_FILL,
    }
    if not set(np.unique(arrays['robust_source']).tolist()).issubset(valid_sources):
        msg = 'robust payload robust_source contains unsupported values'
        raise ValueError(msg)
    if np.any(
        arrays['used_theoretical_mask']
        & (arrays['robust_source'] != np.uint8(ROBUST_SOURCE_THEORETICAL))
    ):
        msg = 'robust payload used_theoretical_mask requires robust_source == 1'
        raise ValueError(msg)
    if np.asarray(payload['lineage']).ndim != 0:
        msg = 'robust payload lineage must be scalar'
        raise ValueError(msg)


def build_fbpick_final_payload(
    *,
    coarse_payload: dict[str, np.ndarray],
    robust_payload: dict[str, np.ndarray],
    fine_payload: dict[str, np.ndarray],
    high_conf_threshold: float,
    lineage,
) -> dict[str, np.ndarray]:
    coarse_n_traces, coarse_dt_sec, coarse_n_samples = _extract_alignment_scalars(
        coarse_payload,
        payload_name='coarse_payload',
    )
    robust_n_traces, robust_dt_sec, robust_n_samples = _extract_alignment_scalars(
        robust_payload,
        payload_name='robust_payload',
    )
    fine_n_traces, fine_dt_sec, fine_n_samples = _extract_alignment_scalars(
        fine_payload,
        payload_name='fine_payload',
    )
    if coarse_n_traces != robust_n_traces or coarse_n_traces != fine_n_traces:
        msg = 'n_traces must match across coarse_payload, robust_payload, and fine_payload'
        raise ValueError(msg)
    if coarse_dt_sec != robust_dt_sec or coarse_dt_sec != fine_dt_sec:
        msg = 'dt_sec must match across coarse_payload, robust_payload, and fine_payload'
        raise ValueError(msg)
    if coarse_n_samples != robust_n_samples or coarse_n_samples != fine_n_samples:
        msg = (
            'n_samples_orig must match across coarse_payload, robust_payload, '
            'and fine_payload'
        )
        raise ValueError(msg)
    coarse_trace_indices = _require_payload_vector(
        coarse_payload,
        'trace_indices',
        dtype=np.int64,
        length=coarse_n_traces,
    )
    robust_trace_indices = _require_payload_vector(
        robust_payload,
        'trace_indices',
        dtype=np.int64,
        length=robust_n_traces,
    )
    fine_trace_indices = _require_payload_vector(
        fine_payload,
        'trace_indices',
        dtype=np.int64,
        length=fine_n_traces,
    )
    if not np.array_equal(coarse_trace_indices, robust_trace_indices) or not np.array_equal(
        coarse_trace_indices,
        fine_trace_indices,
    ):
        msg = 'trace_indices must match across coarse_payload, robust_payload, and fine_payload'
        raise ValueError(msg)

    _validate_coarse_payload_for_final(coarse_payload)
    _validate_robust_payload_for_final(robust_payload)
    validate_fine_result_payload(fine_payload)
    threshold = _validate_high_conf_threshold(high_conf_threshold)

    for key in ('ffid_values', 'chno_values', 'offsets_m'):
        if not np.array_equal(np.asarray(coarse_payload[key]), np.asarray(robust_payload[key])):
            msg = f'{key} must match between coarse_payload and robust_payload'
            raise ValueError(msg)

    window_start_i = np.asarray(fine_payload['window_start_i'], dtype=np.int32)
    window_end_i = (window_start_i.astype(np.int64) + 255).astype(np.int32)
    fine_pick_local_i = np.asarray(fine_payload['fine_pick_local_i'], dtype=np.int32)
    fine_pick_local_f = np.asarray(fine_payload['fine_pick_local_f'], dtype=np.float32)
    final_pick_f = window_start_i.astype(np.float32) + fine_pick_local_f
    final_pick_i = (
        window_start_i.astype(np.int64) + fine_pick_local_i.astype(np.int64)
    ).astype(np.int32)
    final_pick_t_sec = final_pick_f * coarse_dt_sec

    robust_conf = np.asarray(robust_payload['robust_conf'], dtype=np.float32)
    fine_pmax = np.asarray(fine_payload['fine_pmax'], dtype=np.float32)
    final_conf = np.clip(
        np.minimum(fine_pmax, robust_conf),
        0.0,
        1.0,
    ).astype(np.float32)

    robust_source = np.asarray(robust_payload['robust_source'], dtype=np.uint8)
    reason_mask = np.asarray(robust_payload['reason_mask'], dtype=np.uint8)
    reject_mask = (
        (robust_source != np.uint8(ROBUST_SOURCE_COARSE_OBSERVED))
        | (reason_mask != 0)
    ).astype(np.bool_)
    high_conf_mask = ((~reject_mask) & (final_conf >= threshold)).astype(np.bool_)

    out = {
        'dt_sec': np.asarray(coarse_dt_sec, dtype=np.float32),
        'n_samples_orig': np.asarray(coarse_n_samples, dtype=np.int32),
        'n_traces': np.asarray(coarse_n_traces, dtype=np.int32),
        'ffid_values': np.asarray(coarse_payload['ffid_values'], dtype=np.int32),
        'chno_values': np.asarray(coarse_payload['chno_values'], dtype=np.int32),
        'offsets_m': np.asarray(coarse_payload['offsets_m'], dtype=np.float32),
        'trace_indices': np.asarray(coarse_trace_indices, dtype=np.int64),
        'coarse_pick_i': np.asarray(coarse_payload['coarse_pick_i'], dtype=np.int32),
        'coarse_pmax': np.asarray(coarse_payload['coarse_pmax'], dtype=np.float32),
        'robust_pick_i': np.asarray(robust_payload['robust_pick_i'], dtype=np.int32),
        'robust_conf': robust_conf,
        'robust_source': robust_source,
        'used_theoretical_mask': np.asarray(
            robust_payload['used_theoretical_mask'],
            dtype=np.bool_,
        ),
        'reason_mask': reason_mask,
        'window_start_i': window_start_i,
        'window_end_i': window_end_i,
        'fine_pick_local_f': fine_pick_local_f,
        'fine_pick_local_i': fine_pick_local_i,
        'fine_pmax': fine_pmax,
        'final_pick_f': final_pick_f.astype(np.float32, copy=False),
        'final_pick_i': final_pick_i,
        'final_pick_t_sec': final_pick_t_sec.astype(np.float32, copy=False),
        'final_conf': final_conf,
        'high_conf_mask': high_conf_mask,
        'reject_mask': reject_mask,
        'lineage': _coerce_lineage(lineage),
    }
    validate_fbpick_final_payload(out, high_conf_threshold=float(threshold))
    return out
