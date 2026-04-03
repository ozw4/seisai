from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import (
    ROBUST_SOURCE_COARSE_OBSERVED,
    ROBUST_SOURCE_THEORETICAL,
    ROBUST_SOURCE_TREND_FILL,
)

__all__ = [
    'COARSE_REQUIRED_KEYS',
    'FINE_RESULT_REQUIRED_KEYS',
    'ROBUST_REQUIRED_KEYS',
    'load_coarse_npz',
    'load_robust_npz',
    'save_coarse_npz',
    'save_robust_npz',
    'validate_fine_result_payload',
]


COARSE_REQUIRED_KEYS = (
    'dt_sec',
    'n_samples_orig',
    'n_traces',
    'ffid_values',
    'chno_values',
    'offsets_m',
    'trace_indices',
    'coarse_pick_i',
    'coarse_pick_t_sec',
    'coarse_pmax',
    'coarse_prob_summary',
    'lineage',
)

ROBUST_REQUIRED_KEYS = (
    'dt_sec',
    'n_samples_orig',
    'n_traces',
    'ffid_values',
    'chno_values',
    'offsets_m',
    'trace_indices',
    'robust_pick_i',
    'robust_pick_t_sec',
    'robust_conf',
    'robust_source',
    'used_theoretical_mask',
    'reason_mask',
    'conf_prob1',
    'conf_trend1',
    'conf_rs1',
    'lineage',
)

FINE_RESULT_REQUIRED_KEYS = (
    'dt_sec',
    'n_samples_orig',
    'n_traces',
    'trace_indices',
    'fine_pick_local_i',
    'fine_pick_local_f',
    'fine_pmax',
    'final_pick_i',
    'final_pick_f',
    'final_pick_t_sec',
    'final_conf',
    'window_start_i',
    'window_end_i',
)


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


def _require_exact_dtype(name: str, arr: np.ndarray, *, dtype) -> None:
    if arr.dtype != np.dtype(dtype):
        msg = f'{name} dtype must be {np.dtype(dtype)}, got {arr.dtype}'
        raise ValueError(msg)


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
    if np.any(arrays['used_theoretical_mask'] & (arrays['robust_source'] != ROBUST_SOURCE_THEORETICAL)):
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

    n_traces = int(np.asarray(payload['n_traces']).item())
    n_samples_orig = int(np.asarray(payload['n_samples_orig']).item())
    if n_traces <= 0:
        msg = 'n_traces must be positive'
        raise ValueError(msg)
    if n_samples_orig <= 0:
        msg = 'n_samples_orig must be positive'
        raise ValueError(msg)

    _require_exact_dtype('dt_sec', np.asarray(payload['dt_sec']), dtype=np.float32)
    _require_exact_dtype('n_samples_orig', np.asarray(payload['n_samples_orig']), dtype=np.int32)
    _require_exact_dtype('n_traces', np.asarray(payload['n_traces']), dtype=np.int32)

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
    for key, dtype in vector_specs:
        arr = np.asarray(payload[key])
        if arr.ndim != 1 or int(arr.shape[0]) != n_traces:
            msg = f'{key} must be 1D with length n_traces'
            raise ValueError(msg)
        _require_exact_dtype(key, arr, dtype=dtype)

    trace_indices = np.asarray(payload['trace_indices'], dtype=np.int64)
    if not np.array_equal(trace_indices, np.arange(n_traces, dtype=np.int64)):
        msg = 'trace_indices must equal np.arange(n_traces)'
        raise ValueError(msg)

    fine_pick_local_i = np.asarray(payload['fine_pick_local_i'], dtype=np.int32)
    if np.any(fine_pick_local_i < 0) or np.any(fine_pick_local_i >= 256):
        msg = 'fine_pick_local_i must lie in [0, 256)'
        raise ValueError(msg)

    final_pick_i = np.asarray(payload['final_pick_i'], dtype=np.int32)
    if np.any(final_pick_i < 0) or np.any(final_pick_i >= n_samples_orig):
        msg = 'final_pick_i must lie in [0, n_samples_orig)'
        raise ValueError(msg)

    window_start_i = np.asarray(payload['window_start_i'], dtype=np.int32)
    window_end_i = np.asarray(payload['window_end_i'], dtype=np.int32)
    if np.any((window_end_i - window_start_i) != 256):
        msg = 'window_end_i - window_start_i must equal 256 for every trace'
        raise ValueError(msg)

    _validate_unit_interval('fine_pmax', np.asarray(payload['fine_pmax'], dtype=np.float32))
    _validate_unit_interval('final_conf', np.asarray(payload['final_conf'], dtype=np.float32))

    return payload
