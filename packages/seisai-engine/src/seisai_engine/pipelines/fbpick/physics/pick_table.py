from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from seisai_engine.pipelines.fbpick.common import COARSE_REQUIRED_KEYS

__all__ = ['CoarsePickTable', 'normalize_coarse_pick_table']


@dataclass(frozen=True)
class CoarsePickTable:
    n_traces: int
    n_samples_orig: int
    dt_scalar_sec: float
    shot_id: np.ndarray
    trace_id: np.ndarray
    ffid: np.ndarray
    chno: np.ndarray
    offset_m: np.ndarray
    dt_sec: np.ndarray
    coarse_pick_i: np.ndarray
    coarse_pick_t_sec: np.ndarray
    coarse_pmax: np.ndarray


def _require_key(data: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        msg = f'coarse npz missing key: {key}'
        raise KeyError(msg)
    return np.asarray(data[key])


def _require_scalar(name: str, value: np.ndarray, *, dtype, positive: bool = False):
    arr = np.asarray(value)
    if arr.ndim != 0:
        msg = f'{name} must be scalar'
        raise ValueError(msg)
    if arr.dtype != np.dtype(dtype):
        msg = f'{name} dtype must be {np.dtype(dtype)}, got {arr.dtype}'
        raise ValueError(msg)
    scalar = arr.item()
    if positive and scalar <= 0:
        msg = f'{name} must be positive'
        raise ValueError(msg)
    return scalar


def _require_vector(
    name: str,
    value: np.ndarray,
    *,
    dtype,
    length: int,
) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        msg = f'{name} must be 1D'
        raise ValueError(msg)
    if arr.dtype != np.dtype(dtype):
        msg = f'{name} dtype must be {np.dtype(dtype)}, got {arr.dtype}'
        raise ValueError(msg)
    if int(arr.shape[0]) != int(length):
        msg = f'{name} length {int(arr.shape[0])} != n_traces {int(length)}'
        raise ValueError(msg)
    return arr


def normalize_coarse_pick_table(data: Mapping[str, np.ndarray]) -> CoarsePickTable:
    if not isinstance(data, Mapping):
        msg = 'data must be a mapping'
        raise TypeError(msg)

    for key in COARSE_REQUIRED_KEYS:
        _require_key(data, key)

    dt_scalar_sec = float(
        _require_scalar('dt_sec', _require_key(data, 'dt_sec'), dtype=np.float32, positive=True)
    )
    n_samples_orig = int(
        _require_scalar(
            'n_samples_orig',
            _require_key(data, 'n_samples_orig'),
            dtype=np.int32,
            positive=True,
        )
    )
    n_traces = int(
        _require_scalar(
            'n_traces',
            _require_key(data, 'n_traces'),
            dtype=np.int32,
            positive=True,
        )
    )

    ffid = _require_vector('ffid_values', _require_key(data, 'ffid_values'), dtype=np.int32, length=n_traces)
    chno = _require_vector('chno_values', _require_key(data, 'chno_values'), dtype=np.int32, length=n_traces)
    offset_m = _require_vector('offsets_m', _require_key(data, 'offsets_m'), dtype=np.float32, length=n_traces)
    trace_id = _require_vector(
        'trace_indices',
        _require_key(data, 'trace_indices'),
        dtype=np.int64,
        length=n_traces,
    )
    coarse_pick_i = _require_vector(
        'coarse_pick_i',
        _require_key(data, 'coarse_pick_i'),
        dtype=np.int32,
        length=n_traces,
    )
    coarse_pick_t_sec = _require_vector(
        'coarse_pick_t_sec',
        _require_key(data, 'coarse_pick_t_sec'),
        dtype=np.float32,
        length=n_traces,
    )
    coarse_pmax = _require_vector(
        'coarse_pmax',
        _require_key(data, 'coarse_pmax'),
        dtype=np.float32,
        length=n_traces,
    )

    if np.any(coarse_pick_i < 0) or np.any(coarse_pick_i >= n_samples_orig):
        msg = 'coarse_pick_i must lie in [0, n_samples_orig)'
        raise ValueError(msg)
    if not np.all(np.isfinite(offset_m)):
        msg = 'offsets_m must be finite'
        raise ValueError(msg)
    if not np.all(np.isfinite(coarse_pmax)):
        msg = 'coarse_pmax must be finite'
        raise ValueError(msg)

    expected_t_sec = coarse_pick_i.astype(np.float32) * np.float32(dt_scalar_sec)
    atol = max(1.0e-6, float(dt_scalar_sec) * 1.0e-4)
    if not np.allclose(coarse_pick_t_sec, expected_t_sec, atol=atol, rtol=0.0):
        msg = 'coarse_pick_t_sec must match coarse_pick_i * dt_sec'
        raise ValueError(msg)

    dt_col = np.full((n_traces,), np.float32(dt_scalar_sec), dtype=np.float32)
    return CoarsePickTable(
        n_traces=n_traces,
        n_samples_orig=n_samples_orig,
        dt_scalar_sec=dt_scalar_sec,
        shot_id=ffid.copy(),
        trace_id=trace_id.copy(),
        ffid=ffid.copy(),
        chno=chno.copy(),
        offset_m=offset_m.copy(),
        dt_sec=dt_col,
        coarse_pick_i=coarse_pick_i.copy(),
        coarse_pick_t_sec=expected_t_sec.astype(np.float32, copy=False),
        coarse_pmax=coarse_pmax.copy(),
    )
