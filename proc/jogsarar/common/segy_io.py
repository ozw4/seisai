from __future__ import annotations

from pathlib import Path

import numpy as np
import segyio


def read_trace_field(
    src: segyio.SegyFile,
    field,
    *,
    dtype,
    name: str = 'trace_field',
) -> np.ndarray:
    """Read a SEG-Y trace header field for all traces."""
    n_tr = int(src.tracecount)
    v = np.asarray(src.attributes(field)[:], dtype=dtype)
    if v.ndim != 1 or v.shape[0] != n_tr:
        msg = f'{name} must be (n_traces,), got {v.shape}, n_traces={n_tr}'
        raise ValueError(msg)
    return v


def read_basic_segy_info(
    src: segyio.SegyFile,
    *,
    path: Path,
    name: str,
) -> tuple[int, int, int, float]:
    n_traces = int(src.tracecount)
    if n_traces <= 0:
        if name:
            msg = f'no traces in {name} segy: {path}'
        else:
            msg = f'no traces: {path}'
        raise ValueError(msg)

    n_samples = int(src.samples.size)
    if n_samples <= 0:
        msg = f'invalid n_samples: {n_samples}'
        raise ValueError(msg)

    dt_us = int(src.bin[segyio.BinField.Interval])
    if dt_us <= 0:
        msg = f'invalid dt_us: {dt_us}'
        raise ValueError(msg)

    dt_sec = float(dt_us) * 1.0e-6
    return n_traces, n_samples, dt_us, dt_sec


def require_matching_tracecount(
    raw: segyio.SegyFile,
    win: segyio.SegyFile,
    *,
    raw_path: Path,
    win_path: Path,
) -> int:
    n_tr_raw = int(raw.tracecount)
    n_tr_win = int(win.tracecount)
    if n_tr_raw != n_tr_win:
        msg = (
            f'tracecount mismatch raw={n_tr_raw} win512={n_tr_win} '
            f'raw={raw_path} win={win_path}'
        )
        raise ValueError(msg)
    return n_tr_raw


def require_expected_samples(
    win: segyio.SegyFile,
    *,
    expected: int,
    win_path: Path,
) -> None:
    n_samples_win = int(win.samples.size)
    if n_samples_win != int(expected):
        msg = (
            f'win512 segy must have {expected} samples, '
            f'got {n_samples_win}: {win_path}'
        )
        raise ValueError(msg)


def is_contiguous(idx: np.ndarray) -> bool:
    i = np.asarray(idx, dtype=np.int64)
    if i.size <= 1:
        return True
    return bool(np.all(np.diff(i) == 1))


def load_traces_by_indices(segy_obj: segyio.SegyFile, idx: np.ndarray) -> np.ndarray:
    i = np.asarray(idx, dtype=np.int64)
    if i.ndim != 1:
        msg = f'idx must be 1D, got {i.shape}'
        raise ValueError(msg)
    if i.size == 0:
        msg = 'idx must be non-empty'
        raise ValueError(msg)

    if is_contiguous(i):
        sl = slice(int(i[0]), int(i[-1]) + 1)
        data = np.asarray(segy_obj.trace.raw[sl], dtype=np.float32)
    else:
        rows = [np.asarray(segy_obj.trace.raw[int(j)], dtype=np.float32) for j in i]
        data = np.asarray(rows, dtype=np.float32)

    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2:
        msg = f'loaded trace block must be 2D, got {data.shape}'
        raise ValueError(msg)
    return data.astype(np.float32, copy=False)


__all__ = [
    'is_contiguous',
    'load_traces_by_indices',
    'read_basic_segy_info',
    'read_trace_field',
    'require_expected_samples',
    'require_matching_tracecount',
]
