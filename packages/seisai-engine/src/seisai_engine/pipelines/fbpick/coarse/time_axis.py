from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import resample_poly

__all__ = [
    'CoarseTimeGrid',
    'build_coarse_fb_labels_for_anchors',
    'build_coarse_time_grid',
    'build_time_channel',
    'project_coarse_indices_to_raw_time',
    'project_fb_indices_to_coarse_time',
    'resample_waveform_time_axis',
]


@dataclass(frozen=True)
class CoarseTimeGrid:
    raw_time_len: int
    coarse_time_len: int
    dt_sec: float
    dt_eff_sec: float
    raw_to_coarse_factor: float
    coarse_to_raw_factor: float
    time_view_sec: np.ndarray


def _require_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        msg = f'{name} must be int'
        raise TypeError(msg)
    try:
        return int(operator.index(value))
    except TypeError as exc:
        msg = f'{name} must be int'
        raise TypeError(msg) from exc


def _validate_time_lengths(
    *,
    raw_time_len: int,
    coarse_time_len: int,
) -> tuple[int, int]:
    raw = _require_int('raw_time_len', raw_time_len)
    coarse = _require_int('coarse_time_len', coarse_time_len)
    if raw < 2:
        msg = 'raw_time_len must be >= 2'
        raise ValueError(msg)
    if coarse < 2:
        msg = 'coarse_time_len must be >= 2'
        raise ValueError(msg)
    return raw, coarse


def _validate_dt_sec(dt_sec: float) -> float:
    dt = float(dt_sec)
    if not math.isfinite(dt) or dt <= 0.0:
        msg = 'dt_sec must be finite and > 0'
        raise ValueError(msg)
    return dt


def resample_waveform_time_axis(
    waveform: np.ndarray,
    *,
    out_time_len: int,
) -> np.ndarray:
    """Resample ``(Hvalid, W0)`` waveform rows onto an endpoint-aligned time grid."""
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim != 2:
        msg = f'waveform must have shape (Hvalid, W0), got {x.shape}'
        raise ValueError(msg)
    h_valid, raw_time_len = int(x.shape[0]), int(x.shape[1])
    if h_valid == 0:
        msg = 'Hvalid must be > 0'
        raise ValueError(msg)
    out_len = _require_int('out_time_len', out_time_len)
    if raw_time_len < 2:
        msg = 'waveform time axis length must be >= 2'
        raise ValueError(msg)
    if out_len < 2:
        msg = 'out_time_len must be >= 2'
        raise ValueError(msg)
    if not np.all(np.isfinite(x)):
        msg = 'waveform must contain only finite values'
        raise ValueError(msg)

    if raw_time_len == out_len:
        return x.astype(np.float32, copy=True)

    up = out_len - 1
    down = raw_time_len - 1
    gcd = math.gcd(up, down)
    up //= gcd
    down //= gcd
    y = resample_poly(x, up, down, axis=1, padtype='line')
    if int(y.shape[1]) < out_len:
        msg = (
            'resampled waveform time axis is shorter than requested: '
            f'{int(y.shape[1])} < {out_len}'
        )
        raise RuntimeError(msg)
    out = np.asarray(y[:, :out_len], dtype=np.float32)
    out[:, 0] = x[:, 0]
    out[:, -1] = x[:, -1]
    if out.shape != (h_valid, out_len):
        msg = f'resampled waveform shape {out.shape} != {(h_valid, out_len)}'
        raise RuntimeError(msg)
    if not np.all(np.isfinite(out)):
        msg = 'resampled waveform contains non-finite values'
        raise RuntimeError(msg)
    return np.ascontiguousarray(out, dtype=np.float32)


def build_coarse_time_grid(
    *,
    raw_time_len: int,
    coarse_time_len: int,
    dt_sec: float,
) -> CoarseTimeGrid:
    raw, coarse = _validate_time_lengths(
        raw_time_len=raw_time_len,
        coarse_time_len=coarse_time_len,
    )
    dt = _validate_dt_sec(dt_sec)

    raw_to_coarse_factor = float(coarse - 1) / float(raw - 1)
    coarse_to_raw_factor = float(raw - 1) / float(coarse - 1)
    dt_eff_sec = (float(raw - 1) * dt) / float(coarse - 1)
    time_view_sec = np.linspace(
        0.0,
        float(raw - 1) * dt,
        coarse,
        dtype=np.float32,
    )
    if time_view_sec.shape != (coarse,):
        msg = 'time_view_sec has invalid shape'
        raise RuntimeError(msg)
    if float(time_view_sec[0]) != 0.0:
        msg = 'time_view_sec must start at 0.0'
        raise RuntimeError(msg)
    if not np.isclose(float(time_view_sec[-1]), float(raw - 1) * dt):
        msg = 'time_view_sec endpoint does not match raw time endpoint'
        raise RuntimeError(msg)

    return CoarseTimeGrid(
        raw_time_len=raw,
        coarse_time_len=coarse,
        dt_sec=dt,
        dt_eff_sec=float(dt_eff_sec),
        raw_to_coarse_factor=float(raw_to_coarse_factor),
        coarse_to_raw_factor=float(coarse_to_raw_factor),
        time_view_sec=time_view_sec,
    )


def _coerce_ignore_index(ignore_index: int) -> int:
    if isinstance(ignore_index, bool):
        msg = 'ignore_index must be int'
        raise TypeError(msg)
    try:
        return int(operator.index(ignore_index))
    except TypeError as exc:
        msg = 'ignore_index must be int'
        raise TypeError(msg) from exc


def _coerce_index_array(name: str, value: np.ndarray, *, ignore_index: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.dtype('bool'):
        msg = f'{name} must contain integer sample indices, not bool'
        raise TypeError(msg)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)

    as_float = np.asarray(arr, dtype=np.float64)
    ignore_mask = as_float == float(ignore_index)
    valid_float = as_float[~ignore_mask]
    if valid_float.size and not np.all(np.isfinite(valid_float)):
        msg = f'{name} must contain only finite values'
        raise ValueError(msg)
    rounded = np.rint(valid_float)
    if valid_float.size and not np.all(valid_float == rounded):
        msg = f'{name} must contain integer-valued sample indices'
        raise ValueError(msg)
    out = np.empty(as_float.shape, dtype=np.int64)
    out[ignore_mask] = int(ignore_index)
    out[~ignore_mask] = rounded.astype(np.int64, copy=False)
    return out


def _project_indices(
    name: str,
    indices: np.ndarray,
    *,
    src_len: int,
    dst_len: int,
    ignore_index: int,
) -> np.ndarray:
    if src_len < 2:
        msg = 'src_len must be >= 2'
        raise ValueError(msg)
    if dst_len < 2:
        msg = 'dst_len must be >= 2'
        raise ValueError(msg)

    ignore = _coerce_ignore_index(ignore_index)
    idx = _coerce_index_array(name, indices, ignore_index=ignore)
    valid = idx != ignore
    if np.any(idx[valid] < 0) or np.any(idx[valid] > src_len - 1):
        msg = f'{name} values must lie in [0, {src_len - 1}] or equal ignore_index'
        raise ValueError(msg)

    out = np.full(idx.shape, ignore, dtype=np.int64)
    if np.any(valid):
        projected = np.rint(idx[valid].astype(np.float64) * (dst_len - 1) / (src_len - 1))
        projected = np.clip(projected, 0, dst_len - 1)
        out[valid] = projected.astype(np.int64, copy=False)
    return out


def project_fb_indices_to_coarse_time(
    fb_idx_raw: np.ndarray,
    *,
    raw_time_len: int,
    coarse_time_len: int,
    ignore_index: int = -1,
) -> np.ndarray:
    raw, coarse = _validate_time_lengths(
        raw_time_len=raw_time_len,
        coarse_time_len=coarse_time_len,
    )
    return _project_indices(
        'fb_idx_raw',
        fb_idx_raw,
        src_len=raw,
        dst_len=coarse,
        ignore_index=ignore_index,
    )


def project_coarse_indices_to_raw_time(
    idx_coarse: np.ndarray,
    *,
    raw_time_len: int,
    coarse_time_len: int,
    ignore_index: int = -1,
) -> np.ndarray:
    raw, coarse = _validate_time_lengths(
        raw_time_len=raw_time_len,
        coarse_time_len=coarse_time_len,
    )
    return _project_indices(
        'idx_coarse',
        idx_coarse,
        src_len=coarse,
        dst_len=raw,
        ignore_index=ignore_index,
    )


def build_time_channel(
    time_view_sec: np.ndarray,
    *,
    trace_len: int,
) -> np.ndarray:
    """Build a raw seconds time channel by broadcasting ``time_view_sec`` over traces."""
    t = np.asarray(time_view_sec, dtype=np.float32)
    if t.ndim != 1:
        msg = f'time_view_sec must be 1D, got shape={t.shape}'
        raise ValueError(msg)
    if int(t.size) < 2:
        msg = 'time_view_sec length must be >= 2'
        raise ValueError(msg)
    if not np.all(np.isfinite(t)):
        msg = 'time_view_sec must contain only finite values'
        raise ValueError(msg)
    h = _require_int('trace_len', trace_len)
    if h <= 0:
        msg = 'trace_len must be > 0'
        raise ValueError(msg)

    return np.repeat(t[None, :], h, axis=0).astype(np.float32, copy=False)


def build_coarse_fb_labels_for_anchors(
    fb_idx_raw_for_anchors: np.ndarray,
    trace_valid: np.ndarray,
    *,
    raw_time_len: int,
    coarse_time_len: int,
    ignore_index: int = -1,
) -> np.ndarray:
    raw, coarse = _validate_time_lengths(
        raw_time_len=raw_time_len,
        coarse_time_len=coarse_time_len,
    )
    ignore = _coerce_ignore_index(ignore_index)
    valid = np.asarray(trace_valid, dtype=np.bool_)
    fb = np.asarray(fb_idx_raw_for_anchors)
    if fb.shape != valid.shape:
        msg = f'fb_idx_raw_for_anchors shape {fb.shape} != trace_valid shape {valid.shape}'
        raise ValueError(msg)

    out = np.full(valid.shape, ignore, dtype=np.int64)
    if np.any(valid):
        out[valid] = project_fb_indices_to_coarse_time(
            fb[valid],
            raw_time_len=raw,
            coarse_time_len=coarse,
            ignore_index=ignore,
        )
    return out
