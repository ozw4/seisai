from __future__ import annotations

import numpy as np
import segyio

GEOMETRY_ARRAY_KEYS = (
    'source_x_m',
    'source_y_m',
    'receiver_x_m',
    'receiver_y_m',
    'offset_abs_geom_m',
    'geometry_valid_mask',
)
GEOMETRY_CACHE_SCALE_KEY = 'geometry_coord_unit_scale_to_m'


def apply_source_group_scalar(values: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    """Apply the SEG-Y SourceGroupScalar convention to coordinate values."""
    arr = np.asarray(values, dtype=np.float64)
    scal_arr = np.asarray(scalar, dtype=np.float64)
    scal_flat = scal_arr.reshape(-1)
    if scal_flat.size not in (1, arr.size):
        msg = (
            'SourceGroupScalar size must be scalar or match values size, '
            f'got {scal_flat.size} for {arr.size} values'
        )
        raise ValueError(msg)

    scale = np.ones_like(scal_flat, dtype=np.float64)
    pos = scal_flat > 0.0
    neg = scal_flat < 0.0
    scale[pos] = scal_flat[pos]
    scale[neg] = 1.0 / np.abs(scal_flat[neg])

    if scale.size == 1:
        return arr * float(scale[0])
    return arr * scale.reshape(arr.shape)


def invalid_geometry_arrays(n_traces: int) -> dict[str, np.ndarray]:
    n = int(n_traces)
    return {
        'source_x_m': np.full((n,), np.nan, dtype=np.float32),
        'source_y_m': np.full((n,), np.nan, dtype=np.float32),
        'receiver_x_m': np.full((n,), np.nan, dtype=np.float32),
        'receiver_y_m': np.full((n,), np.nan, dtype=np.float32),
        'offset_abs_geom_m': np.full((n,), np.nan, dtype=np.float32),
        'geometry_valid_mask': np.zeros((n,), dtype=np.bool_),
    }


def _read_1d_header(
    f: segyio.SegyFile,
    field: int,
    *,
    name: str,
    n_traces: int,
) -> np.ndarray:
    arr = np.asarray(f.attributes(field)[:], dtype=np.float64).reshape(-1)
    if arr.size != int(n_traces):
        msg = f'{name} length mismatch: got {arr.size}, want {int(n_traces)}'
        raise ValueError(msg)
    return arr


def read_geometry_arrays_from_segy(
    f: segyio.SegyFile,
    *,
    coord_unit_scale_to_m: float = 1.0,
) -> dict[str, np.ndarray]:
    n_traces = int(f.tracecount)
    unit_scale = float(coord_unit_scale_to_m)
    if not np.isfinite(unit_scale):
        msg = 'coord_unit_scale_to_m must be finite'
        raise ValueError(msg)

    source_x = _read_1d_header(
        f,
        segyio.TraceField.SourceX,
        name='SourceX',
        n_traces=n_traces,
    )
    source_y = _read_1d_header(
        f,
        segyio.TraceField.SourceY,
        name='SourceY',
        n_traces=n_traces,
    )
    receiver_x = _read_1d_header(
        f,
        segyio.TraceField.GroupX,
        name='GroupX',
        n_traces=n_traces,
    )
    receiver_y = _read_1d_header(
        f,
        segyio.TraceField.GroupY,
        name='GroupY',
        n_traces=n_traces,
    )
    scalar = _read_1d_header(
        f,
        segyio.TraceField.SourceGroupScalar,
        name='SourceGroupScalar',
        n_traces=n_traces,
    )

    source_x_m = apply_source_group_scalar(source_x, scalar) * unit_scale
    source_y_m = apply_source_group_scalar(source_y, scalar) * unit_scale
    receiver_x_m = apply_source_group_scalar(receiver_x, scalar) * unit_scale
    receiver_y_m = apply_source_group_scalar(receiver_y, scalar) * unit_scale
    dx = receiver_x_m - source_x_m
    dy = receiver_y_m - source_y_m
    offset_abs_geom_m = np.sqrt(dx * dx + dy * dy)

    source_x_f = source_x_m.astype(np.float32)
    source_y_f = source_y_m.astype(np.float32)
    receiver_x_f = receiver_x_m.astype(np.float32)
    receiver_y_f = receiver_y_m.astype(np.float32)
    offset_abs_f = offset_abs_geom_m.astype(np.float32)
    geometry_valid_mask = (
        np.isfinite(source_x_f)
        & np.isfinite(source_y_f)
        & np.isfinite(receiver_x_f)
        & np.isfinite(receiver_y_f)
        & np.isfinite(offset_abs_f)
        & (offset_abs_f >= 0.0)
    )

    return {
        'source_x_m': source_x_f,
        'source_y_m': source_y_f,
        'receiver_x_m': receiver_x_f,
        'receiver_y_m': receiver_y_f,
        'offset_abs_geom_m': offset_abs_f,
        'geometry_valid_mask': geometry_valid_mask.astype(np.bool_),
    }
