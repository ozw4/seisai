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
    'offset_signed_geom_m',
)
GEOMETRY_CACHE_SCALE_KEY = 'geometry_coord_unit_scale_to_m'


def validate_coord_unit_scale_to_m(coord_unit_scale_to_m: float) -> float:
    if isinstance(coord_unit_scale_to_m, bool):
        msg = 'coord_unit_scale_to_m must be float'
        raise TypeError(msg)
    try:
        unit_scale = float(coord_unit_scale_to_m)
    except (TypeError, ValueError) as exc:
        msg = 'coord_unit_scale_to_m must be float'
        raise TypeError(msg) from exc
    if (not np.isfinite(unit_scale)) or unit_scale <= 0.0:
        msg = 'coord_unit_scale_to_m must be finite and > 0'
        raise ValueError(msg)
    return unit_scale


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
        'offset_signed_geom_m': np.full((n,), np.nan, dtype=np.float32),
    }


def _orient_principal_direction(u: np.ndarray) -> np.ndarray:
    out = np.asarray(u, dtype=np.float64).copy()
    dominant = int(np.argmax(np.abs(out)))
    if out[dominant] < 0.0:
        out *= -1.0
    return out


def estimate_signed_geometry_offset_m(
    *,
    source_x_m: np.ndarray,
    source_y_m: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
    geometry_valid_mask: np.ndarray,
    min_receiver_spread_m: float = 1.0e-3,
) -> np.ndarray:
    valid = np.asarray(geometry_valid_mask, dtype=np.bool_)
    n_traces = int(valid.shape[0])
    signed = np.full((n_traces,), np.nan, dtype=np.float32)
    if int(np.count_nonzero(valid)) < 2:
        return signed

    receiver_xy = np.stack(
        [
            np.asarray(receiver_x_m, dtype=np.float64)[valid],
            np.asarray(receiver_y_m, dtype=np.float64)[valid],
        ],
        axis=1,
    )
    centered_receiver_xy = receiver_xy - np.mean(receiver_xy, axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered_receiver_xy, full_matrices=False)
    if singular_values.size == 0 or float(singular_values[0]) <= 0.0:
        return signed

    u = _orient_principal_direction(vh[0])
    receiver_projection = centered_receiver_xy @ u
    if float(np.ptp(receiver_projection)) <= float(min_receiver_spread_m):
        return signed

    source_xy = np.stack(
        [
            np.asarray(source_x_m, dtype=np.float64)[valid],
            np.asarray(source_y_m, dtype=np.float64)[valid],
        ],
        axis=1,
    )
    signed[valid] = np.sum((receiver_xy - source_xy) * u[None, :], axis=1).astype(
        np.float32
    )
    return signed


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
    unit_scale = validate_coord_unit_scale_to_m(coord_unit_scale_to_m)

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
    offset_signed_f = estimate_signed_geometry_offset_m(
        source_x_m=source_x_f,
        source_y_m=source_y_f,
        receiver_x_m=receiver_x_f,
        receiver_y_m=receiver_y_f,
        geometry_valid_mask=geometry_valid_mask,
    )

    return {
        'source_x_m': source_x_f,
        'source_y_m': source_y_f,
        'receiver_x_m': receiver_x_f,
        'receiver_y_m': receiver_y_f,
        'offset_abs_geom_m': offset_abs_f,
        'geometry_valid_mask': geometry_valid_mask.astype(np.bool_),
        'offset_signed_geom_m': offset_signed_f,
    }
