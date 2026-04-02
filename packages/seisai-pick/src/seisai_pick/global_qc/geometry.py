from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from seisai_utils.validator import (
    require_all_finite,
    require_boolint_array,
    require_float_array,
    require_same_shape_and_backend,
    validate_array,
)
from torch import Tensor

ArrayLike = np.ndarray | Tensor
Backend = Literal['numpy', 'torch']

_TORCH_INT_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


def _infer_backend(*xs: ArrayLike | None) -> Backend:
    seen: set[Backend] = set()
    for x in xs:
        if x is None:
            continue
        if isinstance(x, np.ndarray):
            seen.add('numpy')
            continue
        if isinstance(x, torch.Tensor):
            seen.add('torch')
            continue
        msg = 'geometry arrays must be numpy.ndarray or torch.Tensor'
        raise TypeError(msg)
    if not seen:
        msg = 'at least one geometry array is required'
        raise ValueError(msg)
    if len(seen) != 1:
        msg = 'geometry arrays must use a single backend'
        raise TypeError(msg)
    return next(iter(seen))


def _to_float32_1d(x: ArrayLike, *, name: str, backend: Backend) -> ArrayLike:
    validate_array(x, allowed_ndims=(1,), name=name, backend=backend, shape_hint='(N,)')
    require_float_array(x, name=name, backend=backend)
    require_all_finite(x, name=name, backend=backend)
    if backend == 'numpy':
        return np.asarray(x, dtype=np.float32)
    return x.to(dtype=torch.float32)


def _to_int64_1d(x: ArrayLike, *, name: str, backend: Backend) -> ArrayLike:
    validate_array(x, allowed_ndims=(1,), name=name, backend=backend, shape_hint='(N,)')
    if backend == 'numpy':
        arr = np.asarray(x)
        if not np.issubdtype(arr.dtype, np.integer):
            msg = f'{name} must be an integer array'
            raise TypeError(msg)
        return arr.astype(np.int64, copy=False)

    if x.dtype not in _TORCH_INT_DTYPES:
        msg = f'{name} must be an integer tensor'
        raise TypeError(msg)
    return x.to(dtype=torch.int64)


def _to_bool_1d(x: ArrayLike, *, name: str, backend: Backend) -> ArrayLike:
    validate_array(x, allowed_ndims=(1,), name=name, backend=backend, shape_hint='(N,)')
    require_boolint_array(x, name=name, backend=backend)
    if backend == 'numpy':
        return np.asarray(x, dtype=bool)
    return x.to(dtype=torch.bool)


def _count_unique(x: ArrayLike, *, backend: Backend) -> int:
    if backend == 'numpy':
        return int(np.unique(x).shape[0])
    return int(torch.unique(x).numel())


def _any_less_than(x: ArrayLike, value: int, *, backend: Backend) -> bool:
    if backend == 'numpy':
        return bool(np.any(x < value))
    return bool((x < value).any().item())


def _full_true_mask(ref: ArrayLike) -> ArrayLike:
    n = int(ref.shape[0])
    if isinstance(ref, np.ndarray):
        return np.ones((n,), dtype=bool)
    return torch.ones((n,), dtype=torch.bool, device=ref.device)


def _stack_xyz(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
    if isinstance(x, np.ndarray):
        return np.stack((x, y, z), axis=-1).astype(np.float32, copy=False)
    return torch.stack((x, y, z), dim=-1).to(dtype=torch.float32)


def _sqrt(x: ArrayLike) -> ArrayLike:
    if isinstance(x, np.ndarray):
        return np.sqrt(x).astype(np.float32, copy=False)
    return torch.sqrt(x)


def _apply_xy_normalization(
    x: ArrayLike,
    *,
    origin: float,
    scale_to_m: float,
) -> ArrayLike:
    if isinstance(x, np.ndarray):
        out = (x.astype(np.float32, copy=False) - np.float32(origin)) * np.float32(
            scale_to_m
        )
        return out.astype(np.float32, copy=False)
    return (x.to(dtype=torch.float32) - float(origin)) * float(scale_to_m)


def _apply_z_normalization(
    x: ArrayLike,
    *,
    origin: float,
    scale_to_m: float,
    flip_sign: bool,
) -> ArrayLike:
    out = _apply_xy_normalization(x, origin=origin, scale_to_m=scale_to_m)
    if flip_sign:
        return -out
    return out


@dataclass(frozen=True)
class GeometryNormalization:
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_z: float = 0.0
    xy_scale_to_m: float = 1.0
    z_scale_to_m: float = 1.0
    flip_z_sign: bool = False

    def __post_init__(self) -> None:
        for field_name in ('origin_x', 'origin_y', 'origin_z'):
            value = float(getattr(self, field_name))
            if not np.isfinite(value):
                msg = f'{field_name} must be finite'
                raise ValueError(msg)
            object.__setattr__(self, field_name, value)

        for field_name in ('xy_scale_to_m', 'z_scale_to_m'):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value <= 0.0:
                msg = f'{field_name} must be finite and > 0'
                raise ValueError(msg)
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True)
class SurveyGeometry:
    survey_id: str
    shot_x: ArrayLike
    shot_y: ArrayLike
    shot_z: ArrayLike
    recv_x: ArrayLike
    recv_y: ArrayLike
    recv_z: ArrayLike
    raw_trace_idx: ArrayLike
    trace_valid: ArrayLike | None = None
    shot_elevation: ArrayLike | None = None
    recv_elevation: ArrayLike | None = None
    shot_datum: ArrayLike | None = None
    recv_datum: ArrayLike | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.survey_id, str) or self.survey_id.strip() == '':
            msg = 'survey_id must be a non-empty string'
            raise ValueError(msg)

        backend = _infer_backend(
            self.shot_x,
            self.shot_y,
            self.shot_z,
            self.recv_x,
            self.recv_y,
            self.recv_z,
            self.raw_trace_idx,
            self.trace_valid,
            self.shot_elevation,
            self.recv_elevation,
            self.shot_datum,
            self.recv_datum,
        )

        for name in (
            'shot_x',
            'shot_y',
            'shot_z',
            'recv_x',
            'recv_y',
            'recv_z',
        ):
            object.__setattr__(
                self,
                name,
                _to_float32_1d(getattr(self, name), name=name, backend=backend),
            )

        object.__setattr__(
            self,
            'raw_trace_idx',
            _to_int64_1d(self.raw_trace_idx, name='raw_trace_idx', backend=backend),
        )

        if self.trace_valid is not None:
            object.__setattr__(
                self,
                'trace_valid',
                _to_bool_1d(self.trace_valid, name='trace_valid', backend=backend),
            )

        for name in ('shot_elevation', 'recv_elevation', 'shot_datum', 'recv_datum'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _to_float32_1d(value, name=name, backend=backend),
                )

        base = self.shot_x
        for name in (
            'shot_y',
            'shot_z',
            'recv_x',
            'recv_y',
            'recv_z',
            'raw_trace_idx',
        ):
            require_same_shape_and_backend(
                base,
                getattr(self, name),
                name_a='shot_x',
                name_b=name,
                backend=backend,
                shape_hint='(N,)',
            )

        for name in (
            'trace_valid',
            'shot_elevation',
            'recv_elevation',
            'shot_datum',
            'recv_datum',
        ):
            value = getattr(self, name)
            if value is None:
                continue
            require_same_shape_and_backend(
                base,
                value,
                name_a='shot_x',
                name_b=name,
                backend=backend,
                shape_hint='(N,)',
            )

        if int(base.shape[0]) <= 0:
            msg = 'survey geometry must contain at least one trace'
            raise ValueError(msg)

        if _any_less_than(self.raw_trace_idx, 0, backend=backend):
            msg = 'raw_trace_idx must be >= 0'
            raise ValueError(msg)

        if _count_unique(self.raw_trace_idx, backend=backend) != int(
            self.raw_trace_idx.shape[0]
        ):
            msg = 'raw_trace_idx must be unique within a survey'
            raise ValueError(msg)

    @property
    def backend(self) -> Backend:
        return 'numpy' if isinstance(self.shot_x, np.ndarray) else 'torch'

    @property
    def n_traces(self) -> int:
        return int(self.shot_x.shape[0])


def build_survey_geometry(
    *,
    survey_id: str,
    shot_x: ArrayLike,
    shot_y: ArrayLike,
    shot_z: ArrayLike,
    recv_x: ArrayLike,
    recv_y: ArrayLike,
    recv_z: ArrayLike,
    raw_trace_idx: ArrayLike,
    trace_valid: ArrayLike | None = None,
    shot_elevation: ArrayLike | None = None,
    recv_elevation: ArrayLike | None = None,
    shot_datum: ArrayLike | None = None,
    recv_datum: ArrayLike | None = None,
) -> SurveyGeometry:
    return SurveyGeometry(
        survey_id=survey_id,
        shot_x=shot_x,
        shot_y=shot_y,
        shot_z=shot_z,
        recv_x=recv_x,
        recv_y=recv_y,
        recv_z=recv_z,
        raw_trace_idx=raw_trace_idx,
        trace_valid=trace_valid,
        shot_elevation=shot_elevation,
        recv_elevation=recv_elevation,
        shot_datum=shot_datum,
        recv_datum=recv_datum,
    )


def effective_trace_valid(geometry: SurveyGeometry) -> ArrayLike:
    if geometry.trace_valid is None:
        return _full_true_mask(geometry.raw_trace_idx)
    return geometry.trace_valid


def source_points_xyz(geometry: SurveyGeometry) -> ArrayLike:
    return _stack_xyz(geometry.shot_x, geometry.shot_y, geometry.shot_z)


def receiver_points_xyz(geometry: SurveyGeometry) -> ArrayLike:
    return _stack_xyz(geometry.recv_x, geometry.recv_y, geometry.recv_z)


def absolute_offset_m(geometry: SurveyGeometry) -> ArrayLike:
    dx = geometry.recv_x - geometry.shot_x
    dy = geometry.recv_y - geometry.shot_y
    return _sqrt((dx * dx) + (dy * dy))


def source_receiver_distance_m(geometry: SurveyGeometry) -> ArrayLike:
    dx = geometry.recv_x - geometry.shot_x
    dy = geometry.recv_y - geometry.shot_y
    dz = geometry.recv_z - geometry.shot_z
    return _sqrt((dx * dx) + (dy * dy) + (dz * dz))


def normalize_survey_geometry(
    geometry: SurveyGeometry,
    normalization: GeometryNormalization | None = None,
) -> SurveyGeometry:
    if normalization is None:
        return geometry

    return SurveyGeometry(
        survey_id=geometry.survey_id,
        shot_x=_apply_xy_normalization(
            geometry.shot_x,
            origin=normalization.origin_x,
            scale_to_m=normalization.xy_scale_to_m,
        ),
        shot_y=_apply_xy_normalization(
            geometry.shot_y,
            origin=normalization.origin_y,
            scale_to_m=normalization.xy_scale_to_m,
        ),
        shot_z=_apply_z_normalization(
            geometry.shot_z,
            origin=normalization.origin_z,
            scale_to_m=normalization.z_scale_to_m,
            flip_sign=normalization.flip_z_sign,
        ),
        recv_x=_apply_xy_normalization(
            geometry.recv_x,
            origin=normalization.origin_x,
            scale_to_m=normalization.xy_scale_to_m,
        ),
        recv_y=_apply_xy_normalization(
            geometry.recv_y,
            origin=normalization.origin_y,
            scale_to_m=normalization.xy_scale_to_m,
        ),
        recv_z=_apply_z_normalization(
            geometry.recv_z,
            origin=normalization.origin_z,
            scale_to_m=normalization.z_scale_to_m,
            flip_sign=normalization.flip_z_sign,
        ),
        raw_trace_idx=geometry.raw_trace_idx,
        trace_valid=geometry.trace_valid,
        shot_elevation=None
        if geometry.shot_elevation is None
        else _apply_z_normalization(
            geometry.shot_elevation,
            origin=normalization.origin_z,
            scale_to_m=normalization.z_scale_to_m,
            flip_sign=normalization.flip_z_sign,
        ),
        recv_elevation=None
        if geometry.recv_elevation is None
        else _apply_z_normalization(
            geometry.recv_elevation,
            origin=normalization.origin_z,
            scale_to_m=normalization.z_scale_to_m,
            flip_sign=normalization.flip_z_sign,
        ),
        shot_datum=None
        if geometry.shot_datum is None
        else _apply_z_normalization(
            geometry.shot_datum,
            origin=normalization.origin_z,
            scale_to_m=normalization.z_scale_to_m,
            flip_sign=normalization.flip_z_sign,
        ),
        recv_datum=None
        if geometry.recv_datum is None
        else _apply_z_normalization(
            geometry.recv_datum,
            origin=normalization.origin_z,
            scale_to_m=normalization.z_scale_to_m,
            flip_sign=normalization.flip_z_sign,
        ),
    )
