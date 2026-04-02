from __future__ import annotations

from abc import ABC, abstractmethod
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

from .geometry import SurveyGeometry

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
        msg = 'adapter arrays must be numpy.ndarray or torch.Tensor'
        raise TypeError(msg)
    if not seen:
        msg = 'at least one adapter array is required'
        raise ValueError(msg)
    if len(seen) != 1:
        msg = 'adapter arrays must use a single backend'
        raise TypeError(msg)
    return next(iter(seen))


def _to_float32_1d(x: ArrayLike, *, name: str, backend: Backend) -> ArrayLike:
    validate_array(x, allowed_ndims=(1,), name=name, backend=backend, shape_hint='(N,)')
    require_float_array(x, name=name, backend=backend)
    require_all_finite(x, name=name, backend=backend)
    if backend == 'numpy':
        return np.asarray(x, dtype=np.float32)
    return x.to(dtype=torch.float32)


def _to_bool_1d(x: ArrayLike, *, name: str, backend: Backend) -> ArrayLike:
    validate_array(x, allowed_ndims=(1,), name=name, backend=backend, shape_hint='(N,)')
    require_boolint_array(x, name=name, backend=backend)
    if backend == 'numpy':
        return np.asarray(x, dtype=bool)
    return x.to(dtype=torch.bool)


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


def _shape_len(x: ArrayLike) -> int:
    return int(x.shape[0])


def _shape_mismatch(name: str, expected: int, actual: int) -> None:
    if actual != expected:
        msg = f'{name} must have length {expected}, got {actual}'
        raise ValueError(msg)


def _require_positive_float_array(x: ArrayLike, *, name: str, backend: Backend) -> ArrayLike:
    out = _to_float32_1d(x, name=name, backend=backend)
    if backend == 'numpy':
        if np.any(out <= 0.0):
            msg = f'{name} must be > 0'
            raise ValueError(msg)
        return out
    if bool((out <= 0.0).any().item()):
        msg = f'{name} must be > 0'
        raise ValueError(msg)
    return out


def _validate_trace_count(trace_count: int) -> int:
    out = int(trace_count)
    if out <= 0:
        msg = f'trace_count must be > 0, got {trace_count}'
        raise ValueError(msg)
    return out


def _validate_label(value: str, *, name: str) -> str:
    if not isinstance(value, str) or value.strip() == '':
        msg = f'{name} must be a non-empty string'
        raise ValueError(msg)
    return value


def _validate_pick_idx(pick_idx: ArrayLike, *, trace_count: int, backend: Backend) -> ArrayLike:
    out = _to_int64_1d(pick_idx, name='pick_idx', backend=backend)
    _shape_mismatch('pick_idx', trace_count, _shape_len(out))
    if backend == 'numpy':
        if np.any(out < -1):
            msg = 'pick_idx may contain -1 for invalid traces, but no value < -1'
            raise ValueError(msg)
        return out

    if bool((out < -1).any().item()):
        msg = 'pick_idx may contain -1 for invalid traces, but no value < -1'
        raise ValueError(msg)
    return out


@dataclass(frozen=True)
class TravelTimeSummary:
    survey_id: str
    backend_name: str
    trace_count: int
    travel_time_sec: ArrayLike | None = None
    travel_time_uncertainty_sec: ArrayLike | None = None
    apparent_velocity_mps: ArrayLike | None = None
    slowness_s_per_m: ArrayLike | None = None
    trace_valid: ArrayLike | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'survey_id', _validate_label(self.survey_id, name='survey_id'))
        object.__setattr__(
            self,
            'backend_name',
            _validate_label(self.backend_name, name='backend_name'),
        )
        object.__setattr__(self, 'trace_count', _validate_trace_count(self.trace_count))

        if (
            self.travel_time_sec is None
            and self.apparent_velocity_mps is None
            and self.slowness_s_per_m is None
        ):
            msg = (
                'TravelTimeSummary requires at least one of travel_time_sec, '
                'apparent_velocity_mps, or slowness_s_per_m'
            )
            raise ValueError(msg)

        backend = _infer_backend(
            self.travel_time_sec,
            self.travel_time_uncertainty_sec,
            self.apparent_velocity_mps,
            self.slowness_s_per_m,
            self.trace_valid,
        )

        for name in ('travel_time_sec', 'travel_time_uncertainty_sec'):
            value = getattr(self, name)
            if value is None:
                continue
            out = _to_float32_1d(value, name=name, backend=backend)
            _shape_mismatch(name, self.trace_count, _shape_len(out))
            object.__setattr__(self, name, out)

        for name in ('apparent_velocity_mps', 'slowness_s_per_m'):
            value = getattr(self, name)
            if value is None:
                continue
            out = _require_positive_float_array(value, name=name, backend=backend)
            _shape_mismatch(name, self.trace_count, _shape_len(out))
            object.__setattr__(self, name, out)

        if self.trace_valid is not None:
            out = _to_bool_1d(self.trace_valid, name='trace_valid', backend=backend)
            _shape_mismatch('trace_valid', self.trace_count, _shape_len(out))
            object.__setattr__(self, 'trace_valid', out)

        if self.travel_time_uncertainty_sec is not None:
            unc = self.travel_time_uncertainty_sec
            if backend == 'numpy':
                if np.any(unc <= 0.0):
                    msg = 'travel_time_uncertainty_sec must be > 0'
                    raise ValueError(msg)
            elif bool((unc <= 0.0).any().item()):
                msg = 'travel_time_uncertainty_sec must be > 0'
                raise ValueError(msg)

    @property
    def backend(self) -> Backend:
        return _infer_backend(
            self.travel_time_sec,
            self.travel_time_uncertainty_sec,
            self.apparent_velocity_mps,
            self.slowness_s_per_m,
            self.trace_valid,
        )


@dataclass(frozen=True)
class ExpectedArrival:
    survey_id: str
    backend_name: str
    trace_count: int
    center_idx: ArrayLike | None = None
    center_time_sec: ArrayLike | None = None
    uncertainty_idx: ArrayLike | None = None
    uncertainty_sec: ArrayLike | None = None
    trace_valid: ArrayLike | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'survey_id', _validate_label(self.survey_id, name='survey_id'))
        object.__setattr__(
            self,
            'backend_name',
            _validate_label(self.backend_name, name='backend_name'),
        )
        object.__setattr__(self, 'trace_count', _validate_trace_count(self.trace_count))

        if self.center_idx is None and self.center_time_sec is None:
            msg = 'ExpectedArrival requires center_idx or center_time_sec'
            raise ValueError(msg)
        if self.uncertainty_idx is None and self.uncertainty_sec is None:
            msg = 'ExpectedArrival requires uncertainty_idx or uncertainty_sec'
            raise ValueError(msg)

        backend = _infer_backend(
            self.center_idx,
            self.center_time_sec,
            self.uncertainty_idx,
            self.uncertainty_sec,
            self.trace_valid,
        )

        for name in ('center_idx', 'center_time_sec'):
            value = getattr(self, name)
            if value is None:
                continue
            out = _to_float32_1d(value, name=name, backend=backend)
            _shape_mismatch(name, self.trace_count, _shape_len(out))
            object.__setattr__(self, name, out)

        for name in ('uncertainty_idx', 'uncertainty_sec'):
            value = getattr(self, name)
            if value is None:
                continue
            out = _require_positive_float_array(value, name=name, backend=backend)
            _shape_mismatch(name, self.trace_count, _shape_len(out))
            object.__setattr__(self, name, out)

        if self.trace_valid is not None:
            out = _to_bool_1d(self.trace_valid, name='trace_valid', backend=backend)
            _shape_mismatch('trace_valid', self.trace_count, _shape_len(out))
            object.__setattr__(self, 'trace_valid', out)

        if self.center_idx is not None and self.center_time_sec is not None:
            require_same_shape_and_backend(
                self.center_idx,
                self.center_time_sec,
                name_a='center_idx',
                name_b='center_time_sec',
                backend=backend,
                shape_hint='(N,)',
            )

        if self.uncertainty_idx is not None and self.uncertainty_sec is not None:
            require_same_shape_and_backend(
                self.uncertainty_idx,
                self.uncertainty_sec,
                name_a='uncertainty_idx',
                name_b='uncertainty_sec',
                backend=backend,
                shape_hint='(N,)',
            )

    @property
    def backend(self) -> Backend:
        return _infer_backend(
            self.center_idx,
            self.center_time_sec,
            self.uncertainty_idx,
            self.uncertainty_sec,
            self.trace_valid,
        )


class InversionBackend(ABC):
    name: str

    @abstractmethod
    def fit_travel_time_summary(
        self,
        *,
        geometry: SurveyGeometry,
        pick_idx: ArrayLike,
        sample_interval_sec: float,
        trace_valid: ArrayLike | None = None,
        pick_confidence: ArrayLike | None = None,
    ) -> TravelTimeSummary:
        raise NotImplementedError

    @abstractmethod
    def predict_expected_arrivals(
        self,
        *,
        geometry: SurveyGeometry,
        sample_interval_sec: float,
        summary: TravelTimeSummary,
    ) -> ExpectedArrival:
        raise NotImplementedError


@dataclass(frozen=True)
class MissingInversionBackend(InversionBackend):
    name: str = 'unimplemented'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'name', _validate_label(self.name, name='name'))

    def fit_travel_time_summary(
        self,
        *,
        geometry: SurveyGeometry,
        pick_idx: ArrayLike,
        sample_interval_sec: float,
        trace_valid: ArrayLike | None = None,
        pick_confidence: ArrayLike | None = None,
    ) -> TravelTimeSummary:
        msg = (
            'No inversion backend is implemented for global_qc. '
            f'backend={self.name!r}, survey_id={geometry.survey_id!r}'
        )
        raise NotImplementedError(msg)

    def predict_expected_arrivals(
        self,
        *,
        geometry: SurveyGeometry,
        sample_interval_sec: float,
        summary: TravelTimeSummary,
    ) -> ExpectedArrival:
        msg = (
            'No inversion backend is implemented for global_qc expected-arrival prediction. '
            f'backend={self.name!r}, survey_id={geometry.survey_id!r}'
        )
        raise NotImplementedError(msg)


def require_inversion_backend(backend: InversionBackend | None) -> InversionBackend:
    if backend is None:
        msg = 'global_qc requires a concrete inversion backend; got None'
        raise ValueError(msg)
    if isinstance(backend, MissingInversionBackend):
        msg = (
            'global_qc requires a concrete inversion backend; '
            f'got MissingInversionBackend(name={backend.name!r})'
        )
        raise NotImplementedError(msg)
    return backend


def validate_backend_pick_idx(
    pick_idx: ArrayLike,
    *,
    trace_count: int,
    backend: Backend,
) -> ArrayLike:
    return _validate_pick_idx(pick_idx, trace_count=trace_count, backend=backend)
