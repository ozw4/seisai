from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from seisai_pick.gaussian_prob import gaussian_probs1d_np, gaussian_probs1d_torch
from seisai_utils.validator import (
    require_all_finite,
    require_boolint_array,
    require_float_array,
    require_same_shape_and_backend,
    validate_array,
)
from torch import Tensor

from .inversion_adapter import ExpectedArrival

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
        msg = 'arrival-band arrays must be numpy.ndarray or torch.Tensor'
        raise TypeError(msg)
    if not seen:
        msg = 'at least one arrival-band array is required'
        raise ValueError(msg)
    if len(seen) != 1:
        msg = 'arrival-band arrays must use a single backend'
        raise TypeError(msg)
    return next(iter(seen))


def _to_float32_1d(x: ArrayLike, *, name: str, backend: Backend) -> ArrayLike:
    validate_array(x, allowed_ndims=(1,), name=name, backend=backend, shape_hint='(N,)')
    require_float_array(x, name=name, backend=backend)
    require_all_finite(x, name=name, backend=backend)
    if backend == 'numpy':
        return np.asarray(x, dtype=np.float32)
    return x.to(dtype=torch.float32)


def _to_float32_2d(x: ArrayLike, *, name: str, backend: Backend) -> ArrayLike:
    validate_array(
        x,
        allowed_ndims=(2,),
        name=name,
        backend=backend,
        shape_hint='(N,W)',
    )
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


def _to_bool_2d(x: ArrayLike, *, name: str, backend: Backend) -> ArrayLike:
    validate_array(
        x,
        allowed_ndims=(2,),
        name=name,
        backend=backend,
        shape_hint='(N,W)',
    )
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


def _validate_sample_axis_len(sample_axis_len: int) -> int:
    out = int(sample_axis_len)
    if out <= 0:
        msg = f'sample_axis_len must be > 0, got {sample_axis_len}'
        raise ValueError(msg)
    return out


def _validate_positive_scalar(value: float, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        msg = f'{name} must be finite and > 0'
        raise ValueError(msg)
    return out


def _validate_non_negative_scalar(value: float, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        msg = f'{name} must be finite and >= 0'
        raise ValueError(msg)
    return out


def _shape_len(x: ArrayLike) -> int:
    return int(x.shape[0])


def _allclose(a: ArrayLike, b: ArrayLike, *, backend: Backend, atol: float) -> bool:
    if backend == 'numpy':
        return bool(np.allclose(a, b, atol=atol, rtol=0.0))
    return bool(torch.allclose(a, b, atol=atol, rtol=0.0))


def _normalize_rows(x: ArrayLike, *, backend: Backend) -> ArrayLike:
    if backend == 'numpy':
        denom = np.sum(x, axis=-1, keepdims=True, dtype=np.float32)
        if np.any(denom <= 0.0):
            msg = 'arrival prior rows must have positive mass'
            raise ValueError(msg)
        return (x / denom).astype(np.float32, copy=False)

    denom = x.sum(dim=-1, keepdim=True)
    if bool((denom <= 0.0).any().item()):
        msg = 'arrival prior rows must have positive mass'
        raise ValueError(msg)
    return x / denom


def _validate_center_idx(center_idx: ArrayLike, *, sample_axis_len: int, backend: Backend) -> None:
    if backend == 'numpy':
        if np.any(center_idx < 0.0) or np.any(center_idx > float(sample_axis_len - 1)):
            msg = 'center_idx must stay within [0, sample_axis_len - 1]'
            raise ValueError(msg)
        return
    if bool((center_idx < 0.0).any().item()) or bool(
        (center_idx > float(sample_axis_len - 1)).any().item()
    ):
        msg = 'center_idx must stay within [0, sample_axis_len - 1]'
        raise ValueError(msg)


def _shape_mismatch(name: str, expected: int, actual: int) -> None:
    if expected != actual:
        msg = f'{name} must have length {expected}, got {actual}'
        raise ValueError(msg)


def _resolve_expected_arrival(
    expected: ExpectedArrival,
    *,
    sample_axis_len: int,
    sample_interval_sec: float | None,
) -> tuple[ArrayLike, ArrayLike | None, ArrayLike, ArrayLike | None, Backend]:
    backend = expected.backend
    dt = None if sample_interval_sec is None else _validate_positive_scalar(sample_interval_sec, name='sample_interval_sec')

    if expected.center_idx is None:
        if dt is None:
            msg = 'sample_interval_sec is required when ExpectedArrival only provides center_time_sec'
            raise ValueError(msg)
        center_idx = expected.center_time_sec / dt
    else:
        center_idx = expected.center_idx

    if expected.center_time_sec is None:
        center_time_sec = None if dt is None else center_idx * dt
    else:
        center_time_sec = expected.center_time_sec

    if expected.uncertainty_idx is None:
        if dt is None:
            msg = (
                'sample_interval_sec is required when ExpectedArrival only provides '
                'uncertainty_sec'
            )
            raise ValueError(msg)
        uncertainty_idx = expected.uncertainty_sec / dt
    else:
        uncertainty_idx = expected.uncertainty_idx

    if expected.uncertainty_sec is None:
        uncertainty_sec = None if dt is None else uncertainty_idx * dt
    else:
        uncertainty_sec = expected.uncertainty_sec

    if expected.center_idx is not None and expected.center_time_sec is not None and dt is not None:
        if not _allclose(expected.center_idx, expected.center_time_sec / dt, backend=backend, atol=1e-3):
            msg = 'center_idx and center_time_sec are inconsistent for the provided sample_interval_sec'
            raise ValueError(msg)

    if (
        expected.uncertainty_idx is not None
        and expected.uncertainty_sec is not None
        and dt is not None
    ):
        if not _allclose(
            expected.uncertainty_idx,
            expected.uncertainty_sec / dt,
            backend=backend,
            atol=1e-3,
        ):
            msg = (
                'uncertainty_idx and uncertainty_sec are inconsistent for the provided '
                'sample_interval_sec'
            )
            raise ValueError(msg)

    _validate_center_idx(center_idx, sample_axis_len=sample_axis_len, backend=backend)
    return center_idx, center_time_sec, uncertainty_idx, uncertainty_sec, backend


def _build_index_mask(
    pick_idx: ArrayLike,
    *,
    sample_axis_len: int,
    backend: Backend,
) -> tuple[ArrayLike, ArrayLike]:
    idx = _to_int64_1d(pick_idx, name='pick_idx', backend=backend)
    valid = None
    if backend == 'numpy':
        if np.any(idx < -1):
            msg = 'pick_idx may contain -1 for invalid traces, but no value < -1'
            raise ValueError(msg)
        if np.any(idx >= sample_axis_len):
            msg = 'pick_idx must be < sample_axis_len'
            raise ValueError(msg)
        valid = idx >= 0
        safe = idx.copy()
        safe[~valid] = 0
        return safe, valid

    if bool((idx < -1).any().item()):
        msg = 'pick_idx may contain -1 for invalid traces, but no value < -1'
        raise ValueError(msg)
    if bool((idx >= sample_axis_len).any().item()):
        msg = 'pick_idx must be < sample_axis_len'
        raise ValueError(msg)
    valid = idx >= 0
    safe = idx.clone()
    safe[~valid] = 0
    return safe, valid


@dataclass(frozen=True)
class ArrivalBand:
    survey_id: str
    backend_name: str
    sample_axis_len: int
    center_idx: ArrayLike
    uncertainty_idx: ArrayLike
    band_half_width_idx: ArrayLike
    feasible_mask: ArrayLike
    prior: ArrayLike
    center_time_sec: ArrayLike | None = None
    uncertainty_sec: ArrayLike | None = None
    trace_valid: ArrayLike | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.survey_id, str) or self.survey_id.strip() == '':
            msg = 'survey_id must be a non-empty string'
            raise ValueError(msg)
        if not isinstance(self.backend_name, str) or self.backend_name.strip() == '':
            msg = 'backend_name must be a non-empty string'
            raise ValueError(msg)

        object.__setattr__(self, 'sample_axis_len', _validate_sample_axis_len(self.sample_axis_len))

        backend = _infer_backend(
            self.center_idx,
            self.uncertainty_idx,
            self.band_half_width_idx,
            self.feasible_mask,
            self.prior,
            self.center_time_sec,
            self.uncertainty_sec,
            self.trace_valid,
        )

        object.__setattr__(
            self,
            'center_idx',
            _to_float32_1d(self.center_idx, name='center_idx', backend=backend),
        )
        object.__setattr__(
            self,
            'uncertainty_idx',
            _to_float32_1d(self.uncertainty_idx, name='uncertainty_idx', backend=backend),
        )
        object.__setattr__(
            self,
            'band_half_width_idx',
            _to_int64_1d(
                self.band_half_width_idx,
                name='band_half_width_idx',
                backend=backend,
            ),
        )
        object.__setattr__(
            self,
            'feasible_mask',
            _to_bool_2d(self.feasible_mask, name='feasible_mask', backend=backend),
        )
        object.__setattr__(
            self,
            'prior',
            _to_float32_2d(self.prior, name='prior', backend=backend),
        )

        if self.center_time_sec is not None:
            object.__setattr__(
                self,
                'center_time_sec',
                _to_float32_1d(self.center_time_sec, name='center_time_sec', backend=backend),
            )
        if self.uncertainty_sec is not None:
            object.__setattr__(
                self,
                'uncertainty_sec',
                _to_float32_1d(
                    self.uncertainty_sec,
                    name='uncertainty_sec',
                    backend=backend,
                ),
            )
        if self.trace_valid is not None:
            object.__setattr__(
                self,
                'trace_valid',
                _to_bool_1d(self.trace_valid, name='trace_valid', backend=backend),
            )

        n_traces = _shape_len(self.center_idx)
        for name in ('uncertainty_idx', 'band_half_width_idx'):
            _shape_mismatch(name, n_traces, _shape_len(getattr(self, name)))

        if tuple(int(v) for v in self.feasible_mask.shape) != (n_traces, self.sample_axis_len):
            msg = (
                'feasible_mask must have shape '
                f'({n_traces}, {self.sample_axis_len}), got {tuple(int(v) for v in self.feasible_mask.shape)}'
            )
            raise ValueError(msg)
        if tuple(int(v) for v in self.prior.shape) != (n_traces, self.sample_axis_len):
            msg = (
                'prior must have shape '
                f'({n_traces}, {self.sample_axis_len}), got {tuple(int(v) for v in self.prior.shape)}'
            )
            raise ValueError(msg)

        if self.center_time_sec is not None:
            _shape_mismatch('center_time_sec', n_traces, _shape_len(self.center_time_sec))
        if self.uncertainty_sec is not None:
            _shape_mismatch('uncertainty_sec', n_traces, _shape_len(self.uncertainty_sec))
        if self.trace_valid is not None:
            _shape_mismatch('trace_valid', n_traces, _shape_len(self.trace_valid))

        _validate_center_idx(self.center_idx, sample_axis_len=self.sample_axis_len, backend=backend)
        if backend == 'numpy':
            if np.any(self.uncertainty_idx <= 0.0):
                msg = 'uncertainty_idx must be > 0'
                raise ValueError(msg)
            if np.any(self.band_half_width_idx < 0):
                msg = 'band_half_width_idx must be >= 0'
                raise ValueError(msg)
            if np.any(self.prior < 0.0):
                msg = 'prior must be >= 0'
                raise ValueError(msg)
            row_sum = np.sum(self.prior, axis=-1, dtype=np.float32)
            if np.any(row_sum <= 0.0) or not np.allclose(row_sum, 1.0, atol=1e-4, rtol=0.0):
                msg = 'prior rows must be normalized to sum=1'
                raise ValueError(msg)
        else:
            if bool((self.uncertainty_idx <= 0.0).any().item()):
                msg = 'uncertainty_idx must be > 0'
                raise ValueError(msg)
            if bool((self.band_half_width_idx < 0).any().item()):
                msg = 'band_half_width_idx must be >= 0'
                raise ValueError(msg)
            if bool((self.prior < 0.0).any().item()):
                msg = 'prior must be >= 0'
                raise ValueError(msg)
            row_sum = self.prior.sum(dim=-1)
            if bool((row_sum <= 0.0).any().item()) or not bool(
                torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4, rtol=0.0)
            ):
                msg = 'prior rows must be normalized to sum=1'
                raise ValueError(msg)

    @property
    def backend(self) -> Backend:
        return _infer_backend(
            self.center_idx,
            self.uncertainty_idx,
            self.band_half_width_idx,
            self.feasible_mask,
            self.prior,
            self.center_time_sec,
            self.uncertainty_sec,
            self.trace_valid,
        )


def build_gaussian_arrival_prior(
    center_idx: ArrayLike,
    uncertainty_idx: ArrayLike,
    *,
    sample_axis_len: int,
    clip_floor: float = 0.0,
) -> ArrayLike:
    backend = _infer_backend(center_idx, uncertainty_idx)
    W = _validate_sample_axis_len(sample_axis_len)
    floor = _validate_non_negative_scalar(clip_floor, name='clip_floor')

    mu = _to_float32_1d(center_idx, name='center_idx', backend=backend)
    sigma = _to_float32_1d(uncertainty_idx, name='uncertainty_idx', backend=backend)
    require_same_shape_and_backend(
        mu,
        sigma,
        name_a='center_idx',
        name_b='uncertainty_idx',
        backend=backend,
        shape_hint='(N,)',
    )
    _validate_center_idx(mu, sample_axis_len=W, backend=backend)

    if backend == 'numpy':
        if np.any(sigma <= 0.0):
            msg = 'uncertainty_idx must be > 0'
            raise ValueError(msg)
        prior = gaussian_probs1d_np(mu, sigma, W)
        if floor > 0.0:
            prior = np.clip(prior, np.float32(floor), None)
            prior = _normalize_rows(prior, backend=backend)
        return prior.astype(np.float32, copy=False)

    if bool((sigma <= 0.0).any().item()):
        msg = 'uncertainty_idx must be > 0'
        raise ValueError(msg)
    prior = gaussian_probs1d_torch(mu, sigma, W)
    if floor > 0.0:
        prior = torch.clamp(prior, min=float(floor))
        prior = _normalize_rows(prior, backend=backend)
    return prior.to(dtype=torch.float32)


def build_hard_arrival_mask(
    center_idx: ArrayLike,
    band_half_width_idx: ArrayLike,
    *,
    sample_axis_len: int,
) -> ArrayLike:
    backend = _infer_backend(center_idx, band_half_width_idx)
    W = _validate_sample_axis_len(sample_axis_len)
    mu = _to_float32_1d(center_idx, name='center_idx', backend=backend)
    half = _to_int64_1d(
        band_half_width_idx,
        name='band_half_width_idx',
        backend=backend,
    )
    require_same_shape_and_backend(
        mu,
        half,
        name_a='center_idx',
        name_b='band_half_width_idx',
        backend=backend,
        shape_hint='(N,)',
    )
    _validate_center_idx(mu, sample_axis_len=W, backend=backend)

    if backend == 'numpy':
        if np.any(half < 0):
            msg = 'band_half_width_idx must be >= 0'
            raise ValueError(msg)
        bins = np.arange(W, dtype=np.float32)[None, :]
        return np.abs(bins - mu[:, None]) <= half[:, None]

    if bool((half < 0).any().item()):
        msg = 'band_half_width_idx must be >= 0'
        raise ValueError(msg)
    bins = torch.arange(W, device=mu.device, dtype=torch.float32).view(1, W)
    return (bins - mu.unsqueeze(1)).abs() <= half.unsqueeze(1).to(dtype=torch.float32)


def build_arrival_band(
    expected: ExpectedArrival,
    *,
    sample_axis_len: int,
    sample_interval_sec: float | None = None,
    uncertainty_scale: float = 1.0,
    band_radius_sigma: float = 2.0,
    min_half_width_idx: int = 1,
    prior_floor: float = 0.0,
) -> ArrivalBand:
    W = _validate_sample_axis_len(sample_axis_len)
    scale = _validate_positive_scalar(uncertainty_scale, name='uncertainty_scale')
    radius = _validate_positive_scalar(band_radius_sigma, name='band_radius_sigma')
    min_half = int(min_half_width_idx)
    if min_half < 0:
        msg = f'min_half_width_idx must be >= 0, got {min_half_width_idx}'
        raise ValueError(msg)

    center_idx, center_time_sec, uncertainty_idx, uncertainty_sec, backend = _resolve_expected_arrival(
        expected,
        sample_axis_len=W,
        sample_interval_sec=sample_interval_sec,
    )
    scaled_uncertainty_idx = uncertainty_idx * float(scale)
    prior = build_gaussian_arrival_prior(
        center_idx,
        scaled_uncertainty_idx,
        sample_axis_len=W,
        clip_floor=prior_floor,
    )

    if backend == 'numpy':
        band_half_width = np.maximum(
            np.int64(min_half),
            np.ceil(scaled_uncertainty_idx * np.float32(radius)).astype(np.int64),
        )
        scaled_uncertainty_sec = None
        if uncertainty_sec is not None:
            scaled_uncertainty_sec = (uncertainty_sec * np.float32(scale)).astype(
                np.float32,
                copy=False,
            )
    else:
        band_half_width = torch.ceil(
            scaled_uncertainty_idx * float(radius)
        ).clamp_min(float(min_half)).to(dtype=torch.int64)
        scaled_uncertainty_sec = None
        if uncertainty_sec is not None:
            scaled_uncertainty_sec = uncertainty_sec * float(scale)

    feasible_mask = build_hard_arrival_mask(
        center_idx,
        band_half_width,
        sample_axis_len=W,
    )

    return ArrivalBand(
        survey_id=expected.survey_id,
        backend_name=expected.backend_name,
        sample_axis_len=W,
        center_idx=center_idx,
        uncertainty_idx=scaled_uncertainty_idx,
        band_half_width_idx=band_half_width,
        feasible_mask=feasible_mask,
        prior=prior,
        center_time_sec=center_time_sec,
        uncertainty_sec=scaled_uncertainty_sec,
        trace_valid=expected.trace_valid,
    )


def pick_inside_arrival_band(
    arrival_band: ArrivalBand,
    pick_idx: ArrayLike,
) -> ArrayLike:
    backend = arrival_band.backend
    safe_idx, valid = _build_index_mask(
        pick_idx,
        sample_axis_len=arrival_band.sample_axis_len,
        backend=backend,
    )
    _shape_mismatch('pick_idx', arrival_band.center_idx.shape[0], _shape_len(safe_idx))

    rows = np.arange(_shape_len(safe_idx)) if backend == 'numpy' else torch.arange(
        _shape_len(safe_idx), device=safe_idx.device, dtype=torch.int64
    )
    inside = arrival_band.feasible_mask[rows, safe_idx]
    if backend == 'numpy':
        inside = inside.astype(bool, copy=False)
        inside[~valid] = False
        return inside

    inside = inside.to(dtype=torch.bool)
    inside[~valid] = False
    return inside


def gather_arrival_prior_at_pick(
    arrival_band: ArrivalBand,
    pick_idx: ArrayLike,
) -> ArrayLike:
    backend = arrival_band.backend
    safe_idx, valid = _build_index_mask(
        pick_idx,
        sample_axis_len=arrival_band.sample_axis_len,
        backend=backend,
    )
    _shape_mismatch('pick_idx', arrival_band.center_idx.shape[0], _shape_len(safe_idx))

    rows = np.arange(_shape_len(safe_idx)) if backend == 'numpy' else torch.arange(
        _shape_len(safe_idx), device=safe_idx.device, dtype=torch.int64
    )
    out = arrival_band.prior[rows, safe_idx]
    if backend == 'numpy':
        out = out.astype(np.float32, copy=False)
        out[~valid] = np.float32(0.0)
        return out

    out = out.to(dtype=torch.float32)
    out[~valid] = 0.0
    return out
