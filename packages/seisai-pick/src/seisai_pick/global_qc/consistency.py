from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from seisai_utils.convert import to_bool_mask_torch, to_numpy, to_torch
from seisai_utils.validator import (
    require_all_finite,
    require_all_numpy,
    require_boolint_array,
    require_float_array,
    require_same_shape_and_backend,
    validate_array,
)
from torch import Tensor

ArrayLike = Tensor | np.ndarray

_TORCH_INT_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


@dataclass(frozen=True)
class ConsistencyResult:
    score: ArrayLike
    adjacent_smoothness: ArrayLike
    trend_alignment: ArrayLike
    outlier_score: ArrayLike
    adjacent_residual_idx: ArrayLike
    trend_residual_idx: ArrayLike
    robust_outlier_z: ArrayLike
    support_count: ArrayLike
    valid_mask: ArrayLike


def _validate_pick_idx(pick_idx: ArrayLike) -> Tensor:
    t_pick = to_torch(pick_idx)
    validate_array(
        t_pick,
        allowed_ndims=(1,),
        name='pick_idx',
        backend='torch',
        shape_hint='(N,)',
    )
    if t_pick.dtype not in _TORCH_INT_DTYPES:
        msg = 'pick_idx must be an integer array/tensor'
        raise TypeError(msg)
    t_pick = t_pick.to(dtype=torch.int64)
    if bool((t_pick < -1).any().item()):
        msg = 'pick_idx may contain -1 for invalid traces, but no value < -1'
        raise ValueError(msg)
    return t_pick


def _prepare_valid_mask(pick_idx: Tensor, trace_valid: ArrayLike | None) -> Tensor:
    valid = pick_idx >= 0
    if trace_valid is None:
        return valid

    t_valid = to_torch(trace_valid, like=pick_idx)
    validate_array(
        t_valid,
        allowed_ndims=(1,),
        name='trace_valid',
        backend='torch',
        shape_hint='(N,)',
    )
    require_boolint_array(t_valid, name='trace_valid', backend='torch')
    require_same_shape_and_backend(
        pick_idx,
        t_valid,
        name_a='pick_idx',
        name_b='trace_valid',
        backend='torch',
        shape_hint='(N,)',
    )
    return valid & to_bool_mask_torch(t_valid, like=pick_idx)


def _prepare_trend_center(trend_center_idx: ArrayLike, *, like: Tensor) -> Tensor:
    t_trend = to_torch(trend_center_idx)
    if t_trend.device != like.device:
        t_trend = t_trend.to(device=like.device)
    t_trend = t_trend.to(dtype=torch.float32)
    validate_array(
        t_trend,
        allowed_ndims=(1,),
        name='trend_center_idx',
        backend='torch',
        shape_hint='(N,)',
    )
    require_float_array(t_trend, name='trend_center_idx', backend='torch')
    require_all_finite(t_trend, name='trend_center_idx', backend='torch')
    require_same_shape_and_backend(
        like,
        t_trend,
        name_a='pick_idx',
        name_b='trend_center_idx',
        backend='torch',
        shape_hint='(N,)',
    )
    return t_trend


def _validate_positive_float(value: float, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        msg = f'{name} must be finite and > 0'
        raise ValueError(msg)
    return out


def _validate_non_negative_float(value: float, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        msg = f'{name} must be finite and >= 0'
        raise ValueError(msg)
    return out


def _to_output(all_numpy: bool, *xs: Tensor) -> tuple[ArrayLike, ...]:
    if all_numpy:
        out = to_numpy(*xs)
        if isinstance(out, tuple):
            return out
        return (out,)
    return xs


def _adjacent_components(
    pick_idx: Tensor,
    valid_mask: Tensor,
    *,
    neighbor_radius: int,
    sigma_idx: float,
    min_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    n = int(pick_idx.shape[0])
    pick_f = pick_idx.to(dtype=torch.float32)
    score = torch.zeros((n,), dtype=torch.float32, device=pick_idx.device)
    residual = torch.zeros((n,), dtype=torch.float32, device=pick_idx.device)
    support = torch.zeros((n,), dtype=torch.bool, device=pick_idx.device)

    for i in range(n):
        if not bool(valid_mask[i].item()):
            continue
        left = max(0, i - int(neighbor_radius))
        right = min(n, i + int(neighbor_radius) + 1)
        neighbor_valid = valid_mask[left:right].clone()
        neighbor_valid[i - left] = False
        if int(neighbor_valid.count_nonzero()) < int(min_count):
            continue

        neighbors = pick_f[left:right][neighbor_valid]
        local_center = neighbors.median()
        delta = (pick_f[i] - local_center).abs()
        residual[i] = delta
        score[i] = torch.exp(-0.5 * ((delta / float(sigma_idx)) ** 2))
        support[i] = True

    return score, residual, support


def _trend_components(
    pick_idx: Tensor,
    valid_mask: Tensor,
    trend_center_idx: Tensor,
    *,
    sigma_idx: float,
) -> tuple[Tensor, Tensor, Tensor]:
    pick_f = pick_idx.to(dtype=torch.float32)
    residual = (pick_f - trend_center_idx).abs()
    score = torch.exp(-0.5 * ((residual / float(sigma_idx)) ** 2))
    score = torch.where(valid_mask, score, torch.zeros_like(score))
    residual = torch.where(valid_mask, residual, torch.zeros_like(residual))
    return score, residual, valid_mask.clone()


def _outlier_components(
    pick_idx: Tensor,
    valid_mask: Tensor,
    *,
    neighbor_radius: int,
    z_scale: float,
    min_count: int,
    mad_floor_idx: float,
) -> tuple[Tensor, Tensor, Tensor]:
    n = int(pick_idx.shape[0])
    pick_f = pick_idx.to(dtype=torch.float32)
    score = torch.zeros((n,), dtype=torch.float32, device=pick_idx.device)
    robust_z = torch.zeros((n,), dtype=torch.float32, device=pick_idx.device)
    support = torch.zeros((n,), dtype=torch.bool, device=pick_idx.device)

    for i in range(n):
        if not bool(valid_mask[i].item()):
            continue
        left = max(0, i - int(neighbor_radius))
        right = min(n, i + int(neighbor_radius) + 1)
        neighbor_valid = valid_mask[left:right].clone()
        neighbor_valid[i - left] = False
        if int(neighbor_valid.count_nonzero()) < int(min_count):
            continue

        neighbors = pick_f[left:right][neighbor_valid]
        local_center = neighbors.median()
        mad = (neighbors - local_center).abs().median()
        scale = torch.clamp(1.4826 * mad, min=float(mad_floor_idx))
        z = (pick_f[i] - local_center).abs() / scale
        robust_z[i] = z
        score[i] = torch.exp(-0.5 * ((z / float(z_scale)) ** 2))
        support[i] = True

    return score, robust_z, support


def adjacent_trace_smoothness_score(
    pick_idx: ArrayLike,
    *,
    trace_valid: ArrayLike | None = None,
    neighbor_radius: int = 1,
    sigma_idx: float = 4.0,
    min_count: int = 1,
) -> tuple[ArrayLike, ArrayLike]:
    if neighbor_radius < 1:
        msg = f'neighbor_radius must be >= 1, got {neighbor_radius}'
        raise ValueError(msg)
    if min_count < 1:
        msg = f'min_count must be >= 1, got {min_count}'
        raise ValueError(msg)
    sigma = _validate_positive_float(sigma_idx, name='sigma_idx')

    all_numpy = require_all_numpy(pick_idx, trace_valid)
    t_pick = _validate_pick_idx(pick_idx)
    valid_mask = _prepare_valid_mask(t_pick, trace_valid)
    score, residual, _ = _adjacent_components(
        t_pick,
        valid_mask,
        neighbor_radius=int(neighbor_radius),
        sigma_idx=float(sigma),
        min_count=int(min_count),
    )
    return _to_output(all_numpy, score, residual)


def trend_residual_score(
    pick_idx: ArrayLike,
    trend_center_idx: ArrayLike,
    *,
    trace_valid: ArrayLike | None = None,
    sigma_idx: float = 6.0,
) -> tuple[ArrayLike, ArrayLike]:
    sigma = _validate_positive_float(sigma_idx, name='sigma_idx')

    all_numpy = require_all_numpy(pick_idx, trend_center_idx, trace_valid)
    t_pick = _validate_pick_idx(pick_idx)
    valid_mask = _prepare_valid_mask(t_pick, trace_valid)
    t_trend = _prepare_trend_center(trend_center_idx, like=t_pick)
    score, residual, _ = _trend_components(
        t_pick,
        valid_mask,
        t_trend,
        sigma_idx=float(sigma),
    )
    return _to_output(all_numpy, score, residual)


def local_outlier_score(
    pick_idx: ArrayLike,
    *,
    trace_valid: ArrayLike | None = None,
    neighbor_radius: int = 4,
    z_scale: float = 3.0,
    min_count: int = 3,
    mad_floor_idx: float = 1.0,
) -> tuple[ArrayLike, ArrayLike]:
    if neighbor_radius < 1:
        msg = f'neighbor_radius must be >= 1, got {neighbor_radius}'
        raise ValueError(msg)
    if min_count < 1:
        msg = f'min_count must be >= 1, got {min_count}'
        raise ValueError(msg)
    z_scale_v = _validate_positive_float(z_scale, name='z_scale')
    mad_floor_v = _validate_positive_float(mad_floor_idx, name='mad_floor_idx')

    all_numpy = require_all_numpy(pick_idx, trace_valid)
    t_pick = _validate_pick_idx(pick_idx)
    valid_mask = _prepare_valid_mask(t_pick, trace_valid)
    score, robust_z, _ = _outlier_components(
        t_pick,
        valid_mask,
        neighbor_radius=int(neighbor_radius),
        z_scale=float(z_scale_v),
        min_count=int(min_count),
        mad_floor_idx=float(mad_floor_v),
    )
    return _to_output(all_numpy, score, robust_z)


def compute_global_consistency(
    pick_idx: ArrayLike,
    *,
    trace_valid: ArrayLike | None = None,
    trend_center_idx: ArrayLike | None = None,
    adjacent_radius: int = 1,
    adjacent_sigma_idx: float = 4.0,
    adjacent_min_count: int = 1,
    trend_sigma_idx: float = 6.0,
    outlier_radius: int = 4,
    outlier_z_scale: float = 3.0,
    outlier_min_count: int = 3,
    outlier_mad_floor_idx: float = 1.0,
    adjacent_weight: float = 0.4,
    trend_weight: float = 0.35,
    outlier_weight: float = 0.25,
) -> ConsistencyResult:
    if adjacent_radius < 1:
        msg = f'adjacent_radius must be >= 1, got {adjacent_radius}'
        raise ValueError(msg)
    if adjacent_min_count < 1:
        msg = f'adjacent_min_count must be >= 1, got {adjacent_min_count}'
        raise ValueError(msg)
    if outlier_radius < 1:
        msg = f'outlier_radius must be >= 1, got {outlier_radius}'
        raise ValueError(msg)
    if outlier_min_count < 1:
        msg = f'outlier_min_count must be >= 1, got {outlier_min_count}'
        raise ValueError(msg)

    adj_sigma = _validate_positive_float(adjacent_sigma_idx, name='adjacent_sigma_idx')
    trend_sigma = _validate_positive_float(trend_sigma_idx, name='trend_sigma_idx')
    outlier_sigma = _validate_positive_float(outlier_z_scale, name='outlier_z_scale')
    mad_floor = _validate_positive_float(
        outlier_mad_floor_idx,
        name='outlier_mad_floor_idx',
    )
    adj_w = _validate_non_negative_float(adjacent_weight, name='adjacent_weight')
    tr_w = _validate_non_negative_float(trend_weight, name='trend_weight')
    out_w = _validate_non_negative_float(outlier_weight, name='outlier_weight')
    if adj_w == 0.0 and tr_w == 0.0 and out_w == 0.0:
        msg = 'at least one consistency weight must be > 0'
        raise ValueError(msg)

    all_numpy = require_all_numpy(pick_idx, trace_valid, trend_center_idx)
    t_pick = _validate_pick_idx(pick_idx)
    valid_mask = _prepare_valid_mask(t_pick, trace_valid)

    adjacent_score, adjacent_resid, adjacent_support = _adjacent_components(
        t_pick,
        valid_mask,
        neighbor_radius=int(adjacent_radius),
        sigma_idx=float(adj_sigma),
        min_count=int(adjacent_min_count),
    )

    if trend_center_idx is None:
        trend_score = torch.zeros_like(adjacent_score)
        trend_resid = torch.zeros_like(adjacent_score)
        trend_support = torch.zeros_like(valid_mask)
    else:
        t_trend = _prepare_trend_center(trend_center_idx, like=t_pick)
        trend_score, trend_resid, trend_support = _trend_components(
            t_pick,
            valid_mask,
            t_trend,
            sigma_idx=float(trend_sigma),
        )

    outlier_score, robust_z, outlier_support = _outlier_components(
        t_pick,
        valid_mask,
        neighbor_radius=int(outlier_radius),
        z_scale=float(outlier_sigma),
        min_count=int(outlier_min_count),
        mad_floor_idx=float(mad_floor),
    )

    denom = (
        adjacent_support.to(dtype=torch.float32) * float(adj_w)
        + trend_support.to(dtype=torch.float32) * float(tr_w)
        + outlier_support.to(dtype=torch.float32) * float(out_w)
    )
    numer = (
        adjacent_score * float(adj_w) * adjacent_support.to(dtype=torch.float32)
        + trend_score * float(tr_w) * trend_support.to(dtype=torch.float32)
        + outlier_score * float(out_w) * outlier_support.to(dtype=torch.float32)
    )
    score = torch.where(denom > 0.0, numer / denom, torch.zeros_like(numer))
    score = torch.where(valid_mask, score, torch.zeros_like(score))

    support_count = (
        adjacent_support.to(dtype=torch.int32)
        + trend_support.to(dtype=torch.int32)
        + outlier_support.to(dtype=torch.int32)
    )

    out_score, out_adj, out_trend, out_outlier, out_adj_resid, out_trend_resid, out_z, out_support, out_valid = _to_output(
        all_numpy,
        score,
        adjacent_score,
        trend_score,
        outlier_score,
        adjacent_resid,
        trend_resid,
        robust_z,
        support_count,
        valid_mask,
    )
    return ConsistencyResult(
        score=out_score,
        adjacent_smoothness=out_adj,
        trend_alignment=out_trend,
        outlier_score=out_outlier,
        adjacent_residual_idx=out_adj_resid,
        trend_residual_idx=out_trend_resid,
        robust_outlier_z=out_z,
        support_count=out_support,
        valid_mask=out_valid,
    )
