from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from seisai_pick.score.confidence_from_prob import trace_confidence_from_prob
from seisai_pick.score.confidence_from_trend_resid import (
    trace_confidence_from_trend_resid_gaussian,
)
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

from .arrival_band import ArrivalBand
from .consistency import ConsistencyResult

ArrayLike = Tensor | np.ndarray

_TORCH_INT_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


@dataclass(frozen=True)
class GlobalQCConfidence:
    confidence: ArrayLike
    probability_confidence: ArrayLike
    band_confidence: ArrayLike
    trend_confidence: ArrayLike
    consistency_confidence: ArrayLike
    valid_mask: ArrayLike


def _prepare_prob(prob: ArrayLike) -> Tensor:
    t_prob = to_torch(prob).to(dtype=torch.float32)
    validate_array(
        t_prob,
        allowed_ndims=(2,),
        name='prob',
        backend='torch',
        shape_hint='(N,W)',
    )
    require_float_array(t_prob, name='prob', backend='torch')
    require_all_finite(t_prob, name='prob', backend='torch')
    if bool((t_prob < 0.0).any().item()):
        msg = 'prob must be >= 0'
        raise ValueError(msg)
    return t_prob


def _prepare_pick_idx(pick_idx: ArrayLike) -> Tensor:
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


def _prepare_float_term(x: ArrayLike, *, like: Tensor, name: str) -> Tensor:
    t_x = to_torch(x, like=like).to(dtype=torch.float32)
    validate_array(
        t_x,
        allowed_ndims=(1,),
        name=name,
        backend='torch',
        shape_hint='(N,)',
    )
    require_float_array(t_x, name=name, backend='torch')
    require_all_finite(t_x, name=name, backend='torch')
    require_same_shape_and_backend(
        like,
        t_x,
        name_a='pick_idx',
        name_b=name,
        backend='torch',
        shape_hint='(N,)',
    )
    if bool((t_x < 0.0).any().item()) or bool((t_x > 1.0).any().item()):
        msg = f'{name} must stay within [0, 1]'
        raise ValueError(msg)
    return t_x


def _normalize_prob_rows(prob: Tensor, *, valid_mask: Tensor) -> Tensor:
    row_sum = prob.sum(dim=-1)
    bad = valid_mask & (row_sum <= 0.0)
    if bool(bad.any().item()):
        bad_rows = torch.nonzero(bad, as_tuple=False).view(-1).tolist()
        msg = f'prob rows must have positive mass for valid traces: {bad_rows}'
        raise ValueError(msg)
    out = torch.zeros_like(prob)
    if bool(valid_mask.any().item()):
        idx = torch.nonzero(valid_mask, as_tuple=False).view(-1)
        out[idx] = prob[idx] / row_sum[idx].unsqueeze(-1)
    return out


def _gather_prob_at_pick(prob: Tensor, pick_idx: Tensor, valid_mask: Tensor) -> Tensor:
    n_traces, width = int(prob.shape[0]), int(prob.shape[1])
    if bool((pick_idx >= width).any().item()):
        msg = f'pick_idx must be < prob.shape[1] ({width})'
        raise ValueError(msg)
    safe_idx = pick_idx.clone()
    safe_idx[~valid_mask] = 0
    rows = torch.arange(n_traces, device=prob.device, dtype=torch.int64)
    out = prob[rows, safe_idx].to(dtype=torch.float32)
    out[~valid_mask] = 0.0
    return out


def _zeros_like(valid_mask: Tensor) -> Tensor:
    return torch.zeros_like(valid_mask, dtype=torch.float32)


def _weight_value(value: float, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        msg = f'{name} must be finite and >= 0'
        raise ValueError(msg)
    return out


def probability_confidence_from_pick(
    prob: ArrayLike,
    pick_idx: ArrayLike,
    *,
    trace_valid: ArrayLike | None = None,
    entropy_floor: float = 0.2,
    entropy_power: float = 0.5,
) -> ArrayLike:
    floor = float(entropy_floor)
    power = float(entropy_power)
    if not np.isfinite(floor) or not (0.0 <= floor <= 1.0):
        msg = f'entropy_floor must be in [0, 1], got {entropy_floor}'
        raise ValueError(msg)
    if not np.isfinite(power) or power <= 0.0:
        msg = f'entropy_power must be finite and > 0, got {entropy_power}'
        raise ValueError(msg)

    all_numpy = require_all_numpy(prob, pick_idx, trace_valid)
    t_prob = _prepare_prob(prob)
    t_pick = _prepare_pick_idx(pick_idx)
    if int(t_prob.shape[0]) != int(t_pick.shape[0]):
        msg = f'prob and pick_idx length mismatch: {int(t_prob.shape[0])} vs {int(t_pick.shape[0])}'
        raise ValueError(msg)
    valid_mask = _prepare_valid_mask(t_pick, trace_valid)
    norm_prob = _normalize_prob_rows(t_prob, valid_mask=valid_mask)

    entropy_conf = trace_confidence_from_prob(
        norm_prob.unsqueeze(0),
        floor=float(floor),
        power=float(power),
    )
    entropy_conf_t = to_torch(entropy_conf, like=norm_prob).squeeze(0).to(dtype=torch.float32)
    pick_mass = _gather_prob_at_pick(norm_prob, t_pick, valid_mask)
    confidence = torch.sqrt(torch.clamp(entropy_conf_t * pick_mass, min=0.0))
    confidence = torch.where(valid_mask, confidence, torch.zeros_like(confidence))
    return to_numpy(confidence) if all_numpy else confidence


def band_confidence_from_arrival_band(
    arrival_band: ArrivalBand,
    pick_idx: ArrayLike,
    *,
    trace_valid: ArrayLike | None = None,
) -> ArrayLike:
    all_numpy = require_all_numpy(
        pick_idx,
        trace_valid,
        arrival_band.center_idx,
        arrival_band.uncertainty_idx,
        arrival_band.feasible_mask,
        arrival_band.trace_valid,
    )
    t_pick = _prepare_pick_idx(pick_idx)
    float_ref = t_pick.to(dtype=torch.float32)
    t_center = to_torch(arrival_band.center_idx, like=float_ref).to(dtype=torch.float32)
    t_unc = to_torch(arrival_band.uncertainty_idx, like=float_ref).to(dtype=torch.float32)
    validate_array(
        t_center,
        allowed_ndims=(1,),
        name='arrival_band.center_idx',
        backend='torch',
        shape_hint='(N,)',
    )
    validate_array(
        t_unc,
        allowed_ndims=(1,),
        name='arrival_band.uncertainty_idx',
        backend='torch',
        shape_hint='(N,)',
    )
    require_float_array(t_center, name='arrival_band.center_idx', backend='torch')
    require_float_array(t_unc, name='arrival_band.uncertainty_idx', backend='torch')
    require_all_finite(t_center, name='arrival_band.center_idx', backend='torch')
    require_all_finite(t_unc, name='arrival_band.uncertainty_idx', backend='torch')
    require_same_shape_and_backend(
        t_pick,
        t_center,
        t_unc,
        name_a='pick_idx',
        name_b='arrival_band.center_idx',
        other_names=['arrival_band.uncertainty_idx'],
        backend='torch',
        shape_hint='(N,)',
    )
    valid_mask = _prepare_valid_mask(t_pick, trace_valid)
    if arrival_band.trace_valid is not None:
        band_valid = _prepare_valid_mask(t_pick, arrival_band.trace_valid)
        valid_mask = valid_mask & band_valid

    t_feasible = to_torch(arrival_band.feasible_mask, like=t_pick.unsqueeze(-1))
    validate_array(
        t_feasible,
        allowed_ndims=(2,),
        name='arrival_band.feasible_mask',
        backend='torch',
        shape_hint='(N,W)',
    )
    require_boolint_array(t_feasible, name='arrival_band.feasible_mask', backend='torch')
    if int(t_feasible.shape[0]) != int(t_pick.shape[0]):
        msg = (
            'arrival_band.feasible_mask trace count must match pick_idx length, '
            f'got {int(t_feasible.shape[0])} vs {int(t_pick.shape[0])}'
        )
        raise ValueError(msg)
    safe_idx = t_pick.clone()
    safe_idx[~valid_mask] = 0
    if bool((safe_idx >= int(t_feasible.shape[1])).any().item()):
        msg = (
            'pick_idx must be < arrival_band.sample_axis_len, '
            f'got max={int(safe_idx.max().item())}, width={int(t_feasible.shape[1])}'
        )
        raise ValueError(msg)
    rows = torch.arange(int(t_pick.shape[0]), device=t_pick.device, dtype=torch.int64)
    inside = to_bool_mask_torch(t_feasible[rows, safe_idx], like=t_pick)
    delta = (t_pick.to(dtype=torch.float32) - t_center).abs()
    confidence = torch.exp(-0.5 * ((delta / t_unc) ** 2))
    confidence = torch.where(inside & valid_mask, confidence, torch.zeros_like(confidence))
    return to_numpy(confidence) if all_numpy else confidence


def trend_confidence_from_pick(
    pick_idx: ArrayLike,
    trend_center_idx: ArrayLike,
    *,
    trace_valid: ArrayLike | None = None,
    sample_interval_sec: float,
    sigma_idx: float = 6.0,
) -> ArrayLike:
    dt = float(sample_interval_sec)
    sigma = float(sigma_idx)
    if not np.isfinite(dt) or dt <= 0.0:
        msg = f'sample_interval_sec must be finite and > 0, got {sample_interval_sec}'
        raise ValueError(msg)
    if not np.isfinite(sigma) or sigma <= 0.0:
        msg = f'sigma_idx must be finite and > 0, got {sigma_idx}'
        raise ValueError(msg)

    all_numpy = require_all_numpy(pick_idx, trend_center_idx, trace_valid)
    t_pick = _prepare_pick_idx(pick_idx)
    valid_mask = _prepare_valid_mask(t_pick, trace_valid)
    float_ref = t_pick.to(dtype=torch.float32)
    t_trend = to_torch(trend_center_idx, like=float_ref).to(dtype=torch.float32)
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
        t_pick,
        t_trend,
        name_a='pick_idx',
        name_b='trend_center_idx',
        backend='torch',
        shape_hint='(N,)',
    )

    pick_time_sec = t_pick.to(dtype=torch.float32) * float(dt)
    trend_time_sec = t_trend * float(dt)
    sigma_ms = float(sigma) * float(dt) * 1e3
    confidence = trace_confidence_from_trend_resid_gaussian(
        pick_time_sec,
        trend_time_sec,
        valid=valid_mask,
        sigma_ms=float(sigma_ms),
    )
    confidence_t = to_torch(confidence, like=float_ref).to(dtype=torch.float32)
    confidence_t = torch.where(valid_mask, confidence_t, torch.zeros_like(confidence_t))
    return to_numpy(confidence_t) if all_numpy else confidence_t


def combine_confidence_terms(
    *,
    probability_confidence: ArrayLike | None = None,
    band_confidence: ArrayLike | None = None,
    trend_confidence: ArrayLike | None = None,
    consistency_confidence: ArrayLike | None = None,
    valid_mask: ArrayLike | None = None,
    probability_weight: float = 0.5,
    band_weight: float = 0.2,
    trend_weight: float = 0.15,
    consistency_weight: float = 0.15,
) -> ArrayLike:
    all_numpy = require_all_numpy(
        probability_confidence,
        band_confidence,
        trend_confidence,
        consistency_confidence,
        valid_mask,
    )

    ref = None
    for candidate in (
        probability_confidence,
        band_confidence,
        trend_confidence,
        consistency_confidence,
        valid_mask,
    ):
        if candidate is not None:
            ref = to_torch(candidate)
            break
    if ref is None:
        msg = 'at least one confidence term is required'
        raise ValueError(msg)

    validate_array(ref, allowed_ndims=(1,), name='reference', backend='torch', shape_hint='(N,)')
    n_traces = int(ref.shape[0])
    trace_ref = ref.to(dtype=torch.float32)

    prob_term = None if probability_confidence is None else _prepare_float_term(
        probability_confidence,
        like=trace_ref,
        name='probability_confidence',
    )
    band_term = None if band_confidence is None else _prepare_float_term(
        band_confidence,
        like=trace_ref,
        name='band_confidence',
    )
    trend_term = None if trend_confidence is None else _prepare_float_term(
        trend_confidence,
        like=trace_ref,
        name='trend_confidence',
    )
    consistency_term = None if consistency_confidence is None else _prepare_float_term(
        consistency_confidence,
        like=trace_ref,
        name='consistency_confidence',
    )

    weights: list[float] = []
    terms: list[Tensor] = []
    for term, weight, name in (
        (prob_term, probability_weight, 'probability_weight'),
        (band_term, band_weight, 'band_weight'),
        (trend_term, trend_weight, 'trend_weight'),
        (consistency_term, consistency_weight, 'consistency_weight'),
    ):
        w = _weight_value(weight, name=name)
        if term is not None and w > 0.0:
            weights.append(w)
            terms.append(term)

    if len(terms) == 0:
        msg = 'at least one confidence term with positive weight is required'
        raise ValueError(msg)

    denom = float(sum(weights))
    confidence = torch.zeros((n_traces,), dtype=torch.float32, device=trace_ref.device)
    for term, weight in zip(terms, weights, strict=True):
        confidence = confidence + term * float(weight / denom)

    if valid_mask is not None:
        t_valid = to_torch(valid_mask)
        if t_valid.device != trace_ref.device:
            t_valid = t_valid.to(device=trace_ref.device)
        validate_array(
            t_valid,
            allowed_ndims=(1,),
            name='valid_mask',
            backend='torch',
            shape_hint='(N,)',
        )
        require_boolint_array(t_valid, name='valid_mask', backend='torch')
        if int(t_valid.shape[0]) != n_traces:
            msg = f'valid_mask must have length {n_traces}, got {int(t_valid.shape[0])}'
            raise ValueError(msg)
        t_valid = to_bool_mask_torch(t_valid, like=trace_ref)
        confidence = torch.where(t_valid, confidence, torch.zeros_like(confidence))

    return to_numpy(confidence) if all_numpy else confidence


def compute_global_qc_confidence(
    pick_idx: ArrayLike,
    *,
    prob: ArrayLike | None = None,
    probability_confidence: ArrayLike | None = None,
    arrival_band: ArrivalBand | None = None,
    consistency: ConsistencyResult | ArrayLike | None = None,
    trend_center_idx: ArrayLike | None = None,
    trace_valid: ArrayLike | None = None,
    sample_interval_sec: float | None = None,
    probability_weight: float = 0.5,
    band_weight: float = 0.2,
    trend_weight: float = 0.15,
    consistency_weight: float = 0.15,
    entropy_floor: float = 0.2,
    entropy_power: float = 0.5,
    trend_sigma_idx: float = 6.0,
) -> GlobalQCConfidence:
    all_numpy = require_all_numpy(
        pick_idx,
        prob,
        probability_confidence,
        None if arrival_band is None else arrival_band.prior,
        None if arrival_band is None else arrival_band.feasible_mask,
        None if arrival_band is None else arrival_band.trace_valid,
        None if isinstance(consistency, ConsistencyResult) else consistency,
        trend_center_idx,
        trace_valid,
    )

    t_pick = _prepare_pick_idx(pick_idx)
    valid_mask = _prepare_valid_mask(t_pick, trace_valid)
    n_traces = int(t_pick.shape[0])
    float_ref = t_pick.to(dtype=torch.float32)

    prob_term = _zeros_like(valid_mask)
    if probability_confidence is not None:
        prob_term = _prepare_float_term(
            probability_confidence,
            like=float_ref,
            name='probability_confidence',
        )
    elif prob is not None:
        prob_term = to_torch(
            probability_confidence_from_pick(
                prob,
                t_pick,
                trace_valid=valid_mask,
                entropy_floor=float(entropy_floor),
                entropy_power=float(entropy_power),
            ),
            like=float_ref,
        ).to(dtype=torch.float32)

    band_term = _zeros_like(valid_mask)
    if arrival_band is not None:
        band_term = to_torch(
            band_confidence_from_arrival_band(
                arrival_band,
                t_pick,
                trace_valid=valid_mask,
            ),
            like=float_ref,
        ).to(dtype=torch.float32)

    trend_term = _zeros_like(valid_mask)
    if trend_center_idx is not None:
        if sample_interval_sec is None:
            msg = 'sample_interval_sec is required when trend_center_idx is provided'
            raise ValueError(msg)
        trend_term = to_torch(
            trend_confidence_from_pick(
                t_pick,
                trend_center_idx,
                trace_valid=valid_mask,
                sample_interval_sec=float(sample_interval_sec),
                sigma_idx=float(trend_sigma_idx),
            ),
            like=float_ref,
        ).to(dtype=torch.float32)

    consistency_term = _zeros_like(valid_mask)
    if consistency is not None:
        if isinstance(consistency, ConsistencyResult):
            consistency_term = _prepare_float_term(
                consistency.score,
                like=float_ref,
                name='consistency.score',
            )
        else:
            consistency_term = _prepare_float_term(
                consistency,
                like=float_ref,
                name='consistency_confidence',
            )

    confidence = combine_confidence_terms(
        probability_confidence=prob_term,
        band_confidence=band_term,
        trend_confidence=trend_term,
        consistency_confidence=consistency_term,
        valid_mask=valid_mask,
        probability_weight=float(probability_weight),
        band_weight=float(band_weight),
        trend_weight=float(trend_weight),
        consistency_weight=float(consistency_weight),
    )
    t_conf = to_torch(confidence, like=float_ref).to(dtype=torch.float32)

    if int(t_conf.shape[0]) != n_traces:
        msg = f'confidence must have length {n_traces}, got {int(t_conf.shape[0])}'
        raise ValueError(msg)

    out_conf, out_prob, out_band, out_trend, out_consistency, out_valid = (
        to_numpy(t_conf),
        to_numpy(prob_term),
        to_numpy(band_term),
        to_numpy(trend_term),
        to_numpy(consistency_term),
        to_numpy(valid_mask),
    ) if all_numpy else (t_conf, prob_term, band_term, trend_term, consistency_term, valid_mask)

    return GlobalQCConfidence(
        confidence=out_conf,
        probability_confidence=out_prob,
        band_confidence=out_band,
        trend_confidence=out_trend,
        consistency_confidence=out_consistency,
        valid_mask=out_valid,
    )
