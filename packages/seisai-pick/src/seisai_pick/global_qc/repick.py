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

from .arrival_band import ArrivalBand

ArrayLike = Tensor | np.ndarray


@dataclass(frozen=True)
class RepickResult:
    pick_idx: ArrayLike
    confidence: ArrayLike
    reject_mask: ArrayLike
    reweighted_prob: ArrayLike | None = None


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


def _prepare_prior(prior: ArrayLike, *, like: Tensor) -> Tensor:
    t_prior = to_torch(prior, like=like).to(dtype=torch.float32)
    validate_array(
        t_prior,
        allowed_ndims=(2,),
        name='prior',
        backend='torch',
        shape_hint='(N,W)',
    )
    require_float_array(t_prior, name='prior', backend='torch')
    require_all_finite(t_prior, name='prior', backend='torch')
    require_same_shape_and_backend(
        like,
        t_prior,
        name_a='prob',
        name_b='prior',
        backend='torch',
        shape_hint='(N,W)',
    )
    if bool((t_prior < 0.0).any().item()):
        msg = 'prior must be >= 0'
        raise ValueError(msg)
    return t_prior


def _prepare_mask(mask: ArrayLike, *, like: Tensor, name: str, shape_hint: str) -> Tensor:
    t_mask = to_torch(mask)
    if t_mask.device != like.device:
        t_mask = t_mask.to(device=like.device)
    allowed_ndims = (2,) if shape_hint == '(N,W)' else (1,)
    validate_array(
        t_mask,
        allowed_ndims=allowed_ndims,
        name=name,
        backend='torch',
        shape_hint=shape_hint,
    )
    require_boolint_array(t_mask, name=name, backend='torch')
    require_same_shape_and_backend(
        like,
        t_mask,
        name_a='prob' if like.ndim == 2 else 'trace_ref',
        name_b=name,
        backend='torch',
        shape_hint=shape_hint,
    )
    return to_bool_mask_torch(t_mask, like=like)


def _prepare_trace_mask(
    mask: ArrayLike | None,
    *,
    n_traces: int,
    like: Tensor,
    name: str,
) -> Tensor:
    if mask is None:
        return torch.zeros((n_traces,), dtype=torch.bool, device=like.device)

    t_mask = to_torch(mask)
    if t_mask.device != like.device:
        t_mask = t_mask.to(device=like.device)
    validate_array(
        t_mask,
        allowed_ndims=(1,),
        name=name,
        backend='torch',
        shape_hint='(N,)',
    )
    require_boolint_array(t_mask, name=name, backend='torch')
    if int(t_mask.shape[0]) != int(n_traces):
        msg = f'{name} must have length {n_traces}, got {int(t_mask.shape[0])}'
        raise ValueError(msg)
    return to_bool_mask_torch(t_mask, like=like)


def _normalize_active_rows(x: Tensor, *, active_mask: Tensor, name: str) -> Tensor:
    row_sum = x.sum(dim=-1)
    bad = active_mask & (row_sum <= 0.0)
    if bool(bad.any().item()):
        bad_rows = torch.nonzero(bad, as_tuple=False).view(-1).tolist()
        msg = f'{name} produced zero feasible mass for active rows: {bad_rows}'
        raise ValueError(msg)

    out = torch.zeros_like(x)
    if bool(active_mask.any().item()):
        idx = torch.nonzero(active_mask, as_tuple=False).view(-1)
        out[idx] = x[idx] / row_sum[idx].unsqueeze(-1)
    return out


def _combined_inputs(
    prob: Tensor,
    *,
    arrival_band: ArrivalBand | None,
    prior: ArrayLike | None,
    feasible_mask: ArrayLike | None,
    invalid_trace_mask: ArrayLike | None,
    reject_trace_mask: ArrayLike | None,
) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor]:
    n_traces = int(prob.shape[0])
    combined_prior = None if prior is None else _prepare_prior(prior, like=prob)
    combined_mask = None
    if feasible_mask is not None:
        combined_mask = _prepare_mask(
            feasible_mask,
            like=prob,
            name='feasible_mask',
            shape_hint='(N,W)',
        )

    invalid_mask = _prepare_trace_mask(
        invalid_trace_mask,
        n_traces=n_traces,
        like=prob,
        name='invalid_trace_mask',
    )
    reject_mask = _prepare_trace_mask(
        reject_trace_mask,
        n_traces=n_traces,
        like=prob,
        name='reject_trace_mask',
    )

    if arrival_band is not None:
        if int(arrival_band.center_idx.shape[0]) != n_traces:
            msg = (
                'arrival_band trace count must match prob rows, '
                f'got {int(arrival_band.center_idx.shape[0])} vs {n_traces}'
            )
            raise ValueError(msg)
        if int(arrival_band.sample_axis_len) != int(prob.shape[1]):
            msg = (
                'arrival_band.sample_axis_len must match prob width, '
                f'got {int(arrival_band.sample_axis_len)} vs {int(prob.shape[1])}'
            )
            raise ValueError(msg)

        band_prior = to_torch(arrival_band.prior, like=prob).to(dtype=torch.float32)
        band_mask = to_bool_mask_torch(arrival_band.feasible_mask, like=prob)
        require_same_shape_and_backend(
            prob,
            band_prior,
            band_mask,
            name_a='prob',
            name_b='arrival_band.prior',
            other_names=['arrival_band.feasible_mask'],
            backend='torch',
            shape_hint='(N,W)',
        )

        combined_prior = band_prior if combined_prior is None else combined_prior * band_prior
        combined_mask = band_mask if combined_mask is None else (combined_mask & band_mask)

        if arrival_band.trace_valid is not None:
            band_valid = _prepare_trace_mask(
                arrival_band.trace_valid,
                n_traces=n_traces,
                like=prob,
                name='arrival_band.trace_valid',
            )
            reject_mask = reject_mask | (~band_valid)

    return combined_prior, combined_mask, invalid_mask, reject_mask


def reweight_probability_with_prior(
    prob: ArrayLike,
    *,
    prior: ArrayLike | None = None,
    feasible_mask: ArrayLike | None = None,
    invalid_trace_mask: ArrayLike | None = None,
    reject_trace_mask: ArrayLike | None = None,
    prior_power: float = 1.0,
) -> ArrayLike:
    power = float(prior_power)
    if not np.isfinite(power) or power <= 0.0:
        msg = f'prior_power must be finite and > 0, got {prior_power}'
        raise ValueError(msg)

    all_numpy = require_all_numpy(
        prob,
        prior,
        feasible_mask,
        invalid_trace_mask,
        reject_trace_mask,
    )
    t_prob = _prepare_prob(prob)
    n_traces = int(t_prob.shape[0])

    prior_t = None if prior is None else _prepare_prior(prior, like=t_prob)
    mask_t = None
    if feasible_mask is not None:
        mask_t = _prepare_mask(
            feasible_mask,
            like=t_prob,
            name='feasible_mask',
            shape_hint='(N,W)',
        )

    invalid_mask = _prepare_trace_mask(
        invalid_trace_mask,
        n_traces=n_traces,
        like=t_prob,
        name='invalid_trace_mask',
    )
    reject_mask = _prepare_trace_mask(
        reject_trace_mask,
        n_traces=n_traces,
        like=t_prob,
        name='reject_trace_mask',
    )
    active = ~(invalid_mask | reject_mask)

    base = _normalize_active_rows(t_prob, active_mask=active, name='prob')
    if prior_t is not None:
        base = base * torch.pow(prior_t, float(power))
    if mask_t is not None:
        base = torch.where(mask_t, base, torch.zeros_like(base))
    out = _normalize_active_rows(base, active_mask=active, name='prob * prior')

    return to_numpy(out) if all_numpy else out


def argmax_with_hard_mask(
    prob: ArrayLike,
    feasible_mask: ArrayLike,
    *,
    invalid_trace_mask: ArrayLike | None = None,
    reject_trace_mask: ArrayLike | None = None,
) -> ArrayLike:
    all_numpy = require_all_numpy(prob, feasible_mask, invalid_trace_mask, reject_trace_mask)
    reweighted = reweight_probability_with_prior(
        prob,
        feasible_mask=feasible_mask,
        invalid_trace_mask=invalid_trace_mask,
        reject_trace_mask=reject_trace_mask,
    )
    t_reweighted = to_torch(reweighted).to(dtype=torch.float32)
    n_traces = int(t_reweighted.shape[0])
    invalid_mask = _prepare_trace_mask(
        invalid_trace_mask,
        n_traces=n_traces,
        like=t_reweighted,
        name='invalid_trace_mask',
    )
    reject_mask = _prepare_trace_mask(
        reject_trace_mask,
        n_traces=n_traces,
        like=t_reweighted,
        name='reject_trace_mask',
    )
    active = ~(invalid_mask | reject_mask)

    pick_idx = torch.argmax(t_reweighted, dim=-1).to(dtype=torch.int64)
    pick_idx = torch.where(active, pick_idx, torch.full_like(pick_idx, -1))
    return to_numpy(pick_idx) if all_numpy else pick_idx


def repick_with_arrival_band(
    prob: ArrayLike,
    *,
    arrival_band: ArrivalBand | None = None,
    prior: ArrayLike | None = None,
    feasible_mask: ArrayLike | None = None,
    invalid_trace_mask: ArrayLike | None = None,
    reject_trace_mask: ArrayLike | None = None,
    prior_power: float = 1.0,
    return_reweighted_prob: bool = False,
) -> RepickResult:
    power = float(prior_power)
    if not np.isfinite(power) or power <= 0.0:
        msg = f'prior_power must be finite and > 0, got {prior_power}'
        raise ValueError(msg)

    all_numpy = require_all_numpy(
        prob,
        prior,
        feasible_mask,
        invalid_trace_mask,
        reject_trace_mask,
        None if arrival_band is None else arrival_band.prior,
        None if arrival_band is None else arrival_band.feasible_mask,
        None if arrival_band is None else arrival_band.trace_valid,
    )
    t_prob = _prepare_prob(prob)
    n_traces = int(t_prob.shape[0])

    combined_prior, combined_mask, invalid_mask, reject_mask = _combined_inputs(
        t_prob,
        arrival_band=arrival_band,
        prior=prior,
        feasible_mask=feasible_mask,
        invalid_trace_mask=invalid_trace_mask,
        reject_trace_mask=reject_trace_mask,
    )
    active = ~(invalid_mask | reject_mask)

    normalized = _normalize_active_rows(t_prob, active_mask=active, name='prob')
    if combined_prior is not None:
        normalized = normalized * torch.pow(combined_prior, float(power))
    if combined_mask is not None:
        normalized = torch.where(combined_mask, normalized, torch.zeros_like(normalized))
    reweighted = _normalize_active_rows(normalized, active_mask=active, name='prob * prior')

    pick_idx = torch.argmax(reweighted, dim=-1).to(dtype=torch.int64)
    confidence = torch.max(reweighted, dim=-1).values.to(dtype=torch.float32)
    pick_idx = torch.where(active, pick_idx, torch.full_like(pick_idx, -1))
    confidence = torch.where(active, confidence, torch.zeros_like(confidence))

    out_pick, out_conf, out_reject = (
        to_numpy(pick_idx),
        to_numpy(confidence),
        to_numpy(reject_mask),
    ) if all_numpy else (pick_idx, confidence, reject_mask)

    out_reweighted = None
    if return_reweighted_prob:
        out_reweighted = to_numpy(reweighted) if all_numpy else reweighted

    return RepickResult(
        pick_idx=out_pick,
        confidence=out_conf,
        reject_mask=out_reject,
        reweighted_prob=out_reweighted,
    )
