from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from seisai_engine.loss import composite
from seisai_engine.loss.soft_label_ce import build_pixel_mask_from_batch

__all__ = ['build_coarse_criterion']


def _normalize_scope(scope: str, *, label: str) -> str:
    if not isinstance(scope, str):
        raise TypeError(f'{label} must be str')
    scope_norm = scope.lower()
    if scope_norm not in ('all', 'masked_only'):
        raise ValueError(f'{label} must be "all" or "masked_only"')
    return scope_norm


def _validate_params_keys(
    params: dict[str, Any],
    *,
    allowed: tuple[str, ...],
    required: tuple[str, ...],
    label: str,
) -> None:
    if not isinstance(params, dict):
        raise TypeError(f'{label}.params must be dict')
    allowed_set = set(allowed)
    required_set = set(required)

    missing = required_set.difference(params.keys())
    if missing:
        missing_txt = ', '.join(sorted(missing))
        raise ValueError(f'{label}.params missing required keys: {missing_txt}')

    extra = set(params.keys()).difference(allowed_set)
    if extra:
        extra_txt = ', '.join(sorted(extra))
        raise ValueError(f'{label}.params has unknown keys: {extra_txt}')


def _prepare_logits_target(
    *,
    logits: torch.Tensor,
    target: torch.Tensor,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(logits, torch.Tensor) or int(logits.ndim) != 4:
        raise ValueError(f'{label}: logits must be torch.Tensor with shape (B,1,H,W)')
    if not isinstance(target, torch.Tensor) or int(target.ndim) != 4:
        raise ValueError(f'{label}: target must be torch.Tensor with shape (B,1,H,W)')
    if tuple(target.shape) != tuple(logits.shape):
        raise ValueError(
            f'{label}: target shape {tuple(target.shape)} must match logits shape {tuple(logits.shape)}'
        )
    if int(logits.shape[1]) != 1:
        raise ValueError(
            f'{label}: coarse logits channel dim must be 1, got {int(logits.shape[1])}'
        )
    if int(target.shape[1]) != 1:
        raise ValueError(
            f'{label}: coarse target channel dim must be 1, got {int(target.shape[1])}'
        )

    target_tensor = target
    if target_tensor.dtype != logits.dtype:
        target_tensor = target_tensor.to(dtype=logits.dtype)
    if target_tensor.device != logits.device:
        target_tensor = target_tensor.to(device=logits.device, non_blocking=True)
    return logits, target_tensor


def _resolve_trace_mask(
    *,
    scope: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    batch: dict[str, Any],
    use_label_valid: bool,
    label: str,
) -> torch.Tensor:
    if not isinstance(batch, dict):
        raise TypeError('batch must be dict[str, Any]')

    mask_bool_key = '__ignored_mask_bool_for_scope_all__'
    if scope == 'masked_only':
        mask_bool_key = 'mask_bool'

    batch_for_mask = batch
    if 'target' not in batch and 'input' not in batch:
        batch_for_mask = dict(batch)
        batch_for_mask['target'] = target

    pixel_mask = build_pixel_mask_from_batch(
        batch_for_mask,
        use_trace_valid=True,
        use_label_valid=use_label_valid,
        mask_bool_key=mask_bool_key,
    )
    if pixel_mask.device != logits.device:
        pixel_mask = pixel_mask.to(device=logits.device, non_blocking=True)

    trace_any = pixel_mask.any(dim=-1)
    trace_all = pixel_mask.all(dim=-1)
    if not torch.equal(trace_any, trace_all):
        raise ValueError(
            f'{label}: coarse probability loss only supports full-trace masking along W'
        )
    return trace_any


def _normalize_target_prob(
    *,
    logits: torch.Tensor,
    target: torch.Tensor,
    trace_mask: torch.Tensor,
    label: str,
) -> torch.Tensor:
    logits_tensor, target_tensor = _prepare_logits_target(
        logits=logits,
        target=target,
        label=label,
    )
    if not isinstance(trace_mask, torch.Tensor) or trace_mask.dtype is not torch.bool:
        raise TypeError(f'{label}: trace mask must be bool tensor')
    if tuple(trace_mask.shape) != (int(logits_tensor.shape[0]), int(logits_tensor.shape[2])):
        raise ValueError(
            f'{label}: trace mask shape mismatch: {tuple(trace_mask.shape)} vs '
            f'({int(logits_tensor.shape[0])},{int(logits_tensor.shape[2])})'
        )

    mass = target_tensor.sum(dim=-1, keepdim=True)
    mass_bh = mass[:, 0, :, 0]
    if torch.any(trace_mask & (mass_bh <= 0.0)):
        raise ValueError(
            f'{label}: selected traces must have positive target mass along W; '
            'enable label_valid masking or fix the coarse target construction'
        )

    eps = torch.finfo(logits_tensor.dtype).eps
    return torch.where(
        mass > 0.0,
        target_tensor / mass.clamp_min(eps),
        torch.zeros_like(target_tensor),
    )


def _masked_mean(
    *,
    per_trace_loss: torch.Tensor,
    trace_mask: torch.Tensor,
    label: str,
) -> torch.Tensor:
    if not isinstance(per_trace_loss, torch.Tensor) or int(per_trace_loss.ndim) != 2:
        raise ValueError(f'{label}: per-trace loss must have shape (B,H)')
    if tuple(per_trace_loss.shape) != tuple(trace_mask.shape):
        raise ValueError(
            f'{label}: trace mask shape mismatch: {tuple(trace_mask.shape)} vs {tuple(per_trace_loss.shape)}'
        )

    selected = per_trace_loss[trace_mask]
    if int(selected.numel()) <= 0:
        raise ValueError(f'{label}: no traces selected by the coarse loss mask')
    return selected.mean()


def _build_coarse_term(
    *,
    kind: str,
    params: dict[str, Any],
    label: str,
) -> Callable[[torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor], torch.Tensor]:
    kind_norm = str(kind).lower()

    if kind_norm == 'time_softmax_kl':
        _validate_params_keys(params, allowed=(), required=(), label=label)

        def _term(
            logits: torch.Tensor,
            target: torch.Tensor,
            _batch: dict[str, Any],
            trace_mask: torch.Tensor,
        ) -> torch.Tensor:
            target_prob = _normalize_target_prob(
                logits=logits,
                target=target,
                trace_mask=trace_mask,
                label=label,
            )
            log_p = F.log_softmax(logits, dim=-1)
            eps = torch.finfo(logits.dtype).eps
            log_q = target_prob.clamp_min(eps).log()
            per_trace = (target_prob * (log_q - log_p)).sum(dim=-1).squeeze(1)
            return _masked_mean(
                per_trace_loss=per_trace,
                trace_mask=trace_mask,
                label=label,
            )

        return _term

    if kind_norm == 'prob_l1':
        _validate_params_keys(params, allowed=(), required=(), label=label)

        def _term(
            logits: torch.Tensor,
            target: torch.Tensor,
            _batch: dict[str, Any],
            trace_mask: torch.Tensor,
        ) -> torch.Tensor:
            target_prob = _normalize_target_prob(
                logits=logits,
                target=target,
                trace_mask=trace_mask,
                label=label,
            )
            pred_prob = torch.softmax(logits, dim=-1)
            per_trace = (pred_prob - target_prob).abs().mean(dim=-1).squeeze(1)
            return _masked_mean(
                per_trace_loss=per_trace,
                trace_mask=trace_mask,
                label=label,
            )

        return _term

    if kind_norm == 'prob_mse':
        _validate_params_keys(params, allowed=(), required=(), label=label)

        def _term(
            logits: torch.Tensor,
            target: torch.Tensor,
            _batch: dict[str, Any],
            trace_mask: torch.Tensor,
        ) -> torch.Tensor:
            target_prob = _normalize_target_prob(
                logits=logits,
                target=target,
                trace_mask=trace_mask,
                label=label,
            )
            pred_prob = torch.softmax(logits, dim=-1)
            per_trace = ((pred_prob - target_prob) ** 2).mean(dim=-1).squeeze(1)
            return _masked_mean(
                per_trace_loss=per_trace,
                trace_mask=trace_mask,
                label=label,
            )

        return _term

    if kind_norm == 'prob_huber':
        _validate_params_keys(
            params,
            allowed=('huber_delta',),
            required=(),
            label=label,
        )
        delta = params.get('huber_delta', 1.0)
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            raise TypeError(f'{label}.params.huber_delta must be float')
        delta_f = float(delta)
        if delta_f <= 0.0:
            raise ValueError(f'{label}.params.huber_delta must be > 0')

        def _term(
            logits: torch.Tensor,
            target: torch.Tensor,
            _batch: dict[str, Any],
            trace_mask: torch.Tensor,
        ) -> torch.Tensor:
            target_prob = _normalize_target_prob(
                logits=logits,
                target=target,
                trace_mask=trace_mask,
                label=label,
            )
            pred_prob = torch.softmax(logits, dim=-1)
            loss_map = F.huber_loss(
                pred_prob,
                target_prob,
                reduction='none',
                delta=delta_f,
            )
            per_trace = loss_map.mean(dim=-1).squeeze(1)
            return _masked_mean(
                per_trace_loss=per_trace,
                trace_mask=trace_mask,
                label=label,
            )

        return _term

    supported = ('time_softmax_kl', 'prob_l1', 'prob_mse', 'prob_huber')
    raise ValueError(
        f'unknown coarse loss kind "{kind}"; supported: {", ".join(supported)}'
    )


def build_coarse_criterion(
    loss_specs: list[composite.LossSpec],
    *,
    use_label_valid: bool = True,
) -> Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor]:
    if not isinstance(loss_specs, list):
        raise TypeError('loss_specs must be list[LossSpec]')
    if len(loss_specs) == 0:
        raise ValueError('loss_specs must be non-empty')
    if not isinstance(use_label_valid, bool):
        raise TypeError('use_label_valid must be bool')

    terms: list[
        tuple[
            float,
            str,
            Callable[[torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor], torch.Tensor],
        ]
    ] = []
    for i, spec in enumerate(loss_specs):
        if not isinstance(spec, composite.LossSpec):
            raise TypeError(f'loss_specs[{i}] must be LossSpec')
        scope = _normalize_scope(spec.scope, label=f'loss_specs[{i}].scope')
        term = _build_coarse_term(
            kind=spec.kind,
            params=dict(spec.params),
            label=f'losses[{i}]',
        )
        terms.append((float(spec.weight), scope, term))

    def _criterion(
        logits: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        mask_cache: dict[str, torch.Tensor] = {}
        total = logits.new_tensor(0.0)
        for weight, scope, term in terms:
            if scope not in mask_cache:
                mask_cache[scope] = _resolve_trace_mask(
                    scope=scope,
                    logits=logits,
                    target=target,
                    batch=batch,
                    use_label_valid=use_label_valid,
                    label=f'scope[{scope}]',
                )
            trace_mask = mask_cache[scope]
            total = total + term(logits, target, batch, trace_mask) * weight
        return total

    return _criterion
