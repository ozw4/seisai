from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from seisai_engine.loss import composite
from seisai_engine.loss.soft_label_ce import build_pixel_mask_from_batch, soft_label_ce_map

__all__ = ['build_psn_criterion']


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


def _prepare_target(*, logits: torch.Tensor, target: torch.Tensor, label: str) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or int(logits.ndim) != 4:
        raise ValueError(f'{label}: logits must be torch.Tensor with shape (B,C,H,W)')
    if not isinstance(target, torch.Tensor) or tuple(target.shape) != tuple(logits.shape):
        raise ValueError(f'{label}: target must be torch.Tensor with same shape as logits (B,C,H,W)')

    target_tensor = target
    if target_tensor.dtype != logits.dtype:
        target_tensor = target_tensor.to(dtype=logits.dtype)
    if target_tensor.device != logits.device:
        target_tensor = target_tensor.to(device=logits.device, non_blocking=True)
    return target_tensor


def _resolve_pixel_mask(
    *,
    scope: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    batch: dict[str, Any],
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
        use_label_valid=True,
        mask_bool_key=mask_bool_key,
    )
    if pixel_mask.device != logits.device:
        pixel_mask = pixel_mask.to(device=logits.device, non_blocking=True)
    return pixel_mask


def _masked_mean_from_map(
    *,
    loss_map: torch.Tensor,
    pixel_mask: torch.Tensor,
    label: str,
) -> torch.Tensor:
    if not isinstance(loss_map, torch.Tensor) or int(loss_map.ndim) != 3:
        raise ValueError(f'{label}: loss map must be torch.Tensor with shape (B,H,W)')
    if not isinstance(pixel_mask, torch.Tensor):
        pixel_mask = torch.as_tensor(pixel_mask)
    if pixel_mask.dtype is not torch.bool:
        raise TypeError(f'{label}: pixel mask must be bool tensor')
    if tuple(pixel_mask.shape) != tuple(loss_map.shape):
        raise ValueError(
            f'{label}: pixel mask shape mismatch: {tuple(pixel_mask.shape)} vs {tuple(loss_map.shape)}'
        )

    selected = loss_map[pixel_mask]
    loss_sum = selected.sum(dtype=torch.float32)
    denom = pixel_mask.sum(dtype=torch.float32)
    return loss_sum / denom.clamp_min(1.0)


def _build_psn_term(
    *,
    kind: str,
    params: dict[str, Any],
    label: str,
) -> Callable[[torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor], torch.Tensor]:
    kind_norm = str(kind).lower()

    if kind_norm == 'soft_label_ce':
        _validate_params_keys(params, allowed=(), required=(), label=label)

        def _term(
            logits: torch.Tensor,
            target: torch.Tensor,
            _batch: dict[str, Any],
            pixel_mask: torch.Tensor,
        ) -> torch.Tensor:
            target_tensor = _prepare_target(logits=logits, target=target, label=label)
            loss_map = soft_label_ce_map(logits, target_tensor, class_dim=1)
            return _masked_mean_from_map(
                loss_map=loss_map,
                pixel_mask=pixel_mask,
                label=label,
            )

        return _term

    if kind_norm == 'prob_l1':
        _validate_params_keys(params, allowed=(), required=(), label=label)

        def _term(
            logits: torch.Tensor,
            target: torch.Tensor,
            _batch: dict[str, Any],
            pixel_mask: torch.Tensor,
        ) -> torch.Tensor:
            target_tensor = _prepare_target(logits=logits, target=target, label=label)
            pred_prob = torch.softmax(logits, dim=1)
            loss_map = (pred_prob - target_tensor).abs().mean(dim=1)
            return _masked_mean_from_map(
                loss_map=loss_map,
                pixel_mask=pixel_mask,
                label=label,
            )

        return _term

    if kind_norm == 'prob_mse':
        _validate_params_keys(params, allowed=(), required=(), label=label)

        def _term(
            logits: torch.Tensor,
            target: torch.Tensor,
            _batch: dict[str, Any],
            pixel_mask: torch.Tensor,
        ) -> torch.Tensor:
            target_tensor = _prepare_target(logits=logits, target=target, label=label)
            pred_prob = torch.softmax(logits, dim=1)
            loss_map = ((pred_prob - target_tensor) ** 2).mean(dim=1)
            return _masked_mean_from_map(
                loss_map=loss_map,
                pixel_mask=pixel_mask,
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
            pixel_mask: torch.Tensor,
        ) -> torch.Tensor:
            target_tensor = _prepare_target(logits=logits, target=target, label=label)
            pred_prob = torch.softmax(logits, dim=1)
            loss_map_bchw = F.huber_loss(
                pred_prob,
                target_tensor,
                reduction='none',
                delta=delta_f,
            )
            loss_map = loss_map_bchw.mean(dim=1)
            return _masked_mean_from_map(
                loss_map=loss_map,
                pixel_mask=pixel_mask,
                label=label,
            )

        return _term

    supported = ('soft_label_ce', 'prob_l1', 'prob_mse', 'prob_huber')
    msg = f'unknown PSN loss kind "{kind}"; supported: {", ".join(supported)}'
    raise ValueError(msg)


def build_psn_criterion(
    loss_specs: list[composite.LossSpec],
) -> Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor]:
    if not isinstance(loss_specs, list):
        raise TypeError('loss_specs must be list[LossSpec]')
    if len(loss_specs) == 0:
        raise ValueError('loss_specs must be non-empty')

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
        term = _build_psn_term(
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
                mask_cache[scope] = _resolve_pixel_mask(
                    scope=scope,
                    logits=logits,
                    target=target,
                    batch=batch,
                )
            pixel_mask = mask_cache[scope]
            total = total + term(logits, target, batch, pixel_mask) * weight
        return total

    return _criterion
