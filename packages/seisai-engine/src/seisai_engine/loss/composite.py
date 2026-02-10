from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .fx_mag import FxMagPerTraceMSE
from .pixelwise_loss import build_criterion
from .shift_pertrace_mse import ShiftRobustPerTraceMSE

__all__ = [
    'LossSpec',
    'build_loss_term',
    'build_weighted_criterion',
    'parse_loss_specs',
]


@dataclass(frozen=True)
class LossSpec:
    kind: str
    weight: float
    scope: str
    params: dict[str, Any]


def _normalize_scope(scope: str, label: str) -> str:
    if not isinstance(scope, str):
        raise TypeError(f'{label} must be str')
    scope_norm = scope.lower()
    if scope_norm not in ('masked_only', 'all'):
        raise ValueError(f'{label} must be "masked_only" or "all"')
    return scope_norm


def _resolve_trace_mask(
    scope: str, pred: torch.Tensor, batch: dict[str, Any]
) -> torch.Tensor:
    if not isinstance(pred, torch.Tensor):
        raise TypeError('pred must be a torch.Tensor')
    if pred.ndim != 4:
        raise ValueError('pred must be (B,C,H,W)')

    B, _C, H, _W = pred.shape

    if scope == 'all':
        return torch.ones((B, H), dtype=torch.bool, device=pred.device)

    if 'mask_bool' not in batch:
        raise KeyError("batch['mask_bool'] is required for masked_only loss")

    mask_bool = batch['mask_bool']
    if not isinstance(mask_bool, torch.Tensor):
        mask_bool = torch.as_tensor(mask_bool)

    if mask_bool.dtype != torch.bool:
        raise TypeError("batch['mask_bool'] must be a bool tensor")

    if mask_bool.ndim == 3:
        if mask_bool.shape != (B, H, pred.shape[-1]):
            raise ValueError("batch['mask_bool'] must be (B,H,W)")
        trace_mask = mask_bool.any(dim=-1)
    elif mask_bool.ndim == 2:
        if mask_bool.shape != (B, H):
            raise ValueError("batch['mask_bool'] must be (B,H)")
        trace_mask = mask_bool
    else:
        raise ValueError("batch['mask_bool'] must be (B,H,W) or (B,H)")

    if trace_mask.device != pred.device:
        trace_mask = trace_mask.to(device=pred.device, non_blocking=True)

    return trace_mask


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


def _int_like(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{label} must be int-like')
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f'{label} must be an integer value')
    return int(value)


def build_loss_term(
    kind: str, params: dict[str, Any], label: str
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor]:
    kind_norm = str(kind).lower()

    if kind_norm in ('l1', 'mse', 'huber'):
        if kind_norm == 'huber':
            _validate_params_keys(
                params, allowed=('huber_delta',), required=(), label=label
            )
            delta = params.get('huber_delta', 1.0)
            if isinstance(delta, bool) or not isinstance(delta, (int, float)):
                raise TypeError(f'{label}.params.huber_delta must be float')
            if float(delta) <= 0.0:
                raise ValueError(f'{label}.params.huber_delta must be > 0')
            base_criterion = build_criterion('huber', huber_delta=float(delta))
        else:
            _validate_params_keys(params, allowed=(), required=(), label=label)
            base_criterion = build_criterion(kind_norm)

        def _term(
            pred: torch.Tensor,
            target: torch.Tensor,
            trace_mask: torch.Tensor,
            _batch: dict[str, Any],
        ) -> torch.Tensor:
            return base_criterion(pred, target, {'mask_bool': trace_mask})

        return _term

    if kind_norm in ('shift_mse', 'shift_robust_mse'):
        _validate_params_keys(
            params, allowed=('shift_max',), required=('shift_max',), label=label
        )
        shift_max = _int_like(params['shift_max'], label=f'{label}.params.shift_max')
        if shift_max < 0:
            raise ValueError(f'{label}.params.shift_max must be >= 0')
        shift_loss = ShiftRobustPerTraceMSE(max_shift=shift_max, ch_reduce='all')

        def _term(
            pred: torch.Tensor,
            target: torch.Tensor,
            trace_mask: torch.Tensor,
            _batch: dict[str, Any],
        ) -> torch.Tensor:
            if not isinstance(target, torch.Tensor):
                raise TypeError(f'{label}: target must be torch.Tensor')
            return shift_loss(
                pred, {'target': target, 'trace_mask': trace_mask}, reduction='mean'
            )

        return _term

    if kind_norm == 'fx_mag_mse':
        _validate_params_keys(
            params,
            allowed=('use_log', 'eps', 'f_lo', 'f_hi'),
            required=(),
            label=label,
        )
        use_log = params.get('use_log', True)
        if not isinstance(use_log, bool):
            raise TypeError(f'{label}.params.use_log must be bool')
        eps = params.get('eps', 1.0e-6)
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise TypeError(f'{label}.params.eps must be float')
        if float(eps) <= 0.0:
            raise ValueError(f'{label}.params.eps must be > 0')
        f_lo = params.get('f_lo', 0)
        f_lo_val = _int_like(f_lo, label=f'{label}.params.f_lo')
        if f_lo_val < 0:
            raise ValueError(f'{label}.params.f_lo must be >= 0')
        f_hi = params.get('f_hi', None)
        f_hi_val = None
        if f_hi is not None:
            f_hi_val = _int_like(f_hi, label=f'{label}.params.f_hi')
            if f_hi_val <= f_lo_val:
                raise ValueError(f'{label}.params.f_hi must be > f_lo')

        fx_loss = FxMagPerTraceMSE(
            use_log=bool(use_log),
            eps=float(eps),
            f_lo=f_lo_val,
            f_hi=f_hi_val,
        )

        def _term(
            pred: torch.Tensor,
            target: torch.Tensor,
            trace_mask: torch.Tensor,
            _batch: dict[str, Any],
        ) -> torch.Tensor:
            return fx_loss(
                pred, {'target': target, 'trace_mask': trace_mask}, reduction='mean'
            )

        return _term

    supported = (
        'l1',
        'mse',
        'huber',
        'shift_mse',
        'shift_robust_mse',
        'fx_mag_mse',
    )
    msg = f'unknown loss kind "{kind}"; supported: {", ".join(supported)}'
    raise ValueError(msg)


def parse_loss_specs(losses: Any, *, default_scope: str) -> list[LossSpec]:
    if losses is None:
        raise ValueError('train.losses is required')
    if not isinstance(losses, list):
        raise TypeError('train.losses must be list[dict]')
    if len(losses) == 0:
        raise ValueError('train.losses must be non-empty')

    default_scope_norm = _normalize_scope(default_scope, 'train.loss_scope')
    specs: list[LossSpec] = []

    for i, item in enumerate(losses):
        label = f'losses[{i}]'
        if not isinstance(item, dict):
            raise TypeError(f'{label} must be dict')

        if 'kind' not in item:
            raise ValueError(f'{label}.kind is required')
        if not isinstance(item['kind'], str):
            raise TypeError(f'{label}.kind must be str')
        kind = item['kind'].lower()

        if 'weight' not in item:
            raise ValueError(f'{label}.weight is required')
        weight = item['weight']
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError(f'{label}.weight must be float')
        weight_f = float(weight)

        if 'scope' in item and item['scope'] is not None:
            scope = _normalize_scope(item['scope'], f'{label}.scope')
        else:
            scope = default_scope_norm

        params_raw = item.get('params', {})
        if params_raw is None:
            params = {}
        elif not isinstance(params_raw, dict):
            raise TypeError(f'{label}.params must be dict')
        else:
            params = dict(params_raw)

        specs.append(
            LossSpec(
                kind=kind,
                weight=weight_f,
                scope=scope,
                params=params,
            )
        )

    return specs


def build_weighted_criterion(
    loss_specs: list[LossSpec],
) -> Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor]:
    if not isinstance(loss_specs, list):
        raise TypeError('loss_specs must be list[LossSpec]')
    if len(loss_specs) == 0:
        raise ValueError('loss_specs must be non-empty')

    terms: list[tuple[float, str, Callable[..., torch.Tensor]]] = []
    for i, spec in enumerate(loss_specs):
        if not isinstance(spec, LossSpec):
            raise TypeError(f'loss_specs[{i}] must be LossSpec')
        term = build_loss_term(spec.kind, spec.params, f'losses[{i}]')
        terms.append((float(spec.weight), spec.scope, term))

    def _criterion(
        pred: torch.Tensor, target: torch.Tensor, batch: dict[str, Any]
    ) -> torch.Tensor:
        mask_cache: dict[str, torch.Tensor] = {}
        total = pred.new_tensor(0.0)
        for weight, scope, term in terms:
            if scope not in mask_cache:
                mask_cache[scope] = _resolve_trace_mask(scope, pred, batch)
            trace_mask = mask_cache[scope]
            total = total + term(pred, target, trace_mask, batch) * weight
        return total

    return _criterion
