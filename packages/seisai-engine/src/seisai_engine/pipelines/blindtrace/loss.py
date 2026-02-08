from __future__ import annotations

import torch

from seisai_engine.loss.pixelwise_loss import build_criterion
from seisai_engine.loss.shift_pertrace_mse import ShiftRobustPerTraceMSE

__all__ = ['build_masked_criterion']


def build_masked_criterion(*, loss_kind: str, loss_scope: str, shift_max: int):
    scope = str(loss_scope).lower()
    if scope not in ('masked_only', 'all'):
        msg = 'train.loss_scope must be "masked_only" or "all"'
        raise ValueError(msg)

    kind = str(loss_kind).lower()
    if kind not in ('l1', 'mse', 'shift_mse', 'shift_robust_mse'):
        msg = 'train.loss_kind must be "l1", "mse", "shift_mse", or "shift_robust_mse"'
        raise ValueError(msg)

    use_shift = kind in ('shift_mse', 'shift_robust_mse')
    if use_shift:
        shift_loss = ShiftRobustPerTraceMSE(max_shift=int(shift_max), ch_reduce='all')
    else:
        base_criterion = build_criterion(kind)

    def _get_trace_mask(pred: torch.Tensor, batch: dict) -> torch.Tensor:
        if scope == 'all':
            B, _C, H, _W = pred.shape
            return torch.ones((B, H), dtype=torch.bool, device=pred.device)

        if 'mask_bool' not in batch:
            msg = "batch['mask_bool'] is required for masked_only loss"
            raise KeyError(msg)

        mask_bool = batch['mask_bool']
        if not isinstance(mask_bool, torch.Tensor):
            mask_bool = torch.as_tensor(mask_bool)

        if mask_bool.dtype != torch.bool:
            msg = 'mask_bool must be bool'
            raise ValueError(msg)

        if mask_bool.ndim == 3:
            trace_mask = mask_bool.any(dim=-1)  # (B,H)
        elif mask_bool.ndim == 2:
            trace_mask = mask_bool  # (B,H)
        else:
            msg = 'mask_bool must be (B,H,W) or (B,H)'
            raise ValueError(msg)

        if trace_mask.device != pred.device:
            trace_mask = trace_mask.to(device=pred.device, non_blocking=True)

        return trace_mask

    def _criterion(
        pred: torch.Tensor, target: torch.Tensor, batch: dict
    ) -> torch.Tensor:
        if not use_shift:
            if scope == 'all':
                return base_criterion(pred, target, {})

            trace_mask = _get_trace_mask(pred, batch)
            return base_criterion(pred, target, {'mask_bool': trace_mask})

        # ShiftRobustPerTraceMSE reads batch['target'] and batch['trace_mask'].
        # Ensure they exist and are on the same device as pred.
        if not isinstance(target, torch.Tensor):
            msg = 'target must be torch.Tensor for shift loss'
            raise TypeError(msg)

        if target.device != pred.device:
            target = target.to(device=pred.device, non_blocking=True)

        batch['target'] = target
        batch['trace_mask'] = _get_trace_mask(pred, batch)

        return shift_loss(pred, batch, reduction='mean')

    return _criterion
