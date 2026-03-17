from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

__all__ = [
    'reconstruct_pair_prediction',
    'resolve_pair_residual_learning',
    'wrap_pair_criterion',
]


def resolve_pair_residual_learning(cfg: dict[str, Any]) -> bool:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    pair_cfg = cfg.get('pair')
    if pair_cfg is None:
        return False
    if not isinstance(pair_cfg, dict):
        msg = 'pair must be dict'
        raise TypeError(msg)

    residual_learning = pair_cfg.get('residual_learning', False)
    if not isinstance(residual_learning, bool):
        msg = 'pair.residual_learning must be bool'
        raise TypeError(msg)
    return bool(residual_learning)


def reconstruct_pair_prediction(
    pred_raw: torch.Tensor,
    x_in: torch.Tensor,
    *,
    residual_learning: bool,
) -> torch.Tensor:
    if not isinstance(pred_raw, torch.Tensor):
        msg = 'pred_raw must be torch.Tensor'
        raise TypeError(msg)
    if not residual_learning:
        return pred_raw

    if not isinstance(x_in, torch.Tensor):
        msg = 'x_in must be torch.Tensor'
        raise TypeError(msg)
    if pred_raw.shape != x_in.shape:
        msg = f'pred_raw.shape {tuple(pred_raw.shape)} != x_in.shape {tuple(x_in.shape)}'
        raise ValueError(msg)
    if pred_raw.dtype != x_in.dtype:
        msg = f'pred_raw.dtype {pred_raw.dtype} != x_in.dtype {x_in.dtype}'
        raise TypeError(msg)
    if pred_raw.device != x_in.device:
        msg = f'pred_raw.device {pred_raw.device} != x_in.device {x_in.device}'
        raise ValueError(msg)
    return x_in + pred_raw


def wrap_pair_criterion(
    criterion: Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor],
    *,
    residual_learning: bool,
) -> Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor]:
    if not callable(criterion):
        msg = 'criterion must be callable'
        raise TypeError(msg)

    def _criterion(
        pred_raw: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        if not isinstance(pred_raw, torch.Tensor):
            msg = 'pred_raw must be torch.Tensor'
            raise TypeError(msg)
        if not isinstance(batch, dict):
            msg = 'batch must be dict'
            raise TypeError(msg)
        if not residual_learning:
            return criterion(pred_raw, target, batch)
        if 'input' not in batch:
            msg = "batch['input'] is required for pair residual reconstruction"
            raise KeyError(msg)

        x_in = batch['input']
        if not isinstance(x_in, torch.Tensor):
            msg = "batch['input'] must be torch.Tensor"
            raise TypeError(msg)
        x_in = x_in.to(
            device=pred_raw.device,
            dtype=pred_raw.dtype,
            non_blocking=bool(pred_raw.device.type == 'cuda'),
        )
        pred = reconstruct_pair_prediction(
            pred_raw,
            x_in,
            residual_learning=True,
        )
        return criterion(pred, target, batch)

    return _criterion
