from __future__ import annotations

import torch

from seisai_engine.loss.fbseg_kl_loss import FbSegKLLossView

__all__ = ['build_criterion']


def build_criterion(*, tau: float = 1.0, eps: float = 0.0):
    loss_fn = FbSegKLLossView(tau=float(tau), eps=float(eps))
    loss_fn.eps = float(eps)

    def _criterion(
        pred: torch.Tensor,
        target: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        if not isinstance(target, torch.Tensor):
            msg = 'target must be torch.Tensor'
            raise TypeError(msg)
        batch_dev = dict(batch)
        batch_dev['target'] = target
        return loss_fn(pred, batch_dev, reduction='mean')

    return _criterion
