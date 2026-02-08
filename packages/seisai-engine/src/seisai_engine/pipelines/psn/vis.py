from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from seisai_utils.viz_phase import make_title_from_batch_meta, save_psn_debug_png

from seisai_engine.loss.soft_label_ce import (
    build_pixel_mask_from_batch,
    soft_label_ce_masked_mean,
)
from seisai_engine.metrics.phase_pick_metrics import compute_ps_metrics_from_batch

from .loss import criterion

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

__all__ = ['run_epoch_debug']


@torch.no_grad()
def run_epoch_debug(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    epoch: int,
    out_dir: str,
) -> None:
    model.eval()
    batch = next(iter(loader))

    x = batch['input']
    y = batch['target']
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        msg = "batch['input']/batch['target'] must be torch.Tensor"
        raise TypeError(msg)
    if x.ndim != 4 or y.ndim != 4:
        msg = f'expected input/target batched tensors: input={tuple(x.shape)} target={tuple(y.shape)}'
        raise ValueError(
            msg
        )

    x_dev = x.to(device=device, non_blocking=(device.type == 'cuda'))
    logits_dev = model(x_dev)

    print(
        f'[debug] epoch={epoch} input={tuple(x.shape)} target={tuple(y.shape)} logits={tuple(logits_dev.shape)}'
    )

    metrics = compute_ps_metrics_from_batch(logits_dev, batch, thresholds=(5, 10, 20))
    print(
        '[metrics] '
        + ' '.join(
            f'{k}={v:.4f}' if np.isfinite(v) else f'{k}=nan' for k, v in metrics.items()
        )
    )

    pixel_mask = build_pixel_mask_from_batch(batch)
    pixel_mask_sum = int(pixel_mask.sum().item())
    y_dev = y.to(device=device, non_blocking=(device.type == 'cuda'))
    loss_masked = criterion(logits_dev, y_dev, batch)
    loss_empty = soft_label_ce_masked_mean(
        logits_dev, y_dev, torch.zeros_like(pixel_mask, dtype=torch.bool)
    )
    print(
        f'[debug] pixel_mask_sum={pixel_mask_sum} loss_masked={float(loss_masked.detach().cpu().item()):.6f} '
        f'loss_empty_mask={float(loss_empty.detach().cpu().item()):.6f}'
    )

    logits = logits_dev.detach().cpu()

    title = make_title_from_batch_meta(batch, b=0)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    png = outp / f'psn_debug_epoch{int(epoch):04d}.png'
    save_psn_debug_png(
        png,
        x_bchw=x,
        target_b3hw=y,
        logits_b3hw=logits,
        b=0,
        title=title,
    )
    print(f'[saved] {png}')
