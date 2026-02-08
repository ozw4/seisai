from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from seisai_engine.infer.runner import TiledHConfig, infer_batch_tiled_h

from .vis import save_triptych_step

__all__ = ['run_infer_epoch']


def run_infer_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion,
    tiled_cfg: TiledHConfig,
    vis_cfg,
    vis_out_dir: str,
    vis_n: int,
    max_batches: int,
) -> float:
    non_blocking = bool(device.type == 'cuda')
    infer_loss_sum = 0.0
    infer_samples = 0

    Path(vis_out_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= int(max_batches):
                break

            x_in = batch['input'].to(device=device, non_blocking=non_blocking)
            x_tg = batch['target'].to(device=device, non_blocking=non_blocking)

            x_pr = infer_batch_tiled_h(model, x_in, cfg=tiled_cfg)
            loss = criterion(x_pr, x_tg, batch)

            bsize = int(x_in.shape[0])
            infer_loss_sum += float(loss.detach().item()) * bsize
            infer_samples += bsize

            if step < int(vis_n):
                x_in_wave = x_in[:, :1, :, :]
                save_triptych_step(
                    out_dir=str(vis_out_dir),
                    step=step,
                    x_in_bchw=x_in_wave.detach().cpu(),
                    x_tg_bchw=x_tg.detach().cpu(),
                    x_pr_bchw=x_pr.detach().cpu(),
                    cfg=vis_cfg,
                    batch=batch,
                    c=0,
                )

    if infer_samples <= 0:
        msg = 'no inference samples were processed'
        raise RuntimeError(msg)

    return infer_loss_sum / float(infer_samples)
