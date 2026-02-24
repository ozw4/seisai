from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from seisai_engine.infer.runner import TiledHConfig, infer_batch_tiled_h

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

SaveTiledStepFn = Callable[
    [int, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]], None
]
SelectVisInputFn = Callable[[torch.Tensor], torch.Tensor]

__all__ = ['run_tiled_infer_epoch']


def run_tiled_infer_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion,
    tiled_cfg: TiledHConfig,
    vis_out_dir: str,
    vis_n: int,
    max_batches: int,
    save_step_fn: SaveTiledStepFn,
    pass_device_batch_to_criterion: bool,
    select_vis_input_fn: SelectVisInputFn | None = None,
) -> float:
    non_blocking = bool(device.type == 'cuda')
    infer_loss_sum = 0.0
    infer_samples = 0

    Path(vis_out_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= int(max_batches):
                break

            if pass_device_batch_to_criterion:
                batch_dev = {
                    k: (
                        v.to(device=device, non_blocking=non_blocking)
                        if torch.is_tensor(v)
                        else v
                    )
                    for k, v in batch.items()
                }
                x_in = batch_dev['input']
                x_tg = batch_dev['target']
                criterion_batch = batch_dev
            else:
                x_in = batch['input'].to(device=device, non_blocking=non_blocking)
                x_tg = batch['target'].to(device=device, non_blocking=non_blocking)
                criterion_batch = batch

            x_pr = infer_batch_tiled_h(model, x_in, cfg=tiled_cfg)
            loss = criterion(x_pr, x_tg, criterion_batch)

            bsize = int(x_in.shape[0])
            infer_loss_sum += float(loss.detach().item()) * bsize
            infer_samples += bsize

            if step < int(vis_n):
                x_in_vis = x_in
                if select_vis_input_fn is not None:
                    x_in_vis = select_vis_input_fn(x_in)
                if not torch.is_tensor(x_in_vis):
                    msg = 'select_vis_input_fn must return torch.Tensor'
                    raise TypeError(msg)
                save_step_fn(
                    int(step),
                    x_in_vis.detach().cpu(),
                    x_tg.detach().cpu(),
                    x_pr.detach().cpu(),
                    batch,
                )

    if infer_samples <= 0:
        msg = 'no inference samples were processed'
        raise RuntimeError(msg)

    return infer_loss_sum / float(infer_samples)
