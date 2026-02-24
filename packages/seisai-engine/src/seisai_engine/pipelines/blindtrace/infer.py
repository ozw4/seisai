from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from seisai_engine.infer.runner import TiledHConfig
from seisai_engine.pipelines.common.tiled_infer import run_tiled_infer_epoch

from .vis import save_triptych_step

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

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
    def _save_step(
        step: int,
        x_in_bchw: torch.Tensor,
        x_tg_bchw: torch.Tensor,
        x_pr_bchw: torch.Tensor,
        batch: dict,
    ) -> None:
        save_triptych_step(
            out_dir=str(vis_out_dir),
            step=step,
            x_in_bchw=x_in_bchw,
            x_tg_bchw=x_tg_bchw,
            x_pr_bchw=x_pr_bchw,
            cfg=vis_cfg,
            batch=batch,
            c=0,
        )

    return run_tiled_infer_epoch(
        model=model,
        loader=loader,
        device=device,
        criterion=criterion,
        tiled_cfg=tiled_cfg,
        vis_out_dir=vis_out_dir,
        vis_n=int(vis_n),
        max_batches=int(max_batches),
        save_step_fn=_save_step,
        pass_device_batch_to_criterion=False,
        select_vis_input_fn=lambda x: x[:, :1, :, :],
    )
