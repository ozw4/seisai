from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from seisai_utils.viz_pair import PairTriptychVisConfig, save_pair_triptych_step_png
from torch.utils.data import DataLoader, Subset

from seisai_engine.infer.runner import TiledHConfig, infer_batch_tiled_h
from seisai_engine.pipelines.common import load_checkpoint
from seisai_engine.pipelines.common.config_io import resolve_relpath
from seisai_engine.pipelines.common.seed import seed_all
from seisai_engine.pipelines.common.tiled_infer import run_tiled_infer_epoch

from .build_dataset import build_infer_transform, build_pair_dataset
from .build_model import build_model
from .build_plan import build_plan
from .config import load_infer_config

__all__ = ['main', 'run_infer_epoch']

DEFAULT_CONFIG_PATH = Path('examples/config_train_pair.yaml')


def _resolve_ckpt(*, base_dir: Path, override: str | None, fallback: str) -> str:
    if override is None:
        return fallback
    return resolve_relpath(base_dir, override)


def run_infer_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion,
    tiled_cfg: TiledHConfig,
    vis_cfg: PairTriptychVisConfig,
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
        save_pair_triptych_step_png(
            vis_out_dir,
            step=step,
            x_in_bchw=x_in_bchw,
            x_tg_bchw=x_tg_bchw,
            x_pr_bchw=x_pr_bchw,
            cfg=vis_cfg,
            batch=batch,
            prefix='step_',
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
        pass_device_batch_to_criterion=True,
    )


def main(argv: list[str] | None = None) -> None:
    """Run tiled-h inference on paired SEG-Y gathers and save triptych PNGs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument('--ckpt', default=None)
    args, _unknown = parser.parse_known_args(argv)

    cfg = load_infer_config(args.config)
    base_dir = Path(args.config).expanduser().resolve().parent
    default_ckpt = str(Path(cfg.paths.out_dir) / 'ckpt' / 'best.pt')
    ckpt_path = _resolve_ckpt(
        base_dir=base_dir, override=args.ckpt, fallback=default_ckpt
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_all(cfg.infer.seed)

    if cfg.dataset.waveform_mode == 'mmap' and int(cfg.infer.num_workers) > 0:
        msg = 'dataset.waveform_mode="mmap" requires infer.num_workers=0'
        raise ValueError(msg)

    ckpt = load_checkpoint(ckpt_path)
    if ckpt['pipeline'] != 'pair':
        msg = 'checkpoint pipeline must be "pair"'
        raise ValueError(msg)
    if not isinstance(ckpt.get('model_sig'), dict):
        msg = 'checkpoint model_sig must be dict'
        raise ValueError(msg)
    if ckpt['model_sig'] != asdict(cfg.model):
        msg = 'checkpoint model_sig does not match config model'
        raise ValueError(msg)

    model = build_model(cfg.model)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device)
    model.eval()

    plan = build_plan()
    standardize_eps = 1e-8
    infer_transform = build_infer_transform(eps=standardize_eps)

    ds_infer_full = build_pair_dataset(
        paths=cfg.paths,
        ds_cfg=cfg.dataset,
        transform=infer_transform,
        plan=plan,
        subset_traces=cfg.infer.subset_traces,
        secondary_key_fixed=True,
        input_segy_endian=str(cfg.dataset.infer_input_endian),
        target_segy_endian=str(cfg.dataset.infer_target_endian),
        standardize_eps=standardize_eps,
    )

    try:
        infer_ds = Subset(
            ds_infer_full, range(cfg.infer.batch_size * cfg.infer.max_batches)
        )
        infer_loader = DataLoader(
            infer_ds,
            batch_size=cfg.infer.batch_size,
            shuffle=False,
            num_workers=cfg.infer.num_workers,
            pin_memory=(device.type == 'cuda'),
        )

        vis_out_dir = Path(cfg.paths.out_dir) / cfg.vis.out_subdir
        vis_out_dir.mkdir(parents=True, exist_ok=True)

        tiled_cfg = TiledHConfig(
            tile_h=cfg.tile.tile_h,
            overlap_h=cfg.tile.overlap_h,
            tiles_per_batch=cfg.tile.tiles_per_batch,
            amp=cfg.tile.amp,
            use_tqdm=cfg.tile.use_tqdm,
        )

        triptych_cfg = PairTriptychVisConfig(
            cmap=cfg.vis.cmap,
            vmin=cfg.vis.vmin,
            vmax=cfg.vis.vmax,
            transpose_for_trace_time=cfg.vis.transpose_for_trace_time,
            per_trace_norm=cfg.vis.per_trace_norm,
            per_trace_eps=cfg.vis.per_trace_eps,
            figsize=cfg.vis.figsize,
            dpi=cfg.vis.dpi,
        )

        non_blocking = bool(device.type == 'cuda')

        with torch.no_grad():
            for step, batch in enumerate(infer_loader):
                x_in = batch['input'].to(device=device, non_blocking=non_blocking)
                x_tg = batch['target'].to(device=device, non_blocking=non_blocking)

                x_pr = infer_batch_tiled_h(model, x_in, cfg=tiled_cfg)

                if step < cfg.vis.n:
                    save_pair_triptych_step_png(
                        str(vis_out_dir),
                        step=step,
                        x_in_bchw=x_in.detach().cpu(),
                        x_tg_bchw=x_tg.detach().cpu(),
                        x_pr_bchw=x_pr.detach().cpu(),
                        cfg=triptych_cfg,
                        batch=batch,
                    )
    finally:
        ds_infer_full.close()

    print(f'completed inference. outputs: {vis_out_dir}')


if __name__ == '__main__':
    main()
