from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from seisai_utils.config import optional_str, require_dict
from seisai_utils.viz_pair import PairTriptychVisConfig

from seisai_engine.infer.runner import TiledHConfig
from seisai_engine.pipelines.common import (
    TrainSkeletonSpec,
    expand_cfg_listfiles,
    load_cfg_with_base_dir,
    resolve_cfg_paths,
    resolve_device,
    resolve_out_dir,
    run_train_skeleton,
    seed_all,
)

from .build_dataset import (
    build_infer_transform,
    build_pair_dataset,
    build_train_transform,
)
from .build_model import build_model
from .build_plan import build_plan
from .config import PairPaths, load_pair_train_config
from .infer import run_infer_epoch
from .loss import build_criterion

__all__ = ['main']

DEFAULT_CONFIG_PATH = Path('examples/config_train_pair.yaml')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    args, _unknown = parser.parse_known_args(argv)

    cfg, base_dir = load_cfg_with_base_dir(Path(args.config))
    augment_cfg = cfg.get('augment')
    resolve_cfg_paths(
        cfg,
        base_dir,
        keys=[
            'paths.input_segy_files',
            'paths.target_segy_files',
            'paths.infer_input_segy_files',
            'paths.infer_target_segy_files',
        ],
    )
    expand_cfg_listfiles(
        cfg,
        keys=[
            'paths.input_segy_files',
            'paths.target_segy_files',
            'paths.infer_input_segy_files',
            'paths.infer_target_segy_files',
        ],
    )

    train_cfg = require_dict(cfg, 'train')
    device_str = optional_str(train_cfg, 'device', 'auto')
    typed = load_pair_train_config(cfg)
    common = typed.common

    out_dir_path = resolve_out_dir(cfg, base_dir)

    if not typed.ckpt.save_best_only:
        msg = 'ckpt.save_best_only must be true'
        raise ValueError(msg)
    if typed.ckpt.metric != 'infer_loss':
        msg = 'ckpt.metric must be "infer_loss"'
        raise ValueError(msg)
    if typed.ckpt.mode != 'min':
        msg = 'ckpt.mode must be "min"'
        raise ValueError(msg)

    if typed.train.loss_kind not in ('l1', 'mse'):
        msg = 'train.loss_kind must be "l1" or "mse"'
        raise ValueError(msg)

    if typed.tile.tile_h > typed.infer.subset_traces:
        msg = 'tile.tile_h must be <= infer.subset_traces'
        raise ValueError(msg)
    if common.train.samples_per_epoch <= 0:
        msg = 'train.samples_per_epoch must be positive'
        raise ValueError(msg)
    if common.infer.infer_max_batches <= 0:
        msg = 'infer.max_batches must be positive'
        raise ValueError(msg)

    device = resolve_device(device_str)
    seed_all(common.seeds.seed_train)

    train_transform = build_train_transform(
        int(typed.train.time_len),
        augment_cfg=augment_cfg,
    )
    infer_transform = build_infer_transform()

    plan = build_plan()
    criterion = build_criterion(typed.train.loss_kind)

    paths_cfg = PairPaths(
        input_segy_files=list(typed.paths.input_segy_files),
        target_segy_files=list(typed.paths.target_segy_files),
        out_dir=str(out_dir_path),
    )
    dataset_cfg = typed.dataset
    if dataset_cfg.waveform_mode == 'mmap' and int(common.train.train_num_workers) > 0:
        msg = 'dataset.waveform_mode="mmap" requires train.num_workers=0'
        raise ValueError(msg)
    if dataset_cfg.waveform_mode == 'mmap' and int(common.infer.infer_num_workers) > 0:
        msg = 'dataset.waveform_mode="mmap" requires infer.num_workers=0'
        raise ValueError(msg)

    ds_train_full = build_pair_dataset(
        paths=paths_cfg,
        ds_cfg=dataset_cfg,
        transform=train_transform,
        plan=plan,
        subset_traces=int(typed.train.subset_traces),
        secondary_key_fixed=bool(typed.dataset.secondary_key_fixed),
    )

    infer_paths_cfg = PairPaths(
        input_segy_files=list(typed.infer_paths.input_segy_files),
        target_segy_files=list(typed.infer_paths.target_segy_files),
        out_dir=str(out_dir_path),
    )

    ds_infer_full = build_pair_dataset(
        paths=infer_paths_cfg,
        ds_cfg=dataset_cfg,
        transform=infer_transform,
        plan=plan,
        subset_traces=int(typed.infer.subset_traces),
        secondary_key_fixed=True,
    )

    model_sig = asdict(typed.model)
    model = build_model(typed.model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(typed.train.lr))

    tiled_cfg = TiledHConfig(
        tile_h=int(typed.tile.tile_h),
        overlap_h=int(typed.tile.overlap_h),
        tiles_per_batch=int(typed.tile.tiles_per_batch),
        amp=bool(typed.tile.amp),
        use_tqdm=bool(typed.tile.use_tqdm),
    )

    triptych_cfg = PairTriptychVisConfig(
        cmap=typed.vis.cmap,
        vmin=float(typed.vis.vmin),
        vmax=float(typed.vis.vmax),
        transpose_for_trace_time=bool(typed.vis.transpose_for_trace_time),
        per_trace_norm=bool(typed.vis.per_trace_norm),
        per_trace_eps=float(typed.vis.per_trace_eps),
        figsize=typed.vis.figsize,
        dpi=int(typed.vis.dpi),
    )

    def infer_epoch_fn(model, loader, device, vis_epoch_dir, vis_n, max_batches):
        return (run_infer_epoch(
                model=model,
                loader=loader,
                device=device,
                criterion=criterion,
                tiled_cfg=tiled_cfg,
                vis_cfg=triptych_cfg,
                vis_out_dir=str(vis_epoch_dir),
                vis_n=vis_n,
                max_batches=max_batches,
            ))

    spec = TrainSkeletonSpec(
        pipeline='pair',
        cfg=cfg,
        base_dir=base_dir,
        out_dir=out_dir_path,
        vis_subdir=str(common.output.vis_subdir),
        model_sig=model_sig,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        ds_train_full=ds_train_full,
        ds_infer_full=ds_infer_full,
        device=device,
        seed_train=common.seeds.seed_train,
        seed_infer=common.seeds.seed_infer,
        epochs=common.train.epochs,
        train_batch_size=common.train.train_batch_size,
        train_num_workers=common.train.train_num_workers,
        samples_per_epoch=common.train.samples_per_epoch,
        max_norm=common.train.max_norm,
        use_amp_train=common.train.use_amp_train,
        infer_batch_size=common.infer.infer_batch_size,
        infer_num_workers=common.infer.infer_num_workers,
        infer_max_batches=common.infer.infer_max_batches,
        vis_n=common.infer.vis_n,
        infer_epoch_fn=infer_epoch_fn,
        print_freq=common.train.print_freq,
    )

    run_train_skeleton(spec)


if __name__ == '__main__':
    main()
