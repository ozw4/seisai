from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from seisai_utils.config import optional_bool, require_dict, require_list_str
from seisai_utils.viz_phase import make_title_from_batch_meta, save_psn_debug_png

from seisai_engine.pipelines.common import (
    TrainSkeletonSpec,
    expand_cfg_listfiles,
    load_cfg_with_base_dir,
    resolve_cfg_paths,
    resolve_out_dir,
    run_train_skeleton,
    seed_all,
)

from .build_dataset import build_dataset, build_infer_transform, build_train_transform
from .build_model import build_model
from .config import load_psn_train_config
from .loss import criterion

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

__all__ = ['main']

DEFAULT_CONFIG_PATH = Path('examples/config_train_psn.yaml')


def _build_dataset_for_subset(
    cfg: dict,
    subset_traces: int,
    *,
    segy_files: list[str],
    phase_pick_files: list[str],
    transform,
    secondary_key_fixed: bool,
):
    cfg_copy = copy.deepcopy(cfg)
    train_cfg = require_dict(cfg_copy, 'train')
    train_cfg['subset_traces'] = int(subset_traces)
    paths = require_dict(cfg_copy, 'paths')
    paths['segy_files'] = list(segy_files)
    paths['phase_pick_files'] = list(phase_pick_files)
    ds_cfg = require_dict(cfg_copy, 'dataset')
    ds_cfg['secondary_key_fixed'] = bool(secondary_key_fixed)
    return build_dataset(cfg_copy, transform=transform)


def _run_infer_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
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

            logits = model(x_in)
            loss = criterion(logits, x_tg, batch_dev)

            bsize = int(x_in.shape[0])
            infer_loss_sum += float(loss.detach().item()) * bsize
            infer_samples += bsize

            if step < int(vis_n):
                title = make_title_from_batch_meta(batch, b=0)
                out_path = Path(vis_out_dir) / f'step_{int(step):04d}.png'
                save_psn_debug_png(
                    out_path,
                    x_bchw=batch['input'],
                    target_b3hw=batch['target'],
                    logits_b3hw=logits.detach().cpu(),
                    b=0,
                    title=title,
                )

    if infer_samples <= 0:
        msg = 'no inference samples were processed'
        raise RuntimeError(msg)

    return infer_loss_sum / float(infer_samples)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    args, _unknown = parser.parse_known_args(argv)

    cfg, base_dir = load_cfg_with_base_dir(Path(args.config))
    resolve_cfg_paths(
        cfg,
        base_dir,
        keys=[
            'paths.segy_files',
            'paths.phase_pick_files',
            'paths.infer_segy_files',
            'paths.infer_phase_pick_files',
        ],
    )
    expand_cfg_listfiles(
        cfg,
        keys=[
            'paths.segy_files',
            'paths.phase_pick_files',
            'paths.infer_segy_files',
            'paths.infer_phase_pick_files',
        ],
    )

    typed = load_psn_train_config(cfg)
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

    model_sig = asdict(typed.model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_all(common.seeds.seed_train)

    train_transform = build_train_transform(cfg)
    infer_transform = build_infer_transform(cfg)

    paths_cfg = require_dict(cfg, 'paths')
    train_segy_files = require_list_str(paths_cfg, 'segy_files')
    train_phase_pick_files = require_list_str(paths_cfg, 'phase_pick_files')
    infer_segy_files = require_list_str(paths_cfg, 'infer_segy_files')
    infer_phase_pick_files = require_list_str(paths_cfg, 'infer_phase_pick_files')
    if len(train_segy_files) != len(train_phase_pick_files):
        msg = 'paths.segy_files and paths.phase_pick_files must have same length'
        raise ValueError(msg)
    if len(infer_segy_files) != len(infer_phase_pick_files):
        msg = (
            'paths.infer_segy_files and paths.infer_phase_pick_files '
            'must have same length'
        )
        raise ValueError(msg)

    ds_cfg = require_dict(cfg, 'dataset')
    train_secondary_key_fixed = optional_bool(
        ds_cfg, 'secondary_key_fixed', default=False
    )

    ds_train_full = _build_dataset_for_subset(
        cfg,
        typed.train.subset_traces,
        segy_files=list(train_segy_files),
        phase_pick_files=list(train_phase_pick_files),
        transform=train_transform,
        secondary_key_fixed=bool(train_secondary_key_fixed),
    )
    ds_infer_full = _build_dataset_for_subset(
        cfg,
        typed.infer.subset_traces,
        segy_files=list(infer_segy_files),
        phase_pick_files=list(infer_phase_pick_files),
        transform=infer_transform,
        secondary_key_fixed=True,
    )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(typed.train.lr))

    def infer_epoch_fn(model, loader, device, vis_epoch_dir, vis_n, max_batches):
        return (_run_infer_epoch(
                model=model,
                loader=loader,
                device=device,
                vis_out_dir=str(vis_epoch_dir),
                vis_n=vis_n,
                max_batches=max_batches,
            ))

    spec = TrainSkeletonSpec(
        pipeline='psn',
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
