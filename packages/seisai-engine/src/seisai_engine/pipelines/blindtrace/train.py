from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch
from seisai_utils.config import (
    optional_bool,
    optional_float,
    optional_int,
    optional_str,
    optional_tuple2_float,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_list_str,
)

from seisai_engine.infer.runner import TiledHConfig
from seisai_engine.pipelines.common import (
    TrainSkeletonSpec,
    expand_cfg_listfiles,
    load_cfg_with_base_dir,
    resolve_cfg_paths,
    resolve_out_dir,
    run_train_skeleton,
    seed_all,
)
from seisai_engine.pipelines.common.validate_primary_keys import validate_primary_keys

from .build_dataset import (
    build_dataset,
    build_fbgate,
    build_infer_transform,
    build_train_transform,
)
from .build_model import build_model
from .build_plan import build_plan
from .infer import run_infer_epoch
from .loss import build_masked_criterion
from .vis import build_triptych_cfg

__all__ = ['main']

DEFAULT_CONFIG_PATH = Path('examples/config_train_blindtrace.yaml')


def _validate_mask_ratio_for_subset(
    *, mask_ratio: float, subset_traces: int, label: str
) -> None:
    masked = round(float(mask_ratio) * int(subset_traces))
    if masked < 1:
        msg = (
            f'{label}: round(mask.ratio * subset_traces) must be >= 1 for masked_only '
            f'(ratio={float(mask_ratio)}, subset_traces={int(subset_traces)})'
        )
        raise ValueError(msg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    args, _unknown = parser.parse_known_args(argv)

    cfg, base_dir = load_cfg_with_base_dir(Path(args.config))
    paths_raw = require_dict(cfg, 'paths')
    path_keys = ['paths.segy_files', 'paths.infer_segy_files']
    if 'phase_pick_files' in paths_raw:
        path_keys.append('paths.phase_pick_files')
    if 'infer_phase_pick_files' in paths_raw:
        path_keys.append('paths.infer_phase_pick_files')

    resolve_cfg_paths(cfg, base_dir, keys=path_keys)
    expand_cfg_listfiles(cfg, keys=path_keys)

    paths = require_dict(cfg, 'paths')
    ds_cfg = require_dict(cfg, 'dataset')
    transform_cfg = require_dict(cfg, 'transform')
    fbgate_cfg = require_dict(cfg, 'fbgate')
    mask_cfg = require_dict(cfg, 'mask')
    input_cfg = require_dict(cfg, 'input')
    train_cfg = require_dict(cfg, 'train')
    infer_cfg = require_dict(cfg, 'infer')
    tile_cfg = require_dict(cfg, 'tile')
    vis_cfg = require_dict(cfg, 'vis')
    model_cfg = require_dict(cfg, 'model')
    ckpt_cfg = require_dict(cfg, 'ckpt')

    segy_files = require_list_str(paths, 'segy_files')
    infer_segy_files = require_list_str(paths, 'infer_segy_files')
    phase_pick_files = (
        require_list_str(paths, 'phase_pick_files')
        if 'phase_pick_files' in paths
        else None
    )
    infer_phase_pick_files = (
        require_list_str(paths, 'infer_phase_pick_files')
        if 'infer_phase_pick_files' in paths
        else None
    )
    out_dir_path = resolve_out_dir(cfg, base_dir)

    max_trials = optional_int(ds_cfg, 'max_trials', 2048)
    use_header_cache = optional_bool(ds_cfg, 'use_header_cache', default=True)
    verbose = optional_bool(ds_cfg, 'verbose', default=True)
    progress = optional_bool(ds_cfg, 'progress', default=bool(verbose))
    primary_keys_list = ds_cfg.get('primary_keys', ['ffid'])
    primary_keys = validate_primary_keys(primary_keys_list)

    time_len = require_int(transform_cfg, 'time_len')
    per_trace_standardize = optional_bool(
        transform_cfg, 'per_trace_standardize', default=True
    )

    apply_on = optional_str(fbgate_cfg, 'apply_on', 'on')
    min_pick_ratio = optional_float(fbgate_cfg, 'min_pick_ratio', 0.0)

    mask_ratio = require_float(mask_cfg, 'ratio')
    mask_mode = optional_str(mask_cfg, 'mode', 'replace').lower()
    noise_std = optional_float(mask_cfg, 'noise_std', 1.0)
    if mask_mode not in ('replace', 'add'):
        msg = 'mask.mode must be "replace" or "add"'
        raise ValueError(msg)

    use_offset_ch = optional_bool(input_cfg, 'use_offset_ch', default=False)
    offset_normalize = optional_bool(input_cfg, 'offset_normalize', default=True)
    use_time_ch = optional_bool(input_cfg, 'use_time_ch', default=False)

    seed_train = optional_int(train_cfg, 'seed', 42)
    loss_scope = optional_str(train_cfg, 'loss_scope', 'masked_only').lower()
    loss_kind = optional_str(train_cfg, 'loss_kind', 'l1').lower()
    shift_max = optional_int(train_cfg, 'shift_max', 8)
    train_batch_size = require_int(train_cfg, 'batch_size')
    train_num_workers = optional_int(train_cfg, 'num_workers', 0)
    train_use_amp = require_bool(train_cfg, 'use_amp')
    max_norm = optional_float(train_cfg, 'max_norm', 1.0)
    lr = require_float(train_cfg, 'lr')
    weight_decay = require_float(train_cfg, 'weight_decay')
    epochs = require_int(train_cfg, 'epochs')
    samples_per_epoch = require_int(train_cfg, 'samples_per_epoch')
    train_subset_traces = require_int(train_cfg, 'subset_traces')

    seed_infer = optional_int(infer_cfg, 'seed', 43)
    infer_batch_size = require_int(infer_cfg, 'batch_size')
    infer_num_workers = optional_int(infer_cfg, 'num_workers', 0)
    infer_max_batches = require_int(infer_cfg, 'max_batches')
    infer_subset_traces = require_int(infer_cfg, 'subset_traces')

    tile_h = require_int(tile_cfg, 'tile_h')
    overlap_h = require_int(tile_cfg, 'overlap_h')
    tiles_per_batch = require_int(tile_cfg, 'tiles_per_batch')
    tile_amp = optional_bool(tile_cfg, 'amp', default=True)
    use_tqdm = optional_bool(tile_cfg, 'use_tqdm', default=False)

    vis_n = require_int(vis_cfg, 'n')
    vis_subdir = optional_str(vis_cfg, 'out_subdir', 'vis')
    cmap = optional_str(vis_cfg, 'cmap', 'seismic')
    vmin = optional_float(vis_cfg, 'vmin', -3.0)
    vmax = optional_float(vis_cfg, 'vmax', 3.0)
    transpose_for_trace_time = optional_bool(
        vis_cfg, 'transpose_for_trace_time', default=True
    )
    per_trace_norm = optional_bool(vis_cfg, 'per_trace_norm', default=True)
    per_trace_eps = optional_float(vis_cfg, 'per_trace_eps', 1e-8)
    figsize = optional_tuple2_float(vis_cfg, 'figsize', (20.0, 15.0))
    dpi = optional_int(vis_cfg, 'dpi', 300)

    backbone = optional_str(model_cfg, 'backbone', 'resnet18')
    pretrained = optional_bool(model_cfg, 'pretrained', default=False)

    ckpt_best_only = optional_bool(ckpt_cfg, 'save_best_only', default=True)
    ckpt_metric = optional_str(ckpt_cfg, 'metric', 'infer_loss')
    ckpt_mode = optional_str(ckpt_cfg, 'mode', 'min')
    if not ckpt_best_only:
        msg = 'ckpt.save_best_only must be true'
        raise ValueError(msg)
    if ckpt_metric != 'infer_loss':
        msg = 'ckpt.metric must be "infer_loss"'
        raise ValueError(msg)
    if ckpt_mode != 'min':
        msg = 'ckpt.mode must be "min"'
        raise ValueError(msg)

    if loss_kind not in ('l1', 'mse', 'shift_mse', 'shift_robust_mse'):
        msg = 'train.loss_kind must be "l1", "mse", "shift_mse", or "shift_robust_mse"'
        raise ValueError(msg)

    if loss_scope == 'masked_only':
        _validate_mask_ratio_for_subset(
            mask_ratio=mask_ratio, subset_traces=train_subset_traces, label='train'
        )
        _validate_mask_ratio_for_subset(
            mask_ratio=mask_ratio, subset_traces=infer_subset_traces, label='infer'
        )

    if tile_h > infer_subset_traces:
        msg = 'tile.tile_h must be <= infer.subset_traces'
        raise ValueError(msg)
    if samples_per_epoch <= 0:
        msg = 'train.samples_per_epoch must be positive'
        raise ValueError(msg)
    if infer_max_batches <= 0:
        msg = 'infer.max_batches must be positive'
        raise ValueError(msg)

    in_chans = 1 + int(bool(use_offset_ch)) + int(bool(use_time_ch))
    out_chans = 1
    model_sig = {
        'backbone': str(backbone),
        'pretrained': bool(pretrained),
        'in_chans': int(in_chans),
        'out_chans': int(out_chans),
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_all(seed_train)

    if phase_pick_files is not None and len(segy_files) != len(phase_pick_files):
        msg = 'paths.segy_files and paths.phase_pick_files must have same length'
        raise ValueError(msg)
    if infer_phase_pick_files is not None and len(infer_segy_files) != len(
        infer_phase_pick_files
    ):
        msg = (
            'paths.infer_segy_files and paths.infer_phase_pick_files '
            'must have same length'
        )
        raise ValueError(msg)

    train_transform = build_train_transform(
        time_len=int(time_len), per_trace_standardize=bool(per_trace_standardize)
    )
    infer_transform = build_infer_transform(
        time_len=int(time_len), per_trace_standardize=bool(per_trace_standardize)
    )
    plan = build_plan(
        mask_ratio=mask_ratio,
        mask_mode=mask_mode,
        noise_std=noise_std,
        use_offset_ch=bool(use_offset_ch),
        offset_normalize=bool(offset_normalize),
        use_time_ch=bool(use_time_ch),
    )
    if phase_pick_files is None:
        warnings.warn(
            'train fb_files is None; using fbgate apply_on="off" and min_pick_ratio=0.0',
            UserWarning,
        )
        fbgate_train = build_fbgate(
            apply_on='off', min_pick_ratio=0.0, verbose=bool(verbose)
        )
    else:
        fbgate_train = build_fbgate(
            apply_on=apply_on, min_pick_ratio=min_pick_ratio, verbose=bool(verbose)
        )

    if infer_phase_pick_files is None:
        warnings.warn(
            'infer fb_files is None; using fbgate apply_on="off" and min_pick_ratio=0.0',
            UserWarning,
        )
        fbgate_infer = build_fbgate(
            apply_on='off', min_pick_ratio=0.0, verbose=bool(verbose)
        )
    else:
        fbgate_infer = build_fbgate(
            apply_on=apply_on, min_pick_ratio=min_pick_ratio, verbose=bool(verbose)
        )

    criterion = build_masked_criterion(
        loss_kind=loss_kind,
        loss_scope=loss_scope,
        shift_max=int(shift_max),
    )

    ds_train_full = build_dataset(
        segy_files=segy_files,
        fb_files=phase_pick_files,
        transform=train_transform,
        fbgate=fbgate_train,
        plan=plan,
        subset_traces=int(train_subset_traces),
        primary_keys=primary_keys,
        secondary_key_fixed=False,
        verbose=bool(verbose),
        progress=bool(progress),
        max_trials=int(max_trials),
        use_header_cache=bool(use_header_cache),
    )

    ds_infer_full = build_dataset(
        segy_files=infer_segy_files,
        fb_files=infer_phase_pick_files,
        transform=infer_transform,
        fbgate=fbgate_infer,
        plan=plan,
        subset_traces=int(infer_subset_traces),
        primary_keys=primary_keys,
        secondary_key_fixed=True,
        verbose=bool(verbose),
        progress=bool(progress),
        max_trials=int(max_trials),
        use_header_cache=bool(use_header_cache),
    )

    model = build_model(
        backbone=backbone,
        in_chans=int(in_chans),
        out_chans=int(out_chans),
        pretrained=bool(pretrained),
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    tiled_cfg = TiledHConfig(
        tile_h=int(tile_h),
        overlap_h=int(overlap_h),
        tiles_per_batch=int(tiles_per_batch),
        amp=bool(tile_amp),
        use_tqdm=bool(use_tqdm),
    )

    triptych_cfg = build_triptych_cfg(
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        transpose_for_trace_time=bool(transpose_for_trace_time),
        per_trace_norm=bool(per_trace_norm),
        per_trace_eps=float(per_trace_eps),
        figsize=figsize,
        dpi=int(dpi),
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
        pipeline='blindtrace',
        cfg=cfg,
        base_dir=base_dir,
        out_dir=out_dir_path,
        vis_subdir=str(vis_subdir),
        model_sig=model_sig,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        ds_train_full=ds_train_full,
        ds_infer_full=ds_infer_full,
        device=device,
        seed_train=int(seed_train),
        seed_infer=int(seed_infer),
        epochs=int(epochs),
        train_batch_size=int(train_batch_size),
        train_num_workers=int(train_num_workers),
        samples_per_epoch=int(samples_per_epoch),
        max_norm=float(max_norm),
        use_amp_train=bool(train_use_amp),
        infer_batch_size=int(infer_batch_size),
        infer_num_workers=int(infer_num_workers),
        infer_max_batches=int(infer_max_batches),
        vis_n=int(vis_n),
        infer_epoch_fn=infer_epoch_fn,
        print_freq=10,
    )

    run_train_skeleton(spec)


if __name__ == '__main__':
    main()
