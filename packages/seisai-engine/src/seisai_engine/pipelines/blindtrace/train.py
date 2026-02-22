from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import torch
from seisai_utils.config import (
    optional_bool,
    optional_float,
    optional_int,
    optional_str,
    optional_tuple2_float,
    optional_value,
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
    maybe_load_init_weights,
    resolve_device,
    resolve_out_dir,
    run_train_skeleton,
    seed_all,
)
from seisai_engine.pipelines.common.encdec2d_cfg import build_encdec2d_kwargs
from seisai_engine.pipelines.common.validate_primary_keys import validate_primary_keys
from seisai_engine.loss import composite
from seisai_engine.optim import build_optimizer

from .build_dataset import (
    build_dataset,
    build_fbgate,
    build_infer_transform,
    build_train_transform,
)
from .build_model import build_model
from .build_plan import build_plan
from .infer import run_infer_epoch
from .vis import build_triptych_cfg

__all__ = ['main']

DEFAULT_CONFIG_PATH = Path('examples/config_train_blindtrace.yaml')


def _normalize_endian(*, value: str, key_name: str) -> str:
    endian = str(value).strip().lower()
    if endian not in ('big', 'little'):
        msg = f'{key_name} must be "big" or "little"'
        raise ValueError(msg)
    return endian


def _format_key(section: str, key: str) -> str:
    return f'{section}.{key}'


def _raise_if_deprecated_time_len_keys(*, train_cfg: object, transform_cfg: object) -> None:
    if isinstance(train_cfg, dict) and 'time_len' in train_cfg:
        msg = (
            f'deprecated key: {_format_key("train", "time_len")}; '
            f'use {_format_key("transform", "time_len")}'
        )
        raise ValueError(msg)
    if isinstance(transform_cfg, dict) and 'target_len' in transform_cfg:
        msg = (
            f'deprecated key: {_format_key("transform", "target_len")}; '
            f'use {_format_key("transform", "time_len")}'
        )
        raise ValueError(msg)


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


def _build_loss_specs_from_cfg(
    cfg: dict[str, Any], *, label_prefix: str, default_scope: str
) -> list[composite.LossSpec]:
    if not isinstance(cfg, dict):
        raise TypeError(f'{label_prefix} must be dict')

    loss_scope = optional_str(cfg, 'loss_scope', default_scope)
    losses = cfg.get('losses', None)
    if losses is not None:
        return composite.parse_loss_specs(
            losses,
            default_scope=loss_scope,
            label=f'{label_prefix}.losses',
            scope_label=f'{label_prefix}.loss_scope',
        )

    loss_kind = optional_str(cfg, 'loss_kind', 'l1').lower()
    if loss_kind not in (
        'l1',
        'mse',
        'shift_mse',
        'shift_robust_mse',
        'shift_robust_l1',
    ):
        msg = (
            f'{label_prefix}.loss_kind must be "l1", "mse", "shift_mse", '
            '"shift_robust_mse", or "shift_robust_l1"'
        )
        raise ValueError(msg)

    loss_items: list[dict[str, Any]] = []
    loss_params: dict[str, Any] = {}
    if loss_kind in ('shift_mse', 'shift_robust_mse', 'shift_robust_l1'):
        shift_max = optional_int(cfg, 'shift_max', 8)
        loss_params = {'shift_max': int(shift_max)}

    loss_items.append(
        {
            'kind': loss_kind,
            'weight': 1.0,
            'scope': loss_scope,
            'params': loss_params,
        }
    )

    fx_weight = optional_float(cfg, 'fx_weight', 0.0)
    if fx_weight < 0.0:
        raise ValueError(f'{label_prefix}.fx_weight must be >= 0')

    if fx_weight > 0.0:
        fx_use_log = optional_bool(cfg, 'fx_use_log', default=True)
        fx_eps = optional_float(cfg, 'fx_eps', 1.0e-6)
        fx_f_lo = optional_int(cfg, 'fx_f_lo', 0)
        fx_f_hi_raw = cfg.get('fx_f_hi', None)
        if fx_f_hi_raw is None:
            fx_f_hi = None
        else:
            if isinstance(fx_f_hi_raw, bool) or not isinstance(
                fx_f_hi_raw, (int, float)
            ):
                raise TypeError(f'{label_prefix}.fx_f_hi must be int or null')
            if isinstance(fx_f_hi_raw, float) and not fx_f_hi_raw.is_integer():
                raise ValueError(f'{label_prefix}.fx_f_hi must be int or null')
            fx_f_hi = int(fx_f_hi_raw)

        loss_items.append(
            {
                'kind': 'fx_mag_mse',
                'weight': float(fx_weight),
                'scope': loss_scope,
                'params': {
                    'use_log': bool(fx_use_log),
                    'eps': float(fx_eps),
                    'f_lo': int(fx_f_lo),
                    'f_hi': fx_f_hi,
                },
            }
        )

    return composite.parse_loss_specs(
        loss_items,
        default_scope=loss_scope,
        label=f'{label_prefix}.losses',
        scope_label=f'{label_prefix}.loss_scope',
    )


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

    expand_cfg_listfiles(cfg, keys=path_keys)

    _raise_if_deprecated_time_len_keys(
        train_cfg=cfg.get('train'),
        transform_cfg=cfg.get('transform'),
    )

    paths = require_dict(cfg, 'paths')
    ds_cfg = require_dict(cfg, 'dataset')
    transform_cfg = require_dict(cfg, 'transform')
    augment_cfg = cfg.get('augment')
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
    waveform_mode = optional_str(ds_cfg, 'waveform_mode', 'eager').lower()
    if waveform_mode not in ('eager', 'mmap'):
        msg = 'dataset.waveform_mode must be "eager" or "mmap"'
        raise ValueError(msg)
    ds_cfg['waveform_mode'] = waveform_mode
    train_endian = _normalize_endian(
        value=optional_str(ds_cfg, 'train_endian', 'big'),
        key_name='dataset.train_endian',
    )
    infer_endian = _normalize_endian(
        value=optional_str(ds_cfg, 'infer_endian', 'big'),
        key_name='dataset.infer_endian',
    )

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
    train_loss_scope = optional_str(train_cfg, 'loss_scope', 'masked_only')
    loss_specs_train = _build_loss_specs_from_cfg(
        train_cfg, label_prefix='train', default_scope=train_loss_scope
    )
    train_batch_size = require_int(train_cfg, 'batch_size')
    train_num_workers = optional_int(train_cfg, 'num_workers', 0)
    gradient_accumulation_steps = optional_value(
        train_cfg,
        'gradient_accumulation_steps',
        1,
        int,
        type_message='config.train.gradient_accumulation_steps must be int',
        coerce=int,
        coerce_default=True,
    )
    if int(gradient_accumulation_steps) <= 0:
        msg = 'config.train.gradient_accumulation_steps must be positive'
        raise ValueError(msg)
    train_use_amp = require_bool(train_cfg, 'use_amp')
    max_norm = optional_float(train_cfg, 'max_norm', 1.0)
    lr = require_float(train_cfg, 'lr')
    weight_decay = require_float(train_cfg, 'weight_decay')
    epochs = require_int(train_cfg, 'epochs')
    samples_per_epoch = require_int(train_cfg, 'samples_per_epoch')
    train_subset_traces = require_int(train_cfg, 'subset_traces')
    device_str = optional_str(train_cfg, 'device', 'auto')

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

    eval_cfg = cfg.get('eval')
    if eval_cfg is None:
        loss_specs_eval = loss_specs_train
    else:
        if not isinstance(eval_cfg, dict):
            raise TypeError('eval must be dict')
        loss_specs_eval = _build_loss_specs_from_cfg(
            eval_cfg, label_prefix='eval', default_scope=train_loss_scope
        )

    if any(spec.scope == 'masked_only' for spec in loss_specs_train):
        _validate_mask_ratio_for_subset(
            mask_ratio=mask_ratio, subset_traces=train_subset_traces, label='train'
        )
    if any(spec.scope == 'masked_only' for spec in loss_specs_eval):
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
    if waveform_mode == 'mmap' and int(train_num_workers) > 0:
        msg = 'dataset.waveform_mode="mmap" requires train.num_workers=0'
        raise ValueError(msg)
    if waveform_mode == 'mmap' and int(infer_num_workers) > 0:
        msg = 'dataset.waveform_mode="mmap" requires infer.num_workers=0'
        raise ValueError(msg)

    in_chans = 1 + int(bool(use_offset_ch)) + int(bool(use_time_ch))
    out_chans = 1
    if 'in_chans' in model_cfg:
        in_chans_cfg = model_cfg['in_chans']
        if not isinstance(in_chans_cfg, int):
            raise TypeError('config.model.in_chans must be int')
        if int(in_chans_cfg) != int(in_chans):
            msg = 'config.model.in_chans must match computed in_chans'
            raise ValueError(msg)
    if 'out_chans' in model_cfg:
        out_chans_cfg = model_cfg['out_chans']
        if not isinstance(out_chans_cfg, int):
            raise TypeError('config.model.out_chans must be int')
        if int(out_chans_cfg) != int(out_chans):
            msg = 'config.model.out_chans must match computed out_chans'
            raise ValueError(msg)

    encdec_kwargs = build_encdec2d_kwargs(
        model_cfg,
        in_chans=int(in_chans),
        out_chans=int(out_chans),
        defaults={'pretrained': False},
    )
    model_sig = dict(encdec_kwargs)

    device = resolve_device(device_str)
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
        time_len=int(time_len),
        per_trace_standardize=bool(per_trace_standardize),
        augment_cfg=augment_cfg,
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

    criterion_train = composite.build_weighted_criterion(loss_specs_train)
    criterion_eval = composite.build_weighted_criterion(loss_specs_eval)

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
        waveform_mode=str(waveform_mode),
        segy_endian=str(train_endian),
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
        waveform_mode=str(waveform_mode),
        segy_endian=str(infer_endian),
    )

    model = build_model(encdec_kwargs)
    model.to(device)
    maybe_load_init_weights(
        cfg=cfg,
        base_dir=base_dir,
        model=model,
        model_sig=model_sig,
    )

    optimizer = build_optimizer(
        cfg,
        model,
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
        return run_infer_epoch(
            model=model,
            loader=loader,
            device=device,
            criterion=criterion_eval,
            tiled_cfg=tiled_cfg,
            vis_cfg=triptych_cfg,
            vis_out_dir=str(vis_epoch_dir),
            vis_n=vis_n,
            max_batches=max_batches,
        )

    spec = TrainSkeletonSpec(
        pipeline='blindtrace',
        cfg=cfg,
        base_dir=base_dir,
        out_dir=out_dir_path,
        vis_subdir=str(vis_subdir),
        model_sig=model_sig,
        model=model,
        optimizer=optimizer,
        criterion=criterion_train,
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
        gradient_accumulation_steps=int(gradient_accumulation_steps),
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
