from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

from seisai_utils.config import (
    optional_bool,
    optional_float,
    optional_int,
    optional_str,
    require_dict,
    require_list_str,
)
from seisai_utils.listfiles import expand_cfg_listfiles, get_cfg_listfile_meta

from seisai_engine.pipelines.common import (
    TrainSkeletonSpec,
    load_cfg_with_base_dir,
    maybe_load_init_weights,
    parse_train_eval_loss_specs,
    resolve_device,
    resolve_out_dir,
    run_train_skeleton,
    seed_all,
)
from seisai_engine.pipelines.common.config_keys import normalize_endian
from seisai_engine.pipelines.common.validate_primary_keys import validate_primary_keys
from seisai_engine.optim import build_optimizer

from .build_dataset import (
    build_dataset,
    build_fbgate,
    build_infer_transform,
    build_train_transform,
)
from .build_model import build_model
from .build_plan import build_plan_from_config
from .config import CoarseTrainConfig, load_coarse_train_config
from .infer import build_tiled_w_cfg, run_infer_epoch
from .loss import build_coarse_criterion

__all__ = ['main']

DEFAULT_CONFIG_PATH = Path('examples/config_train_fbpick_coarse.yaml')


def _parse_trace_decimation_cfg(train_cfg: dict) -> tuple[float, tuple[int, int]]:
    raw = train_cfg.get('trace_decimation')
    if raw is None:
        return 0.0, (1, 1)
    if not isinstance(raw, dict):
        raise TypeError('train.trace_decimation must be dict')
    unknown = set(raw) - {'prob', 'stride_range'}
    if unknown:
        raise ValueError(
            'train.trace_decimation has unsupported keys: '
            f'{sorted(unknown)}'
        )

    prob_raw = raw.get('prob', 0.0)
    if isinstance(prob_raw, bool) or not isinstance(prob_raw, (int, float)):
        raise TypeError('train.trace_decimation.prob must be float in [0, 1]')
    prob = float(prob_raw)
    if not math.isfinite(prob):
        raise ValueError('train.trace_decimation.prob must be finite')
    if prob < 0.0 or prob > 1.0:
        raise ValueError('train.trace_decimation.prob must be in [0, 1]')

    stride_range_raw = raw.get('stride_range', (1, 1))
    if not isinstance(stride_range_raw, (list, tuple)) or len(stride_range_raw) != 2:
        raise TypeError('train.trace_decimation.stride_range must be [min_int, max_int]')
    min_stride_raw, max_stride_raw = stride_range_raw
    if isinstance(min_stride_raw, bool) or not isinstance(min_stride_raw, int):
        raise TypeError('train.trace_decimation.stride_range[0] must be int')
    if isinstance(max_stride_raw, bool) or not isinstance(max_stride_raw, int):
        raise TypeError('train.trace_decimation.stride_range[1] must be int')
    min_stride = int(min_stride_raw)
    max_stride = int(max_stride_raw)
    if min_stride < 1:
        raise ValueError('train.trace_decimation.stride_range[0] must be >= 1')
    if min_stride > max_stride:
        raise ValueError('train.trace_decimation.stride_range requires min <= max')
    return prob, (min_stride, max_stride)


def _validate_runtime_contract(typed: CoarseTrainConfig) -> None:
    if not isinstance(typed, CoarseTrainConfig):
        raise TypeError('typed must be CoarseTrainConfig')
    if str(typed.input.input_key) != 'input':
        raise ValueError('config.input.input_key must be "input" for coarse runtime')
    if str(typed.target.target_key) != 'target':
        raise ValueError('config.target.target_key must be "target" for coarse runtime')
    if str(typed.target.fb_index_key) != 'fb_idx_view':
        raise ValueError(
            'config.target.fb_index_key must be "fb_idx_view" to match coarse build_plan'
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    args, _unknown = parser.parse_known_args(argv)

    cfg, base_dir = load_cfg_with_base_dir(Path(args.config))

    path_keys = [
        'paths.segy_files',
        'paths.fb_files',
        'paths.infer_segy_files',
        'paths.infer_fb_files',
    ]
    augment_raw = cfg.get('augment')
    if isinstance(augment_raw, dict):
        noise_add_raw = augment_raw.get('noise_add')
        if isinstance(noise_add_raw, dict) and 'segy_files' in noise_add_raw:
            path_keys.append('augment.noise_add.segy_files')
    expand_cfg_listfiles(cfg, keys=path_keys)

    train_cfg = require_dict(cfg, 'train')
    device_str = optional_str(train_cfg, 'device', 'auto')
    train_trace_decimate_prob, train_trace_decimate_stride_range = (
        _parse_trace_decimation_cfg(train_cfg)
    )

    typed = load_coarse_train_config(cfg, base_dir=base_dir)
    _validate_runtime_contract(typed)
    common = typed.common

    out_dir_path = resolve_out_dir(cfg, base_dir)

    if not typed.ckpt.save_best_only:
        raise ValueError('ckpt.save_best_only must be true for coarse training')
    ckpt_metric = str(typed.ckpt.metric).strip()
    ckpt_mode = str(typed.ckpt.mode).strip()
    if not ckpt_metric:
        raise ValueError('ckpt.metric must be non-empty')
    if ckpt_mode not in ('min', 'max'):
        raise ValueError(f'ckpt.mode must be "min" or "max" (got {ckpt_mode})')
    if ckpt_metric in ('infer_loss', 'infer/loss') and ckpt_mode != 'min':
        raise ValueError('ckpt.mode must be "min" when ckpt.metric is infer_loss')

    model_sig = asdict(typed.model)
    device = resolve_device(device_str)
    seed_all(common.seeds.seed_train)

    loss_specs_train, loss_specs_eval = parse_train_eval_loss_specs(
        cfg,
        train_cfg=train_cfg,
        default_scope='all',
        scope_key='loss_scope',
        losses_key='losses',
        train_label='train.losses',
        eval_label='eval.losses',
    )
    criterion_train = build_coarse_criterion(
        list(loss_specs_train),
        use_label_valid=bool(typed.train.use_label_valid_mask),
    )
    criterion_eval = build_coarse_criterion(
        list(loss_specs_eval),
        use_label_valid=bool(typed.eval.use_label_valid_mask),
    )

    paths_cfg = require_dict(cfg, 'paths')
    train_segy_files = require_list_str(paths_cfg, 'segy_files')
    train_fb_files = require_list_str(paths_cfg, 'fb_files')
    infer_segy_files = require_list_str(paths_cfg, 'infer_segy_files')
    infer_fb_files = require_list_str(paths_cfg, 'infer_fb_files')
    train_sampling_overrides = get_cfg_listfile_meta(cfg, key_path='paths.segy_files')
    infer_sampling_overrides = get_cfg_listfile_meta(
        cfg,
        key_path='paths.infer_segy_files',
    )
    if len(train_segy_files) != len(train_fb_files):
        raise ValueError('paths.segy_files and paths.fb_files must have same length')
    if len(infer_segy_files) != len(infer_fb_files):
        raise ValueError(
            'paths.infer_segy_files and paths.infer_fb_files must have same length'
        )

    ds_cfg = require_dict(cfg, 'dataset')
    max_trials = optional_int(ds_cfg, 'max_trials', 2048)
    use_header_cache = optional_bool(ds_cfg, 'use_header_cache', default=True)
    verbose = optional_bool(ds_cfg, 'verbose', default=True)
    progress = optional_bool(ds_cfg, 'progress', default=bool(verbose))
    primary_keys = validate_primary_keys(ds_cfg.get('primary_keys', ['ffid']))
    train_secondary_key_fixed = optional_bool(
        ds_cfg,
        'secondary_key_fixed',
        default=False,
    )
    waveform_mode = optional_str(ds_cfg, 'waveform_mode', 'eager').lower()
    if waveform_mode not in ('eager', 'mmap'):
        raise ValueError('dataset.waveform_mode must be "eager" or "mmap"')
    ds_cfg['waveform_mode'] = waveform_mode
    train_endian = normalize_endian(
        value=optional_str(ds_cfg, 'train_endian', 'big'),
        key_name='dataset.train_endian',
    )
    infer_endian = normalize_endian(
        value=optional_str(ds_cfg, 'infer_endian', 'big'),
        key_name='dataset.infer_endian',
    )
    if waveform_mode == 'mmap' and int(common.train.train_num_workers) > 0:
        raise ValueError('dataset.waveform_mode="mmap" requires train.num_workers=0')
    if waveform_mode == 'mmap' and int(common.infer.infer_num_workers) > 0:
        raise ValueError('dataset.waveform_mode="mmap" requires infer.num_workers=0')

    fbgate = build_fbgate(cfg.get('fbgate'))
    plan = build_plan_from_config(typed)
    tile_cfg = build_tiled_w_cfg(require_dict(cfg, 'tile'))

    train_transform = build_train_transform(
        cfg,
        noise_provider_ctx={
            'subset_traces': int(typed.train.subset_traces),
            'primary_keys': primary_keys,
            'secondary_key_fixed': bool(train_secondary_key_fixed),
            'waveform_mode': str(waveform_mode),
            'segy_endian': str(train_endian),
            'header_cache_dir': None,
            'use_header_cache': bool(use_header_cache),
        },
    )
    infer_transform = build_infer_transform(cfg)

    ds_train_full = build_dataset(
        segy_files=list(train_segy_files),
        fb_files=list(train_fb_files),
        sampling_overrides=train_sampling_overrides,
        transform=train_transform,
        fbgate=fbgate,
        plan=plan,
        subset_traces=int(typed.train.subset_traces),
        trace_decimate_prob=float(train_trace_decimate_prob),
        trace_decimate_stride_range=tuple(train_trace_decimate_stride_range),
        primary_keys=primary_keys,
        secondary_key_fixed=bool(train_secondary_key_fixed),
        verbose=bool(verbose),
        progress=bool(progress),
        max_trials=int(max_trials),
        use_header_cache=bool(use_header_cache),
        waveform_mode=str(waveform_mode),
        segy_endian=str(train_endian),
        input_cfg=typed.input,
        require_target=True,
    )
    ds_infer_full = build_dataset(
        segy_files=list(infer_segy_files),
        fb_files=list(infer_fb_files),
        sampling_overrides=infer_sampling_overrides,
        transform=infer_transform,
        fbgate=fbgate,
        plan=plan,
        subset_traces=int(typed.infer.subset_traces),
        trace_decimate_prob=0.0,
        trace_decimate_stride_range=(1, 1),
        primary_keys=primary_keys,
        secondary_key_fixed=True,
        verbose=bool(verbose),
        progress=bool(progress),
        max_trials=int(max_trials),
        use_header_cache=bool(use_header_cache),
        waveform_mode=str(waveform_mode),
        segy_endian=str(infer_endian),
        input_cfg=typed.input,
        require_target=True,
    )

    model = build_model(typed.model).to(device)
    maybe_load_init_weights(
        cfg=cfg,
        base_dir=base_dir,
        model=model,
        model_sig=model_sig,
    )

    weight_decay = optional_float(train_cfg, 'weight_decay', 0.01)
    optimizer = build_optimizer(
        cfg,
        model,
        lr=float(typed.train.lr),
        weight_decay=float(weight_decay),
    )

    def infer_epoch_fn(model, loader, device, vis_epoch_dir, vis_n, max_batches):
        return run_infer_epoch(
            model=model,
            loader=loader,
            device=device,
            criterion=criterion_eval,
            tiled_cfg=tile_cfg,
            vis_out_dir=str(vis_epoch_dir),
            vis_n=vis_n,
            max_batches=max_batches,
        )

    spec = TrainSkeletonSpec(
        pipeline='fbpick_coarse',
        cfg=cfg,
        base_dir=base_dir,
        out_dir=out_dir_path,
        vis_subdir=str(common.output.vis_subdir),
        model_sig=model_sig,
        model=model,
        optimizer=optimizer,
        criterion=criterion_train,
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
        gradient_accumulation_steps=common.train.gradient_accumulation_steps,
        infer_batch_size=common.infer.infer_batch_size,
        infer_num_workers=common.infer.infer_num_workers,
        infer_max_batches=common.infer.infer_max_batches,
        vis_n=common.infer.vis_n,
        infer_epoch_fn=infer_epoch_fn,
        ckpt_metric=ckpt_metric,
        ckpt_mode=ckpt_mode,
        ckpt_extra={'output_ids': ['FB'], 'softmax_axis': 'time'},
        print_freq=common.train.print_freq,
    )
    run_train_skeleton(spec)
