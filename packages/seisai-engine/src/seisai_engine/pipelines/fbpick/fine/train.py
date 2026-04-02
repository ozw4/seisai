from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from seisai_utils.config import optional_float, optional_str, require_dict, require_list_str
from seisai_utils.listfiles import expand_cfg_listfiles

from seisai_engine.optim import build_optimizer
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
from seisai_engine.pipelines.common.config_io import resolve_relpath

from .build_dataset import build_dataset
from .build_model import build_model
from .build_plan import build_plan_from_config
from .config import FineInputCfg, FineTrainConfig, load_fine_train_config
from .infer import run_infer_epoch
from .loss import build_fine_criterion

__all__ = ['main']

DEFAULT_CONFIG_PATH = Path('examples/config_train_fbpick_fine.yaml')


def _validate_runtime_contract(typed: FineTrainConfig) -> None:
    if not isinstance(typed, FineTrainConfig):
        raise TypeError('typed must be FineTrainConfig')
    if str(typed.input.input_key) != 'input':
        raise ValueError('config.input.input_key must be "input" for fine runtime')
    if str(typed.target.target_key) != 'target':
        raise ValueError('config.target.target_key must be "target" for fine runtime')
    if str(typed.target.local_pick_idx_key) != 'local_pick_idx':
        raise ValueError(
            'config.target.local_pick_idx_key must be "local_pick_idx" to match fine build_plan'
        )
    if int(typed.model.in_chans) != 1 or int(typed.model.out_chans) != 1:
        raise ValueError('fine runtime requires model in_chans=1 and out_chans=1')


def _resolve_paths(base_dir: Path, paths: list[str]) -> list[str]:
    resolved: list[str] = []
    for idx, item in enumerate(paths):
        if not isinstance(item, str):
            raise TypeError(f'path[{idx}] must be str')
        resolved.append(resolve_relpath(base_dir, item))
    return resolved


def _resolve_dataset_paths(
    *,
    cfg: dict[str, Any],
    base_dir: Path,
) -> tuple[list[str], list[str], list[str], list[str]]:
    paths_cfg = require_dict(cfg, 'paths')
    train_segy_files = _resolve_paths(base_dir, require_list_str(paths_cfg, 'segy_files'))
    train_fb_files = _resolve_paths(base_dir, require_list_str(paths_cfg, 'fb_files'))

    has_eval_segy = 'infer_segy_files' in paths_cfg
    has_eval_fb = 'infer_fb_files' in paths_cfg
    if has_eval_segy != has_eval_fb:
        raise ValueError(
            'paths.infer_segy_files and paths.infer_fb_files must either both be set or both be omitted'
        )
    if has_eval_segy:
        eval_segy_files = _resolve_paths(base_dir, require_list_str(paths_cfg, 'infer_segy_files'))
        eval_fb_files = _resolve_paths(base_dir, require_list_str(paths_cfg, 'infer_fb_files'))
    else:
        print('fine eval paths are absent; reusing training segy/fb files for eval')
        eval_segy_files = list(train_segy_files)
        eval_fb_files = list(train_fb_files)
    return train_segy_files, train_fb_files, eval_segy_files, eval_fb_files


def _validate_dataset_runtime_sample(
    *,
    sample: dict[str, Any],
    input_cfg: FineInputCfg,
    require_target: bool,
    label: str,
) -> None:
    if not isinstance(sample, dict):
        raise TypeError(f'{label}: dataset sample must be dict')
    if input_cfg.amplitude_key not in sample:
        raise KeyError(f'{label}: dataset sample must contain {input_cfg.amplitude_key!r}')
    if 'input' not in sample:
        raise KeyError(f'{label}: dataset sample must contain "input"')
    if 'raw_sample_idx_local' not in sample:
        raise KeyError(f'{label}: dataset sample must contain "raw_sample_idx_local"')
    if 'meta' not in sample or not isinstance(sample['meta'], dict):
        raise KeyError(f'{label}: dataset sample must contain dict meta')
    if 'raw_sample_idx_local' not in sample['meta']:
        raise KeyError(f'{label}: dataset sample meta must contain "raw_sample_idx_local"')

    amplitude = sample[input_cfg.amplitude_key]
    x_input = sample['input']
    raw_sample_idx_local = sample['raw_sample_idx_local']
    meta_raw_sample_idx_local = sample['meta']['raw_sample_idx_local']

    if not isinstance(amplitude, torch.Tensor):
        amplitude = torch.as_tensor(amplitude)
    if not isinstance(x_input, torch.Tensor):
        raise TypeError(f'{label}: sample["input"] must be torch.Tensor')
    if not isinstance(raw_sample_idx_local, torch.Tensor):
        raw_sample_idx_local = torch.as_tensor(raw_sample_idx_local)
    if not isinstance(meta_raw_sample_idx_local, torch.Tensor):
        meta_raw_sample_idx_local = torch.as_tensor(meta_raw_sample_idx_local)

    if tuple(x_input.shape[:2]) != (1, 1):
        raise ValueError(f'{label}: sample["input"] must have shape (1,1,W), got {tuple(x_input.shape)}')
    if amplitude.ndim != 2 or int(amplitude.shape[0]) != 1:
        raise ValueError(
            f'{label}: sample[{input_cfg.amplitude_key!r}] must have shape (1,W), got {tuple(amplitude.shape)}'
        )
    if int(raw_sample_idx_local.ndim) != 1:
        raise ValueError(
            f'{label}: sample["raw_sample_idx_local"] must have shape (W,), got {tuple(raw_sample_idx_local.shape)}'
        )
    if not torch.equal(x_input[0], amplitude):
        raise ValueError(
            f'{label}: sample["input"][0] must match sample[{input_cfg.amplitude_key!r}] exactly'
        )
    if not torch.equal(raw_sample_idx_local.to(dtype=torch.int64), meta_raw_sample_idx_local.to(dtype=torch.int64)):
        raise ValueError(
            f'{label}: sample["raw_sample_idx_local"] must match meta["raw_sample_idx_local"] exactly'
        )
    if require_target and 'target' not in sample:
        raise KeyError(f'{label}: dataset sample must contain "target"')


def _validate_loop_sizes(typed: FineTrainConfig, *, ds_train_full, ds_eval_full) -> None:
    train_required = int(typed.common.train.samples_per_epoch)
    eval_required = int(typed.common.infer.infer_batch_size) * int(typed.common.infer.infer_max_batches)
    if len(ds_train_full) < train_required:
        raise ValueError(
            f'train dataset length {len(ds_train_full)} is smaller than train.samples_per_epoch={train_required}'
        )
    if len(ds_eval_full) < eval_required:
        raise ValueError(
            'eval dataset length '
            f'{len(ds_eval_full)} is smaller than '
            f'infer.batch_size * infer.max_batches = {eval_required}'
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    args, _unknown = parser.parse_known_args(argv)

    cfg, base_dir = load_cfg_with_base_dir(Path(args.config))
    path_keys = ['paths.segy_files', 'paths.fb_files']
    paths_cfg_raw = require_dict(cfg, 'paths')
    if 'infer_segy_files' in paths_cfg_raw or 'infer_fb_files' in paths_cfg_raw:
        path_keys.extend(['paths.infer_segy_files', 'paths.infer_fb_files'])
    expand_cfg_listfiles(cfg, keys=path_keys)

    typed = load_fine_train_config(cfg, base_dir=base_dir)
    _validate_runtime_contract(typed)
    common = typed.common
    train_cfg_raw = require_dict(cfg, 'train')

    out_dir_path = resolve_out_dir(cfg, base_dir)
    device = resolve_device(optional_str(train_cfg_raw, 'device', 'auto'))
    seed_all(common.seeds.seed_train)

    if int(common.train.train_num_workers) != 0:
        raise ValueError('fine train.num_workers must be 0 for the LocalWindowDataset mmap flow')
    if not typed.ckpt.save_best_only:
        raise ValueError('ckpt.save_best_only must be true for fine training')
    ckpt_metric = str(typed.ckpt.metric).strip()
    ckpt_mode = str(typed.ckpt.mode).strip()
    if not ckpt_metric:
        raise ValueError('ckpt.metric must be non-empty')
    if ckpt_mode not in ('min', 'max'):
        raise ValueError(f'ckpt.mode must be "min" or "max" (got {ckpt_mode})')
    if ckpt_metric in ('infer_loss', 'infer/loss') and ckpt_mode != 'min':
        raise ValueError('ckpt.mode must be "min" when ckpt.metric is infer_loss')

    loss_specs_train, loss_specs_eval = parse_train_eval_loss_specs(
        cfg,
        train_cfg=train_cfg_raw,
        default_scope='all',
        scope_key='loss_scope',
        losses_key='losses',
        train_label='train.losses',
        eval_label='eval.losses',
    )
    criterion_train = build_fine_criterion(
        list(loss_specs_train),
        use_label_valid=bool(typed.train.use_label_valid_mask),
    )
    criterion_eval = build_fine_criterion(
        list(loss_specs_eval),
        use_label_valid=bool(typed.eval.use_label_valid_mask),
    )

    train_segy_files, train_fb_files, eval_segy_files, eval_fb_files = _resolve_dataset_paths(
        cfg=cfg,
        base_dir=base_dir,
    )
    plan = build_plan_from_config(typed)
    ds_train_full = build_dataset(
        segy_files=list(train_segy_files),
        fb_files=list(train_fb_files),
        transform=None,
        plan=plan,
        input_cfg=typed.input,
        window_cfg=typed.window,
        mode='train',
    )
    ds_eval_full = build_dataset(
        segy_files=list(eval_segy_files),
        fb_files=list(eval_fb_files),
        transform=None,
        plan=plan,
        input_cfg=typed.input,
        window_cfg=typed.window,
        mode='eval',
    )
    _validate_dataset_runtime_sample(
        sample=ds_train_full[0],
        input_cfg=typed.input,
        require_target=True,
        label='train_dataset',
    )
    _validate_dataset_runtime_sample(
        sample=ds_eval_full[0],
        input_cfg=typed.input,
        require_target=True,
        label='eval_dataset',
    )
    _validate_loop_sizes(typed, ds_train_full=ds_train_full, ds_eval_full=ds_eval_full)
    print(f'fine dataset sizes: train={len(ds_train_full)} eval={len(ds_eval_full)}')

    model_sig = asdict(typed.model)
    model = build_model(typed.model).to(device)
    maybe_load_init_weights(
        cfg=cfg,
        base_dir=base_dir,
        model=model,
        model_sig=model_sig,
    )

    weight_decay = optional_float(train_cfg_raw, 'weight_decay', 0.01)
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
            vis_out_dir=str(vis_epoch_dir),
            vis_n=vis_n,
            max_batches=max_batches,
        )

    spec = TrainSkeletonSpec(
        pipeline='fbpick_fine',
        cfg=cfg,
        base_dir=base_dir,
        out_dir=out_dir_path,
        vis_subdir=str(common.output.vis_subdir),
        model_sig=model_sig,
        model=model,
        optimizer=optimizer,
        criterion=criterion_train,
        ds_train_full=ds_train_full,
        ds_infer_full=ds_eval_full,
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
        ckpt_extra={
            'output_ids': ['FB_LOCAL_PROB'],
            'softmax_axis': 'time',
            'input_semantics': 'amplitude_only_1ch',
            'raw_pick_restore_key': 'raw_sample_idx_local',
            'invalid_index': -1,
        },
        print_freq=common.train.print_freq,
    )
    run_train_skeleton(spec)


if __name__ == '__main__':
    main()
