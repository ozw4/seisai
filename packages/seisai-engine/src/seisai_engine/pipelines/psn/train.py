from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from seisai_utils.config import (
    optional_bool,
    optional_float,
    optional_str,
    require_dict,
    require_list_str,
)
from seisai_utils.viz_phase import make_title_from_batch_meta, save_psn_debug_png

from seisai_engine.pipelines.common import (
    InferEpochResult,
    TrainSkeletonSpec,
    expand_cfg_listfiles,
    get_cfg_listfile_meta,
    load_cfg_with_base_dir,
    maybe_load_init_weights,
    resolve_device,
    resolve_out_dir,
    run_train_skeleton,
    seed_all,
)
from seisai_engine.optim import build_optimizer

from .build_dataset import build_dataset, build_infer_transform, build_train_transform
from .build_model import build_model
from .config import load_psn_train_config
from .loss import build_psn_criterion

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

__all__ = ['main']

DEFAULT_CONFIG_PATH = Path('examples/config_train_psn.yaml')


def _normalize_endian(*, value: str, key_name: str) -> str:
    endian = str(value).strip().lower()
    if endian not in ('big', 'little'):
        msg = f'{key_name} must be "big" or "little"'
        raise ValueError(msg)
    return endian


def _build_dataset_for_subset(
    cfg: dict,
    subset_traces: int,
    *,
    segy_files: list[str],
    phase_pick_files: list[str],
    transform,
    secondary_key_fixed: bool,
    segy_endian: str,
    sampling_overrides: list[dict[str, object] | None] | None,
):
    cfg_copy = copy.deepcopy(cfg)
    train_cfg = require_dict(cfg_copy, 'train')
    train_cfg['subset_traces'] = int(subset_traces)
    paths = require_dict(cfg_copy, 'paths')
    paths['segy_files'] = list(segy_files)
    paths['phase_pick_files'] = list(phase_pick_files)
    ds_cfg = require_dict(cfg_copy, 'dataset')
    ds_cfg['secondary_key_fixed'] = bool(secondary_key_fixed)
    return build_dataset(
        cfg_copy,
        transform=transform,
        segy_endian=segy_endian,
        sampling_overrides=sampling_overrides,
    )


def _to_tensor_bh(*, value, name: str, device: torch.device) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if int(tensor.ndim) != 2:
        msg = f'{name} must be (B,H), got shape={tuple(tensor.shape)}'
        raise ValueError(msg)
    return tensor.to(device=device, non_blocking=True)


def _phase_metrics(
    prefix: str,
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    if not isinstance(prefix, str) or not prefix:
        msg = 'prefix must be non-empty str'
        raise TypeError(msg)
    if not isinstance(pred, torch.Tensor) or int(pred.ndim) != 2:
        msg = 'pred must be torch.Tensor with shape (B,H)'
        raise ValueError(msg)
    if not isinstance(gt, torch.Tensor) or int(gt.ndim) != 2:
        msg = 'gt must be torch.Tensor with shape (B,H)'
        raise ValueError(msg)
    if not isinstance(valid, torch.Tensor) or int(valid.ndim) != 2:
        msg = 'valid must be torch.Tensor with shape (B,H)'
        raise ValueError(msg)
    if valid.dtype is not torch.bool:
        msg = f'valid must be bool tensor, got dtype={valid.dtype}'
        raise TypeError(msg)
    if tuple(pred.shape) != tuple(gt.shape) or tuple(pred.shape) != tuple(valid.shape):
        msg = (
            'shape mismatch: '
            f'pred={tuple(pred.shape)} gt={tuple(gt.shape)} valid={tuple(valid.shape)}'
        )
        raise ValueError(msg)

    n = int(valid.sum().item())
    if n == 0:
        return {}

    err = pred.to(dtype=torch.float32) - gt.to(
        dtype=torch.float32, device=pred.device, non_blocking=True
    )
    err_valid = err[valid]
    abs_err = err_valid.abs()
    rmse = torch.sqrt((err_valid * err_valid).mean())

    return {
        f'infer/{prefix}_rmse': float(rmse.item()),
        f'infer/{prefix}_within_0': float((abs_err <= 0.0).float().mean().item()),
        f'infer/{prefix}_within_2': float((abs_err <= 2.0).float().mean().item()),
        f'infer/{prefix}_within_4': float((abs_err <= 4.0).float().mean().item()),
        f'infer/{prefix}_within_6': float((abs_err <= 6.0).float().mean().item()),
        f'infer/{prefix}_n': float(n),
    }


def _run_infer_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion,
    vis_out_dir: str,
    vis_n: int,
    max_batches: int,
) -> InferEpochResult:
    non_blocking = bool(device.type == 'cuda')
    infer_loss_sum = 0.0
    infer_samples = 0
    p_pred_chunks: list[torch.Tensor] = []
    p_gt_chunks: list[torch.Tensor] = []
    p_valid_chunks: list[torch.Tensor] = []
    s_pred_chunks: list[torch.Tensor] = []
    s_gt_chunks: list[torch.Tensor] = []
    s_valid_chunks: list[torch.Tensor] = []

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
            if int(logits.ndim) != 4 or int(logits.shape[1]) != 3:
                msg = f'logits must have shape (B,3,H,W), got {tuple(logits.shape)}'
                raise ValueError(msg)
            loss = criterion(logits, x_tg, batch_dev)

            bsize = int(x_in.shape[0])
            infer_loss_sum += float(loss.detach().item()) * bsize
            infer_samples += bsize

            if 'trace_valid' not in batch_dev:
                msg = "batch must contain 'trace_valid'"
                raise KeyError(msg)
            if 'label_valid' not in batch_dev:
                msg = "batch must contain 'label_valid'"
                raise KeyError(msg)
            if 'meta' not in batch:
                msg = "batch must contain 'meta'"
                raise KeyError(msg)
            meta = batch['meta']
            if not isinstance(meta, Mapping):
                msg = "batch['meta'] must be mapping"
                raise TypeError(msg)
            if 'p_idx_view' not in meta:
                msg = "batch['meta'] must contain 'p_idx_view'"
                raise KeyError(msg)
            if 's_idx_view' not in meta:
                msg = "batch['meta'] must contain 's_idx_view'"
                raise KeyError(msg)

            trace_valid = _to_tensor_bh(
                value=batch_dev['trace_valid'],
                name='trace_valid',
                device=logits.device,
            )
            label_valid = _to_tensor_bh(
                value=batch_dev['label_valid'],
                name='label_valid',
                device=logits.device,
            )
            if trace_valid.dtype is not torch.bool:
                msg = f"batch['trace_valid'] must be bool, got {trace_valid.dtype}"
                raise TypeError(msg)
            if label_valid.dtype is not torch.bool:
                msg = f"batch['label_valid'] must be bool, got {label_valid.dtype}"
                raise TypeError(msg)

            p_gt = _to_tensor_bh(
                value=meta['p_idx_view'],
                name="meta['p_idx_view']",
                device=logits.device,
            ).to(dtype=torch.int64)
            s_gt = _to_tensor_bh(
                value=meta['s_idx_view'],
                name="meta['s_idx_view']",
                device=logits.device,
            ).to(dtype=torch.int64)

            probs = torch.softmax(logits, dim=1)
            pred_p = torch.argmax(probs[:, 0], dim=-1).to(dtype=torch.int64)
            pred_s = torch.argmax(probs[:, 1], dim=-1).to(dtype=torch.int64)
            if tuple(pred_p.shape) != tuple(p_gt.shape):
                msg = (
                    'pred_p and p_gt shape mismatch: '
                    f'pred_p={tuple(pred_p.shape)} p_gt={tuple(p_gt.shape)}'
                )
                raise ValueError(msg)
            if tuple(pred_s.shape) != tuple(s_gt.shape):
                msg = (
                    'pred_s and s_gt shape mismatch: '
                    f'pred_s={tuple(pred_s.shape)} s_gt={tuple(s_gt.shape)}'
                )
                raise ValueError(msg)

            valid_common = trace_valid & label_valid
            valid_p = valid_common & (p_gt > 0)
            valid_s = valid_common & (s_gt > 0)

            p_pred_chunks.append(pred_p)
            p_gt_chunks.append(p_gt)
            p_valid_chunks.append(valid_p)
            s_pred_chunks.append(pred_s)
            s_gt_chunks.append(s_gt)
            s_valid_chunks.append(valid_s)

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

    if not p_pred_chunks or not s_pred_chunks:
        msg = 'no inference batches were processed'
        raise RuntimeError(msg)

    p_pred = torch.cat(p_pred_chunks, dim=0)
    p_gt = torch.cat(p_gt_chunks, dim=0)
    p_valid = torch.cat(p_valid_chunks, dim=0)
    s_pred = torch.cat(s_pred_chunks, dim=0)
    s_gt = torch.cat(s_gt_chunks, dim=0)
    s_valid = torch.cat(s_valid_chunks, dim=0)

    metrics: dict[str, float] = {}
    metrics.update(_phase_metrics(prefix='p', pred=p_pred, gt=p_gt, valid=p_valid))
    metrics.update(_phase_metrics(prefix='s', pred=s_pred, gt=s_gt, valid=s_valid))

    return InferEpochResult(
        loss=infer_loss_sum / float(infer_samples),
        metrics=metrics,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    args, _unknown = parser.parse_known_args(argv)

    cfg, base_dir = load_cfg_with_base_dir(Path(args.config))
    expand_cfg_listfiles(
        cfg,
        keys=[
            'paths.segy_files',
            'paths.phase_pick_files',
            'paths.infer_segy_files',
            'paths.infer_phase_pick_files',
        ],
    )

    train_cfg = require_dict(cfg, 'train')
    device_str = optional_str(train_cfg, 'device', 'auto')
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

    device = resolve_device(device_str)
    seed_all(common.seeds.seed_train)

    train_transform = build_train_transform(cfg)
    infer_transform = build_infer_transform(cfg)
    criterion_train = build_psn_criterion(list(typed.loss_specs_train))
    criterion_eval = build_psn_criterion(list(typed.loss_specs_eval))

    paths_cfg = require_dict(cfg, 'paths')
    train_segy_files = require_list_str(paths_cfg, 'segy_files')
    train_phase_pick_files = require_list_str(paths_cfg, 'phase_pick_files')
    infer_segy_files = require_list_str(paths_cfg, 'infer_segy_files')
    infer_phase_pick_files = require_list_str(paths_cfg, 'infer_phase_pick_files')
    train_sampling_overrides = get_cfg_listfile_meta(
        cfg, key_path='paths.segy_files'
    )
    infer_sampling_overrides = get_cfg_listfile_meta(
        cfg, key_path='paths.infer_segy_files'
    )
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
    if waveform_mode == 'mmap' and int(common.train.train_num_workers) > 0:
        msg = 'dataset.waveform_mode="mmap" requires train.num_workers=0'
        raise ValueError(msg)
    if waveform_mode == 'mmap' and int(common.infer.infer_num_workers) > 0:
        msg = 'dataset.waveform_mode="mmap" requires infer.num_workers=0'
        raise ValueError(msg)

    ds_train_full = _build_dataset_for_subset(
        cfg,
        typed.train.subset_traces,
        segy_files=list(train_segy_files),
        phase_pick_files=list(train_phase_pick_files),
        transform=train_transform,
        secondary_key_fixed=bool(train_secondary_key_fixed),
        segy_endian=str(train_endian),
        sampling_overrides=train_sampling_overrides,
    )
    ds_infer_full = _build_dataset_for_subset(
        cfg,
        typed.infer.subset_traces,
        segy_files=list(infer_segy_files),
        phase_pick_files=list(infer_phase_pick_files),
        transform=infer_transform,
        secondary_key_fixed=True,
        segy_endian=str(infer_endian),
        sampling_overrides=infer_sampling_overrides,
    )

    model = build_model(typed.model).to(device)
    maybe_load_init_weights(
        cfg=cfg,
        base_dir=base_dir,
        model=model,
        model_sig=model_sig,
    )

    train_cfg = require_dict(cfg, 'train')
    weight_decay = optional_float(train_cfg, 'weight_decay', 0.01)
    optimizer = build_optimizer(
        cfg,
        model,
        lr=float(typed.train.lr),
        weight_decay=float(weight_decay),
    )

    def infer_epoch_fn(model, loader, device, vis_epoch_dir, vis_n, max_batches):
        return _run_infer_epoch(
            model=model,
            loader=loader,
            device=device,
            criterion=criterion_eval,
            vis_out_dir=str(vis_epoch_dir),
            vis_n=vis_n,
            max_batches=max_batches,
        )

    spec = TrainSkeletonSpec(
        pipeline='psn',
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
        ckpt_extra={'output_ids': ['P', 'S', 'N'], 'softmax_axis': 'channel'},
        print_freq=common.train.print_freq,
    )

    run_train_skeleton(spec)


if __name__ == '__main__':
    main()
