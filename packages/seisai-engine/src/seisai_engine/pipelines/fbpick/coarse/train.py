from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from seisai_utils.listfiles import expand_cfg_listfiles, get_cfg_listfile_meta

from seisai_engine.optim import build_optimizer
from seisai_engine.pipelines.common import (
    TrainSkeletonSpec,
    load_cfg_with_base_dir,
    maybe_load_init_weights,
    resolve_cfg_paths,
    resolve_device,
    resolve_out_dir,
    run_train_skeleton,
    seed_all,
)
from seisai_engine.viewer.fbpick import save_fbpick_debug_png

from .build_dataset import (
    build_fbgate,
    build_labeled_infer_dataset,
    build_train_dataset,
)
from .build_model import build_model
from .build_plan import build_plan
from .config import (
    COARSE_IN_CHANS,
    COARSE_INPUT_CHANNELS,
    COARSE_TIME_LEN,
    COARSE_TRACE_LEN,
    INFER_PAIR_MESSAGE,
    CoarseTrainConfig,
    load_coarse_train_config,
)
from .loss import build_criterion

__all__ = [
    'CoarseTrainBundle',
    'build_train_bundle',
    'build_train_spec',
    'load_train_bundle',
    'load_train_spec',
    'run_train',
]


@dataclass
class CoarseTrainBundle:
    cfg: dict[str, Any]
    base_dir: Path
    typed: CoarseTrainConfig
    out_dir: Path
    model_sig: dict[str, Any]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion: Any
    ds_train_full: Any
    ds_infer_full: Any
    device: torch.device


def _prepare_cfg(
    cfg: dict,
    *,
    base_dir: Path,
) -> tuple[
    dict,
    list[dict[str, object] | None] | None,
    list[dict[str, object] | None] | None,
]:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    paths = cfg.get('paths')
    if not isinstance(paths, dict):
        msg = 'paths must be dict'
        raise TypeError(msg)

    path_keys: list[str] = ['paths.segy_files', 'paths.fb_files']
    if 'infer_segy_files' in paths:
        path_keys.append('paths.infer_segy_files')
    if 'infer_fb_files' in paths:
        path_keys.append('paths.infer_fb_files')

    expand_cfg_listfiles(cfg, keys=path_keys)
    train_sampling_overrides = get_cfg_listfile_meta(cfg, key_path='paths.segy_files')
    if 'infer_segy_files' in paths:
        infer_sampling_overrides = get_cfg_listfile_meta(
            cfg,
            key_path='paths.infer_segy_files',
        )
    else:
        infer_sampling_overrides = train_sampling_overrides

    resolve_cfg_paths(cfg, base_dir, keys=path_keys)
    return cfg, train_sampling_overrides, infer_sampling_overrides


def _build_fbgate_from_cfg(cfg: dict, *, default_verbose: bool):
    fbgate_cfg = cfg.get('fbgate')
    if fbgate_cfg is None:
        return build_fbgate(apply_on='off', min_pick_ratio=0.0, verbose=default_verbose)
    if not isinstance(fbgate_cfg, dict):
        msg = 'fbgate must be dict'
        raise TypeError(msg)
    apply_on = str(fbgate_cfg.get('apply_on', 'off'))
    min_pick_ratio = float(fbgate_cfg.get('min_pick_ratio', 0.0))
    verbose = bool(fbgate_cfg.get('verbose', default_verbose))
    return build_fbgate(
        apply_on=apply_on,
        min_pick_ratio=min_pick_ratio,
        verbose=verbose,
    )


def _run_infer_epoch(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    criterion,
    vis_out_dir: str | Path,
    vis_n: int,
    max_batches: int,
) -> float:
    non_blocking = bool(device.type == 'cuda')
    infer_loss_sum = 0.0
    infer_samples = 0

    if int(vis_n) > 0:
        Path(vis_out_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= int(max_batches):
                break
            if not isinstance(batch, dict):
                msg = 'infer loader must yield dict batches'
                raise TypeError(msg)
            x = batch['input'].to(device=device, non_blocking=non_blocking)
            y = batch['target'].to(device=device, non_blocking=non_blocking)
            pred = model(x)
            loss = criterion(pred, y, batch)

            bsize = int(x.shape[0])
            infer_loss_sum += float(loss.detach().item()) * bsize
            infer_samples += bsize

            if step < int(vis_n):
                out_path = Path(vis_out_dir) / f'step_{int(step):04d}.png'
                save_fbpick_debug_png(
                    out_path,
                    x_bchw=batch['input'],
                    target_bchw=batch['target'],
                    pred_bchw=pred.detach().cpu(),
                    batch=batch,
                )

    if infer_samples <= 0:
        msg = 'no inference samples were processed'
        raise RuntimeError(msg)
    return infer_loss_sum / float(infer_samples)


def build_train_bundle(
    cfg: dict,
    *,
    base_dir: Path,
    device: torch.device,
) -> CoarseTrainBundle:
    cfg_prepared, train_sampling_overrides, infer_sampling_overrides = _prepare_cfg(
        cfg,
        base_dir=base_dir,
    )
    typed = load_coarse_train_config(cfg_prepared)
    if typed.paths.fb_files is None:
        msg = 'paths.fb_files is required for coarse training'
        raise ValueError(msg)
    if typed.paths.infer_segy_files is None or typed.paths.infer_fb_files is None:
        raise ValueError(INFER_PAIR_MESSAGE)

    plan = build_plan(
        sigma_ms=typed.train.fb_sigma_ms,
        time_ref_sec=typed.norm_refs.time_ref_sec,
        offset_ref_m=typed.norm_refs.offset_ref_m,
    )
    fbgate = _build_fbgate_from_cfg(cfg_prepared, default_verbose=typed.dataset.verbose)

    ds_train_full = build_train_dataset(
        segy_files=list(typed.paths.segy_files),
        fb_files=list(typed.paths.fb_files),
        sampling_overrides=train_sampling_overrides,
        plan=plan,
        fbgate=fbgate,
        subset_traces=typed.train.subset_traces,
        trace_len=typed.transform.trace_len,
        time_len=typed.transform.time_len,
        standardize_eps=typed.transform.standardize_eps,
        anchor_mode=typed.trace_anchor.train_mode,
        gap_ratio=typed.trace_anchor.gap_ratio,
        min_gap_m=typed.trace_anchor.min_gap_m,
        trace_decimate_prob=typed.train.trace_decimate_prob,
        trace_decimate_stride_range=typed.train.trace_decimate_stride_range,
        primary_keys=typed.dataset.primary_keys,
        secondary_key_fixed=typed.dataset.secondary_key_fixed,
        verbose=typed.dataset.verbose,
        progress=typed.dataset.progress,
        max_trials=typed.dataset.max_trials,
        use_header_cache=typed.dataset.use_header_cache,
        waveform_mode=typed.dataset.waveform_mode,
        segy_endian=typed.dataset.train_endian,
    )
    ds_infer_full = build_labeled_infer_dataset(
        segy_files=list(typed.paths.infer_segy_files),
        fb_files=list(typed.paths.infer_fb_files),
        sampling_overrides=infer_sampling_overrides,
        plan=plan,
        fbgate=fbgate,
        subset_traces=typed.infer.subset_traces,
        trace_len=typed.transform.trace_len,
        time_len=typed.transform.time_len,
        standardize_eps=typed.transform.standardize_eps,
        anchor_mode=typed.trace_anchor.infer_mode,
        gap_ratio=typed.trace_anchor.gap_ratio,
        min_gap_m=typed.trace_anchor.min_gap_m,
        primary_keys=typed.dataset.primary_keys,
        secondary_key_fixed=typed.dataset.secondary_key_fixed,
        verbose=typed.dataset.verbose,
        progress=typed.dataset.progress,
        max_trials=typed.dataset.max_trials,
        use_header_cache=typed.dataset.use_header_cache,
        waveform_mode=typed.dataset.waveform_mode,
        segy_endian=typed.dataset.infer_endian,
    )

    model_sig = dict(typed.model_sig)
    model = build_model(model_sig)
    maybe_load_init_weights(
        cfg=cfg_prepared,
        base_dir=base_dir,
        model=model,
        model_sig=model_sig,
    )
    model.to(device)

    optimizer = build_optimizer(
        cfg_prepared,
        model,
        lr=typed.train.lr,
        weight_decay=typed.train.weight_decay,
    )
    criterion = build_criterion()

    return CoarseTrainBundle(
        cfg=cfg_prepared,
        base_dir=base_dir,
        typed=typed,
        out_dir=resolve_out_dir(cfg_prepared, base_dir),
        model_sig=model_sig,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        ds_train_full=ds_train_full,
        ds_infer_full=ds_infer_full,
        device=device,
    )


def build_train_spec(
    cfg: dict,
    *,
    base_dir: Path,
    device: torch.device,
) -> TrainSkeletonSpec:
    bundle = build_train_bundle(cfg, base_dir=base_dir, device=device)
    typed = bundle.typed
    common = typed.common
    criterion_eval = build_criterion()

    def infer_epoch_fn(model, loader, device, vis_epoch_dir, vis_n, max_batches):
        return _run_infer_epoch(
            model=model,
            loader=loader,
            device=device,
            criterion=criterion_eval,
            vis_out_dir=vis_epoch_dir,
            vis_n=vis_n,
            max_batches=max_batches,
        )

    return TrainSkeletonSpec(
        pipeline=str(typed.ckpt.pipeline),
        cfg=bundle.cfg,
        base_dir=bundle.base_dir,
        out_dir=bundle.out_dir,
        vis_subdir=str(common.output.vis_subdir),
        model_sig=bundle.model_sig,
        model=bundle.model,
        optimizer=bundle.optimizer,
        criterion=bundle.criterion,
        ds_train_full=bundle.ds_train_full,
        ds_infer_full=bundle.ds_infer_full,
        device=bundle.device,
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
        ckpt_metric=str(typed.ckpt.metric),
        ckpt_mode=str(typed.ckpt.mode),
        ckpt_extra={
            'output_ids': list(typed.ckpt.output_ids),
            'softmax_axis': str(typed.ckpt.softmax_axis),
            'coarse_input_mode': str(typed.coarse.input_mode),
            'coarse_trace_len': COARSE_TRACE_LEN,
            'coarse_time_len': COARSE_TIME_LEN,
            'coarse_in_chans': COARSE_IN_CHANS,
            'coarse_input_channels': list(COARSE_INPUT_CHANNELS),
        },
        print_freq=common.train.print_freq,
    )


def load_train_bundle(
    config_path: str | Path,
    *,
    device: torch.device | None = None,
) -> CoarseTrainBundle:
    cfg, base_dir = load_cfg_with_base_dir(Path(config_path))
    if device is None:
        device = resolve_device(None)
    return build_train_bundle(cfg, base_dir=base_dir, device=device)


def load_train_spec(
    config_path: str | Path,
    *,
    device: torch.device | None = None,
) -> TrainSkeletonSpec:
    cfg, base_dir = load_cfg_with_base_dir(Path(config_path))
    if device is None:
        device = resolve_device(None)
    return build_train_spec(cfg, base_dir=base_dir, device=device)


def run_train(
    config_path: str | Path,
    *,
    device: torch.device | None = None,
) -> None:
    spec = load_train_spec(config_path, device=device)
    seed_all(spec.seed_train)
    run_train_skeleton(spec)
