from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from seisai_engine.pipelines.common import (
    TrainSkeletonSpec,
    load_cfg_with_base_dir,
    resolve_device,
    run_train_skeleton,
    seed_all,
)

from .config import (
    COARSE_CKPT_OUTPUT_IDS,
    COARSE_CKPT_SOFTMAX_AXIS,
    COARSE_IN_CHANS,
    COARSE_INPUT_CHANNELS,
    COARSE_INPUT_MODE_GLOBAL_ANCHOR_RESIZE,
    COARSE_TIME_LEN,
    COARSE_TRACE_LEN,
    CoarseTrainConfig,
    load_coarse_train_config,
)

__all__ = [
    'CoarseTrainBundle',
    'build_coarse_ckpt_extra',
    'build_train_bundle',
    'build_train_spec',
    'load_train_config',
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


def build_coarse_ckpt_extra(
    typed: CoarseTrainConfig | None = None,
) -> dict[str, Any]:
    if typed is not None:
        if typed.coarse.input_mode != COARSE_INPUT_MODE_GLOBAL_ANCHOR_RESIZE:
            msg = (
                'coarse.input_mode must be '
                f'{COARSE_INPUT_MODE_GLOBAL_ANCHOR_RESIZE!r}'
            )
            raise ValueError(msg)
        if typed.transform.trace_len != COARSE_TRACE_LEN:
            msg = f'transform.trace_len must be {COARSE_TRACE_LEN}'
            raise ValueError(msg)
        if typed.transform.time_len != COARSE_TIME_LEN:
            msg = f'transform.time_len must be {COARSE_TIME_LEN}'
            raise ValueError(msg)
        if int(typed.model_sig.get('in_chans', -1)) != COARSE_IN_CHANS:
            msg = f'model_sig.in_chans must be {COARSE_IN_CHANS}'
            raise ValueError(msg)

    return {
        'output_ids': list(COARSE_CKPT_OUTPUT_IDS),
        'softmax_axis': COARSE_CKPT_SOFTMAX_AXIS,
        'coarse_input_mode': COARSE_INPUT_MODE_GLOBAL_ANCHOR_RESIZE,
        'coarse_trace_len': COARSE_TRACE_LEN,
        'coarse_time_len': COARSE_TIME_LEN,
        'coarse_in_chans': COARSE_IN_CHANS,
        'coarse_input_channels': list(COARSE_INPUT_CHANNELS),
    }


def load_train_config(config_path: str | Path) -> CoarseTrainConfig:
    cfg, _ = load_cfg_with_base_dir(Path(config_path))
    return load_coarse_train_config(cfg)


def build_train_bundle(
    cfg: dict[str, Any],
    *,
    base_dir: Path,
    device: torch.device,
) -> CoarseTrainBundle:
    _ = (base_dir, device)
    load_coarse_train_config(cfg)
    msg = (
        'fbpick coarse training datasets are not implemented in this branch; '
        'issue #18 defines the config and checkpoint metadata contract only'
    )
    raise NotImplementedError(msg)


def _infer_epoch_not_implemented(*args: Any, **kwargs: Any) -> float:
    _ = (args, kwargs)
    msg = (
        'fbpick coarse inference epoch is not implemented in this branch; '
        'issue #18 defines the config and checkpoint metadata contract only'
    )
    raise NotImplementedError(msg)


def build_train_spec(
    cfg: dict[str, Any],
    *,
    base_dir: Path,
    device: torch.device,
) -> TrainSkeletonSpec:
    bundle = build_train_bundle(cfg, base_dir=base_dir, device=device)
    typed = bundle.typed
    common = typed.common
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
        infer_epoch_fn=_infer_epoch_not_implemented,
        ckpt_metric=str(typed.ckpt.metric),
        ckpt_mode=str(typed.ckpt.mode),
        ckpt_extra=build_coarse_ckpt_extra(typed),
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


def run_train(config_path: str | Path) -> None:
    spec = load_train_spec(config_path)
    seed_all(spec.seed_train)
    run_train_skeleton(spec)
