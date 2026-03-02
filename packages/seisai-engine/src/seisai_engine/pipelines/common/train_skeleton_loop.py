from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from seisai_utils.config import (
    optional_bool,
    optional_float,
    optional_int,
    optional_str,
    require_dict,
)
from torch.utils.data import DataLoader, Subset

from seisai_engine.ema_controller import EmaConfig, EmaController
from seisai_engine.schedulers import build_lr_scheduler
from seisai_engine.train_loop import train_one_epoch

from .skeleton_helpers import (
    epoch_vis_dir,
    make_train_worker_init_fn,
    maybe_save_best,
    set_dataset_rng,
)
from .train_skeleton_checkpoint import build_ckpt_payload
from .train_skeleton_tracking import TrackingRunState, log_epoch_tracking

if TYPE_CHECKING:
    from .train_skeleton import InferEpochResult, TrainSkeletonSpec

__all__ = [
    'LoopStats',
    'build_ema_from_cfg',
    'extract_time_len',
    'normalize_infer_epoch_output',
    'resolve_device_index',
    'run_training_loop',
]


@dataclass
class LoopStats:
    best_ckpt_value: float | None
    best_epoch: int | None
    global_step: int
    total_train_steps: int
    total_train_samples: int


def extract_time_len(cfg: dict) -> int | None:
    if not isinstance(cfg, dict):
        return None
    section = cfg.get('transform')
    if not isinstance(section, dict):
        return None
    val = section.get('time_len')
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return int(val)
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return None


def resolve_device_index(device: torch.device) -> int | None:
    if not isinstance(device, torch.device):
        return None
    if device.type != 'cuda':
        return None
    if not torch.cuda.is_available():
        return None
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    return int(idx)


def normalize_infer_epoch_output(
    output: float | InferEpochResult,
) -> tuple[float, dict[str, float]]:
    # Avoid import cycle by importing runtime type here.
    from .train_skeleton import InferEpochResult as InferEpochResultType

    if isinstance(output, InferEpochResultType):
        raw_loss = output.loss
        raw_metrics = output.metrics
    else:
        raw_loss = output
        raw_metrics = {}

    if isinstance(raw_loss, bool) or not isinstance(raw_loss, (int, float)):
        msg = (
            'infer_epoch_fn must return float '
            'or InferEpochResult(loss=<numeric>, metrics=<dict>)'
        )
        raise TypeError(msg)
    infer_loss = float(raw_loss)

    if not isinstance(raw_metrics, dict):
        msg = 'InferEpochResult.metrics must be dict[str, float]'
        raise TypeError(msg)

    metrics: dict[str, float] = {}
    for key, value in raw_metrics.items():
        if not isinstance(key, str) or not key:
            msg = 'InferEpochResult.metrics keys must be non-empty str'
            raise TypeError(msg)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = f'InferEpochResult.metrics[{key!r}] must be numeric'
            raise TypeError(msg)
        metrics[key] = float(value)
    return infer_loss, metrics


def build_ema_from_cfg(
    spec: TrainSkeletonSpec,
) -> tuple[EmaConfig | None, EmaController | None]:
    ema_cfg_obj: EmaConfig | None = None
    ema_controller: EmaController | None = None
    if 'ema' not in spec.cfg:
        return ema_cfg_obj, ema_controller

    ema_cfg = require_dict(spec.cfg, 'ema')
    ema_enabled = optional_bool(ema_cfg, 'enabled', default=True)
    if not ema_enabled:
        return ema_cfg_obj, ema_controller

    decay = float(optional_float(ema_cfg, 'decay', 0.999))
    start_step = int(optional_int(ema_cfg, 'start_step', 0))
    update_every = int(optional_int(ema_cfg, 'update_every', 1))
    use_for_infer = bool(optional_bool(ema_cfg, 'use_for_infer', default=True))

    device_str = str(optional_str(ema_cfg, 'device', ''))
    device: torch.device | None = None
    if device_str:
        if device_str not in {'cpu', 'cuda'}:
            msg = 'ema.device must be "cpu" or "cuda" when provided'
            raise ValueError(msg)
        if device_str == 'cpu':
            device = torch.device('cpu')
        else:
            if spec.device.type != 'cuda':
                msg = 'ema.device="cuda" requires CUDA device'
                raise ValueError(msg)
            device = spec.device

    ema_cfg_obj = EmaConfig(
        decay=float(decay),
        start_step=int(start_step),
        update_every=int(update_every),
        use_for_infer=bool(use_for_infer),
        device=device,
    )
    ema_controller = EmaController(spec.model, ema_cfg_obj, initial_step=0)
    return ema_cfg_obj, ema_controller


def run_training_loop(
    *,
    spec: TrainSkeletonSpec,
    ckpt_dir: Path,
    vis_root: Path,
    ema_cfg_obj: EmaConfig | None,
    ema_controller: EmaController | None,
    tracking_state: TrackingRunState,
) -> LoopStats:
    lr_sched_spec = None
    best_ckpt_value: float | None = None
    best_epoch: int | None = None
    global_step = 0
    total_train_steps = 0
    total_train_samples = 0

    for epoch in range(spec.epochs):
        seed_epoch = spec.seed_train + epoch

        if spec.train_num_workers == 0:
            set_dataset_rng(spec.ds_train_full, seed_epoch)
            train_worker_init_fn = None
        else:
            train_worker_init_fn = make_train_worker_init_fn(seed_epoch)

        train_ds = Subset(spec.ds_train_full, range(spec.samples_per_epoch))
        train_loader = DataLoader(
            train_ds,
            batch_size=spec.train_batch_size,
            shuffle=False,
            num_workers=spec.train_num_workers,
            pin_memory=(spec.device.type == 'cuda'),
            worker_init_fn=train_worker_init_fn,
        )

        if lr_sched_spec is None:
            acc = int(spec.gradient_accumulation_steps)
            if acc <= 0:
                msg = 'gradient_accumulation_steps must be positive'
                raise ValueError(msg)
            optimizer_steps_per_epoch = (len(train_loader) + acc - 1) // acc
            if optimizer_steps_per_epoch <= 0:
                msg = 'steps_per_epoch must be positive'
                raise ValueError(msg)
            lr_sched_spec = build_lr_scheduler(
                spec.optimizer,
                spec.cfg,
                steps_per_epoch=int(optimizer_steps_per_epoch),
                epochs=spec.epochs,
            )

        lr_scheduler_step = (
            lr_sched_spec.scheduler
            if (lr_sched_spec is not None and lr_sched_spec.interval == 'step')
            else None
        )

        if ema_controller is not None:
            ema_controller.set_step(int(global_step))

        spec.model.train()
        stats = train_one_epoch(
            spec.model,
            train_loader,
            spec.optimizer,
            spec.criterion,
            device=spec.device,
            lr_scheduler=lr_scheduler_step,
            gradient_accumulation_steps=int(spec.gradient_accumulation_steps),
            max_norm=spec.max_norm,
            use_amp=spec.use_amp_train,
            scaler=None,
            ema=ema_controller,
            step_offset=0,
            print_freq=spec.print_freq,
            on_step=None,
        )

        if lr_sched_spec is not None and lr_sched_spec.interval == 'epoch':
            if lr_sched_spec.monitor is None:
                lr_sched_spec.scheduler.step()
        print(
            f'epoch={epoch} loss={stats["loss"]:.4E} steps={int(stats["steps"])} '
            f'samples={int(stats["samples"])}'
        )
        global_step += int(stats['steps'])
        total_train_steps += int(stats['steps'])
        total_train_samples += int(stats['samples'])

        set_dataset_rng(spec.ds_infer_full, spec.seed_infer)

        infer_ds = Subset(
            spec.ds_infer_full,
            range(spec.infer_batch_size * spec.infer_max_batches),
        )
        infer_use_ema = bool(
            ema_cfg_obj is not None
            and ema_controller is not None
            and ema_cfg_obj.use_for_infer
        )
        infer_model = ema_controller.module if infer_use_ema else spec.model
        infer_device = (
            ema_cfg_obj.device
            if (infer_use_ema and ema_cfg_obj.device is not None)
            else spec.device
        )
        infer_loader = DataLoader(
            infer_ds,
            batch_size=spec.infer_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(infer_device.type == 'cuda'),
        )

        vis_epoch_dir = epoch_vis_dir(vis_root, epoch)

        infer_model.eval()
        infer_output = spec.infer_epoch_fn(
            infer_model,
            infer_loader,
            infer_device,
            vis_epoch_dir,
            spec.vis_n,
            spec.infer_max_batches,
        )
        infer_loss, infer_extra_metrics = normalize_infer_epoch_output(infer_output)
        print(f'epoch={epoch} infer_loss={infer_loss:.6E}')
        if infer_extra_metrics:
            metric_parts: list[str] = []
            for key in sorted(infer_extra_metrics):
                metric_parts.append(f'{key}={infer_extra_metrics[key]:.6E}')
            print(f'epoch={epoch} infer_metrics {" ".join(metric_parts)}')

        if lr_sched_spec is not None and lr_sched_spec.interval == 'epoch':
            if lr_sched_spec.monitor is not None:
                monitor = lr_sched_spec.monitor
                if monitor in ('infer_loss', 'infer/loss'):
                    lr_sched_spec.scheduler.step(float(infer_loss))
                elif monitor in ('train_loss', 'train/loss'):
                    lr_sched_spec.scheduler.step(float(stats['loss']))
                else:
                    msg = (
                        'unknown scheduler.monitor: '
                        f'{monitor} (expected infer_loss/train_loss)'
                    )
                    raise ValueError(msg)

        ckpt_path = ckpt_dir / 'best.pt'
        monitor_key = str(spec.ckpt_metric)
        if monitor_key in ('infer_loss', 'infer/loss'):
            monitor_val = float(infer_loss)
        elif monitor_key in infer_extra_metrics:
            monitor_val = float(infer_extra_metrics[monitor_key])
        else:
            msg = (
                'ckpt.metric not found in infer metrics: '
                f'{monitor_key} (available: infer_loss and {sorted(infer_extra_metrics)})'
            )
            raise ValueError(msg)

        mode = str(spec.ckpt_mode)
        if mode == 'min':
            did_update = best_ckpt_value is None or monitor_val < float(best_ckpt_value)
        elif mode == 'max':
            did_update = best_ckpt_value is None or monitor_val > float(best_ckpt_value)
        else:
            msg = f'unknown ckpt.mode: {mode} (expected min/max)'
            raise ValueError(msg)
        if did_update:
            best_epoch = int(epoch)
        scheduler_state_dict = None
        scheduler_sig = None
        if lr_sched_spec is not None:
            if callable(getattr(lr_sched_spec.scheduler, 'state_dict', None)):
                scheduler_state_dict = lr_sched_spec.scheduler.state_dict()
            scheduler_sig = {
                'name': lr_sched_spec.name,
                'interval': lr_sched_spec.interval,
                'monitor': lr_sched_spec.monitor,
            }
        ckpt_payload = build_ckpt_payload(
            spec=spec,
            epoch=epoch,
            global_step=global_step,
            scheduler_sig=scheduler_sig,
            scheduler_state_dict=scheduler_state_dict,
            ema_cfg_obj=ema_cfg_obj,
            ema_controller=ema_controller,
        )

        best_ckpt_value = maybe_save_best(
            best_ckpt_value,
            monitor_val,
            ckpt_path,
            ckpt_payload,
            mode=spec.ckpt_mode,
        )

        log_epoch_tracking(
            tracking_state,
            epoch=epoch,
            train_loss=float(stats['loss']),
            infer_loss=float(infer_loss),
            infer_extra_metrics=infer_extra_metrics,
            did_update=did_update,
            ckpt_path=ckpt_path,
            vis_epoch_dir=vis_epoch_dir,
        )

    return LoopStats(
        best_ckpt_value=best_ckpt_value,
        best_epoch=best_epoch,
        global_step=int(global_step),
        total_train_steps=int(total_train_steps),
        total_train_samples=int(total_train_samples),
    )
