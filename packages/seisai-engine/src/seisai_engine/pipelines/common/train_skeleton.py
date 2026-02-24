from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from seisai_engine.schedulers import load_lr_scheduler_cfg

from .skeleton_helpers import (
    ensure_fixed_infer_num_workers,
    prepare_output_dirs,
)
from .train_skeleton_checkpoint import build_ckpt_payload
from .train_skeleton_loop import (
    build_ema_from_cfg,
    extract_time_len,
    resolve_device_index,
    run_training_loop,
)
from .train_skeleton_tracking import (
    TrackingRunState,
    finalize_tracking_failed,
    finalize_tracking_finished,
    init_tracking_run,
)

__all__ = [
    'InferEpochFn',
    'InferEpochResult',
    'TrainSkeletonSpec',
    'build_ckpt_payload',
    'run_train_skeleton',
]


@dataclass
class InferEpochResult:
    loss: float
    metrics: dict[str, float]


InferEpochFn = Callable[
    [torch.nn.Module, DataLoader, torch.device, Path, int, int],
    float | InferEpochResult,
]


@dataclass
class TrainSkeletonSpec:
    pipeline: str
    cfg: dict
    base_dir: Path
    out_dir: Path
    vis_subdir: str
    model_sig: dict
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion: Callable[..., Any]
    ds_train_full: Any
    ds_infer_full: Any
    device: torch.device
    seed_train: int
    seed_infer: int
    epochs: int
    train_batch_size: int
    train_num_workers: int
    samples_per_epoch: int
    max_norm: float
    use_amp_train: bool
    gradient_accumulation_steps: int
    infer_batch_size: int
    infer_num_workers: int
    infer_max_batches: int
    vis_n: int
    infer_epoch_fn: InferEpochFn
    ckpt_extra: dict[str, Any] | None = None
    print_freq: int = 10


def run_train_skeleton(spec: TrainSkeletonSpec) -> None:
    if not callable(getattr(spec.ds_train_full, 'close', None)):
        msg = 'ds_train_full must provide a callable close()'
        raise TypeError(msg)
    if not callable(getattr(spec.ds_infer_full, 'close', None)):
        msg = 'ds_infer_full must provide a callable close()'
        raise TypeError(msg)
    if not isinstance(spec.base_dir, Path):
        msg = 'base_dir must be Path'
        raise TypeError(msg)

    ckpt_dir, vis_root = prepare_output_dirs(spec.out_dir, spec.vis_subdir)
    ensure_fixed_infer_num_workers(spec.infer_num_workers)

    scheduler_cfg = load_lr_scheduler_cfg(spec.cfg)

    wall_start = time.perf_counter()

    device_index = resolve_device_index(spec.device)
    if device_index is not None:
        torch.cuda.reset_peak_memory_stats(device_index)

    tracking_state: TrackingRunState | None = None
    try:
        ema_cfg_obj, ema_controller = build_ema_from_cfg(spec)
        tracking_state = init_tracking_run(
            spec=spec,
            scheduler_cfg=scheduler_cfg,
            ema_cfg_obj=ema_cfg_obj,
        )
        loop_stats = run_training_loop(
            spec=spec,
            ckpt_dir=ckpt_dir,
            vis_root=vis_root,
            ema_cfg_obj=ema_cfg_obj,
            ema_controller=ema_controller,
            tracking_state=tracking_state,
        )
    except Exception:
        if tracking_state is not None:
            finalize_tracking_failed(tracking_state)
        raise
    else:
        if tracking_state is None:
            msg = 'tracking state must be initialized before summarizing run'
            raise RuntimeError(msg)

        wall_time_sec = time.perf_counter() - wall_start
        if device_index is None:
            num_gpus = 0
            gpu_name = 'cpu'
            peak_mem_gb = 0.0
        else:
            num_gpus = int(torch.cuda.device_count())
            gpu_name = torch.cuda.get_device_name(device_index)
            peak_mem_gb = float(
                torch.cuda.max_memory_allocated(device_index) / (1024**3)
            )
        gpu_hours = float(wall_time_sec) / 3600.0 * float(num_gpus)
        if wall_time_sec > 0:
            train_samples_per_sec = float(loop_stats.total_train_samples) / float(
                wall_time_sec
            )
        else:
            train_samples_per_sec = 0.0

        time_len = extract_time_len(spec.cfg)
        run_summary = {
            'meta': {
                'pipeline': spec.pipeline,
                'seed_train': int(spec.seed_train),
                'exp_name': tracking_state.tracking_cfg.exp_name,
                'run_name': tracking_state.run_name,
                'gpu_name': gpu_name,
                'num_gpus': int(num_gpus),
            },
            'budget': {
                'time_len': time_len,
                'epochs': int(spec.epochs),
                'samples_per_epoch': int(spec.samples_per_epoch),
                'train_batch_size': int(spec.train_batch_size),
                'infer_batch_size': int(spec.infer_batch_size),
                'infer_max_batches': int(spec.infer_max_batches),
                'use_amp_train': bool(spec.use_amp_train),
            },
            'cost': {
                'wall_time_sec': float(wall_time_sec),
                'gpu_hours': float(gpu_hours),
                'total_train_steps': int(loop_stats.total_train_steps),
                'total_train_samples': int(loop_stats.total_train_samples),
                'train_samples_per_sec': float(train_samples_per_sec),
                'peak_mem_gb': float(peak_mem_gb),
            },
            'results': {
                'best_infer_loss': (
                    None
                    if loop_stats.best_infer_loss is None
                    else float(loop_stats.best_infer_loss)
                ),
                'best_epoch': loop_stats.best_epoch,
            },
        }

        summary_path = Path(spec.out_dir) / 'run_summary.json'
        summary_path.write_text(
            json.dumps(
                run_summary,
                sort_keys=True,
                separators=(',', ':'),
                ensure_ascii=True,
            )
            + '\n',
            encoding='utf-8',
        )
        finalize_tracking_finished(
            tracking_state,
            artifacts={'run_summary.json': summary_path},
        )
    finally:
        spec.ds_train_full.close()
        spec.ds_infer_full.close()
