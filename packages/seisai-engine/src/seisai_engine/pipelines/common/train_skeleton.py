from __future__ import annotations

import getpass
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from seisai_utils.config import (
    optional_bool,
    optional_float,
    optional_int,
    optional_str,
    require_dict,
)
from torch.utils.data import DataLoader, Subset

from seisai_engine.ema_controller import EmaConfig, EmaController
from seisai_engine.schedulers import build_lr_scheduler, load_lr_scheduler_cfg
from seisai_engine.tracking.config import load_tracking_config
from seisai_engine.tracking.data_id import build_data_manifest, calc_data_id
from seisai_engine.tracking.factory import build_tracker
from seisai_engine.tracking.sanitize import sanitize_key
from seisai_engine.train_loop import train_one_epoch

from .skeleton_helpers import (
    ensure_fixed_infer_num_workers,
    epoch_vis_dir,
    make_train_worker_init_fn,
    maybe_save_best_min,
    prepare_output_dirs,
    set_dataset_rng,
)


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
    lr_sched_spec = None

    def _extract_time_len(cfg: dict) -> int | None:
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

    def _resolve_device_index(device: torch.device) -> int | None:
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

    def _normalize_infer_epoch_output(
        output: float | InferEpochResult,
    ) -> tuple[float, dict[str, float]]:
        if isinstance(output, InferEpochResult):
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

    wall_start = time.perf_counter()
    total_train_steps = 0
    total_train_samples = 0
    best_epoch: int | None = None

    device_index = _resolve_device_index(spec.device)
    if device_index is not None:
        torch.cuda.reset_peak_memory_stats(device_index)

    best_infer_loss: float | None = None
    global_step = 0
    tracking_enabled = False
    run_started = False
    tracker = None
    ema_cfg_obj: EmaConfig | None = None
    ema_controller: EmaController | None = None
    try:
        tracking_cfg = load_tracking_config(spec.cfg, spec.base_dir)
        tracking_enabled = tracking_cfg.enabled
        run_started = False
        tracker = build_tracker(tracking_cfg)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        run_name = f'{timestamp}__{tracking_cfg.exp_name}__s{int(spec.seed_train)}'

        ema_cfg_obj = None
        ema_controller = None
        if 'ema' in spec.cfg:
            ema_cfg = require_dict(spec.cfg, 'ema')
            ema_enabled = optional_bool(ema_cfg, 'enabled', default=True)
            if ema_enabled:
                decay = float(optional_float(ema_cfg, 'decay', 0.999))
                start_step = int(optional_int(ema_cfg, 'start_step', 0))
                update_every = int(optional_int(ema_cfg, 'update_every', 1))
                use_for_infer = bool(
                    optional_bool(ema_cfg, 'use_for_infer', default=True)
                )

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

        if tracking_enabled:
            tracking_dir = Path(spec.out_dir) / 'tracking'
            tracking_dir.mkdir(parents=True, exist_ok=True)

            config_resolved_path = tracking_dir / 'config.resolved.yaml'
            config_resolved_path.write_text(
                yaml.safe_dump(spec.cfg, sort_keys=False),
                encoding='utf-8',
            )

            manifest = build_data_manifest(spec.cfg)
            data_manifest_path = tracking_dir / 'data_manifest.json'
            data_manifest_path.write_text(
                json.dumps(
                    manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=True
                ),
                encoding='utf-8',
            )

            data_id = calc_data_id(manifest)
            data_nfiles = len(manifest['files'])
            data_id_human = f'nfiles={data_nfiles}'

            git_sha = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=str(spec.base_dir),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            git_branch = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=str(spec.base_dir),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            git_status = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=str(spec.base_dir),
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            dirty = bool(git_status.strip())

            git_txt_path = tracking_dir / 'git.txt'
            git_txt_path.write_text(
                '\n'.join(
                    [
                        f'sha={git_sha}',
                        f'branch={git_branch}',
                        f'dirty={str(dirty).lower()}',
                        'status:',
                        git_status.rstrip(),
                    ]
                )
                + '\n',
                encoding='utf-8',
            )

            env_txt_path = tracking_dir / 'env.txt'
            env_lines = [
                f'python={sys.version.split()[0]}',
                f'torch={torch.__version__}',
                f'cuda_available={torch.cuda.is_available()}',
                f'cuda_version={torch.version.cuda}',
            ]
            env_txt_path.write_text('\n'.join(env_lines) + '\n', encoding='utf-8')

            tags = {
                'pipeline': spec.pipeline,
                'user': getpass.getuser(),
                'git_sha': git_sha,
                'dirty': str(dirty).lower(),
                'data_id': data_id,
                'data_id_human': data_id_human,
                'data_nfiles': str(data_nfiles),
            }

            params = {
                'train/batch_size': int(spec.train_batch_size),
                'train/epochs': int(spec.epochs),
                'train/samples_per_epoch': int(spec.samples_per_epoch),
                'seeds/seed_train': int(spec.seed_train),
                'seeds/seed_infer': int(spec.seed_infer),
            }

            if scheduler_cfg is not None:
                sched_type = scheduler_cfg.get('type')
                if isinstance(sched_type, str):
                    params['scheduler/type'] = sched_type
                sched_interval = scheduler_cfg.get('interval')
                if isinstance(sched_interval, str):
                    params['scheduler/interval'] = sched_interval

            if ema_cfg_obj is not None:
                params['ema/enabled'] = True
                params['ema/decay'] = float(ema_cfg_obj.decay)
                params['ema/start_step'] = int(ema_cfg_obj.start_step)
                params['ema/update_every'] = int(ema_cfg_obj.update_every)
                params['ema/use_for_infer'] = bool(ema_cfg_obj.use_for_infer)
                if ema_cfg_obj.device is None:
                    params['ema/device'] = 'same'
                else:
                    params['ema/device'] = str(ema_cfg_obj.device)

            if not spec.optimizer.param_groups:
                msg = 'optimizer.param_groups must be non-empty'
                raise ValueError(msg)
            group0 = spec.optimizer.param_groups[0]
            if 'lr' in group0:
                params['optimizer/lr'] = float(group0['lr'])
            if 'weight_decay' in group0:
                params['optimizer/weight_decay'] = float(group0['weight_decay'])

            has_groupwise_lr = False
            has_groupwise_weight_decay = False
            for group in spec.optimizer.param_groups[1:]:
                if 'lr' in group0 and 'lr' in group:
                    if float(group['lr']) != float(group0['lr']):
                        has_groupwise_lr = True
                if 'weight_decay' in group0 and 'weight_decay' in group:
                    if float(group['weight_decay']) != float(group0['weight_decay']):
                        has_groupwise_weight_decay = True
            params['optimizer/has_groupwise_lr'] = has_groupwise_lr
            params['optimizer/has_groupwise_weight_decay'] = has_groupwise_weight_decay

            optimizer_groups: list[dict[str, object]] = []
            for idx, group in enumerate(spec.optimizer.param_groups):
                entry: dict[str, object] = {'idx': int(idx)}
                if 'lr' in group:
                    entry['lr'] = float(group['lr'])
                if 'weight_decay' in group:
                    entry['weight_decay'] = float(group['weight_decay'])
                optimizer_groups.append(entry)
            optimizer_groups_payload = {
                'groups': optimizer_groups,
                'has_groupwise_lr': has_groupwise_lr,
                'has_groupwise_weight_decay': has_groupwise_weight_decay,
            }
            optimizer_groups_path = tracking_dir / 'optimizer_groups.json'
            optimizer_groups_path.write_text(
                json.dumps(
                    optimizer_groups_payload,
                    sort_keys=True,
                    separators=(',', ':'),
                    ensure_ascii=True,
                ),
                encoding='utf-8',
            )

            if isinstance(spec.model_sig, dict):
                if 'backbone' in spec.model_sig:
                    params['model/backbone'] = str(spec.model_sig['backbone'])
                if 'in_chans' in spec.model_sig:
                    params['model/in_chans'] = int(spec.model_sig['in_chans'])
                if 'out_chans' in spec.model_sig:
                    params['model/out_chans'] = int(spec.model_sig['out_chans'])

            def _looks_like_structured(value: str) -> bool:
                stripped = value.strip()
                if stripped.startswith('{') or stripped.startswith('['):
                    return True
                if re.search(r'\b\w+\s*:\s', value):
                    return True
                return False

            def _prepare_kv(
                mapping: dict[str, object],
                *,
                max_bytes: int,
                label: str,
            ) -> tuple[dict[str, object], dict[str, str]]:
                sanitized: dict[str, object] = {}
                stored: dict[str, str] = {}
                for key, value in mapping.items():
                    safe_key = sanitize_key(key)
                    if safe_key in sanitized:
                        msg = f'duplicate {label} key after sanitization: {safe_key}'
                        raise ValueError(msg)
                    value_str = value if isinstance(value, str) else str(value)
                    is_overlong = len(value_str.encode('utf-8')) > max_bytes
                    if '\n' in value_str or _looks_like_structured(value_str):
                        is_overlong = True
                    if is_overlong:
                        stored[safe_key] = value_str
                        ref = (
                            f'<stored:tracking/overlong_values.json#{label}.{safe_key}>'
                        )
                        sanitized[safe_key] = ref
                    else:
                        sanitized[safe_key] = value
                return sanitized, stored

            tags, overlong_tags = _prepare_kv(tags, max_bytes=256, label='tags')
            params, overlong_params = _prepare_kv(
                params, max_bytes=1024, label='params'
            )

            overlong_payload: dict[str, dict[str, str]] = {}
            if overlong_tags:
                overlong_payload['tags'] = overlong_tags
            if overlong_params:
                overlong_payload['params'] = overlong_params

            overlong_path: Path | None = None
            if overlong_payload:
                overlong_path = tracking_dir / 'overlong_values.json'
                overlong_path.write_text(
                    json.dumps(
                        overlong_payload,
                        sort_keys=True,
                        separators=(',', ':'),
                        ensure_ascii=True,
                    ),
                    encoding='utf-8',
                )

            experiment = f'{tracking_cfg.experiment_prefix}/{spec.pipeline}'

            artifacts = {
                'config.resolved.yaml': config_resolved_path,
                'git.txt': git_txt_path,
                'env.txt': env_txt_path,
                'data_manifest.json': data_manifest_path,
                'optimizer_groups.json': optimizer_groups_path,
            }
            if overlong_path is not None:
                artifacts['overlong_values.json'] = overlong_path

            tracker.start_run(
                tracking_uri=tracking_cfg.tracking_uri,
                experiment=experiment,
                run_name=run_name,
                tags=tags,
                params=params,
                artifacts=artifacts,
            )
            run_started = True

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
            infer_loss, infer_extra_metrics = _normalize_infer_epoch_output(
                infer_output
            )
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
            did_update = best_infer_loss is None or infer_loss < best_infer_loss
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
            ckpt_payload = {
                'version': 1,
                'pipeline': spec.pipeline,
                'epoch': int(epoch),
                'global_step': int(global_step),
                'model_sig': spec.model_sig,
                'model_state_dict': spec.model.state_dict(),
                'optimizer_state_dict': spec.optimizer.state_dict(),
                'lr_scheduler_sig': scheduler_sig,
                'lr_scheduler_state_dict': scheduler_state_dict,
                'cfg': spec.cfg,
            }
            if ema_cfg_obj is not None and ema_controller is not None:
                dev = None
                if ema_cfg_obj.device is not None:
                    dev = str(ema_cfg_obj.device)
                ckpt_payload['ema_cfg'] = {
                    'decay': float(ema_cfg_obj.decay),
                    'start_step': int(ema_cfg_obj.start_step),
                    'update_every': int(ema_cfg_obj.update_every),
                    'use_for_infer': bool(ema_cfg_obj.use_for_infer),
                    'device': dev,
                }
                ckpt_payload['ema_state_dict'] = ema_controller.state_dict_cpu()
                ckpt_payload['ema_step'] = int(ema_controller.step)
                ckpt_payload['infer_used_ema'] = bool(ema_cfg_obj.use_for_infer)

            best_infer_loss = maybe_save_best_min(
                best_infer_loss,
                infer_loss,
                ckpt_path,
                ckpt_payload,
            )

            if tracking_enabled and run_started:
                tracked_metrics = {
                    'train/loss': float(stats['loss']),
                    'infer/loss': float(infer_loss),
                }
                if infer_extra_metrics:
                    tracked_metrics.update(infer_extra_metrics)
                tracker.log_metrics(
                    tracked_metrics,
                    step=int(epoch),
                )
                if did_update:
                    tracker.log_best(
                        ckpt_path=ckpt_path,
                        vis_epoch_dir=vis_epoch_dir,
                        vis_max_files=tracking_cfg.vis_max_files,
                    )
    except Exception:
        if tracking_enabled and run_started:
            tracker.end_run(status='FAILED')
        raise
    else:
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
            train_samples_per_sec = float(total_train_samples) / float(wall_time_sec)
        else:
            train_samples_per_sec = 0.0

        time_len = _extract_time_len(spec.cfg)
        run_summary = {
            'meta': {
                'pipeline': spec.pipeline,
                'seed_train': int(spec.seed_train),
                'exp_name': tracking_cfg.exp_name,
                'run_name': run_name,
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
                'total_train_steps': int(total_train_steps),
                'total_train_samples': int(total_train_samples),
                'train_samples_per_sec': float(train_samples_per_sec),
                'peak_mem_gb': float(peak_mem_gb),
            },
            'results': {
                'best_infer_loss': (
                    None if best_infer_loss is None else float(best_infer_loss)
                ),
                'best_epoch': best_epoch,
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
        if tracking_enabled and run_started:
            tracker.log_artifacts({'run_summary.json': summary_path})
        if tracking_enabled and run_started:
            tracker.end_run(status='FINISHED')
    finally:
        spec.ds_train_full.close()
        spec.ds_infer_full.close()
