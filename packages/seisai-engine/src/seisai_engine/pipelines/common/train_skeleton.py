from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from seisai_engine.train_loop import train_one_epoch
from seisai_engine.tracking.config import load_tracking_config
from seisai_engine.tracking.data_id import build_data_manifest, calc_data_id
from seisai_engine.tracking.factory import build_tracker
from seisai_engine.tracking.sanitize import sanitize_key

from .skeleton_helpers import (
    ensure_fixed_infer_num_workers,
    epoch_vis_dir,
    make_train_worker_init_fn,
    maybe_save_best_min,
    prepare_output_dirs,
    set_dataset_rng,
)

InferEpochFn = Callable[
    [torch.nn.Module, DataLoader, torch.device, Path, int, int], float
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

    best_infer_loss: float | None = None
    global_step = 0
    tracking_enabled = False
    run_started = False
    tracker = None
    try:
        tracking_cfg = load_tracking_config(spec.cfg, spec.base_dir)
        tracking_enabled = tracking_cfg.enabled
        run_started = False
        tracker = build_tracker(tracking_cfg)

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
            params['optimizer/has_groupwise_weight_decay'] = (
                has_groupwise_weight_decay
            )

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
                            f'<stored:tracking/overlong_values.json#'
                            f'{label}.{safe_key}>'
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
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
            run_name = f'{timestamp}__{tracking_cfg.exp_name}__s{int(spec.seed_train)}'

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

            spec.model.train()
            stats = train_one_epoch(
                spec.model,
                train_loader,
                spec.optimizer,
                spec.criterion,
                device=spec.device,
                lr_scheduler=None,
                gradient_accumulation_steps=1,
                max_norm=spec.max_norm,
                use_amp=spec.use_amp_train,
                scaler=None,
                ema=None,
                step_offset=0,
                print_freq=spec.print_freq,
                on_step=None,
            )
            print(
                f'epoch={epoch} loss={stats["loss"]:.6f} steps={int(stats["steps"])} '
                f'samples={int(stats["samples"])}'
            )
            global_step += int(stats['steps'])

            set_dataset_rng(spec.ds_infer_full, spec.seed_infer)

            infer_ds = Subset(
                spec.ds_infer_full,
                range(spec.infer_batch_size * spec.infer_max_batches),
            )
            infer_loader = DataLoader(
                infer_ds,
                batch_size=spec.infer_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(spec.device.type == 'cuda'),
            )

            vis_epoch_dir = epoch_vis_dir(vis_root, epoch)

            spec.model.eval()
            infer_loss = spec.infer_epoch_fn(
                spec.model,
                infer_loader,
                spec.device,
                vis_epoch_dir,
                spec.vis_n,
                spec.infer_max_batches,
            )
            print(f'epoch={epoch} infer_loss={infer_loss:.6f}')

            ckpt_path = ckpt_dir / 'best.pt'
            did_update = best_infer_loss is None or infer_loss < best_infer_loss
            best_infer_loss = maybe_save_best_min(
                best_infer_loss,
                infer_loss,
                ckpt_path,
                {
                    'version': 1,
                    'pipeline': spec.pipeline,
                    'epoch': int(epoch),
                    'global_step': int(global_step),
                    'model_sig': spec.model_sig,
                    'model_state_dict': spec.model.state_dict(),
                    'optimizer_state_dict': spec.optimizer.state_dict(),
                    'cfg': spec.cfg,
                },
            )

            if tracking_enabled and run_started:
                tracker.log_metrics(
                    {
                        'train/loss': float(stats['loss']),
                        'infer/loss': float(infer_loss),
                    },
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
        if tracking_enabled and run_started:
            tracker.end_run(status='FINISHED')
    finally:
        spec.ds_train_full.close()
        spec.ds_infer_full.close()
