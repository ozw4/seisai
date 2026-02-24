from __future__ import annotations

import getpass
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml

from seisai_engine.ema_controller import EmaConfig
from seisai_engine.tracking.config import TrackingConfig, load_tracking_config
from seisai_engine.tracking.data_id import build_data_manifest, calc_data_id
from seisai_engine.tracking.factory import build_tracker
from seisai_engine.tracking.sanitize import sanitize_key
from seisai_engine.tracking.tracker import BaseTracker

if TYPE_CHECKING:
    from .train_skeleton import TrainSkeletonSpec

__all__ = [
    'TrackingRunState',
    'finalize_tracking_failed',
    'finalize_tracking_finished',
    'init_tracking_run',
    'log_epoch_tracking',
]


@dataclass
class TrackingRunState:
    tracking_cfg: TrackingConfig
    tracker: BaseTracker
    tracking_enabled: bool
    run_started: bool
    run_name: str


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
            ref = f'<stored:tracking/overlong_values.json#{label}.{safe_key}>'
            sanitized[safe_key] = ref
        else:
            sanitized[safe_key] = value
    return sanitized, stored


def init_tracking_run(
    *,
    spec: TrainSkeletonSpec,
    scheduler_cfg: dict[str, Any] | None,
    ema_cfg_obj: EmaConfig | None,
) -> TrackingRunState:
    tracking_cfg = load_tracking_config(spec.cfg, spec.base_dir)
    tracking_enabled = tracking_cfg.enabled
    tracker = build_tracker(tracking_cfg)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    run_name = f'{timestamp}__{tracking_cfg.exp_name}__s{int(spec.seed_train)}'

    state = TrackingRunState(
        tracking_cfg=tracking_cfg,
        tracker=tracker,
        tracking_enabled=tracking_enabled,
        run_started=False,
        run_name=run_name,
    )
    if not tracking_enabled:
        return state

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
        json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=True),
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

    tags, overlong_tags = _prepare_kv(tags, max_bytes=256, label='tags')
    params, overlong_params = _prepare_kv(params, max_bytes=1024, label='params')

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
    state.run_started = True
    return state


def log_epoch_tracking(
    state: TrackingRunState,
    *,
    epoch: int,
    train_loss: float,
    infer_loss: float,
    infer_extra_metrics: dict[str, float],
    did_update: bool,
    ckpt_path: Path,
    vis_epoch_dir: Path,
) -> None:
    if not state.tracking_enabled or not state.run_started:
        return

    tracked_metrics = {
        'train/loss': float(train_loss),
        'infer/loss': float(infer_loss),
    }
    if infer_extra_metrics:
        tracked_metrics.update(infer_extra_metrics)
    state.tracker.log_metrics(
        tracked_metrics,
        step=int(epoch),
    )
    if did_update:
        state.tracker.log_best(
            ckpt_path=ckpt_path,
            vis_epoch_dir=vis_epoch_dir,
            vis_max_files=state.tracking_cfg.vis_max_files,
        )


def finalize_tracking_failed(state: TrackingRunState) -> None:
    if state.tracking_enabled and state.run_started:
        state.tracker.end_run(status='FAILED')


def finalize_tracking_finished(
    state: TrackingRunState, *, artifacts: dict[str, Path] | None = None
) -> None:
    if not (state.tracking_enabled and state.run_started):
        return
    if artifacts:
        state.tracker.log_artifacts(artifacts)
    state.tracker.end_run(status='FINISHED')
