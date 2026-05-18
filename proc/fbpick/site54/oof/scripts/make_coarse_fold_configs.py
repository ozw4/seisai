#!/usr/bin/env python3
"""Render site54 OOF coarse configs."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

FOLDS = [f'fold{i:02d}' for i in range(6)]


def rel_or_abs(p: Path) -> str:
    """Return a path string for YAML serialization."""
    return str(p)


def parse_bool(value: object) -> bool:
    """Parse a command-line boolean value."""
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        msg = f'expected boolean value, got {value!r}'
        raise argparse.ArgumentTypeError(msg)
    lowered = value.lower()
    if lowered in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if lowered in {'0', 'false', 'no', 'n', 'off'}:
        return False
    msg = f'expected boolean value, got {value!r}'
    raise argparse.ArgumentTypeError(msg)


def model_cfg() -> dict:
    """Build the shared coarse model config."""
    return {
        'backbone': 'resnet18',
        'pretrained': False,
        'in_chans': 3,
        'out_chans': 1,
        'pre_stages': 3,
        'pre_stage_strides': [[1, 2], [1, 1], [1, 2]],
        'pre_stage_channels': [16, 32, 32],
        'pre_stage_kernels': [3, 3],
    }


def base_common(args: argparse.Namespace) -> dict:
    """Build shared coarse train/infer config."""
    return {
        'coarse': {'input_mode': 'global_anchor_resize'},
        'dataset': {
            'use_header_cache': True,
            'verbose': True,
            'progress': True,
            'primary_keys': ['ffid'],
            'waveform_mode': 'eager',
            'train_endian': args.train_endian,
            'infer_endian': args.infer_endian,
        },
        'fbgate': {
            'apply_on': args.fbgate_apply_on,
            'min_pick_ratio': args.fbgate_min_pick_ratio,
            'verbose': False,
        },
        'transform': {
            'trace_len': 256,
            'time_len': 2048,
            'standardize_eps': 1.0e-8,
        },
        'trace_anchor': {
            'gap_ratio': 5.0,
            'min_gap_m': None,
            'train_mode': 'random',
            'infer_mode': 'center',
        },
        'norm_refs': {
            'time_ref_sec': args.time_ref_sec,
            'offset_ref_m': args.offset_ref_m,
        },
        'model': model_cfg(),
    }


def train_config(args: argparse.Namespace, fold: str, *, smoke: bool = False) -> dict:
    """Build one fold's coarse training config."""
    root = args.fold_list_root / 'folds' / fold
    train_prefix = (
        'train_all_nonheldout' if args.train_list_mode == 'all_nonheldout' else 'train'
    )
    out_stage = '01_coarse_train_smoke' if smoke else '01_coarse_train'
    cfg = base_common(args)
    cfg['paths'] = {
        'segy_files': rel_or_abs(root / f'{train_prefix}_sgy.txt'),
        'fb_files': rel_or_abs(root / f'{train_prefix}_fb.txt'),
        'infer_segy_files': rel_or_abs(root / 'inner_valid_sgy.txt'),
        'infer_fb_files': rel_or_abs(root / 'inner_valid_fb.txt'),
        'out_dir': rel_or_abs(args.run_root / fold / out_stage),
    }
    cfg['train'] = {
        'seed': args.seed,
        'epochs': 1 if smoke else args.epochs,
        'samples_per_epoch': 4 if smoke else args.samples_per_epoch,
        'batch_size': 1 if smoke else args.batch_size,
        'num_workers': args.num_workers,
        'max_norm': 1.0,
        'use_amp': args.use_amp,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'fb_sigma_ms': args.fb_sigma_ms,
        'print_freq': 1 if smoke else args.print_freq,
    }
    cfg['infer'] = {
        'seed': args.seed,
        'batch_size': 1,
        'num_workers': args.num_workers,
        'max_batches': 1 if smoke else args.inner_valid_max_batches,
    }
    cfg['vis'] = {'n': 0 if smoke else args.vis_n, 'out_subdir': 'vis'}
    cfg['ckpt'] = {'save_best_only': True, 'metric': 'infer_loss', 'mode': 'min'}
    return cfg


def infer_config(args: argparse.Namespace, fold: str) -> dict:
    """Build one fold's coarse heldout inference config."""
    root = args.fold_list_root / 'folds' / fold
    cfg = base_common(args)
    # Raw inference does not need fbgate.
    cfg.pop('fbgate', None)
    cfg['paths'] = {
        'segy_files': rel_or_abs(root / 'heldout_sgy.txt'),
        'out_dir': rel_or_abs(args.run_root / fold / '02_coarse_infer'),
    }
    cfg['infer'] = {
        'ckpt_path': rel_or_abs(
            args.run_root / fold / '01_coarse_train' / 'ckpt' / 'best.pt'
        ),
        'device': args.infer_device,
        'batch_size': 1,
        'num_workers': args.num_workers,
        'amp': args.infer_amp,
        'use_tqdm': args.use_tqdm,
    }
    return cfg


def write_yaml(path: Path, data: dict) -> None:
    """Write one YAML file and print its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding='utf-8',
    )
    print(path)


def main() -> None:
    """Render all fold configs."""
    ap = argparse.ArgumentParser(
        description='Render 6-fold global-anchor fbpick-coarse configs.'
    )
    ap.add_argument(
        '--cv-root',
        type=Path,
        default=Path('/workspace/proc/fbpick/site54/oof'),
    )
    ap.add_argument('--run-id', default='baseline_physical_center')
    ap.add_argument('--run-root', type=Path, default=None)
    ap.add_argument('--fold-list-root', type=Path, default=None)
    ap.add_argument('--config-root', type=Path, default=None)
    ap.add_argument(
        '--legacy-flat-configs',
        nargs='?',
        const=True,
        default=False,
        type=parse_bool,
    )
    ap.add_argument(
        '--train-list-mode',
        choices=['strict', 'all_nonheldout'],
        default='strict',
    )
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--samples-per-epoch', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--inner-valid-max-batches', type=int, default=8)
    ap.add_argument('--vis-n', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1.0e-4)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--fb-sigma-ms', type=float, default=10.0)
    ap.add_argument('--print-freq', type=int, default=10)
    ap.add_argument('--use-amp', action='store_true', default=True)
    ap.add_argument('--no-use-amp', action='store_false', dest='use_amp')
    ap.add_argument('--infer-amp', action='store_true', default=False)
    ap.add_argument('--infer-device', default='auto')
    ap.add_argument('--use-tqdm', action='store_true', default=False)
    ap.add_argument('--train-endian', default='big')
    ap.add_argument('--infer-endian', default='big')
    ap.add_argument('--time-ref-sec', type=float, default=20.0)
    ap.add_argument('--offset-ref-m', type=float, default=2000.0)
    ap.add_argument('--fbgate-apply-on', default='any')
    ap.add_argument('--fbgate-min-pick-ratio', type=float, default=0.3)
    args = ap.parse_args()

    args.cv_root = args.cv_root.resolve()
    args.run_root = (args.run_root or args.cv_root / 'runs' / args.run_id).resolve()
    args.fold_list_root = (args.fold_list_root or args.cv_root / 'fold_lists').resolve()
    args.config_root = (args.config_root or args.run_root / 'configs').resolve()

    if not args.fold_list_root.is_dir():
        raise SystemExit(f'fold list root not found: {args.fold_list_root}')
    for fold in FOLDS:
        fold_config_dir = args.config_root / fold
        full_train = train_config(args, fold, smoke=False)
        smoke_train = train_config(args, fold, smoke=True)
        heldout_infer = infer_config(args, fold)
        write_yaml(fold_config_dir / '01_coarse_train.yaml', full_train)
        write_yaml(fold_config_dir / '01_coarse_train_smoke.yaml', smoke_train)
        write_yaml(fold_config_dir / '02_coarse_infer.yaml', heldout_infer)
        if args.legacy_flat_configs:
            write_yaml(
                args.config_root / f'config_train_fbpick_coarse_{fold}.yaml',
                full_train,
            )
            write_yaml(
                args.config_root / f'config_train_fbpick_coarse_{fold}_smoke.yaml',
                smoke_train,
            )
            write_yaml(
                args.config_root
                / f'config_infer_fbpick_coarse_{fold}_heldout.yaml',
                heldout_infer,
            )

if __name__ == '__main__':
    main()
