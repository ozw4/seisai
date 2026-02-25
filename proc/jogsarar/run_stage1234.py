#!/usr/bin/env python3
"""Run stage1 -> stage2 -> stage3 -> stage4 pipeline with mode switch.

Examples:
  # infer-only mode with CLI
  # python proc/jogsarar/run_stage1234.py --mode infer_only --in /data/jogsarar --stage1-ckpt /w/stage1.pth --stage4-ckpt /w/stage4.pt
  # train mode with YAML
  # python proc/jogsarar/run_stage1234.py --config proc/jogsarar/configs/run1234.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import stage1_fbp_infer_raw as stage1
import stage2_make_psn512_windows as stage2
import stage4_psn512_infer_to_raw as stage4
from jogsarar_shared import find_segy_files

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE3_CONFIG = (
    Path(__file__).resolve().parent / 'configs' / 'config_convnext_prestage2_drop005.yaml'
)
_ALLOWED_MODES = {'train', 'infer_only'}
_ALLOWED_THRESH_MODES = {'global', 'per_segy'}


def _parse_segy_exts(text: str) -> tuple[str, ...]:
    vals = [x.strip() for x in text.split(',')]
    out: list[str] = []
    for v in vals:
        if not v:
            continue
        e = v.lower()
        if not e.startswith('.'):
            e = '.' + e
        out.append(e)
    if len(out) == 0:
        msg = f'--segy-exts produced empty list: {text!r}'
        raise ValueError(msg)
    return tuple(out)


def _resolve_input_paths(
    in_path: Path,
    *,
    segy_exts: tuple[str, ...],
) -> tuple[Path, list[Path]]:
    p = Path(in_path).expanduser().resolve()
    if not p.exists():
        msg = f'--in not found: {p}'
        raise FileNotFoundError(msg)

    if p.is_file():
        if p.suffix.lower() not in segy_exts:
            msg = f'--in file extension must be one of {segy_exts}, got {p.suffix}'
            raise ValueError(msg)
        return p.parent, [p]

    if not p.is_dir():
        msg = f'--in must be file or directory: {p}'
        raise NotADirectoryError(msg)

    segys = find_segy_files(p, exts=segy_exts, recursive=True)
    if len(segys) == 0:
        msg = f'no segy files found under: {p}'
        raise RuntimeError(msg)
    return p, segys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Run stage1 -> stage2 -> stage3 -> stage4 pipeline.'
    )
    p.add_argument(
        '--config',
        type=Path,
        default=None,
        help='YAML config path. CLI options override config values.',
    )
    p.add_argument('--mode', choices=tuple(sorted(_ALLOWED_MODES)), default=None)
    p.add_argument('--in', dest='in_path', type=Path, default=None)
    p.add_argument('--stage1-ckpt', type=Path, default=None)
    p.add_argument('--stage3-config', type=Path, default=None)
    p.add_argument('--stage4-ckpt', type=Path, default=None)
    p.add_argument('--stage4-cfg-yaml', type=Path, default=None)
    p.add_argument('--stage4-standardize-eps', type=float, default=None)
    p.add_argument('--out-root', type=Path, default=None)
    p.add_argument('--segy-exts', type=str, default=None)
    p.add_argument('--thresh-mode', choices=tuple(sorted(_ALLOWED_THRESH_MODES)), default=None)
    p.add_argument('--viz-every-n-shots', type=int, default=None)
    p.add_argument(
        '--skip-stage4',
        dest='skip_stage4',
        action='store_true',
        default=None,
        help='Skip stage4.',
    )
    p.add_argument(
        '--no-skip-stage4',
        dest='skip_stage4',
        action='store_false',
        help='Do not skip stage4.',
    )
    return p


def _coerce_path_value(
    key: str, value: object, *, allow_none: bool = False
) -> Path | None:
    if value is None:
        if allow_none:
            return None
        msg = f'config[{key}] must not be null'
        raise TypeError(msg)
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    msg = f'config[{key}] must be str or Path, got {type(value).__name__}'
    raise TypeError(msg)


def _coerce_segy_exts_value(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return _parse_segy_exts(value)
    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError('config[segy_exts] must not be empty')
        for i, v in enumerate(value):
            if not isinstance(v, str):
                msg = f'config[segy_exts][{i}] must be str, got {type(v).__name__}'
                raise TypeError(msg)
        return _parse_segy_exts(','.join(value))
    msg = f'config[segy_exts] must be str or list[str], got {type(value).__name__}'
    raise TypeError(msg)


def _coerce_mode_value(value: object) -> str:
    if not isinstance(value, str):
        msg = f'config[mode] must be str, got {type(value).__name__}'
        raise TypeError(msg)
    if value not in _ALLOWED_MODES:
        msg = f"config[mode] must be one of {sorted(_ALLOWED_MODES)}, got {value!r}"
        raise ValueError(msg)
    return value


def _coerce_thresh_mode_value(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f'config[thresh_mode] must be str, got {type(value).__name__}'
        raise TypeError(msg)
    if value not in _ALLOWED_THRESH_MODES:
        msg = (
            f"config[thresh_mode] must be one of {sorted(_ALLOWED_THRESH_MODES)}, "
            f'got {value!r}'
        )
        raise ValueError(msg)
    return value


def _coerce_optional_int_value(key: str, value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f'config[{key}] must be int, got {type(value).__name__}'
        raise TypeError(msg)
    return int(value)


def _coerce_optional_bool_value(key: str, value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        msg = f'config[{key}] must be bool, got {type(value).__name__}'
        raise TypeError(msg)
    return bool(value)


def _coerce_optional_float_value(key: str, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f'config[{key}] must be float, got {type(value).__name__}'
        raise TypeError(msg)
    return float(value)


def _load_yaml_defaults(config_path: Path) -> dict[str, object]:
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.is_file():
        msg = f'--config not found: {cfg_path}'
        raise FileNotFoundError(msg)

    import yaml

    with cfg_path.open('r', encoding='utf-8') as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        msg = f'config top-level must be dict, got {type(loaded).__name__}'
        raise TypeError(msg)

    allowed_keys = {
        'in_path',
        'out_root',
        'segy_exts',
        'mode',
        'stage1_ckpt',
        'stage3_config',
        'stage4_ckpt',
        'stage4_cfg_yaml',
        'stage4_standardize_eps',
        'thresh_mode',
        'viz_every_n_shots',
        'skip_stage4',
    }
    unknown = sorted(set(loaded) - allowed_keys)
    if unknown:
        msg = f'unknown config keys: {unknown}'
        raise ValueError(msg)

    defaults: dict[str, object] = {}
    for key, value in loaded.items():
        if key in {'in_path', 'stage1_ckpt'}:
            defaults[key] = _coerce_path_value(key, value, allow_none=False)
            continue
        if key in {'out_root', 'stage3_config', 'stage4_ckpt', 'stage4_cfg_yaml'}:
            defaults[key] = _coerce_path_value(key, value, allow_none=True)
            continue
        if key == 'segy_exts':
            defaults[key] = _coerce_segy_exts_value(value)
            continue
        if key == 'mode':
            defaults[key] = _coerce_mode_value(value)
            continue
        if key == 'thresh_mode':
            defaults[key] = _coerce_thresh_mode_value(value)
            continue
        if key == 'viz_every_n_shots':
            defaults[key] = _coerce_optional_int_value(key, value)
            continue
        if key == 'skip_stage4':
            defaults[key] = _coerce_optional_bool_value(key, value)
            continue
        if key == 'stage4_standardize_eps':
            defaults[key] = _coerce_optional_float_value(key, value)
            continue
    return defaults


def _parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=Path, default=None)
    pre_args, _unknown = pre.parse_known_args()

    parser = _build_parser()
    if pre_args.config is not None:
        yaml_defaults = _load_yaml_defaults(pre_args.config)
        parser.set_defaults(**yaml_defaults)
    return parser.parse_args()


def _resolve_existing_file(path: Path, *, context: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        msg = f'{context} not found: {p}'
        raise FileNotFoundError(msg)
    return p


def _run_stage3_train(*, stage3_config: Path) -> None:
    cli_path = REPO_ROOT / 'cli' / 'run_psn_train.py'
    if not cli_path.is_file():
        msg = f'stage3 CLI not found: {cli_path}'
        raise FileNotFoundError(msg)

    cmd = [sys.executable, str(cli_path), '--config', str(stage3_config)]
    print(f'[PIPELINE] stage3 start config={stage3_config}')
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    print('[PIPELINE] stage3 done')


def main() -> None:
    args = _parse_args()

    mode = str(args.mode) if args.mode is not None else 'infer_only'
    if mode not in _ALLOWED_MODES:
        msg = f'mode must be one of {sorted(_ALLOWED_MODES)}, got {mode!r}'
        raise ValueError(msg)

    skip_stage4 = bool(args.skip_stage4) if args.skip_stage4 is not None else False

    if args.in_path is None:
        raise ValueError('missing required argument: --in (or config.in_path)')
    if args.stage1_ckpt is None:
        raise ValueError(
            'missing required argument: --stage1-ckpt (or config.stage1_ckpt)'
        )

    if mode == 'infer_only' and args.thresh_mode is not None:
        msg = 'thresh_mode is not allowed when --mode infer_only'
        raise ValueError(msg)

    if not skip_stage4:
        if args.stage4_cfg_yaml is not None and args.stage4_standardize_eps is not None:
            msg = (
                'stage4_standardize_eps is not allowed when stage4_cfg_yaml is set '
                '(stage4 uses transform.standardize_eps from YAML)'
            )
            raise ValueError(msg)
        if (
            args.stage4_standardize_eps is not None
            and float(args.stage4_standardize_eps) <= 0.0
        ):
            msg = (
                f'--stage4-standardize-eps must be > 0, '
                f'got {args.stage4_standardize_eps}'
            )
            raise ValueError(msg)

    stage1_ckpt = _resolve_existing_file(args.stage1_ckpt, context='--stage1-ckpt')

    stage3_config: Path | None = None
    if mode == 'train':
        stage3_cfg_arg = args.stage3_config
        if stage3_cfg_arg is None:
            stage3_cfg_arg = DEFAULT_STAGE3_CONFIG
        stage3_config = _resolve_existing_file(stage3_cfg_arg, context='--stage3-config')

    stage4_ckpt: Path | None = None
    if args.stage4_ckpt is not None:
        stage4_ckpt = _resolve_existing_file(args.stage4_ckpt, context='--stage4-ckpt')

    stage4_cfg_yaml: Path | None = None
    if args.stage4_cfg_yaml is not None:
        stage4_cfg_yaml = _resolve_existing_file(
            args.stage4_cfg_yaml, context='--stage4-cfg-yaml'
        )

    if not skip_stage4 and stage4_ckpt is None and stage4_cfg_yaml is None:
        msg = (
            'stage4 requires --stage4-ckpt or --stage4-cfg-yaml '
            '(unless --skip-stage4 is set)'
        )
        raise ValueError(msg)

    if args.segy_exts is None:
        segy_exts = tuple(stage2.DEFAULT_STAGE2_CFG.segy_exts)
    elif isinstance(args.segy_exts, tuple):
        segy_exts = tuple(args.segy_exts)
    else:
        segy_exts = _parse_segy_exts(args.segy_exts)

    in_root, segy_paths = _resolve_input_paths(args.in_path, segy_exts=segy_exts)

    if args.out_root is not None:
        out_root = Path(args.out_root).expanduser().resolve()
        if out_root.exists() and not out_root.is_dir():
            msg = f'--out-root must be a directory path: {out_root}'
            raise NotADirectoryError(msg)
        stage1_out = out_root / 'stage1'
        stage2_out = out_root / 'stage2_win512'
        stage4_out = out_root / 'stage4_pred'
    else:
        stage1_out = Path(stage2.DEFAULT_STAGE2_CFG.in_infer_root)
        stage2_out = Path(stage2.DEFAULT_STAGE2_CFG.out_segy_root)
        stage4_out = Path(stage4.DEFAULT_STAGE4_CFG.out_pred_root)

    print(f'[PIPELINE] mode={mode} skip_stage4={int(skip_stage4)}')
    print(f'[PIPELINE] input_root={in_root} files={len(segy_paths)}')
    print(f'[PIPELINE] stage1_out={stage1_out}')
    print(f'[PIPELINE] stage2_out={stage2_out}')
    if not skip_stage4:
        print(f'[PIPELINE] stage4_out={stage4_out}')

    print(f'[PIPELINE] stage1 start target={len(segy_paths)} skip=0')
    stage1.run_stage1(
        in_segy_root=in_root,
        out_infer_root=stage1_out,
        weights_path=stage1_ckpt,
        segy_paths=segy_paths,
        segy_exts=segy_exts,
        recursive=True,
        viz_every_n_shots=stage1.VIZ_EVERY_N_SHOTS,
        viz_dirname=stage1.VIZ_DIRNAME,
    )
    print(f'[PIPELINE] stage1 done processed={len(segy_paths)} skip=0')

    emit_training_artifacts = mode == 'train'
    stage2_cfg = replace(
        stage2.DEFAULT_STAGE2_CFG,
        in_segy_root=in_root,
        in_infer_root=stage1_out,
        out_segy_root=stage2_out,
        segy_exts=segy_exts,
        emit_training_artifacts=emit_training_artifacts,
    )
    if mode == 'train':
        if args.thresh_mode is not None:
            stage2_cfg = replace(stage2_cfg, thresh_mode=str(args.thresh_mode))
    else:
        stage2_cfg = replace(stage2_cfg, thresh_mode='per_segy')

    stage2_skip = 0
    for p in segy_paths:
        infer_npz = stage2.infer_npz_path_for_segy(p, cfg=stage2_cfg)
        if not infer_npz.exists():
            stage2_skip += 1
    stage2_target = len(segy_paths) - stage2_skip

    print(f'[PIPELINE] stage2 start target={stage2_target} skip={stage2_skip}')
    stage2.run_stage2(cfg=stage2_cfg, segy_paths=segy_paths)
    print(f'[PIPELINE] stage2 done processed={stage2_target} skip={stage2_skip}')

    if mode == 'train':
        if stage3_config is None:
            msg = 'internal error: stage3_config must not be None in train mode'
            raise RuntimeError(msg)
        _run_stage3_train(stage3_config=stage3_config)
    else:
        print('[PIPELINE] stage3 skipped (mode=infer_only)')

    if skip_stage4:
        print('[PIPELINE] stage4 skipped by config')
        if mode == 'train':
            print('[PIPELINE] completed stage1->stage2->stage3')
        else:
            print('[PIPELINE] completed stage1->stage2 (infer_only)')
        return

    stage4_cfg = replace(
        stage4.DEFAULT_STAGE4_CFG,
        in_raw_segy_root=in_root,
        in_win512_segy_root=stage2_out,
        out_pred_root=stage4_out,
        segy_exts=segy_exts,
        cfg_yaml=stage4_cfg_yaml,
        ckpt_path=stage4_ckpt,
    )
    if args.viz_every_n_shots is not None:
        stage4_cfg = replace(stage4_cfg, viz_every_n_shots=int(args.viz_every_n_shots))
    if stage4_cfg_yaml is None and args.stage4_standardize_eps is not None:
        stage4_cfg = replace(
            stage4_cfg, standardize_eps=float(args.stage4_standardize_eps)
        )

    win_lookup = stage4._build_win512_lookup(stage4_cfg.in_win512_segy_root, cfg=stage4_cfg)
    stage4_skip = 0
    for raw_path in segy_paths:
        rel = raw_path.relative_to(stage4_cfg.in_raw_segy_root)
        key = (rel.parent.as_posix(), rel.stem)
        win_path = win_lookup.get(key)
        if win_path is None:
            stage4_skip += 1
            continue
        if stage4._resolve_sidecar_path(win_path) is None:
            stage4_skip += 1
            continue
    stage4_target = len(segy_paths) - stage4_skip

    print(f'[PIPELINE] stage4 start target={stage4_target} skip={stage4_skip}')
    stage4.run_stage4(cfg=stage4_cfg, raw_paths=segy_paths)
    print(f'[PIPELINE] stage4 done processed={stage4_target} skip={stage4_skip}')

    if mode == 'train':
        print('[PIPELINE] completed stage1->stage2->stage3->stage4')
    else:
        print('[PIPELINE] completed stage1->stage2->stage4 (infer_only)')


if __name__ == '__main__':
    main()
