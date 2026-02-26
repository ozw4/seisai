#!/usr/bin/env python3
"""Run stage1 -> stage2 -> stage3 -> stage4 pipeline with mode switch.

Examples:
  # infer-only mode with CLI
  # python -m proc.jogsarar.run_stage1234 --mode infer_only --in /data/jogsarar --stage1-ckpt /w/stage1.pth --stage4-ckpt /w/stage4.pt
  # train mode with YAML
  # python -m proc.jogsarar.run_stage1234 --config proc/jogsarar/configs/run1234_train.yaml

"""

from __future__ import annotations

import argparse
from pathlib import Path

from _runner_common import (
    coerce_optional_bool_value,
    coerce_optional_float_value,
    coerce_optional_int_value,
    coerce_path_value,
    load_config,
    main_common,
    normalize_segy_exts,
    parse_args_with_yaml_defaults,
    resolve_existing_file,
)

DEFAULT_STAGE3_CONFIG = (
    Path(__file__).resolve().parent
    / 'configs'
    / 'config_convnext_prestage2_drop005.yaml'
)
_ALLOWED_MODES = {'train', 'infer_only'}
_ALLOWED_THRESH_MODES = {'global', 'per_segy'}


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
    p.add_argument(
        '--thresh-mode', choices=tuple(sorted(_ALLOWED_THRESH_MODES)), default=None
    )
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


def _coerce_segy_exts_value(value: object) -> tuple[str, ...]:
    return normalize_segy_exts(value)


def _coerce_mode_value(value: object) -> str:
    if not isinstance(value, str):
        msg = f'config[mode] must be str, got {type(value).__name__}'
        raise TypeError(msg)
    if value not in _ALLOWED_MODES:
        msg = f'config[mode] must be one of {sorted(_ALLOWED_MODES)}, got {value!r}'
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
            f'config[thresh_mode] must be one of {sorted(_ALLOWED_THRESH_MODES)}, '
            f'got {value!r}'
        )
        raise ValueError(msg)
    return value


def _load_yaml_defaults(config_path: Path) -> dict[str, object]:
    loaded = load_config(config_path)

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
            defaults[key] = coerce_path_value(key, value, allow_none=False)
            continue
        if key in {'out_root', 'stage3_config', 'stage4_ckpt', 'stage4_cfg_yaml'}:
            defaults[key] = coerce_path_value(key, value, allow_none=True)
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
            defaults[key] = coerce_optional_int_value(key, value)
            continue
        if key == 'skip_stage4':
            defaults[key] = coerce_optional_bool_value(key, value)
            continue
        if key == 'stage4_standardize_eps':
            defaults[key] = coerce_optional_float_value(key, value)
            continue
    return defaults


def _parse_args() -> argparse.Namespace:
    parser = _build_parser()
    return parse_args_with_yaml_defaults(
        parser,
        load_yaml_defaults=_load_yaml_defaults,
    )


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

    stage1_ckpt = resolve_existing_file(args.stage1_ckpt, context='--stage1-ckpt')

    stage3_config: Path | None = None
    if mode == 'train':
        stage3_cfg_arg = args.stage3_config
        if stage3_cfg_arg is None:
            stage3_cfg_arg = DEFAULT_STAGE3_CONFIG
        stage3_config = resolve_existing_file(stage3_cfg_arg, context='--stage3-config')

    stage4_ckpt: Path | None = None
    if args.stage4_ckpt is not None:
        stage4_ckpt = resolve_existing_file(args.stage4_ckpt, context='--stage4-ckpt')

    stage4_cfg_yaml: Path | None = None
    if args.stage4_cfg_yaml is not None:
        stage4_cfg_yaml = resolve_existing_file(
            args.stage4_cfg_yaml, context='--stage4-cfg-yaml'
        )

    if not skip_stage4 and stage4_ckpt is None and stage4_cfg_yaml is None:
        msg = (
            'stage4 requires --stage4-ckpt or --stage4-cfg-yaml '
            '(unless --skip-stage4 is set)'
        )
        raise ValueError(msg)

    stage2_emit_training_artifacts = mode == 'train'
    if mode == 'train':
        stage2_thresh_mode = (
            str(args.thresh_mode) if args.thresh_mode is not None else None
        )
        run_stage3 = True
        stage3_skip_message = None
        completion_message = '[PIPELINE] completed stage1->stage2->stage3->stage4'
        completion_message_no_stage4 = '[PIPELINE] completed stage1->stage2->stage3'
    else:
        stage2_thresh_mode = 'per_segy'
        run_stage3 = False
        stage3_skip_message = '[PIPELINE] stage3 skipped (mode=infer_only)'
        completion_message = '[PIPELINE] completed stage1->stage2->stage4 (infer_only)'
        completion_message_no_stage4 = (
            '[PIPELINE] completed stage1->stage2 (infer_only)'
        )

    main_common(
        args,
        stages=('stage1', 'stage2', 'stage3', 'stage4'),
        stage1_ckpt=stage1_ckpt,
        mode=mode,
        skip_stage4=skip_stage4,
        stage2_thresh_mode=stage2_thresh_mode,
        stage2_emit_training_artifacts=stage2_emit_training_artifacts,
        run_stage3=run_stage3,
        stage3_config=stage3_config,
        stage3_skip_message=stage3_skip_message,
        stage4_ckpt=stage4_ckpt,
        stage4_cfg_yaml=stage4_cfg_yaml,
        stage4_standardize_eps=args.stage4_standardize_eps,
        completion_message=completion_message,
        completion_message_no_stage4=completion_message_no_stage4,
    )


if __name__ == '__main__':
    main()
