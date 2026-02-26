# %%
#!/usr/bin/env python3
"""Run stage1 -> stage2 -> stage4 pipeline (without stage3).

Examples:
  # directory input
  # python -m proc.jogsarar.run_stage124 --in /data/jogsarar --stage1-ckpt /w/stage1.pth --stage4-ckpt /w/stage4.pt --out-root /data/out
  # single file input
  # python -m proc.jogsarar.run_stage124 --in /data/jogsarar/a.sgy --stage1-ckpt /w/stage1.pth --stage4-ckpt /w/stage4.pt --stage4-cfg-yaml proc/jogsarar/configs/config_convnext_prestage2_drop005.yaml
  # YAML config input
  # python -m proc.jogsarar.run_stage124 --config proc/jogsarar/configs/run124.yaml

"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from _runner_common import main_common, resolve_existing_file
from config_io import (
    build_yaml_defaults,
    coerce_optional_int,
    coerce_path,
    load_yaml_dict,
    normalize_segy_exts,
    parse_args_with_yaml_defaults,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Run stage1 -> stage2 -> stage4 pipeline without stage3.'
    )
    p.add_argument(
        '--config',
        type=Path,
        default=None,
        help='YAML config path. CLI options override config values.',
    )
    p.add_argument('--in', dest='in_path', type=Path, default=None)
    p.add_argument('--stage1-ckpt', type=Path, default=None)
    p.add_argument('--stage1-cfg-yaml', type=Path, default=None)
    p.add_argument('--stage4-ckpt', type=Path, default=None)
    p.add_argument('--stage4-cfg-yaml', type=Path, default=None)
    p.add_argument('--out-root', type=Path, default=None)
    p.add_argument('--iter-id', type=int, default=None)
    p.add_argument('--segy-exts', type=str, default=None)
    p.add_argument('--thresh-mode', choices=('global', 'per_segy'), default=None)
    p.add_argument('--viz-every-n-shots', type=int, default=None)
    return p


def _coerce_thresh_mode_value(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f'config[thresh_mode] must be str, got {type(value).__name__}'
        raise TypeError(msg)
    if value not in {'global', 'per_segy'}:
        msg = f"config[thresh_mode] must be 'global' or 'per_segy', got {value!r}"
        raise ValueError(msg)
    return value


def _coerce_iter_id_value(value: object) -> int | None:
    out = coerce_optional_int('iter_id', value)
    if out is None:
        return None
    if int(out) < 0:
        msg = f'config[iter_id] must be >= 0, got {out}'
        raise ValueError(msg)
    return int(out)


def _load_yaml_defaults(config_path: Path) -> dict[str, object]:
    loaded = load_yaml_dict(config_path)

    allowed_keys = {
        'in_path',
        'stage1_ckpt',
        'stage1_cfg_yaml',
        'stage4_ckpt',
        'stage4_cfg_yaml',
        'out_root',
        'iter_id',
        'segy_exts',
        'thresh_mode',
        'viz_every_n_shots',
    }
    coercers = {
        'in_path': partial(coerce_path, 'in_path', allow_none=False),
        'stage1_ckpt': partial(coerce_path, 'stage1_ckpt', allow_none=False),
        'stage1_cfg_yaml': partial(coerce_path, 'stage1_cfg_yaml', allow_none=True),
        'stage4_ckpt': partial(coerce_path, 'stage4_ckpt', allow_none=False),
        'stage4_cfg_yaml': partial(coerce_path, 'stage4_cfg_yaml', allow_none=True),
        'out_root': partial(coerce_path, 'out_root', allow_none=True),
        'iter_id': _coerce_iter_id_value,
        'segy_exts': normalize_segy_exts,
        'thresh_mode': _coerce_thresh_mode_value,
        'viz_every_n_shots': partial(coerce_optional_int, 'viz_every_n_shots'),
    }
    return build_yaml_defaults(
        loaded,
        allowed_keys=allowed_keys,
        coercers=coercers,
    )


def _parse_args() -> argparse.Namespace:
    parser = _build_parser()
    return parse_args_with_yaml_defaults(
        parser,
        load_defaults=_load_yaml_defaults,
    )


def main() -> None:
    args = _parse_args()

    if args.in_path is None:
        raise ValueError('missing required argument: --in (or config.in_path)')
    if args.iter_id is not None and int(args.iter_id) < 0:
        msg = f'--iter-id must be >= 0, got {args.iter_id}'
        raise ValueError(msg)
    if args.stage1_ckpt is None:
        raise ValueError(
            'missing required argument: --stage1-ckpt (or config.stage1_ckpt)'
        )
    if args.stage4_ckpt is None:
        raise ValueError(
            'missing required argument: --stage4-ckpt (or config.stage4_ckpt)'
        )

    stage1_ckpt = resolve_existing_file(args.stage1_ckpt, context='--stage1-ckpt')
    stage1_cfg_yaml: Path | None = None
    if args.stage1_cfg_yaml is not None:
        stage1_cfg_yaml = resolve_existing_file(
            args.stage1_cfg_yaml, context='--stage1-cfg-yaml'
        )
    stage4_ckpt = resolve_existing_file(args.stage4_ckpt, context='--stage4-ckpt')

    stage4_cfg_yaml: Path | None = None
    if args.stage4_cfg_yaml is not None:
        stage4_cfg_yaml = resolve_existing_file(
            args.stage4_cfg_yaml, context='--stage4-cfg-yaml'
        )

    main_common(
        args,
        stages=('stage1', 'stage2', 'stage4'),
        stage1_ckpt=stage1_ckpt,
        stage1_cfg_yaml=stage1_cfg_yaml,
        stage2_thresh_mode=str(args.thresh_mode)
        if args.thresh_mode is not None
        else None,
        stage2_emit_training_artifacts=True,
        stage4_ckpt=stage4_ckpt,
        stage4_cfg_yaml=stage4_cfg_yaml,
        completion_message='[PIPELINE] completed stage1->stage2->stage4',
    )


if __name__ == '__main__':
    main()
