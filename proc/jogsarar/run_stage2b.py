#!/usr/bin/env python3
"""Run stage2b (seed from stage4 pred npz) with iteration-aware layout.

Examples:
  python -m proc.jogsarar.run_stage2b \
    --in /data/raw \
    --out-root /data/out \
    --iter-in 0 \
    --iter-out 1

"""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
from pathlib import Path

from common.iter_layout import iter_tag, resolve_iter_layout
from common.model_id import guess_stage4_model_id
from config_io import (
    build_yaml_defaults,
    coerce_optional_bool,
    coerce_optional_float,
    coerce_optional_int,
    coerce_path,
    load_yaml_dict,
    normalize_segy_exts,
    parse_args_with_yaml_defaults,
)
from stage2.cfg import DEFAULT_STAGE2_CFG

_ALLOWED_THRESH_MODES = {'global', 'per_segy'}


def _coerce_nonneg_int(key: str, value: object) -> int | None:
    out = coerce_optional_int(key, value)
    if out is None:
        return None
    if int(out) < 0:
        msg = f'config[{key}] must be >= 0, got {out}'
        raise ValueError(msg)
    return int(out)


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


def _coerce_drop_low_frac_value(value: object) -> float | None:
    out = coerce_optional_float('drop_low_frac', value)
    if out is None:
        return None
    out_f = float(out)
    if out_f < 0.0 or out_f > 1.0:
        msg = f'config[drop_low_frac] must be in [0,1], got {out_f}'
        raise ValueError(msg)
    return out_f


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Run stage2b (seed from stage4_pred npz) with iter layout.'
    )
    p.add_argument(
        '--config',
        type=Path,
        default=None,
        help='YAML config path. CLI options override config values.',
    )
    p.add_argument('--in', '--in-path', dest='in_path', type=Path, default=None)
    p.add_argument('--out-root', type=Path, default=None)
    p.add_argument('--iter-in', type=int, default=None)
    p.add_argument('--iter-out', type=int, default=None)
    p.add_argument('--segy-exts', type=str, default=None)
    p.add_argument(
        '--emit-training-artifacts',
        dest='emit_training_artifacts',
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument('--thresh-mode', choices=tuple(sorted(_ALLOWED_THRESH_MODES)), default=None)
    p.add_argument('--drop-low-frac', type=float, default=None)
    return p


def _load_yaml_defaults(config_path: Path) -> dict[str, object]:
    loaded = load_yaml_dict(config_path)
    allowed_keys = {
        'in_path',
        'out_root',
        'iter_in',
        'iter_out',
        'segy_exts',
        'emit_training_artifacts',
        'thresh_mode',
        'drop_low_frac',
    }
    coercers = {
        'in_path': partial(coerce_path, 'in_path', allow_none=False),
        'out_root': partial(coerce_path, 'out_root', allow_none=False),
        'iter_in': partial(_coerce_nonneg_int, 'iter_in'),
        'iter_out': partial(_coerce_nonneg_int, 'iter_out'),
        'segy_exts': normalize_segy_exts,
        'emit_training_artifacts': partial(
            coerce_optional_bool,
            'emit_training_artifacts',
        ),
        'thresh_mode': _coerce_thresh_mode_value,
        'drop_low_frac': _coerce_drop_low_frac_value,
    }
    return build_yaml_defaults(
        loaded,
        allowed_keys=allowed_keys,
        coercers=coercers,
    )


def _collect_inputs(
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

    segys = sorted(
        x for x in p.rglob('*') if x.is_file() and x.suffix.lower() in segy_exts
    )
    if len(segys) == 0:
        msg = f'no segy files found under: {p}'
        raise RuntimeError(msg)
    return p, segys


def _validate_args(args: argparse.Namespace) -> None:
    if args.in_path is None:
        raise ValueError('missing required argument: --in (or config.in_path)')
    if args.out_root is None:
        raise ValueError('missing required argument: --out-root (or config.out_root)')
    if args.iter_in is None:
        raise ValueError('missing required argument: --iter-in (or config.iter_in)')
    if args.iter_out is None:
        raise ValueError('missing required argument: --iter-out (or config.iter_out)')

    if int(args.iter_in) < 0:
        msg = f'--iter-in must be >= 0, got {args.iter_in}'
        raise ValueError(msg)
    if int(args.iter_out) < 0:
        msg = f'--iter-out must be >= 0, got {args.iter_out}'
        raise ValueError(msg)


def _build_source_model_id(*, stage4_pred_in: Path, iter_in: int) -> str:
    guessed = guess_stage4_model_id(stage4_pred_in)
    if guessed != '':
        return guessed
    return f'{iter_tag(iter_in)}_stage4'


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    if argv is None:
        return parse_args_with_yaml_defaults(
            parser,
            load_defaults=_load_yaml_defaults,
        )

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=Path, default=None)
    pre_args, _ = pre.parse_known_args(argv)
    if pre_args.config is not None:
        parser.set_defaults(**_load_yaml_defaults(pre_args.config))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from stage2b.runner import run_stage2

    args = _parse_args(argv)
    _validate_args(args)

    segy_value = args.segy_exts
    if segy_value is None:
        segy_exts = tuple(DEFAULT_STAGE2_CFG.segy_exts)
    elif isinstance(segy_value, tuple):
        segy_exts = tuple(segy_value)
    else:
        segy_exts = normalize_segy_exts(segy_value)

    in_root, segy_paths = _collect_inputs(args.in_path, segy_exts=segy_exts)

    out_root = Path(args.out_root).expanduser().resolve()
    layout_in = resolve_iter_layout(out_root, iter_id=int(args.iter_in))
    layout_out = resolve_iter_layout(out_root, iter_id=int(args.iter_out))

    stage4_pred_in = layout_in.stage4_out
    if not stage4_pred_in.is_dir():
        msg = f'stage4_pred input directory not found: {stage4_pred_in}'
        raise FileNotFoundError(msg)

    source_model_id = _build_source_model_id(
        stage4_pred_in=stage4_pred_in,
        iter_in=int(args.iter_in),
    )

    cfg = replace(
        DEFAULT_STAGE2_CFG,
        in_segy_root=in_root,
        in_infer_root=stage4_pred_in,
        out_segy_root=layout_out.stage2_out,
        segy_exts=segy_exts,
        iter_id=int(args.iter_out),
        source_model_id=source_model_id,
    )

    if args.emit_training_artifacts is not None:
        cfg = replace(cfg, emit_training_artifacts=bool(args.emit_training_artifacts))
    if args.thresh_mode is not None:
        cfg = replace(cfg, thresh_mode=str(args.thresh_mode))
    if args.drop_low_frac is not None:
        cfg = replace(cfg, drop_low_frac=float(args.drop_low_frac))

    print(f'[STAGE2B] in_root={in_root} files={len(segy_paths)}')
    print(f'[STAGE2B] seed_stage4_pred={stage4_pred_in}')
    print(f'[STAGE2B] out_stage2_win512={layout_out.stage2_out}')
    print(f'[STAGE2B] iter_in={args.iter_in} iter_out={args.iter_out}')

    run_stage2(cfg=cfg, segy_paths=segy_paths)


__all__ = [
    '_build_source_model_id',
    '_collect_inputs',
    '_coerce_drop_low_frac_value',
    '_coerce_nonneg_int',
    '_coerce_thresh_mode_value',
    '_load_yaml_defaults',
    '_parse_args',
    '_validate_args',
    'main',
]


if __name__ == '__main__':
    main()
