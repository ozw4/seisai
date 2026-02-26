from __future__ import annotations

import argparse
from pathlib import Path

from config_io import parse_args_with_yaml_defaults
from stage1.cfg import (
    DEFAULT_STAGE1_CFG,
    _STAGE1_CFG_KEYS,
    _cfg_from_namespace,
    _load_yaml_defaults,
)
from stage1.runner import run_stage1_cfg


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run stage1 FBP inference.')
    p.add_argument(
        '--config',
        type=Path,
        default=None,
        help='YAML config path. CLI options override config values.',
    )
    p.add_argument('--in', dest='in_segy_root', type=Path, default=None)
    p.add_argument('--out', dest='out_infer_root', type=Path, default=None)
    p.add_argument('--weights', dest='weights_path', type=Path, default=None)

    for key in _STAGE1_CFG_KEYS:
        if key in {'in_segy_root', 'out_infer_root', 'weights_path'}:
            p.add_argument(
                f'--{key.replace("_", "-")}',
                dest=key,
                type=Path,
                default=None,
                help=argparse.SUPPRESS,
            )
            continue
        if key == 'segy_exts':
            p.add_argument('--segy-exts', dest='segy_exts', type=str, default=None)
            continue

        flag = f'--{key.replace("_", "-")}'
        default_value = getattr(DEFAULT_STAGE1_CFG, key)
        if isinstance(default_value, bool):
            p.add_argument(
                flag,
                dest=key,
                action=argparse.BooleanOptionalAction,
                default=None,
            )
            continue
        if isinstance(default_value, int):
            p.add_argument(flag, dest=key, type=int, default=None)
            continue
        if isinstance(default_value, float):
            p.add_argument(flag, dest=key, type=float, default=None)
            continue
        p.add_argument(flag, dest=key, type=str, default=None)
    return p


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
    args = _parse_args(argv)
    cfg = _cfg_from_namespace(args)
    run_stage1_cfg(cfg, segy_paths=None)


__all__ = ['main']


if __name__ == '__main__':
    main()
