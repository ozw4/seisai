"""Thin entrypoint for fbpick coarse inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from seisai_engine.pipelines.fbpick.coarse import INFER_MAIN_TARGET

__all__ = ['main']

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / 'examples' / 'config_infer_fbpick_coarse.yaml'


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    args, _unknown = parser.parse_known_args(argv)
    msg = (
        'fbpick coarse infer entrypoint is intentionally unavailable in Phase 1. '
        f'Config: {args.config}. Future implementation target: {INFER_MAIN_TARGET}'
    )
    raise NotImplementedError(msg)


if __name__ == '__main__':
    main()
