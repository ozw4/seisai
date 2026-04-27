"""Thin entrypoint for fbpick coarse training."""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = ['main', 'run_pipeline']


def _load_run_train():
    from seisai_engine.pipelines.fbpick.coarse.train import run_train

    return run_train


def run_pipeline(config_path: str | Path) -> None:
    run_train = _load_run_train()
    run_train(config_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    run_pipeline(args.config)


if __name__ == '__main__':
    main()
