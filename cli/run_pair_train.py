# %%
"""Thin entrypoint for paired SEG-Y training."""

from __future__ import annotations

import argparse
from pathlib import Path

from seisai_engine.pipelines.pair.train import main as pipeline_main

__all__ = ['main']

DEFAULT_CONFIG_PATH = Path(__file__).with_name('config_train_pair.yaml')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    args, unknown = parser.parse_known_args(argv)

    pipeline_args = ['--config', str(args.config)]
    pipeline_args += unknown

    pipeline_main(argv=pipeline_args)


if __name__ == '__main__':
    main()
