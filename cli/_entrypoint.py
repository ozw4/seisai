from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

__all__ = ['run_pipeline_train_entrypoint']

PipelineMainFn = Callable[[list[str] | None], None]


def run_pipeline_train_entrypoint(
    *,
    default_config_path: Path,
    pipeline_main: PipelineMainFn,
    argv: list[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(default_config_path))
    args, unknown = parser.parse_known_args(argv)

    pipeline_args = ['--config', str(args.config)]
    pipeline_args += unknown

    pipeline_main(argv=pipeline_args)
