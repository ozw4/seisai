"""Thin entrypoint for PSN training."""

# %%
from __future__ import annotations

from pathlib import Path

from seisai_engine.pipelines.psn.train import main as pipeline_main

if __package__:
    from ._entrypoint import run_pipeline_train_entrypoint
else:  # pragma: no cover - used when run as a script path
    from _entrypoint import run_pipeline_train_entrypoint

__all__ = ['main']

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / 'examples' / 'config_train_psn.yaml'


def main(argv: list[str] | None = None) -> None:
    run_pipeline_train_entrypoint(
        default_config_path=DEFAULT_CONFIG_PATH,
        pipeline_main=pipeline_main,
        argv=argv,
    )


if __name__ == '__main__':
    main()
