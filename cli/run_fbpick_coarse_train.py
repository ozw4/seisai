"""Thin entrypoint for fbpick coarse training."""

from __future__ import annotations

from pathlib import Path

from seisai_engine.pipelines.fbpick.coarse import TRAIN_MAIN_TARGET

if __package__:
    from ._entrypoint import run_pipeline_train_entrypoint
else:  # pragma: no cover - used when run as a script path
    from _entrypoint import run_pipeline_train_entrypoint

__all__ = ['main']

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / 'examples' / 'config_train_fbpick_coarse.yaml'


def _pipeline_main(argv: list[str] | None = None) -> None:
    _ = argv
    msg = (
        'fbpick coarse train entrypoint is intentionally unavailable in Phase 1. '
        f'Future implementation target: {TRAIN_MAIN_TARGET}'
    )
    raise NotImplementedError(msg)


def main(argv: list[str] | None = None) -> None:
    run_pipeline_train_entrypoint(
        default_config_path=DEFAULT_CONFIG_PATH,
        pipeline_main=_pipeline_main,
        argv=argv,
    )


if __name__ == '__main__':
    main()
