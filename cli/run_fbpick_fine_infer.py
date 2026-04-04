"""Thin entrypoint for fbpick fine inference."""

from __future__ import annotations

import argparse

__all__ = ['main']

pipeline_main = None


def _load_pipeline_main():
    # Delay pipeline import so CLI module import stays lightweight in test envs.
    from seisai_engine.pipelines.fbpick.fine.infer import main as pipeline_main

    return pipeline_main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--save-overview', action='store_true')
    group.add_argument('--no-save-overview', action='store_true')
    args, unknown = parser.parse_known_args(argv)

    pipeline_args = ['--config', str(args.config)]
    if args.save_overview:
        pipeline_args += ['viewer.enabled=true', 'viewer.save_overview_png=true']
    if args.no_save_overview:
        pipeline_args += ['viewer.save_overview_png=false']
    pipeline_args += unknown

    loaded_pipeline_main = pipeline_main or _load_pipeline_main()
    loaded_pipeline_main(argv=pipeline_args)


if __name__ == '__main__':
    main()
