"""Thin entrypoint for fbpick fine inference."""

from __future__ import annotations

import argparse

from seisai_engine.pipelines.fbpick.fine.infer import main as pipeline_main

__all__ = ['main']


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
    pipeline_main(argv=pipeline_args)


if __name__ == '__main__':
    main()
