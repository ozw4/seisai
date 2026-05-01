"""Thin entrypoint for fbpick coarse coverage evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = ['main', 'run_pipeline']


def _load_run_eval_from_config():
    from seisai_engine.pipelines.fbpick.coarse.eval import run_eval_from_config

    return run_eval_from_config


def run_pipeline(config_path: str | Path) -> Path:
    run_eval_from_config = _load_run_eval_from_config()
    report_paths = run_eval_from_config(config_path)
    summary_path = report_paths['summary_json']
    print(summary_path)
    return summary_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    run_pipeline(args.config)


if __name__ == '__main__':
    main()
