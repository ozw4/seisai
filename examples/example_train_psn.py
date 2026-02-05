"""Thin entrypoint for PSN training."""

from __future__ import annotations

import argparse
from pathlib import Path

from seisai_engine.pipelines.psn.train import main as pipeline_main

__all__ = ['main']

DEFAULT_CONFIG_PATH = Path(__file__).with_name('config_train_psn.yaml')


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
	parser.add_argument('--vis_out_dir', default=None)
	args, _unknown = parser.parse_known_args(argv)

	pipeline_args = ['--config', str(args.config)]
	if args.vis_out_dir is not None:
		pipeline_args += ['--vis_out_dir', str(args.vis_out_dir)]

	pipeline_main(argv=pipeline_args)


if __name__ == '__main__':
	main()
