from __future__ import annotations

import argparse
from pathlib import Path

import torch
from seisai_engine.pipelines.common.config_io import (
	load_config,
	resolve_cfg_paths,
	resolve_relpath,
)
from seisai_engine.pipelines.common.seed import seed_all
from seisai_engine.train_loop import train_one_epoch
from seisai_utils.config import (
	optional_bool,
	optional_float,
	optional_int,
	optional_str,
	require_dict,
	require_float,
	require_int,
)
from torch.utils.data import DataLoader, Subset

from .build_dataset import build_dataset
from .build_model import build_model
from .loss import criterion
from .vis import run_epoch_debug

__all__ = ['main']

DEFAULT_CONFIG_PATH = Path('examples/config_train_psn.yaml')


def _resolve_vis_out_dir(
	*, base_dir: Path, vis_cfg: dict, override: str | None
) -> str:
	vis_out_dir = optional_str(vis_cfg, 'out_dir', './_psn_vis')
	if override is not None:
		return resolve_relpath(base_dir, override)
	return resolve_relpath(base_dir, vis_out_dir)


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
	parser.add_argument('--vis_out_dir', default=None)
	args, _unknown = parser.parse_known_args(argv)

	cfg_path = Path(args.config).expanduser().resolve()
	if not cfg_path.is_file():
		msg = f'config not found: {cfg_path}'
		raise FileNotFoundError(msg)

	cfg = load_config(str(cfg_path))
	if cfg is None:
		msg = f'config is empty or failed to load: {cfg_path}'
		raise ValueError(msg)

	base_dir = cfg_path.parent
	resolve_cfg_paths(
		cfg,
		base_dir,
		keys=[
			'paths.segy_files',
			'paths.phase_pick_files',
		],
	)

	train_cfg = require_dict(cfg, 'train')
	vis_cfg = require_dict(cfg, 'vis')

	vis_out_dir = _resolve_vis_out_dir(
		base_dir=base_dir, vis_cfg=vis_cfg, override=args.vis_out_dir
	)

	batch_size = require_int(train_cfg, 'batch_size')
	epochs = require_int(train_cfg, 'epochs')
	lr = require_float(train_cfg, 'lr')
	samples_per_epoch = require_int(train_cfg, 'samples_per_epoch')
	max_norm = optional_float(train_cfg, 'max_norm', 1.0)
	use_amp = optional_bool(train_cfg, 'use_amp', default=True)
	num_workers = optional_int(train_cfg, 'num_workers', 0)

	if 'seed' in train_cfg:
		seed_val = train_cfg['seed']
		if not isinstance(seed_val, int):
			raise TypeError('train.seed must be int')
		seed_all(int(seed_val))

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	ds_full = build_dataset(cfg)

	try:
		train_ds = Subset(ds_full, range(int(samples_per_epoch)))
		train_loader = DataLoader(
			train_ds,
			batch_size=int(batch_size),
			shuffle=True,
			num_workers=int(num_workers),
			pin_memory=(device.type == 'cuda'),
		)

		model = build_model(cfg)
		model.to(device)
		optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))

		Path(vis_out_dir).mkdir(parents=True, exist_ok=True)

		for epoch in range(int(epochs)):
			stats = train_one_epoch(
				model,
				train_loader,
				optimizer,
				criterion,
				device=device,
				lr_scheduler=None,
				gradient_accumulation_steps=1,
				max_norm=float(max_norm),
				use_amp=bool(use_amp),
				scaler=None,
				ema=None,
				step_offset=0,
				print_freq=10,
				on_step=None,
			)
			print(
				f'epoch={epoch} loss={stats["loss"]:.6f} steps={int(stats["steps"])} samples={int(stats["samples"])}'
			)
			run_epoch_debug(
				model,
				train_loader,
				device=device,
				epoch=epoch,
				out_dir=vis_out_dir,
			)
	finally:
		ds_full.close()


if __name__ == '__main__':
	main()
