from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from seisai_engine.pipelines.common.config_io import resolve_relpath
from seisai_engine.pipelines.common.seed import seed_all
from seisai_engine.train_loop import train_one_epoch

from .build_dataset import build_pair_dataset, build_train_transform
from .build_model import build_model
from .build_plan import build_plan
from .checkpoint import save_checkpoint
from .config import load_train_config
from .loss import build_criterion

__all__ = ['main']

DEFAULT_CONFIG_PATH = Path('examples/pair/config_train_pair.yaml')


def _resolve_ckpt_out(*, base_dir: Path, override: str | None, fallback: str) -> str:
	if override is None:
		return fallback
	return resolve_relpath(base_dir, override)


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
	parser.add_argument('--ckpt_out', default=None)
	args, _unknown = parser.parse_known_args(argv)

	cfg = load_train_config(args.config)
	base_dir = Path(args.config).expanduser().resolve().parent
	ckpt_path = _resolve_ckpt_out(
		base_dir=base_dir, override=args.ckpt_out, fallback=cfg.paths.ckpt_path
	)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	seed_all(cfg.train.seed)

	plan = build_plan()
	train_transform = build_train_transform(cfg.train.time_len)

	ds_train_full = build_pair_dataset(
		paths=cfg.paths,
		ds_cfg=cfg.dataset,
		transform=train_transform,
		plan=plan,
		subset_traces=cfg.train.subset_traces,
		secondary_key_fixed=False,
	)

	try:
		train_ds = Subset(ds_train_full, range(cfg.train.samples_per_epoch))
		train_loader = DataLoader(
			train_ds,
			batch_size=cfg.train.batch_size,
			shuffle=True,
			num_workers=cfg.train.num_workers,
			pin_memory=(device.type == 'cuda'),
		)

		model = build_model(cfg.model).to(device)

		criterion = build_criterion(cfg.train.loss_kind)
		optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

		global_step = 0
		for epoch in range(cfg.train.epochs):
			stats = train_one_epoch(
				model,
				train_loader,
				optimizer,
				criterion,
				device=device,
				lr_scheduler=None,
				gradient_accumulation_steps=1,
				max_norm=cfg.train.max_norm,
				use_amp=cfg.train.use_amp,
				scaler=None,
				ema=None,
				step_offset=global_step,
				print_freq=10,
				on_step=None,
			)
			global_step += int(stats['steps'])
			print(
				f'epoch={epoch} loss={stats["loss"]:.6f} steps={int(stats["steps"])} '
				f'samples={int(stats["samples"])}'
			)

		save_checkpoint(
			ckpt_path=ckpt_path,
			model=model,
			model_cfg=cfg.model,
			epoch=cfg.train.epochs - 1,
			global_step=global_step,
			optimizer=optimizer,
		)
	finally:
		ds_train_full.close()

	print(f'saved checkpoint: {ckpt_path}')


if __name__ == '__main__':
	main()
