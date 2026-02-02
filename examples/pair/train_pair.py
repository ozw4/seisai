"""Train EncDec2D on paired SEG-Y gathers and save a checkpoint."""

from __future__ import annotations

import argparse

import torch
from pair_common import (
	build_device,
	build_model,
	build_pair_dataset,
	build_plan,
	build_train_transform,
	load_train_config,
	save_checkpoint,
	seed_all,
)
from seisai_engine.loss.pixelwise_loss import build_criterion
from seisai_engine.train_loop import train_one_epoch
from torch.utils.data import DataLoader, Subset


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default='./config_train_pair.yaml')
	parser.add_argument('--ckpt_out', default=None)
	args = parser.parse_args(argv)

	cfg = load_train_config(args.config)
	ckpt_path = str(args.ckpt_out) if args.ckpt_out is not None else cfg.paths.ckpt_path

	device = build_device()
	seed_all(cfg.train.seed)

	plan = build_plan()
	train_transform = build_train_transform(cfg.train.time_len)

	ds_train_full = build_pair_dataset(
		paths=cfg.paths,
		ds_cfg=cfg.dataset,
		transform=train_transform,
		plan=plan,
		subset_traces=cfg.train.subset_traces,
		valid=False,
	)

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

	ds_train_full.close()
	print(f'saved checkpoint: {ckpt_path}')


if __name__ == '__main__':
	main()
