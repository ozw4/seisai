# %%
"""Run tiled-h inference on paired SEG-Y gathers and save triptych PNGs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from pair_common import (
	build_device,
	build_infer_transform,
	build_model,
	build_pair_dataset,
	build_plan,
	load_checkpoint,
	load_infer_config,
	seed_all,
)
from seisai_engine.infer.runner import TiledHConfig, infer_batch_tiled_h
from seisai_utils.viz_pair import PairTriptychVisConfig, save_pair_triptych_step_png
from torch.utils.data import DataLoader, Subset


def main(argv: list[str] | None = None) -> None:
	"""Run tiled-h inference on paired SEG-Y gathers and save triptych PNGs.

	Parameters
	----------
	argv : list[str] | None, optional
		Command-line arguments (excluding the program name). If None, arguments
		are read from ``sys.argv``.

	Returns
	-------
	None

	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default='./config_train_pair.yaml')
	parser.add_argument('--ckpt', default=None)
	args = parser.parse_args(argv)

	cfg = load_infer_config(args.config)
	ckpt_path = str(args.ckpt) if args.ckpt is not None else cfg.paths.ckpt_path

	device = build_device()
	seed_all(cfg.infer.seed)

	ckpt = load_checkpoint(ckpt_path)
	if not isinstance(ckpt.get('model_cfg'), dict):
		msg = 'checkpoint model_cfg must be dict'
		raise ValueError(msg)
	if ckpt['model_cfg'] != asdict(cfg.model):
		msg = 'checkpoint model_cfg does not match config model'
		raise ValueError(msg)

	model = build_model(cfg.model)
	model.load_state_dict(ckpt['model_state_dict'], strict=True)
	model.to(device)
	model.eval()

	plan = build_plan()
	infer_transform = build_infer_transform()

	ds_infer_full = build_pair_dataset(
		paths=cfg.paths,
		ds_cfg=cfg.dataset,
		transform=infer_transform,
		plan=plan,
		subset_traces=cfg.infer.subset_traces,
		valid=True,
	)

	infer_ds = Subset(
		ds_infer_full, range(cfg.infer.batch_size * cfg.infer.max_batches)
	)
	infer_loader = DataLoader(
		infer_ds,
		batch_size=cfg.infer.batch_size,
		shuffle=False,
		num_workers=cfg.infer.num_workers,
		pin_memory=(device.type == 'cuda'),
	)

	Path(cfg.vis.out_dir).mkdir(parents=True, exist_ok=True)

	tiled_cfg = TiledHConfig(
		tile_h=cfg.tile.tile_h,
		overlap_h=cfg.tile.overlap_h,
		tiles_per_batch=cfg.tile.tiles_per_batch,
		amp=cfg.tile.amp,
		use_tqdm=cfg.tile.use_tqdm,
	)

	triptych_cfg = PairTriptychVisConfig(
		cmap=cfg.vis.cmap,
		vmin=cfg.vis.vmin,
		vmax=cfg.vis.vmax,
		transpose_for_trace_time=cfg.vis.transpose_for_trace_time,
		per_trace_norm=cfg.vis.per_trace_norm,
		per_trace_eps=cfg.vis.per_trace_eps,
		figsize=cfg.vis.figsize,
		dpi=cfg.vis.dpi,
	)

	non_blocking = bool(device.type == 'cuda')

	with torch.no_grad():
		for step, batch in enumerate(infer_loader):
			x_in = batch['input'].to(device=device, non_blocking=non_blocking)
			x_tg = batch['target'].to(device=device, non_blocking=non_blocking)

			x_pr = infer_batch_tiled_h(model, x_in, cfg=tiled_cfg)

			if step < cfg.vis.n:
				save_pair_triptych_step_png(
					cfg.vis.out_dir,
					step=step,
					x_in_bchw=x_in.detach().cpu(),
					x_tg_bchw=x_tg.detach().cpu(),
					x_pr_bchw=x_pr.detach().cpu(),
					cfg=triptych_cfg,
					batch=batch,
				)

	ds_infer_full.close()
	print(f'completed inference. outputs: {cfg.vis.out_dir}')


if __name__ == '__main__':
	main()

# %%
