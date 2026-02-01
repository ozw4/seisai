# %%
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from seisai_dataset import BuildPlan, SegyGatherPairDataset
from seisai_dataset.builder.builder import IdentitySignal, SelectStack
from seisai_engine.infer.runner import TiledHConfig, infer_batch_tiled_h
from seisai_engine.loss.pixelwise_loss import build_criterion
from seisai_engine.train_loop import train_one_epoch
from seisai_models.models.encdec2d import EncDec2D
from seisai_transforms.augment import PerTraceStandardize, RandomCropOrPad, ViewCompose
from seisai_utils.config import (
	load_config,
	optional_bool,
	optional_float,
	optional_int,
	optional_str,
	optional_tuple2_float,
	require_dict,
	require_float,
	require_int,
	require_list_str,
)
from seisai_utils.viz_pair import PairTriptychVisConfig, save_pair_triptych_step_png
from torch.utils.data import DataLoader, Subset


def _build_plan() -> BuildPlan:
	return BuildPlan(
		wave_ops=[
			IdentitySignal(source_key='x_view_input', dst='x_in', copy=False),
			IdentitySignal(source_key='x_view_target', dst='x_tg', copy=False),
		],
		label_ops=[],
		input_stack=SelectStack(
			keys=['x_in'],
			dst='input',
			dtype=np.float32,
			to_torch=True,
		),
		target_stack=SelectStack(
			keys=['x_tg'],
			dst='target',
			dtype=np.float32,
			to_torch=True,
		),
	)


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default='./config_train_pair.yaml')
	args, _unknown = parser.parse_known_args(argv)

	cfg = load_config(args.config)

	paths = require_dict(cfg, 'paths')
	ds_cfg = require_dict(cfg, 'dataset')
	train_cfg = require_dict(cfg, 'train')
	infer_cfg = require_dict(cfg, 'infer')
	tile_cfg = require_dict(cfg, 'tile')
	vis_cfg = require_dict(cfg, 'vis')
	model_cfg = require_dict(cfg, 'model')

	input_segy_files = require_list_str(paths, 'input_segy_files')
	target_segy_files = require_list_str(paths, 'target_segy_files')
	if len(input_segy_files) != len(target_segy_files):
		raise ValueError(
			'paths.input_segy_files and paths.target_segy_files must have same length'
		)

	max_trials = optional_int(ds_cfg, 'max_trials', 2048)
	use_header_cache = optional_bool(ds_cfg, 'use_header_cache', default=True)
	verbose: bool = optional_bool(ds_cfg, 'verbose', default=True)
	primary_keys_list = ds_cfg.get('primary_keys', ['ffid'])
	if not isinstance(primary_keys_list, list) or not all(
		isinstance(x, str) for x in primary_keys_list
	):
		raise ValueError('dataset.primary_keys must be list[str]')
	primary_keys = tuple(primary_keys_list)

	train_batch_size = require_int(train_cfg, 'batch_size')
	epochs = require_int(train_cfg, 'epochs')
	lr = require_float(train_cfg, 'lr')
	subset_traces = require_int(train_cfg, 'subset_traces')
	time_len = require_int(train_cfg, 'time_len')
	samples_per_epoch = require_int(train_cfg, 'samples_per_epoch')
	loss_kind = optional_str(train_cfg, 'loss_kind', 'l1').lower()
	seed_train = optional_int(train_cfg, 'seed', 42)
	use_amp_train = optional_bool(train_cfg, 'use_amp', default=True)
	max_norm = optional_float(train_cfg, 'max_norm', 1.0)
	train_num_workers = optional_int(train_cfg, 'num_workers', 0)

	infer_batch_size = require_int(infer_cfg, 'batch_size')
	infer_max_batches = require_int(infer_cfg, 'max_batches')
	infer_subset_traces = require_int(infer_cfg, 'subset_traces')
	seed_infer = optional_int(infer_cfg, 'seed', 43)
	infer_num_workers = optional_int(infer_cfg, 'num_workers', 0)

	tile_h = require_int(tile_cfg, 'tile_h')
	overlap_h = require_int(tile_cfg, 'overlap_h')
	tiles_per_batch = require_int(tile_cfg, 'tiles_per_batch')
	amp_infer = optional_bool(tile_cfg, 'amp', default=True)
	use_tqdm = optional_bool(tile_cfg, 'use_tqdm', default=False)

	vis_out_dir = optional_str(vis_cfg, 'out_dir', './_pair_vis')
	vis_n = require_int(vis_cfg, 'n')
	cmap = optional_str(vis_cfg, 'cmap', 'seismic')
	vmin = optional_float(vis_cfg, 'vmin', -3.0)
	vmax = optional_float(vis_cfg, 'vmax', 3.0)
	transpose_for_trace_time = optional_bool(
		vis_cfg, 'transpose_for_trace_time', default=True
	)
	per_trace_norm = optional_bool(vis_cfg, 'per_trace_norm', default=True)
	per_trace_eps = optional_float(vis_cfg, 'per_trace_eps', 1e-8)
	figsize = optional_tuple2_float(vis_cfg, 'figsize', (20.0, 15.0))
	dpi = optional_int(vis_cfg, 'dpi', 300)

	backbone = optional_str(model_cfg, 'backbone', 'resnet18')
	pretrained = optional_bool(model_cfg, 'pretrained', default=False)
	in_chans = optional_int(model_cfg, 'in_chans', 1)
	out_chans = optional_int(model_cfg, 'out_chans', 1)

	if loss_kind not in ('l1', 'mse'):
		raise ValueError('train.loss_kind must be "l1" or "mse"')

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	torch.manual_seed(seed_train)
	_ = np.random.default_rng(seed_train)

	train_transform = ViewCompose(
		[
			RandomCropOrPad(target_len=time_len),
			PerTraceStandardize(eps=1e-8),
		]
	)

	infer_transform = ViewCompose(
		[
			PerTraceStandardize(eps=1e-8),
		]
	)

	plan = _build_plan()
	criterion = build_criterion(loss_kind)

	ds_train_full = SegyGatherPairDataset(
		input_segy_files=input_segy_files,
		target_segy_files=target_segy_files,
		transform=train_transform,
		plan=plan,
		subset_traces=int(subset_traces),
		primary_keys=primary_keys,
		valid=False,
		verbose=bool(verbose),
		max_trials=int(max_trials),
		use_header_cache=bool(use_header_cache),
	)

	train_ds = Subset(ds_train_full, range(int(samples_per_epoch)))
	train_loader = DataLoader(
		train_ds,
		batch_size=int(train_batch_size),
		shuffle=True,
		num_workers=int(train_num_workers),
		pin_memory=(device.type == 'cuda'),
	)

	model = EncDec2D(
		backbone=backbone,
		in_chans=int(in_chans),
		out_chans=int(out_chans),
		pretrained=bool(pretrained),
	)
	model.use_tta = False
	model.to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))

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
			use_amp=bool(use_amp_train),
			scaler=None,
			ema=None,
			step_offset=0,
			print_freq=10,
			on_step=None,
		)
		print(
			f'epoch={epoch} loss={stats["loss"]:.6f} steps={int(stats["steps"])} samples={int(stats["samples"])}'
		)

	torch.manual_seed(seed_infer)
	_ = np.random.default_rng(seed_infer)

	ds_infer_full = SegyGatherPairDataset(
		input_segy_files=input_segy_files,
		target_segy_files=target_segy_files,
		transform=infer_transform,
		plan=plan,
		subset_traces=int(infer_subset_traces),
		primary_keys=primary_keys,
		valid=True,
		verbose=bool(verbose),
		max_trials=int(max_trials),
		use_header_cache=bool(use_header_cache),
	)

	infer_ds = Subset(ds_infer_full, range(int(infer_batch_size * infer_max_batches)))
	infer_loader = DataLoader(
		infer_ds,
		batch_size=int(infer_batch_size),
		shuffle=False,
		num_workers=int(infer_num_workers),
		pin_memory=(device.type == 'cuda'),
	)

	# ---------- infer (engine) + visualize (utils) ----------
	Path(vis_out_dir).mkdir(parents=True, exist_ok=True)

	tiled_cfg = TiledHConfig(
		tile_h=int(tile_h),
		overlap_h=int(overlap_h),
		tiles_per_batch=int(tiles_per_batch),
		amp=bool(amp_infer),
		use_tqdm=bool(use_tqdm),
	)

	triptych_cfg = PairTriptychVisConfig(
		cmap=cmap,
		vmin=float(vmin),
		vmax=float(vmax),
		transpose_for_trace_time=bool(transpose_for_trace_time),
		per_trace_norm=bool(per_trace_norm),
		per_trace_eps=float(per_trace_eps),
		figsize=figsize,
		dpi=int(dpi),
	)

	model.eval()
	non_blocking = bool(device.type == 'cuda')

	with torch.no_grad():
		for step, batch in enumerate(infer_loader):
			if step >= int(infer_max_batches):
				break

			x_in = batch['input'].to(device=device, non_blocking=non_blocking)
			x_tg = batch['target'].to(device=device, non_blocking=non_blocking)

			x_pr = infer_batch_tiled_h(model, x_in, cfg=tiled_cfg)

			if step < int(vis_n):
				save_pair_triptych_step_png(
					vis_out_dir,
					step=step,
					x_in_bchw=x_in.detach().cpu(),
					x_tg_bchw=x_tg.detach().cpu(),
					x_pr_bchw=x_pr.detach().cpu(),
					cfg=triptych_cfg,
					batch=batch,
				)

	ds_train_full.close()
	ds_infer_full.close()


if __name__ == '__main__':
	main()
