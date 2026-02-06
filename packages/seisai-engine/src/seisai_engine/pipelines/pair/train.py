from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from seisai_utils.config import (
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
from seisai_utils.viz_pair import PairTriptychVisConfig

from seisai_engine.infer.runner import TiledHConfig
from seisai_engine.pipelines.common import (
	TrainSkeletonSpec,
	ensure_fixed_infer_num_workers,
	load_cfg_with_base_dir,
	resolve_cfg_paths,
	resolve_out_dir,
	run_train_skeleton,
	seed_all,
)
from seisai_engine.pipelines.common.validate_primary_keys import validate_primary_keys

from .build_dataset import (
	build_infer_transform,
	build_pair_dataset,
	build_train_transform,
)
from .build_model import build_model
from .build_plan import build_plan
from .config import PairDatasetCfg, PairModelCfg, PairPaths
from .infer import run_infer_epoch
from .loss import build_criterion

__all__ = ['main']

DEFAULT_CONFIG_PATH = Path('examples/config_train_pair.yaml')


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
	args, _unknown = parser.parse_known_args(argv)

	cfg, base_dir = load_cfg_with_base_dir(Path(args.config))
	resolve_cfg_paths(
		cfg,
		base_dir,
		keys=['paths.input_segy_files', 'paths.target_segy_files'],
	)

	paths = require_dict(cfg, 'paths')
	ds_cfg = require_dict(cfg, 'dataset')
	train_cfg = require_dict(cfg, 'train')
	infer_cfg = require_dict(cfg, 'infer')
	tile_cfg = require_dict(cfg, 'tile')
	vis_cfg = require_dict(cfg, 'vis')
	model_cfg_dict = require_dict(cfg, 'model')
	ckpt_cfg = require_dict(cfg, 'ckpt')

	input_segy_files = require_list_str(paths, 'input_segy_files')
	target_segy_files = require_list_str(paths, 'target_segy_files')
	if len(input_segy_files) != len(target_segy_files):
		msg = 'paths.input_segy_files and paths.target_segy_files must have same length'
		raise ValueError(msg)

	out_dir_path = resolve_out_dir(cfg, base_dir)

	max_trials = optional_int(ds_cfg, 'max_trials', 2048)
	use_header_cache = optional_bool(ds_cfg, 'use_header_cache', default=True)
	verbose: bool = optional_bool(ds_cfg, 'verbose', default=True)
	secondary_key_fixed = optional_bool(ds_cfg, 'secondary_key_fixed', default=False)
	primary_keys_list = ds_cfg.get('primary_keys', ['ffid'])
	primary_keys = validate_primary_keys(primary_keys_list)

	train_batch_size = require_int(train_cfg, 'batch_size')
	epochs = require_int(train_cfg, 'epochs')
	lr = require_float(train_cfg, 'lr')
	train_subset_traces = require_int(train_cfg, 'subset_traces')
	time_len = require_int(train_cfg, 'time_len')
	samples_per_epoch = require_int(train_cfg, 'samples_per_epoch')
	loss_kind = optional_str(train_cfg, 'loss_kind', 'l1').lower()
	seed_train = require_int(train_cfg, 'seed')
	use_amp_train = optional_bool(train_cfg, 'use_amp', default=True)
	max_norm = optional_float(train_cfg, 'max_norm', 1.0)
	train_num_workers = optional_int(train_cfg, 'num_workers', 0)

	seed_infer = require_int(infer_cfg, 'seed')
	infer_batch_size = require_int(infer_cfg, 'batch_size')
	infer_max_batches = require_int(infer_cfg, 'max_batches')
	infer_subset_traces = require_int(infer_cfg, 'subset_traces')
	infer_num_workers = require_int(infer_cfg, 'num_workers')

	ensure_fixed_infer_num_workers(infer_num_workers)

	tile_h = require_int(tile_cfg, 'tile_h')
	overlap_h = require_int(tile_cfg, 'overlap_h')
	tiles_per_batch = require_int(tile_cfg, 'tiles_per_batch')
	amp_infer = optional_bool(tile_cfg, 'amp', default=True)
	use_tqdm = optional_bool(tile_cfg, 'use_tqdm', default=False)

	vis_subdir = optional_str(vis_cfg, 'out_subdir', 'vis')
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

	backbone = optional_str(model_cfg_dict, 'backbone', 'resnet18')
	pretrained = optional_bool(model_cfg_dict, 'pretrained', default=False)
	in_chans = optional_int(model_cfg_dict, 'in_chans', 1)
	out_chans = optional_int(model_cfg_dict, 'out_chans', 1)

	ckpt_best_only = optional_bool(ckpt_cfg, 'save_best_only', default=True)
	ckpt_metric = optional_str(ckpt_cfg, 'metric', 'infer_loss')
	ckpt_mode = optional_str(ckpt_cfg, 'mode', 'min')
	if not ckpt_best_only:
		raise ValueError('ckpt.save_best_only must be true')
	if ckpt_metric != 'infer_loss':
		raise ValueError('ckpt.metric must be "infer_loss"')
	if ckpt_mode != 'min':
		raise ValueError('ckpt.mode must be "min"')

	if loss_kind not in ('l1', 'mse'):
		raise ValueError('train.loss_kind must be "l1" or "mse"')

	if tile_h > infer_subset_traces:
		raise ValueError('tile.tile_h must be <= infer.subset_traces')
	if samples_per_epoch <= 0:
		raise ValueError('train.samples_per_epoch must be positive')
	if infer_max_batches <= 0:
		raise ValueError('infer.max_batches must be positive')

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	seed_all(seed_train)

	train_transform = build_train_transform(int(time_len))
	infer_transform = build_infer_transform()

	plan = build_plan()
	criterion = build_criterion(loss_kind)

	paths_cfg = PairPaths(
		input_segy_files=list(input_segy_files),
		target_segy_files=list(target_segy_files),
		out_dir=str(out_dir_path),
	)
	dataset_cfg = PairDatasetCfg(
		max_trials=int(max_trials),
		use_header_cache=bool(use_header_cache),
		verbose=bool(verbose),
		primary_keys=primary_keys,
	)

	ds_train_full = build_pair_dataset(
		paths=paths_cfg,
		ds_cfg=dataset_cfg,
		transform=train_transform,
		plan=plan,
		subset_traces=int(train_subset_traces),
		secondary_key_fixed=bool(secondary_key_fixed),
	)

	ds_infer_full = build_pair_dataset(
		paths=paths_cfg,
		ds_cfg=dataset_cfg,
		transform=infer_transform,
		plan=plan,
		subset_traces=int(infer_subset_traces),
		secondary_key_fixed=True,
	)

	model_cfg = PairModelCfg(
		backbone=str(backbone),
		pretrained=bool(pretrained),
		in_chans=int(in_chans),
		out_chans=int(out_chans),
	)
	model_sig = asdict(model_cfg)
	model = build_model(model_cfg).to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))

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

	infer_epoch_fn = (
		lambda model,
		loader,
		device,
		vis_epoch_dir,
		vis_n,
		max_batches: run_infer_epoch(
			model=model,
			loader=loader,
			device=device,
			criterion=criterion,
			tiled_cfg=tiled_cfg,
			vis_cfg=triptych_cfg,
			vis_out_dir=str(vis_epoch_dir),
			vis_n=vis_n,
			max_batches=max_batches,
		)
	)

	spec = TrainSkeletonSpec(
		pipeline='pair',
		cfg=cfg,
		out_dir=out_dir_path,
		vis_subdir=str(vis_subdir),
		model_sig=model_sig,
		model=model,
		optimizer=optimizer,
		criterion=criterion,
		ds_train_full=ds_train_full,
		ds_infer_full=ds_infer_full,
		device=device,
		seed_train=int(seed_train),
		seed_infer=int(seed_infer),
		epochs=int(epochs),
		train_batch_size=int(train_batch_size),
		train_num_workers=int(train_num_workers),
		samples_per_epoch=int(samples_per_epoch),
		max_norm=float(max_norm),
		use_amp_train=bool(use_amp_train),
		infer_batch_size=int(infer_batch_size),
		infer_num_workers=int(infer_num_workers),
		infer_max_batches=int(infer_max_batches),
		vis_n=int(vis_n),
		infer_epoch_fn=infer_epoch_fn,
		print_freq=10,
	)

	run_train_skeleton(spec)


if __name__ == '__main__':
	main()
