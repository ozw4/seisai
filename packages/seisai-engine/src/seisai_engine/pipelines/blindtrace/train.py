from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from seisai_engine.infer.runner import TiledHConfig
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
	optional_tuple2_float,
	require_dict,
	require_float,
	require_int,
	require_list_str,
)
from torch.utils.data import DataLoader, Subset, get_worker_info

from .build_dataset import build_dataset, build_fbgate, build_transform, validate_primary_keys
from .build_model import build_model
from .build_plan import build_plan
from .infer import run_infer_epoch
from .loss import build_masked_criterion
from .vis import build_triptych_cfg

__all__ = ['main']

DEFAULT_CONFIG_PATH = Path('examples/config_train_blindtrace.yaml')


def _validate_mask_ratio_for_subset(
	*, mask_ratio: float, subset_traces: int, label: str
) -> None:
	masked = round(float(mask_ratio) * int(subset_traces))
	if masked < 1:
		msg = (
			f'{label}: round(mask.ratio * subset_traces) must be >= 1 for masked_only '
			f'(ratio={float(mask_ratio)}, subset_traces={int(subset_traces)})'
		)
		raise ValueError(msg)


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
	args = parser.parse_args(argv)

	cfg = load_config(args.config)

	base_dir = Path(args.config).expanduser().resolve().parent
	resolve_cfg_paths(
		cfg,
		base_dir,
		keys=['paths.segy_files', 'paths.fb_files'],
	)

	paths = require_dict(cfg, 'paths')
	ds_cfg = require_dict(cfg, 'dataset')
	transform_cfg = require_dict(cfg, 'transform')
	fbgate_cfg = require_dict(cfg, 'fbgate')
	mask_cfg = require_dict(cfg, 'mask')
	input_cfg = require_dict(cfg, 'input')
	train_cfg = require_dict(cfg, 'train')
	infer_cfg = require_dict(cfg, 'infer')
	tile_cfg = require_dict(cfg, 'tile')
	vis_cfg = require_dict(cfg, 'vis')
	model_cfg = require_dict(cfg, 'model')
	ckpt_cfg = require_dict(cfg, 'ckpt')

	segy_files = require_list_str(paths, 'segy_files')
	fb_files = require_list_str(paths, 'fb_files')
	out_dir = optional_str(paths, 'out_dir', './_blindtrace_out')
	out_dir = resolve_relpath(base_dir, out_dir)

	max_trials = optional_int(ds_cfg, 'max_trials', 2048)
	use_header_cache = optional_bool(ds_cfg, 'use_header_cache', default=True)
	verbose = optional_bool(ds_cfg, 'verbose', default=True)
	primary_keys_list = ds_cfg.get('primary_keys', ['ffid'])
	primary_keys = validate_primary_keys(primary_keys_list)

	time_len = require_int(transform_cfg, 'time_len')
	per_trace_standardize = optional_bool(
		transform_cfg, 'per_trace_standardize', default=True
	)

	apply_on = optional_str(fbgate_cfg, 'apply_on', 'on')
	min_pick_ratio = optional_float(fbgate_cfg, 'min_pick_ratio', 0.0)

	mask_ratio = require_float(mask_cfg, 'ratio')
	mask_mode = optional_str(mask_cfg, 'mode', 'replace').lower()
	noise_std = optional_float(mask_cfg, 'noise_std', 1.0)
	if mask_mode not in ('replace', 'add'):
		msg = 'mask.mode must be "replace" or "add"'
		raise ValueError(msg)

	use_offset_ch = optional_bool(input_cfg, 'use_offset_ch', default=False)
	offset_normalize = optional_bool(input_cfg, 'offset_normalize', default=True)
	use_time_ch = optional_bool(input_cfg, 'use_time_ch', default=False)

	seed_train = optional_int(train_cfg, 'seed', 42)
	loss_scope = optional_str(train_cfg, 'loss_scope', 'masked_only').lower()
	loss_kind = optional_str(train_cfg, 'loss_kind', 'l1').lower()
	shift_max = optional_int(train_cfg, 'shift_max', 8)
	train_batch_size = require_int(train_cfg, 'batch_size')
	train_num_workers = optional_int(train_cfg, 'num_workers', 0)
	train_amp = optional_bool(train_cfg, 'amp', default=True)
	max_norm = optional_float(train_cfg, 'max_norm', 1.0)
	lr = require_float(train_cfg, 'lr')
	weight_decay = require_float(train_cfg, 'weight_decay')
	epochs = require_int(train_cfg, 'epochs')
	samples_per_epoch = require_int(train_cfg, 'samples_per_epoch')
	train_subset_traces = require_int(train_cfg, 'subset_traces')

	seed_infer = optional_int(infer_cfg, 'seed', 43)
	infer_batch_size = require_int(infer_cfg, 'batch_size')
	infer_num_workers = optional_int(infer_cfg, 'num_workers', 0)
	infer_max_batches = require_int(infer_cfg, 'max_batches')
	infer_subset_traces = require_int(infer_cfg, 'subset_traces')

	if int(infer_num_workers) != 0:
		msg = (
			'infer.num_workers must be 0 to keep fixed inference samples '
			'(set infer.num_workers: 0)'
		)
		raise ValueError(msg)
	tile_h = require_int(tile_cfg, 'tile_h')
	overlap_h = require_int(tile_cfg, 'overlap_h')
	tiles_per_batch = require_int(tile_cfg, 'tiles_per_batch')
	tile_amp = optional_bool(tile_cfg, 'amp', default=True)
	use_tqdm = optional_bool(tile_cfg, 'use_tqdm', default=False)

	vis_n = require_int(vis_cfg, 'n')
	vis_subdir = optional_str(vis_cfg, 'out_subdir', 'vis')
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

	ckpt_best_only = optional_bool(ckpt_cfg, 'save_best_only', default=True)
	ckpt_metric = optional_str(ckpt_cfg, 'metric', 'infer_loss')
	ckpt_mode = optional_str(ckpt_cfg, 'mode', 'min')
	if not ckpt_best_only:
		msg = 'ckpt.save_best_only must be true'
		raise ValueError(msg)
	if ckpt_metric != 'infer_loss':
		msg = 'ckpt.metric must be "infer_loss"'
		raise ValueError(msg)
	if ckpt_mode != 'min':
		msg = 'ckpt.mode must be "min"'
		raise ValueError(msg)

	if loss_kind not in ('l1', 'mse', 'shift_mse', 'shift_robust_mse'):
		msg = 'train.loss_kind must be "l1", "mse", "shift_mse", or "shift_robust_mse"'
		raise ValueError(msg)

	if loss_scope == 'masked_only':
		_validate_mask_ratio_for_subset(
			mask_ratio=mask_ratio, subset_traces=train_subset_traces, label='train'
		)
		_validate_mask_ratio_for_subset(
			mask_ratio=mask_ratio, subset_traces=infer_subset_traces, label='infer'
		)

	if tile_h > infer_subset_traces:
		msg = 'tile.tile_h must be <= infer.subset_traces'
		raise ValueError(msg)
	if samples_per_epoch <= 0:
		msg = 'train.samples_per_epoch must be positive'
		raise ValueError(msg)
	if infer_max_batches <= 0:
		msg = 'infer.max_batches must be positive'
		raise ValueError(msg)

	in_chans = 1 + int(bool(use_offset_ch)) + int(bool(use_time_ch))
	out_chans = 1

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	seed_all(seed_train)

	transform = build_transform(
		time_len=int(time_len), per_trace_standardize=bool(per_trace_standardize)
	)
	plan = build_plan(
		mask_ratio=mask_ratio,
		mask_mode=mask_mode,
		noise_std=noise_std,
		use_offset_ch=bool(use_offset_ch),
		offset_normalize=bool(offset_normalize),
		use_time_ch=bool(use_time_ch),
	)
	fbgate = build_fbgate(
		apply_on=apply_on, min_pick_ratio=min_pick_ratio, verbose=bool(verbose)
	)

	criterion = build_masked_criterion(
		loss_kind=loss_kind,
		loss_scope=loss_scope,
		shift_max=int(shift_max),
	)

	ds_train_full = build_dataset(
		segy_files=segy_files,
		fb_files=fb_files,
		transform=transform,
		fbgate=fbgate,
		plan=plan,
		subset_traces=int(train_subset_traces),
		primary_keys=primary_keys,
		valid=False,
		verbose=bool(verbose),
		max_trials=int(max_trials),
		use_header_cache=bool(use_header_cache),
	)

	ds_infer_full = build_dataset(
		segy_files=segy_files,
		fb_files=fb_files,
		transform=transform,
		fbgate=fbgate,
		plan=plan,
		subset_traces=int(infer_subset_traces),
		primary_keys=primary_keys,
		valid=True,
		verbose=bool(verbose),
		max_trials=int(max_trials),
		use_header_cache=bool(use_header_cache),
	)

	model = build_model(
		backbone=backbone,
		in_chans=int(in_chans),
		out_chans=int(out_chans),
		pretrained=bool(pretrained),
	)
	model.to(device)

	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=float(lr),
		weight_decay=float(weight_decay),
	)

	out_dir_path = Path(out_dir)
	out_dir_path.mkdir(parents=True, exist_ok=True)
	ckpt_dir = out_dir_path / 'ckpt'
	ckpt_dir.mkdir(parents=True, exist_ok=True)

	tiled_cfg = TiledHConfig(
		tile_h=int(tile_h),
		overlap_h=int(overlap_h),
		tiles_per_batch=int(tiles_per_batch),
		amp=bool(tile_amp),
		use_tqdm=bool(use_tqdm),
	)

	triptych_cfg = build_triptych_cfg(
		cmap=cmap,
		vmin=float(vmin),
		vmax=float(vmax),
		transpose_for_trace_time=bool(transpose_for_trace_time),
		per_trace_norm=bool(per_trace_norm),
		per_trace_eps=float(per_trace_eps),
		figsize=figsize,
		dpi=int(dpi),
	)

	best_infer_loss: float | None = None

	try:
		for epoch in range(int(epochs)):
			seed_epoch = int(seed_train) + int(epoch)

			if int(train_num_workers) == 0:
				ds_train_full._rng = np.random.default_rng(seed_epoch)
				train_worker_init_fn = None
			else:

				def _train_worker_init_fn(worker_id: int) -> None:
					info = get_worker_info()
					if info is None:
						msg = 'get_worker_info() returned None in worker'
						raise RuntimeError(msg)

					ds = info.dataset
					base = ds.dataset if isinstance(ds, Subset) else ds
					seed_worker = int(seed_epoch) + int(worker_id) * 1000

					base._rng = np.random.default_rng(seed_worker)
					np.random.seed(seed_worker)
					torch.manual_seed(seed_worker)

				train_worker_init_fn = _train_worker_init_fn

			train_ds = Subset(ds_train_full, range(int(samples_per_epoch)))
			train_loader = DataLoader(
				train_ds,
				batch_size=int(train_batch_size),
				shuffle=False,
				num_workers=int(train_num_workers),
				pin_memory=(device.type == 'cuda'),
				worker_init_fn=train_worker_init_fn,
			)

			stats = train_one_epoch(
				model,
				train_loader,
				optimizer,
				criterion,
				device=device,
				lr_scheduler=None,
				gradient_accumulation_steps=1,
				max_norm=float(max_norm),
				use_amp=bool(train_amp),
				scaler=None,
				ema=None,
				step_offset=0,
				print_freq=10,
				on_step=None,
			)
			print(
				f'epoch={epoch} train_loss={stats["loss"]:.6f} '
				f'steps={int(stats["steps"])} samples={int(stats["samples"])}'
			)

			ds_infer_full._rng = np.random.default_rng(int(seed_infer))

			infer_ds = Subset(
				ds_infer_full, range(int(infer_batch_size * infer_max_batches))
			)
			infer_loader = DataLoader(
				infer_ds,
				batch_size=int(infer_batch_size),
				shuffle=False,
				num_workers=0,
				pin_memory=(device.type == 'cuda'),
			)

			vis_epoch_dir = out_dir_path / str(vis_subdir) / f'epoch_{epoch:04d}'
			vis_epoch_dir.mkdir(parents=True, exist_ok=True)

			model.eval()

			infer_loss = run_infer_epoch(
				model=model,
				loader=infer_loader,
				device=device,
				criterion=criterion,
				tiled_cfg=tiled_cfg,
				vis_cfg=triptych_cfg,
				vis_out_dir=str(vis_epoch_dir),
				vis_n=int(vis_n),
				max_batches=int(infer_max_batches),
			)
			print(f'epoch={epoch} infer_loss={infer_loss:.6f}')

			is_best = best_infer_loss is None or infer_loss < best_infer_loss
			if is_best:
				best_infer_loss = float(infer_loss)
				ckpt_path = ckpt_dir / 'best.pt'
				torch.save(
					{
						'epoch': int(epoch),
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						'cfg': cfg,
					},
					ckpt_path,
				)
	finally:
		ds_train_full.close()
		ds_infer_full.close()


if __name__ == '__main__':
	main()
