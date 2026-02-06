# %%
"""Train and run tiled-h inference on paired SEG-Y gathers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from seisai_dataset import BuildPlan, SegyGatherPairDataset
from seisai_dataset.builder.builder import IdentitySignal, SelectStack
from seisai_engine.infer.runner import TiledHConfig, infer_batch_tiled_h
from seisai_engine.loss.pixelwise_loss import build_criterion
from seisai_engine.pipelines.common.config_io import (
	load_config,
	resolve_cfg_paths,
	resolve_relpath,
)
from seisai_engine.pipelines.common.seed import seed_all
from seisai_engine.train_loop import train_one_epoch
from seisai_models.models.encdec2d import EncDec2D
from seisai_transforms.augment import PerTraceStandardize, RandomCropOrPad, ViewCompose
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
from seisai_utils.viz_pair import PairTriptychVisConfig, save_pair_triptych_step_png
from torch.utils.data import DataLoader, Subset, get_worker_info

__all__ = ['main']

DEFAULT_CONFIG_PATH = Path(__file__).with_name('config_train_pair.yaml')


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


def _run_infer_epoch(
	*,
	model: torch.nn.Module,
	loader: DataLoader,
	device: torch.device,
	criterion,
	tiled_cfg: TiledHConfig,
	vis_cfg: PairTriptychVisConfig,
	vis_out_dir: str,
	vis_n: int,
	max_batches: int,
) -> float:
	non_blocking = bool(device.type == 'cuda')
	infer_loss_sum = 0.0
	infer_samples = 0

	Path(vis_out_dir).mkdir(parents=True, exist_ok=True)

	with torch.no_grad():
		for step, batch in enumerate(loader):
			if step >= int(max_batches):
				break

			batch_dev = {
				k: (
					v.to(device=device, non_blocking=non_blocking)
					if torch.is_tensor(v)
					else v
				)
				for k, v in batch.items()
			}
			x_in = batch_dev['input']
			x_tg = batch_dev['target']

			x_pr = infer_batch_tiled_h(model, x_in, cfg=tiled_cfg)
			loss = criterion(x_pr, x_tg, batch_dev)

			bsize = int(x_in.shape[0])
			infer_loss_sum += float(loss.detach().item()) * bsize
			infer_samples += bsize

			if step < int(vis_n):
				save_pair_triptych_step_png(
					vis_out_dir,
					step=step,
					x_in_bchw=x_in.detach().cpu(),
					x_tg_bchw=x_tg.detach().cpu(),
					x_pr_bchw=x_pr.detach().cpu(),
					cfg=vis_cfg,
					batch=batch,
					prefix='step_',
				)

	if infer_samples <= 0:
		msg = 'no inference samples were processed'
		raise RuntimeError(msg)

	return infer_loss_sum / float(infer_samples)


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
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
		keys=['paths.input_segy_files', 'paths.target_segy_files'],
	)

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
		msg = 'paths.input_segy_files and paths.target_segy_files must have same length'
		raise ValueError(msg)

	out_dir = optional_str(paths, 'out_dir', './_pair_out')
	out_dir = resolve_relpath(base_dir, out_dir)

	max_trials = optional_int(ds_cfg, 'max_trials', 2048)
	use_header_cache = optional_bool(ds_cfg, 'use_header_cache', default=True)
	verbose: bool = optional_bool(ds_cfg, 'verbose', default=True)
	secondary_key_fixed = optional_bool(ds_cfg, 'secondary_key_fixed', default=False)
	primary_keys_list = ds_cfg.get('primary_keys', ['ffid'])
	if not isinstance(primary_keys_list, list) or not all(
		isinstance(x, str) for x in primary_keys_list
	):
		msg = 'dataset.primary_keys must be list[str]'
		raise ValueError(msg)
	primary_keys = tuple(primary_keys_list)

	train_batch_size = require_int(train_cfg, 'batch_size')
	epochs = require_int(train_cfg, 'epochs')
	lr = require_float(train_cfg, 'lr')
	subset_traces = require_int(train_cfg, 'subset_traces')
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

	if int(infer_num_workers) != 0:
		msg = (
			'infer.num_workers must be 0 to keep fixed inference samples '
			'(set infer.num_workers: 0)'
		)
		raise ValueError(msg)

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

	backbone = optional_str(model_cfg, 'backbone', 'resnet18')
	pretrained = optional_bool(model_cfg, 'pretrained', default=False)
	in_chans = optional_int(model_cfg, 'in_chans', 1)
	out_chans = optional_int(model_cfg, 'out_chans', 1)

	if loss_kind not in ('l1', 'mse'):
		msg = 'train.loss_kind must be "l1" or "mse"'
		raise ValueError(msg)

	if tile_h > infer_subset_traces:
		msg = 'tile.tile_h must be <= infer.subset_traces'
		raise ValueError(msg)
	if samples_per_epoch <= 0:
		msg = 'train.samples_per_epoch must be positive'
		raise ValueError(msg)
	if infer_max_batches <= 0:
		msg = 'infer.max_batches must be positive'
		raise ValueError(msg)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	seed_all(seed_train)

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
		secondary_key_fixed=bool(secondary_key_fixed),
		verbose=bool(verbose),
		max_trials=int(max_trials),
		use_header_cache=bool(use_header_cache),
	)

	ds_infer_full = SegyGatherPairDataset(
		input_segy_files=input_segy_files,
		target_segy_files=target_segy_files,
		transform=infer_transform,
		plan=plan,
		subset_traces=int(infer_subset_traces),
		primary_keys=primary_keys,
		secondary_key_fixed=True,
		verbose=bool(verbose),
		max_trials=int(max_trials),
		use_header_cache=bool(use_header_cache),
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

	out_dir_path = Path(out_dir)
	out_dir_path.mkdir(parents=True, exist_ok=True)
	ckpt_dir = out_dir_path / 'ckpt'
	ckpt_dir.mkdir(parents=True, exist_ok=True)

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
				shuffle=True,
				num_workers=int(train_num_workers),
				pin_memory=(device.type == 'cuda'),
				worker_init_fn=train_worker_init_fn,
			)

			model.train()
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
				f'epoch={epoch} loss={stats["loss"]:.6f} steps={int(stats["steps"])} '
				f'samples={int(stats["samples"])}'
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
			infer_loss = _run_infer_epoch(
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

# %%
