"""Train and run inference on PSN phase-pick datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from seisai_dataset import (
	FirstBreakGate,
	FirstBreakGateConfig,
	SegyGatherPhasePipelineDataset,
)
from seisai_engine.pipelines.common.config_io import (
	load_config,
	resolve_cfg_paths,
	resolve_relpath,
)
from seisai_engine.pipelines.common.seed import seed_all
from seisai_engine.pipelines.psn.build_model import build_model
from seisai_engine.pipelines.psn.build_plan import build_plan
from seisai_engine.pipelines.psn.loss import criterion
from seisai_engine.train_loop import train_one_epoch
from seisai_transforms.augment import PerTraceStandardize, RandomCropOrPad, ViewCompose
from seisai_utils.config import (
	optional_bool,
	optional_float,
	optional_int,
	optional_str,
	require_dict,
	require_float,
	require_int,
	require_list_str,
)
from seisai_utils.viz_phase import make_title_from_batch_meta, save_psn_debug_png
from torch.utils.data import DataLoader, Subset, get_worker_info

__all__ = ['main']

DEFAULT_CONFIG_PATH = Path(__file__).with_name('config_train_psn.yaml')


def _validate_primary_keys(primary_keys_list: object) -> tuple[str, ...]:
	if not isinstance(primary_keys_list, list) or not all(
		isinstance(x, str) for x in primary_keys_list
	):
		raise ValueError('dataset.primary_keys must be list[str]')
	return tuple(primary_keys_list)


def _build_fbgate(fbgate_cfg: dict | None) -> FirstBreakGate:
	if fbgate_cfg is None:
		return FirstBreakGate(
			FirstBreakGateConfig(
				apply_on='off',
				min_pick_ratio=0.0,
				verbose=False,
			)
		)
	if not isinstance(fbgate_cfg, dict):
		raise TypeError('fbgate must be dict')

	apply_on = optional_str(fbgate_cfg, 'apply_on', 'off').lower()
	if apply_on == 'on':
		apply_on = 'any'
	if apply_on not in ('any', 'super_only', 'off'):
		msg = 'fbgate.apply_on must be "any", "super_only", or "off"'
		raise ValueError(msg)

	min_pick_ratio = optional_float(fbgate_cfg, 'min_pick_ratio', 0.0)
	verbose = optional_bool(fbgate_cfg, 'verbose', default=False)
	return FirstBreakGate(
		FirstBreakGateConfig(
			apply_on=apply_on,
			min_pick_ratio=float(min_pick_ratio),
			verbose=bool(verbose),
		)
	)


def _validate_files(segy_files: list[str], phase_pick_files: list[str]) -> None:
	for p in list(segy_files) + list(phase_pick_files):
		if not Path(p).is_file():
			raise FileNotFoundError(f'file not found: {p}')


def _run_infer_epoch(
	*,
	model: torch.nn.Module,
	loader: DataLoader,
	device: torch.device,
	vis_out_dir: str,
	vis_n: int,
	max_batches: int,
	transpose_for_trace_time: bool,
	dpi: int,
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

			logits = model(x_in)
			loss = criterion(logits, x_tg, batch_dev)

			bsize = int(x_in.shape[0])
			infer_loss_sum += float(loss.detach().item()) * bsize
			infer_samples += bsize

			if step < int(vis_n):
				title = make_title_from_batch_meta(batch, b=0)
				out_path = Path(vis_out_dir) / f'step_{int(step):04d}.png'
				save_psn_debug_png(
					out_path,
					x_bchw=batch['input'],
					target_b3hw=batch['target'],
					logits_b3hw=logits.detach().cpu(),
					b=0,
					title=title,
					transpose_for_trace_time=bool(transpose_for_trace_time),
					dpi=int(dpi),
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
		keys=[
			'paths.segy_files',
			'paths.phase_pick_files',
		],
	)

	paths = require_dict(cfg, 'paths')
	ds_cfg = require_dict(cfg, 'dataset')
	train_cfg = require_dict(cfg, 'train')
	infer_cfg = require_dict(cfg, 'infer')
	transform_cfg = require_dict(cfg, 'transform')
	vis_cfg = require_dict(cfg, 'vis')

	segy_files = require_list_str(paths, 'segy_files')
	phase_pick_files = require_list_str(paths, 'phase_pick_files')
	if len(segy_files) != len(phase_pick_files):
		raise ValueError(
			'paths.segy_files and paths.phase_pick_files must have same length'
		)

	out_dir = optional_str(paths, 'out_dir', './_psn_out')
	out_dir = resolve_relpath(base_dir, out_dir)

	_validate_files(segy_files, phase_pick_files)

	max_trials = optional_int(ds_cfg, 'max_trials', 2048)
	use_header_cache = optional_bool(ds_cfg, 'use_header_cache', default=True)
	verbose = optional_bool(ds_cfg, 'verbose', default=True)
	include_empty_gathers = optional_bool(
		ds_cfg, 'include_empty_gathers', default=False
	)
	secondary_key_fixed = optional_bool(
		ds_cfg, 'secondary_key_fixed', default=False
	)
	primary_keys_list = ds_cfg.get('primary_keys', ['ffid'])
	primary_keys = _validate_primary_keys(primary_keys_list)

	train_batch_size = require_int(train_cfg, 'batch_size')
	epochs = require_int(train_cfg, 'epochs')
	lr = require_float(train_cfg, 'lr')
	train_subset_traces = require_int(train_cfg, 'subset_traces')
	time_len = require_int(train_cfg, 'time_len')
	samples_per_epoch = require_int(train_cfg, 'samples_per_epoch')
	psn_sigma = optional_float(train_cfg, 'psn_sigma', 1.5)
	seed_train = require_int(train_cfg, 'seed')
	use_amp = optional_bool(train_cfg, 'use_amp', default=True)
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

	standardize_eps = optional_float(transform_cfg, 'standardize_eps', 1.0e-8)

	vis_subdir = optional_str(vis_cfg, 'out_subdir', 'vis')
	vis_n = optional_int(vis_cfg, 'n', 1)
	transpose_for_trace_time = optional_bool(
		vis_cfg, 'transpose_for_trace_time', default=True
	)
	dpi = optional_int(vis_cfg, 'dpi', 150)

	if samples_per_epoch <= 0:
		msg = 'train.samples_per_epoch must be positive'
		raise ValueError(msg)
	if infer_max_batches <= 0:
		msg = 'infer.max_batches must be positive'
		raise ValueError(msg)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	seed_all(seed_train)

	transform = ViewCompose(
		[
			RandomCropOrPad(target_len=int(time_len)),
			PerTraceStandardize(eps=float(standardize_eps)),
		]
	)
	fbgate_cfg = cfg.get('fbgate')
	fbgate = _build_fbgate(fbgate_cfg)
	plan = build_plan(psn_sigma=float(psn_sigma))

	ds_train_full = SegyGatherPhasePipelineDataset(
		segy_files=segy_files,
		phase_pick_files=phase_pick_files,
		transform=transform,
		fbgate=fbgate,
		plan=plan,
		subset_traces=int(train_subset_traces),
		include_empty_gathers=bool(include_empty_gathers),
		use_header_cache=bool(use_header_cache),
		primary_keys=primary_keys,
		secondary_key_fixed=bool(secondary_key_fixed),
		verbose=bool(verbose),
		max_trials=int(max_trials),
	)

	ds_infer_full = SegyGatherPhasePipelineDataset(
		segy_files=segy_files,
		phase_pick_files=phase_pick_files,
		transform=transform,
		fbgate=fbgate,
		plan=plan,
		subset_traces=int(infer_subset_traces),
		include_empty_gathers=bool(include_empty_gathers),
		use_header_cache=bool(use_header_cache),
		primary_keys=primary_keys,
		secondary_key_fixed=True,
		verbose=bool(verbose),
		max_trials=int(max_trials),
	)

	model = build_model(cfg).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))

	out_dir_path = Path(out_dir)
	out_dir_path.mkdir(parents=True, exist_ok=True)
	ckpt_dir = out_dir_path / 'ckpt'
	ckpt_dir.mkdir(parents=True, exist_ok=True)

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
				use_amp=bool(use_amp),
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
				vis_out_dir=str(vis_epoch_dir),
				vis_n=int(vis_n),
				max_batches=int(infer_max_batches),
				transpose_for_trace_time=bool(transpose_for_trace_time),
				dpi=int(dpi),
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
