from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from seisai_utils.config import (
	optional_bool,
	optional_float,
	optional_int,
	optional_str,
	require_dict,
	require_float,
	require_int,
)
from seisai_utils.viz_phase import make_title_from_batch_meta, save_psn_debug_png
from torch.utils.data import DataLoader, Subset

from seisai_engine.pipelines.common import (
	ensure_fixed_infer_num_workers,
	epoch_vis_dir,
	load_cfg_with_base_dir,
	make_train_worker_init_fn,
	maybe_save_best_min,
	prepare_output_dirs,
	resolve_cfg_paths,
	resolve_out_dir,
	seed_all,
	set_dataset_rng,
)
from seisai_engine.train_loop import train_one_epoch

from .build_dataset import build_dataset
from .build_model import build_model
from .loss import criterion

__all__ = ['main']

DEFAULT_CONFIG_PATH = Path('examples/config_train_psn.yaml')


def _build_dataset_for_subset(cfg: dict, subset_traces: int):
	cfg_copy = copy.deepcopy(cfg)
	train_cfg = require_dict(cfg_copy, 'train')
	train_cfg['subset_traces'] = int(subset_traces)
	return build_dataset(cfg_copy)


def _run_infer_epoch(
	*,
	model: torch.nn.Module,
	loader: DataLoader,
	device: torch.device,
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
				)

	if infer_samples <= 0:
		raise RuntimeError('no inference samples were processed')

	return infer_loss_sum / float(infer_samples)


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
	args, _unknown = parser.parse_known_args(argv)

	cfg, base_dir = load_cfg_with_base_dir(Path(args.config))
	resolve_cfg_paths(
		cfg,
		base_dir,
		keys=[
			'paths.segy_files',
			'paths.phase_pick_files',
		],
	)

	train_cfg = require_dict(cfg, 'train')
	infer_cfg = require_dict(cfg, 'infer')
	vis_cfg = require_dict(cfg, 'vis')
	ckpt_cfg = require_dict(cfg, 'ckpt')

	out_dir_path = resolve_out_dir(cfg, base_dir)

	train_batch_size = require_int(train_cfg, 'batch_size')
	epochs = require_int(train_cfg, 'epochs')
	lr = require_float(train_cfg, 'lr')
	train_subset_traces = require_int(train_cfg, 'subset_traces')
	samples_per_epoch = require_int(train_cfg, 'samples_per_epoch')
	seed_train = require_int(train_cfg, 'seed')
	max_norm = optional_float(train_cfg, 'max_norm', 1.0)
	use_amp = optional_bool(train_cfg, 'use_amp', default=True)
	train_num_workers = optional_int(train_cfg, 'num_workers', 0)

	infer_batch_size = require_int(infer_cfg, 'batch_size')
	infer_max_batches = require_int(infer_cfg, 'max_batches')
	infer_subset_traces = require_int(infer_cfg, 'subset_traces')
	seed_infer = require_int(infer_cfg, 'seed')
	infer_num_workers = require_int(infer_cfg, 'num_workers')

	ensure_fixed_infer_num_workers(infer_num_workers)

	vis_subdir = optional_str(vis_cfg, 'out_subdir', 'vis')
	vis_n = require_int(vis_cfg, 'n')

	ckpt_best_only = optional_bool(ckpt_cfg, 'save_best_only', default=True)
	ckpt_metric = optional_str(ckpt_cfg, 'metric', 'infer_loss')
	ckpt_mode = optional_str(ckpt_cfg, 'mode', 'min')
	if not ckpt_best_only:
		raise ValueError('ckpt.save_best_only must be true')
	if ckpt_metric != 'infer_loss':
		raise ValueError('ckpt.metric must be "infer_loss"')
	if ckpt_mode != 'min':
		raise ValueError('ckpt.mode must be "min"')

	if samples_per_epoch <= 0:
		raise ValueError('train.samples_per_epoch must be positive')
	if infer_max_batches <= 0:
		raise ValueError('infer.max_batches must be positive')

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	seed_all(seed_train)

	ds_train_full = _build_dataset_for_subset(cfg, train_subset_traces)
	ds_infer_full = _build_dataset_for_subset(cfg, infer_subset_traces)

	model = build_model(cfg).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))

	ckpt_dir, vis_root = prepare_output_dirs(out_dir_path, vis_subdir)

	best_infer_loss: float | None = None

	try:
		for epoch in range(int(epochs)):
			seed_epoch = int(seed_train) + int(epoch)

			if int(train_num_workers) == 0:
				set_dataset_rng(ds_train_full, seed_epoch)
				train_worker_init_fn = None
			else:
				train_worker_init_fn = make_train_worker_init_fn(seed_epoch)

			train_ds = Subset(ds_train_full, range(int(samples_per_epoch)))
			train_loader = DataLoader(
				train_ds,
				batch_size=int(train_batch_size),
				shuffle=False,
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

			set_dataset_rng(ds_infer_full, seed_infer)

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

			vis_epoch_dir = epoch_vis_dir(vis_root, epoch)

			model.eval()
			infer_loss = _run_infer_epoch(
				model=model,
				loader=infer_loader,
				device=device,
				vis_out_dir=str(vis_epoch_dir),
				vis_n=int(vis_n),
				max_batches=int(infer_max_batches),
			)
			print(f'epoch={epoch} infer_loss={infer_loss:.6f}')

			ckpt_path = ckpt_dir / 'best.pt'
			best_infer_loss = maybe_save_best_min(
				best_infer_loss,
				infer_loss,
				ckpt_path,
				{
					'epoch': int(epoch),
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'cfg': cfg,
				},
			)
	finally:
		ds_train_full.close()
		ds_infer_full.close()


if __name__ == '__main__':
	main()
