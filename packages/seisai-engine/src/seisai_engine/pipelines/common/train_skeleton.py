from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Subset

from seisai_engine.train_loop import train_one_epoch

from .skeleton_helpers import (
	ensure_fixed_infer_num_workers,
	epoch_vis_dir,
	make_train_worker_init_fn,
	maybe_save_best_min,
	prepare_output_dirs,
	set_dataset_rng,
)

InferEpochFn = Callable[
	[torch.nn.Module, DataLoader, torch.device, Path, int, int], float
]


@dataclass
class TrainSkeletonSpec:
	pipeline: str
	cfg: dict
	out_dir: Path
	vis_subdir: str
	model_sig: dict
	model: torch.nn.Module
	optimizer: torch.optim.Optimizer
	criterion: Callable[..., Any]
	ds_train_full: Any
	ds_infer_full: Any
	device: torch.device
	seed_train: int
	seed_infer: int
	epochs: int
	train_batch_size: int
	train_num_workers: int
	samples_per_epoch: int
	max_norm: float
	use_amp_train: bool
	infer_batch_size: int
	infer_num_workers: int
	infer_max_batches: int
	vis_n: int
	infer_epoch_fn: InferEpochFn
	print_freq: int = 10


def run_train_skeleton(spec: TrainSkeletonSpec) -> None:
	ckpt_dir, vis_root = prepare_output_dirs(spec.out_dir, spec.vis_subdir)

	best_infer_loss: float | None = None
	global_step = 0

	try:
		for epoch in range(int(spec.epochs)):
			seed_epoch = int(spec.seed_train) + int(epoch)

			if int(spec.train_num_workers) == 0:
				set_dataset_rng(spec.ds_train_full, seed_epoch)
				train_worker_init_fn = None
			else:
				train_worker_init_fn = make_train_worker_init_fn(seed_epoch)

			train_ds = Subset(spec.ds_train_full, range(int(spec.samples_per_epoch)))
			train_loader = DataLoader(
				train_ds,
				batch_size=int(spec.train_batch_size),
				shuffle=False,
				num_workers=int(spec.train_num_workers),
				pin_memory=(spec.device.type == 'cuda'),
				worker_init_fn=train_worker_init_fn,
			)

			spec.model.train()
			stats = train_one_epoch(
				spec.model,
				train_loader,
				spec.optimizer,
				spec.criterion,
				device=spec.device,
				lr_scheduler=None,
				gradient_accumulation_steps=1,
				max_norm=float(spec.max_norm),
				use_amp=bool(spec.use_amp_train),
				scaler=None,
				ema=None,
				step_offset=0,
				print_freq=int(spec.print_freq),
				on_step=None,
			)
			print(
				f'epoch={epoch} loss={stats["loss"]:.6f} steps={int(stats["steps"])} '
				f'samples={int(stats["samples"])}'
			)
			global_step += int(stats['steps'])

			set_dataset_rng(spec.ds_infer_full, spec.seed_infer)
			ensure_fixed_infer_num_workers(spec.infer_num_workers)

			infer_ds = Subset(
				spec.ds_infer_full,
				range(int(spec.infer_batch_size * spec.infer_max_batches)),
			)
			infer_loader = DataLoader(
				infer_ds,
				batch_size=int(spec.infer_batch_size),
				shuffle=False,
				num_workers=0,
				pin_memory=(spec.device.type == 'cuda'),
			)

			vis_epoch_dir = epoch_vis_dir(vis_root, epoch)

			spec.model.eval()
			infer_loss = spec.infer_epoch_fn(
				spec.model,
				infer_loader,
				spec.device,
				vis_epoch_dir,
				int(spec.vis_n),
				int(spec.infer_max_batches),
			)
			print(f'epoch={epoch} infer_loss={infer_loss:.6f}')

			ckpt_path = ckpt_dir / 'best.pt'
			best_infer_loss = maybe_save_best_min(
				best_infer_loss,
				infer_loss,
				ckpt_path,
				{
					'version': 1,
					'pipeline': spec.pipeline,
					'epoch': int(epoch),
					'global_step': int(global_step),
					'model_sig': spec.model_sig,
					'model_state_dict': spec.model.state_dict(),
					'optimizer_state_dict': spec.optimizer.state_dict(),
					'cfg': spec.cfg,
				},
			)
	finally:
		spec.ds_train_full.close()
		spec.ds_infer_full.close()
