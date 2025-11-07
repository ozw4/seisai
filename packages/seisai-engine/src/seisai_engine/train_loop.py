# seis_engine/train_loop.py
from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
from seisai_utils.logging import MetricLogger
from torch import nn
from torch.utils.data import DataLoader

__all__ = ['train_one_epoch']


def train_one_epoch(
	model: nn.Module,
	dataloader: DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: Callable[..., torch.Tensor],
	*,
	device: torch.device,
	lr_scheduler: Any | None = None,
	gradient_accumulation_steps: int = 1,
	max_norm: float | None = 1.0,
	use_amp: bool = True,
	scaler: torch.cuda.amp.GradScaler | None = None,
	ema: Any | None = None,  # expects .update(model)
	step_offset: int = 0,
	print_freq: int = 50,
	log_header: str = 'Train',
	on_step: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, float]:
	if not isinstance(device, torch.device):
		raise TypeError('device must be a torch.device')
	if gradient_accumulation_steps <= 0:
		raise ValueError('gradient_accumulation_steps must be > 0')

	model.train()
	meter = MetricLogger(delimiter='\t')

	total_samples = 0
	step = step_offset

	device_type = device.type
	do_amp = bool(use_amp and device_type == 'cuda')
	autocast_ctx = torch.cuda.amp.autocast if do_amp else nullcontext
	local_scaler = (
		scaler
		if (do_amp and scaler is not None)
		else (torch.cuda.amp.GradScaler() if do_amp else None)
	)

	optimizer.zero_grad(set_to_none=True)

	saw_any_batch = False
	for i, batch in enumerate(
		meter.log_every(dataloader, print_freq, header=log_header)
	):
		if not isinstance(batch, dict) or (
			'input' not in batch or 'target' not in batch
		):
			raise KeyError("batch must be a dict containing 'input' and 'target'")

		saw_any_batch = True

		x = batch['input'].to(device, non_blocking=True)
		y = batch['target'].to(device, non_blocking=True)

		with autocast_ctx():
			pred = model(x)
			loss = criterion(pred, y, batch)

		if local_scaler is not None:
			local_scaler.scale(loss).backward()
		else:
			loss.backward()

		total_samples += int(x.shape[0])

		if (i + 1) % gradient_accumulation_steps == 0:
			if local_scaler is not None:
				local_scaler.unscale_(optimizer)
			if max_norm is not None:
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

			if local_scaler is not None:
				local_scaler.step(optimizer)
				local_scaler.update()
			else:
				optimizer.step()

			optimizer.zero_grad(set_to_none=True)

			if lr_scheduler is not None:
				lr_scheduler.step()

			step += 1

			lr0 = float(optimizer.param_groups[0].get('lr', 0.0))
			meter.update(loss=float(loss.detach().item()), lr=lr0)

			if ema is not None:
				ema.update(model)

			if on_step is not None:
				on_step(
					step, {'loss': float(meter.meters['loss'].global_avg), 'lr': lr0}
				)

	meter.synchronize_between_processes()

	if not saw_any_batch:
		raise ValueError('dataloader yielded no batches')

	return {
		'loss': float(meter.meters['loss'].global_avg),
		'steps': float(step - step_offset),
		'samples': float(total_samples),
	}
