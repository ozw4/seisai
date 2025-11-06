# seis_engine/train_loop.py
from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
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
) -> dict[str, float]:
	if gradient_accumulation_steps <= 0:
		raise ValueError('gradient_accumulation_steps must be > 0')

	model.train()
	total_samples = 0
	step = step_offset

	device_type = getattr(device, 'type', None)
	if device_type is None:
		raise TypeError('device must be a torch.device')

	do_amp = use_amp and device_type == 'cuda'
	autocast_ctx = torch.cuda.amp.autocast if do_amp else nullcontext

	local_scaler = None
	if do_amp:
		local_scaler = scaler if scaler is not None else torch.cuda.amp.GradScaler()

	optimizer.zero_grad(set_to_none=True)

	for i, batch in enumerate(dataloader):
		x = batch['input']
		y = batch['target']

		x = x.to(device, non_blocking=True)
		y = y.to(device, non_blocking=True)

		with autocast_ctx():
			pred = model(x)
			# pred = postprocess(x)
			loss = criterion(pred, y, batch)

		if local_scaler is not None:
			local_scaler.scale(loss).backward()
		else:
			loss.backward()

		batch_size = int(x.shape[0])
		total_samples += batch_size

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

			if ema is not None:
				ema.update(model)

			optimizer.zero_grad(set_to_none=True)

			if lr_scheduler is not None:
				lr_scheduler.step()

			step += 1
			if on_step is not None:
				on_step(step, {'loss': meter_loss.avg})

	return {
		'loss': meter_loss.avg,
		'steps': float(step - step_offset),
		'samples': float(total_samples),
	}
