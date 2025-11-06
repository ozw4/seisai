import math

import torch


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
	def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0, last_epoch=-1):
		self.warmup_steps = warmup_steps
		self.total_steps = total_steps
		self.eta_min = eta_min
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		step = self.last_epoch + 1
		if step < self.warmup_steps:
			return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
		if step <= self.total_steps:
			decay_step = step - self.warmup_steps
			decay_total = self.total_steps - self.warmup_steps
			return [
				self.eta_min
				+ (base_lr - self.eta_min)
				* 0.5
				* (1 + math.cos(math.pi * decay_step / decay_total))
				for base_lr in self.base_lrs
			]
		return [self.eta_min for _ in self.base_lrs]
