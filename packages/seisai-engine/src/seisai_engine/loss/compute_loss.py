from collections.abc import Mapping
from typing import Any

import torch


def compute_loss(
	pred: torch.Tensor,
	batch: Mapping[str, Any],
	criterion: torch.nn.Module,
) -> torch.Tensor:
	device = pred.device
	target = batch['target'].to(device, non_blocking=True).to(dtype=pred.dtype)
	assert pred.shape == target.shape, 'shape mismatch: pred vs target'

	loss = criterion(batch)
	return loss
