from collections.abc import Mapping
from typing import Any

import torch


def compute_loss(
	pred: torch.Tensor,
	batch: Mapping[str, Any],
	criterion: torch.nn.Module,
) -> torch.Tensor:
	"""Standard MSE。
	- masked_only=True: batch['mask_bool']==True の画素のみで平均MSE
	- masked_only=False: マスク無視（全画素で標準MSE）
	前提:
	  - batch['target'] は pred と同形状
	  - batch['mask_bool'] は pred と同形状の bool（masked_only=True のとき必須）
	"""
	device = pred.device
	target = batch['target'].to(device, non_blocking=True).to(dtype=pred.dtype)
	assert pred.shape == target.shape, 'shape mismatch: pred vs target'

	loss = criterion(batch)
	return sel.mean()
