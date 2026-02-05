from __future__ import annotations

import torch
from seisai_engine.loss.soft_label_ce import (
	build_pixel_mask_from_batch,
	soft_label_ce_masked_mean,
)

__all__ = ['criterion']


def criterion(
	logits: torch.Tensor, target: torch.Tensor, batch: dict
) -> torch.Tensor:
	pixel_mask = build_pixel_mask_from_batch(
		batch, use_trace_valid=True, use_label_valid=True, mask_bool_key='mask_bool'
	)
	return soft_label_ce_masked_mean(logits, target, pixel_mask)
