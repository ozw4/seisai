from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def pick_argmax_w(prob_bhw: torch.Tensor) -> torch.Tensor:
	"""Pick W-index by argmax for each (B,H).

	Args:
		prob_bhw: (B,H,W) tensor

	Returns:
		idx_bh: (B,H) int64 indices along W
	"""
	if not isinstance(prob_bhw, torch.Tensor) or int(prob_bhw.ndim) != 3:
		raise ValueError('prob_bhw must be torch.Tensor with shape (B,H,W)')
	return prob_bhw.argmax(dim=-1).to(dtype=torch.int64)


def masked_abs_error_1d(
	pred_idx_bh: torch.Tensor,
	gt_idx_bh: torch.Tensor,
	valid_bh: torch.Tensor,
) -> torch.Tensor:
	"""Absolute index error extracted by a (B,H) boolean mask.

	Args:
		pred_idx_bh: (B,H) integer tensor
		gt_idx_bh: (B,H) integer tensor (no-GT values should be filtered by valid_bh)
		valid_bh: (B,H) bool tensor

	Returns:
		err_1d: (N,) int64 absolute errors
	"""
	if not isinstance(pred_idx_bh, torch.Tensor) or int(pred_idx_bh.ndim) != 2:
		raise ValueError('pred_idx_bh must be torch.Tensor with shape (B,H)')
	if not isinstance(gt_idx_bh, torch.Tensor) or int(gt_idx_bh.ndim) != 2:
		raise ValueError('gt_idx_bh must be torch.Tensor with shape (B,H)')
	if not isinstance(valid_bh, torch.Tensor) or int(valid_bh.ndim) != 2:
		raise ValueError('valid_bh must be torch.Tensor with shape (B,H)')
	if valid_bh.dtype is not torch.bool:
		raise TypeError(f'valid_bh must be bool tensor, got dtype={valid_bh.dtype}')
	if tuple(pred_idx_bh.shape) != tuple(gt_idx_bh.shape) or tuple(pred_idx_bh.shape) != tuple(
		valid_bh.shape
	):
		raise ValueError(
			f'shape mismatch: pred={tuple(pred_idx_bh.shape)} gt={tuple(gt_idx_bh.shape)} valid={tuple(valid_bh.shape)}'
		)

	pred = pred_idx_bh.to(dtype=torch.int64)
	gt = gt_idx_bh.to(dtype=torch.int64, device=pred.device, non_blocking=True)
	valid = valid_bh.to(device=pred.device, non_blocking=True)

	err = (pred - gt).abs()
	return err[valid]


def summarize_abs_error(
	err_1d: torch.Tensor,
	*,
	thresholds: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
	"""Summarize absolute errors.

	Returns:
		- mean: float
		- median: float
		- p_le_{thr}: float for each threshold

	Notes:
		- If N==0, all metrics are NaN (to clearly indicate "no eval targets").
	"""
	if not isinstance(err_1d, torch.Tensor):
		err_1d = torch.as_tensor(err_1d)
	if int(err_1d.ndim) != 1:
		raise ValueError(f'err_1d must be 1D, got shape={tuple(err_1d.shape)}')

	n = int(err_1d.numel())
	if n == 0:
		out = {'mean': float('nan'), 'median': float('nan')}
		for thr in thresholds:
			out[f'p_le_{int(thr)}'] = float('nan')
		return out

	err_f = err_1d.to(dtype=torch.float32)
	out = {
		'mean': float(err_f.mean().item()),
		'median': float(err_f.median().item()),
	}
	for thr in thresholds:
		thr_i = int(thr)
		out[f'p_le_{thr_i}'] = float((err_f <= float(thr_i)).float().mean().item())
	return out


def _as_tensor_bh(x: Any, *, name: str, device: torch.device) -> torch.Tensor:
	if isinstance(x, torch.Tensor):
		t = x
	else:
		t = torch.as_tensor(x)
	if int(t.ndim) != 2:
		raise ValueError(f'{name} must be (B,H), got shape={tuple(t.shape)}')
	return t.to(device=device, non_blocking=True)


def compute_ps_metrics_from_batch(
	logits: torch.Tensor,
	batch: Mapping[str, Any],
	*,
	thresholds: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
	"""Compute P/S pick metrics from a training batch.

	Expected:
	- logits: (B,3,H,W) (no softmax applied)
	- batch['meta']['p_idx_view'], batch['meta']['s_idx_view']: (B,H) int tensors
	- batch['trace_valid'], batch['label_valid']: (B,H) bool

	Returns:
		Flattened dict with prefixes:
		- p_mean, p_median, p_p_le_5, ...
		- s_mean, s_median, s_p_le_5, ...
	"""
	if not isinstance(logits, torch.Tensor) or int(logits.ndim) != 4:
		raise ValueError('logits must be torch.Tensor with shape (B,C,H,W)')
	if int(logits.shape[1]) < 2:
		raise ValueError(f'logits must have at least 2 channels for P/S, got C={int(logits.shape[1])}')
	if not isinstance(batch, Mapping):
		raise TypeError('batch must be a Mapping[str, Any]')
	if 'meta' not in batch:
		raise KeyError("batch['meta'] is required")
	if 'trace_valid' not in batch or 'label_valid' not in batch:
		raise KeyError("batch must contain 'trace_valid' and 'label_valid'")

	meta = batch['meta']
	if not isinstance(meta, Mapping):
		raise TypeError("batch['meta'] must be a Mapping")
	if 'p_idx_view' not in meta or 's_idx_view' not in meta:
		raise KeyError("batch['meta'] must contain 'p_idx_view' and 's_idx_view'")

	device = logits.device
	trace_valid = _as_tensor_bh(batch['trace_valid'], name='trace_valid', device=device)
	label_valid = _as_tensor_bh(batch['label_valid'], name='label_valid', device=device)
	if trace_valid.dtype is not torch.bool:
		raise TypeError('trace_valid must be bool')
	if label_valid.dtype is not torch.bool:
		raise TypeError('label_valid must be bool')

	p_gt = _as_tensor_bh(meta['p_idx_view'], name="meta['p_idx_view']", device=device).to(
		dtype=torch.int64
	)
	s_gt = _as_tensor_bh(meta['s_idx_view'], name="meta['s_idx_view']", device=device).to(
		dtype=torch.int64
	)

	probs = torch.softmax(logits, dim=1)
	p_prob = probs[:, 0]
	s_prob = probs[:, 1]
	p_pred = pick_argmax_w(p_prob)
	s_pred = pick_argmax_w(s_prob)

	base_valid = trace_valid & label_valid
	p_valid = base_valid & (p_gt > 0)
	s_valid = base_valid & (s_gt > 0)

	p_err = masked_abs_error_1d(p_pred, p_gt, p_valid)
	s_err = masked_abs_error_1d(s_pred, s_gt, s_valid)

	p_sum = summarize_abs_error(p_err, thresholds=thresholds)
	s_sum = summarize_abs_error(s_err, thresholds=thresholds)

	out: dict[str, float] = {}
	for k, v in p_sum.items():
		out[f'p_{k}'] = float(v)
	for k, v in s_sum.items():
		out[f's_{k}'] = float(v)
	return out


__all__ = [
	'pick_argmax_w',
	'masked_abs_error_1d',
	'summarize_abs_error',
	'compute_ps_metrics_from_batch',
]

