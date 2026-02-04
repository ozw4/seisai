from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def soft_label_ce_map(
	logits: torch.Tensor,
	target: torch.Tensor,
	*,
	class_dim: int = 1,
) -> torch.Tensor:
	"""Soft-label cross entropy map.

	Args:
		logits: (B,C,H,W) logits (do NOT apply softmax beforehand).
		target: (B,C,H,W) probability distribution (soft labels).
		class_dim: class dimension to reduce (default: 1 for (B,C,H,W)).

	Returns:
		loss_map: logits/target summed over class_dim. For class_dim==1, shape is (B,H,W).

	"""
	if not isinstance(logits, torch.Tensor) or int(logits.ndim) != 4:
		raise ValueError('logits must be torch.Tensor with shape (B,C,H,W)')
	if not isinstance(target, torch.Tensor) or tuple(target.shape) != tuple(
		logits.shape
	):
		raise ValueError(
			'target must be torch.Tensor with same shape as logits (B,C,H,W)'
		)

	class_dim = int(class_dim)
	if class_dim < 0:
		class_dim += int(logits.ndim)
	if not (0 <= class_dim < int(logits.ndim)):
		raise ValueError(
			f'class_dim out of range: {class_dim} for ndim={int(logits.ndim)}'
		)

	if target.dtype != logits.dtype:
		target = target.to(dtype=logits.dtype)
	if target.device != logits.device:
		target = target.to(device=logits.device, non_blocking=True)

	log_p = F.log_softmax(logits, dim=class_dim)
	return -(target * log_p).sum(dim=class_dim)


def _require_bool_tensor(
	v: Any,
	*,
	name: str,
	ndim: int | None = None,
) -> torch.Tensor:
	t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
	if not isinstance(t, torch.Tensor):
		raise TypeError(f'{name} must be a torch.Tensor')
	if t.dtype is not torch.bool:
		raise TypeError(f'{name} must be bool tensor, got dtype={t.dtype}')
	if ndim is not None and int(t.ndim) != int(ndim):
		raise ValueError(f'{name} must be {ndim}D, got shape={tuple(t.shape)}')
	return t


def build_pixel_mask_from_batch(
	batch: Mapping[str, Any],
	*,
	use_trace_valid: bool = True,
	use_label_valid: bool = True,
	mask_bool_key: str = 'mask_bool',
) -> torch.Tensor:
	"""Build a (B,H,W) boolean mask by AND-ing trace/label validity and an optional mask.

	Expected keys in `batch`:
	- trace_valid: (B,H) bool (if use_trace_valid)
	- label_valid: (B,H) bool (if use_label_valid)
	- mask_bool (optional): (B,H,W) or (B,H) or (B,W) bool

	Note:
	- W is inferred from batch['target'], batch['input'], or mask_bool.
	- Returned tensor is on the same device as the source tensors (typically CPU).
	  `soft_label_ce_masked_mean` moves it to `logits.device` internally; if you use
	  the mask yourself, move it to the appropriate device.

	"""
	if not isinstance(batch, Mapping):
		raise TypeError('batch must be a Mapping[str, Any]')

	ref_b: int | None = None
	ref_h: int | None = None

	tv_bh: torch.Tensor | None = None
	if use_trace_valid:
		if 'trace_valid' not in batch:
			raise KeyError("batch['trace_valid'] is required when use_trace_valid=True")
		tv_bh = _require_bool_tensor(batch['trace_valid'], name='trace_valid', ndim=2)
		ref_b, ref_h = int(tv_bh.shape[0]), int(tv_bh.shape[1])

	lv_bh: torch.Tensor | None = None
	if use_label_valid:
		if 'label_valid' not in batch:
			raise KeyError("batch['label_valid'] is required when use_label_valid=True")
		lv_bh = _require_bool_tensor(batch['label_valid'], name='label_valid', ndim=2)
		if ref_b is None:
			ref_b, ref_h = int(lv_bh.shape[0]), int(lv_bh.shape[1])
		elif tuple(lv_bh.shape) != (ref_b, ref_h):
			raise ValueError(
				f'label_valid shape mismatch: {tuple(lv_bh.shape)} vs ({ref_b},{ref_h})'
			)

	# Infer (B,H,W) reference from target/input if present.
	w_from_batch: int | None = None
	h_from_batch: int | None = None
	b_from_batch: int | None = None
	for key in ('target', 'input'):
		v = batch.get(key)
		if isinstance(v, torch.Tensor) and int(v.ndim) == 4:
			b_from_batch = int(v.shape[0])
			h_from_batch = int(v.shape[-2])
			w_from_batch = int(v.shape[-1])
			break

	if b_from_batch is not None:
		if ref_b is None:
			ref_b, ref_h = b_from_batch, h_from_batch
		elif ref_b != b_from_batch or ref_h != h_from_batch:
			raise ValueError(
				f'batch {key} shape mismatch vs masks: B/H=({b_from_batch},{h_from_batch}) vs ({ref_b},{ref_h})'
			)

	# Optional mask_bool
	mb = batch.get(mask_bool_key, None)
	mb_bhw: torch.Tensor | None = None
	if mb is not None:
		mb_t = _require_bool_tensor(mb, name=mask_bool_key)
		if int(mb_t.ndim) == 4:
			# Explicitly allow (B,1,H,W) only.
			if int(mb_t.shape[1]) != 1:
				raise ValueError(
					f'{mask_bool_key} 4D must be (B,1,H,W); got shape={tuple(mb_t.shape)}'
				)
			mb_t = mb_t[:, 0]
		if int(mb_t.ndim) == 3:
			mb_bhw = mb_t
			if ref_b is None:
				ref_b, ref_h, w_from_batch = (
					int(mb_bhw.shape[0]),
					int(mb_bhw.shape[1]),
					int(mb_bhw.shape[2]),
				)
			elif int(mb_bhw.shape[0]) != int(ref_b) or int(mb_bhw.shape[1]) != int(
				ref_h
			):
				raise ValueError(
					f'{mask_bool_key} shape mismatch: {tuple(mb_bhw.shape)} vs ({ref_b},{ref_h},W)'
				)
		elif int(mb_t.ndim) == 2:
			if ref_b is None:
				raise ValueError(
					f'{mask_bool_key} is 2D but cannot infer H/W without trace_valid/label_valid or target/input'
				)
			if int(mb_t.shape[0]) != int(ref_b):
				raise ValueError(
					f'{mask_bool_key} batch size mismatch: {tuple(mb_t.shape)} vs B={ref_b}'
				)
			# Disambiguate (B,H) vs (B,W) using known H/W.
			if ref_h is not None and int(mb_t.shape[1]) == int(ref_h):
				# (B,H) -> (B,H,W)
				if w_from_batch is None:
					raise ValueError(
						f'{mask_bool_key} is (B,H) but W is unknown; provide batch[target]/batch[input] or (B,H,W) mask'
					)
				mb_bhw = mb_t[:, :, None].expand(
					int(ref_b), int(ref_h), int(w_from_batch)
				)
			elif w_from_batch is not None and int(mb_t.shape[1]) == int(w_from_batch):
				# (B,W) -> (B,H,W)
				if ref_h is None:
					raise ValueError('internal error: ref_h is None')
				mb_bhw = mb_t[:, None, :].expand(
					int(ref_b), int(ref_h), int(w_from_batch)
				)
			else:
				raise ValueError(
					f'{mask_bool_key} 2D must be (B,H) or (B,W); got shape={tuple(mb_t.shape)} with H={ref_h} W={w_from_batch}'
				)
		else:
			raise ValueError(
				f'{mask_bool_key} must be (B,H,W) or (B,H) or (B,W); got shape={tuple(mb_t.shape)}'
			)

	if ref_b is None or ref_h is None:
		raise ValueError(
			'cannot infer (B,H): provide trace_valid/label_valid, or target/input, or a 3D mask_bool'
		)

	W: int | None = w_from_batch
	if W is None:
		# Fall back to mask_bool if it was 3D.
		if mb_bhw is not None and int(mb_bhw.ndim) == 3:
			W = int(mb_bhw.shape[2])
	if W is None:
		raise ValueError(
			'cannot infer W: provide batch[target]/batch[input] or (B,H,W) mask_bool'
		)

	# Start with all-True, then AND masks explicitly expanded to (B,H,W).
	device = None
	for t in (tv_bh, lv_bh, mb_bhw):
		if isinstance(t, torch.Tensor):
			device = t.device
			break
	if device is None:
		device = torch.device('cpu')

	pixel_mask = torch.ones(
		(int(ref_b), int(ref_h), int(W)), dtype=torch.bool, device=device
	)

	if tv_bh is not None:
		pixel_mask &= tv_bh.to(device=device)[:, :, None].expand(
			int(ref_b), int(ref_h), int(W)
		)
	if lv_bh is not None:
		pixel_mask &= lv_bh.to(device=device)[:, :, None].expand(
			int(ref_b), int(ref_h), int(W)
		)
	if mb_bhw is not None:
		if tuple(mb_bhw.shape) != (int(ref_b), int(ref_h), int(W)):
			raise ValueError(
				f'{mask_bool_key} resolved shape mismatch: {tuple(mb_bhw.shape)} vs ({ref_b},{ref_h},{W})'
			)
		pixel_mask &= mb_bhw.to(device=device)

	return pixel_mask


def soft_label_ce_masked_mean(
	logits: torch.Tensor,
	target: torch.Tensor,
	pixel_mask: torch.Tensor,
) -> torch.Tensor:
	"""Masked mean soft-label cross entropy.

	Args:
		logits: (B,C,H,W)
		target: (B,C,H,W)
		pixel_mask: (B,H,W) bool

	Returns:
		loss: scalar tensor. If no pixels are selected, returns 0 (no NaN / no exception).

	"""
	if not isinstance(logits, torch.Tensor) or int(logits.ndim) != 4:
		raise ValueError('logits must be torch.Tensor with shape (B,C,H,W)')
	if not isinstance(target, torch.Tensor) or tuple(target.shape) != tuple(
		logits.shape
	):
		raise ValueError(
			'target must be torch.Tensor with same shape as logits (B,C,H,W)'
		)

	if not isinstance(pixel_mask, torch.Tensor):
		pixel_mask = torch.as_tensor(pixel_mask)
	if pixel_mask.dtype is not torch.bool:
		raise TypeError(f'pixel_mask must be bool tensor, got dtype={pixel_mask.dtype}')
	if int(pixel_mask.ndim) != 3:
		raise ValueError(
			f'pixel_mask must be (B,H,W), got shape={tuple(pixel_mask.shape)}'
		)

	B, _C, H, W = logits.shape
	if tuple(pixel_mask.shape) != (int(B), int(H), int(W)):
		raise ValueError(
			f'pixel_mask shape mismatch: mask={tuple(pixel_mask.shape)} vs expected={(int(B), int(H), int(W))}'
		)

	if target.dtype != logits.dtype:
		target = target.to(dtype=logits.dtype)
	if target.device != logits.device:
		target = target.to(device=logits.device, non_blocking=True)

	pixel_mask = pixel_mask.to(device=logits.device, non_blocking=True)
	denom = pixel_mask.sum(dtype=torch.float32)

	# Select only masked pixels to avoid 0*NaN propagation when masked-out entries are non-finite.
	logits_bhwc = logits.permute(0, 2, 3, 1)  # (B,H,W,C)
	target_bhwc = target.permute(0, 2, 3, 1)  # (B,H,W,C)
	logits_nc = logits_bhwc[pixel_mask]  # (N,C)
	target_nc = target_bhwc[pixel_mask]  # (N,C)
	log_p_nc = F.log_softmax(logits_nc, dim=-1)
	loss_n = -(target_nc * log_p_nc).sum(dim=-1)  # (N,)
	loss_sum = loss_n.sum(dtype=torch.float32)
	# When denom==0, logits_nc/target_nc are empty => loss_sum==0. Use denom.clamp_min(1)
	# to return an explicit 0 without introducing a GPU sync via denom.item().
	return loss_sum / denom.clamp_min(1.0)


__all__ = [
	'build_pixel_mask_from_batch',
	'soft_label_ce_map',
	'soft_label_ce_masked_mean',
]
