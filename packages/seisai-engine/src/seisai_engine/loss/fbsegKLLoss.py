import warnings
from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn.functional as F


class FbSegKLLossView:
	"""First-break セグメンテーション用 KL 損失（meta['fb_idx_view']>0 のトレースに限定）。
	IF: loss = FbSegKLLossView(tau=1.0, eps=0.0)(pred, batch, reduction='mean')
	  - pred/logits, batch['target']: (B,1,H,W)
	  - batch['meta']['fb_idx_view']: (B,H)；True(>0) のトレースのみで集約
	"""

	def __init__(self, tau: float = 1.0, eps: float = 0.0):
		assert tau > 0
		assert eps >= 0.0
		self.tau = float(tau)
		self.eps = float(eps)

	def __call__(
		self,
		pred: torch.Tensor,
		batch: Mapping[str, Any],
		*,
		reduction: Literal['mean', 'sum', 'none'] = 'mean',
	) -> torch.Tensor:
		assert isinstance(pred, torch.Tensor) and pred.ndim == 4, (
			'pred: (B,1,H,W) tensor expected'
		)
		B, C, H, W = pred.shape
		assert C == 1, 'pred channel must be 1 for fb segmentation'

		# target
		assert 'target' in batch, "batch['target'] is required"
		target = batch['target']
		assert isinstance(target, torch.Tensor) and target.shape == pred.shape, (
			'target must have same shape as pred'
		)
		if target.dtype != pred.dtype:
			target = target.to(dtype=pred.dtype)
		if target.device != pred.device:
			target = target.to(device=pred.device, non_blocking=True)

		# 対象トレース（ビュー）の選択
		meta = batch.get('meta', None)
		assert isinstance(meta, Mapping) and 'fb_idx_view' in meta, (
			"batch['meta']['fb_idx_view'] is required"
		)
		view_mask = meta['fb_idx_view']
		assert isinstance(view_mask, torch.Tensor) and view_mask.dtype in (
			torch.bool,
			torch.int32,
			torch.int64,
		), 'fb_idx_view must be bool or int tensor'
		if view_mask.dtype != torch.bool:
			view_mask = view_mask > 0
		assert view_mask.ndim == 2 and view_mask.shape == (B, H), (
			'fb_idx_view: (B,H) expected'
		)
		view_mask = view_mask.to(device=pred.device, non_blocking=True)

		# KL(q || p) を W 次元で集約 → (B,H)
		# p: pred の softmax（温度 tau）
		log_p = F.log_softmax(pred / self.tau, dim=-1)  # (B,1,H,W)
		# q: target を確率分布に正規化（epsで下駄）
		q_raw = (target + self.eps).clamp_min(0)
		q_sum = q_raw.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(q_raw.dtype).eps)
		q = q_raw / q_sum  # (B,1,H,W)
		# log q
		log_q = (q.clamp_min(torch.finfo(q.dtype).eps)).log()
		# KL per trace
		kl_map = (q * (log_q - log_p)).sum(dim=-1)  # (B,1,H)
		per_trace = kl_map.squeeze(1)  # (B,H)

		# 選択を適用
		sel_vals = per_trace[view_mask]  # (N,)
		if reduction == 'none':
			return sel_vals  # 1D
		if sel_vals.numel() == 0:
			# 既存仕様に寄せて 0 を返す（学習を止めない）
			warnings.warn(
				"FbSegKLLossView: no traces selected by meta['fb_idx_view']; returning 0 fallback.",
				category=UserWarning,
				stacklevel=2,
			)
			return torch.zeros((), dtype=per_trace.dtype, device=per_trace.device)
		if reduction == 'sum':
			return sel_vals.sum()
		return sel_vals.mean()
