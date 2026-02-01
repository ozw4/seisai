# fbsegKLLoss.py
import warnings
from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn.functional as F

Reduction = Literal['mean', 'sum', 'none']


class FbSegKLLossView:
	"""First-break セグメンテーション用 KL 損失。
	- 任意チャネル C に対応（各チャネルで W 方向に正規化して KL を計算し、C 次元で平均）
	- meta['fb_idx_view']>0 の (B,H) トレースのみで集約

	IF:
	  loss = FbSegKLLossView(tau=1.0, eps=0.0)(pred, batch, reduction='mean')

	Args:
	  pred / batch['target']: (B,C,H,W) の同形状テンソル
	  batch['meta']['fb_idx_view']: (B,H)；True(>0) のトレースを採用

	Notes:
	  - target は各 (b,c,h, :) で非負、W 方向に正規化されていなくてもよい（内部で正規化）
	  - C>1 の場合は、各チャネルの KL を平均して (B,H) に集約する

	"""

	def __init__(self, tau: float = 1.0, eps: float = 0.0):
		assert tau > 0.0 and eps >= 0.0
		self.tau = float(tau)
		self.eps = float(eps)

	def __call__(
		self,
		pred: torch.Tensor,
		batch: Mapping[str, Any],
		*,
		reduction: Reduction = 'mean',
	) -> torch.Tensor:
		assert isinstance(pred, torch.Tensor) and pred.ndim == 4, 'pred: (B,C,H,W)'
		assert isinstance(batch, Mapping)
		assert 'target' in batch, "batch['target'] is required"
		target = batch['target']
		assert isinstance(target, torch.Tensor), 'target must be Tensor'
		assert target.shape == pred.shape, 'target must have same shape as pred'

		B, _, H, _ = pred.shape

		# view mask の取得（>0 を True とみなす）。bool 以外なら >0 で変換。
		assert 'meta' in batch and isinstance(batch['meta'], Mapping), (
			"batch['meta'] is required"
		)
		assert 'fb_idx_view' in batch['meta'], (
			"batch['meta']['fb_idx_view'] is required"
		)
		view_mask = batch['meta']['fb_idx_view']
		if not isinstance(view_mask, torch.Tensor):
			raise TypeError("meta['fb_idx_view'] must be a torch.Tensor")
		if view_mask.dtype is not torch.bool:
			view_mask = view_mask > 0
		assert view_mask.shape == (B, H), "meta['fb_idx_view'] must be (B,H)"
		view_mask = view_mask.to(device=pred.device)

		# q, p の計算（W 次元で確率分布化）、KL(q||p) を W で和
		log_p = F.log_softmax(pred / self.tau, dim=-1)  # (B,C,H,W)

		q_raw = (target + self.eps).clamp_min(0)  # (B,C,H,W)
		eps_t = torch.finfo(q_raw.dtype).eps
		q_sum = q_raw.sum(dim=-1, keepdim=True).clamp_min(eps_t)  # (B,C,H,1)
		q = q_raw / q_sum
		log_q = (q.clamp_min(eps_t)).log()

		kl_bchw = q * (log_q - log_p)  # (B,C,H,W)
		kl_bch = kl_bchw.sum(dim=-1)  # (B,C,H)
		per_trace = kl_bch.mean(dim=1)  # (B,H)  ※C 次元を平均で集約

		# (B,H) → view で選択 → 1D
		sel_vals = per_trace[view_mask]
		if reduction == 'none':
			return sel_vals
		if sel_vals.numel() == 0:
			warnings.warn(
				"FbSegKLLossView: no traces selected by meta['fb_idx_view']; returning 0 fallback.",
				category=UserWarning,
				stacklevel=2,
			)
			return torch.zeros((), dtype=per_trace.dtype, device=per_trace.device)
		if reduction == 'sum':
			return sel_vals.sum()
		return sel_vals.mean()
