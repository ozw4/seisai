from __future__ import annotations

import torch
from torch import Tensor


def _argmax_time_parabolic(prob_bhw: Tensor, dt_sec: Tensor) -> Tensor:
	"""(B,H,W) の確率から放物線補間付きargmaxで連続時刻[s]を返す → (B,H)
	端ビン(idx==0 or W-1)では補間量dは0(=そのままargmax)。
	"""
	assert prob_bhw.ndim == 3
	B, H, W = prob_bhw.shape
	idx = prob_bhw.argmax(dim=-1)  # (B,H)

	# 近傍インデックス
	i0 = (idx - 1).clamp_min(0)
	i2 = (idx + 1).clamp_max(W - 1)

	def g(i: Tensor) -> Tensor:
		return prob_bhw.gather(-1, i.unsqueeze(-1)).squeeze(-1)

	y0, y1, y2 = g(i0), g(idx), g(i2)
	den = (y0 - 2.0 * y1 + y2).clamp_min(1e-12)
	d = 0.5 * (y0 - y2) / den  # サブビン補間 [-∞,∞]→後で制限
	d = d.clamp_(-1.0, 1.0)

	# 端ビンは補間なし
	edge = (idx == 0) | (idx == (W - 1))
	d = torch.where(edge, torch.zeros_like(d), d)

	frac = idx.to(prob_bhw.dtype) + d  # (B,H) ビン位置の連続値

	# dt を (B,1) に整形して秒へ
	if dt_sec.ndim == 1 or dt_sec.ndim == 2:
		dt = dt_sec.view(B, 1).to(prob_bhw)
	else:
		assert dt_sec.shape == (B, 1, 1), 'dt_sec must be (B,), (B,1), or (B,1,1)'
		dt = dt_sec.view(B, 1).to(prob_bhw)

	return frac * dt  # (B,H) [s]
