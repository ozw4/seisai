import math

import torch
from torch import Tensor


@torch.no_grad()
def trace_confidence_from_prob(
	prob: Tensor,  # (B,H,W) after softmax
	floor: float = 0.2,  # 最低重み（自己強化の回避用）
	power: float = 0.5,  # 緩和（0.5 = sqrt）
	eps: float = 1e-9,  # 数値下限（log(0)回避）
) -> Tensor:
	"""エントロピーで (B,H) の自信度を算出。
	高いほど分布が尖っており自信が高い。0..1に正規化後、floor適用→powerで緩和。
	"""
	assert isinstance(prob, torch.Tensor) and prob.ndim == 3, 'prob must be (B,H,W)'
	W = prob.size(-1)
	assert W > 1, 'W must be > 1'

	p = prob.clamp_min(eps)
	H = -(p * p.log()).sum(dim=-1)  # (B,H) エントロピー
	Hnorm = H / math.log(W)  # 0..1 に正規化
	w = (1.0 - Hnorm).clamp(0.0, 1.0)  # 尖り＝自信度

	return w.clamp_min(floor) ** power  # (B,H)
