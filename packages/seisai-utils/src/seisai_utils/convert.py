import numpy as np
import torch
from torch import Tensor


def to_torch(x, *, like: Tensor | None = None) -> Tensor:
	"""入力を torch.Tensor に変換するユーティリティ関数。
	like が指定された場合はその dtype/device に変換する。
	"""
	if isinstance(x, torch.Tensor):
		return x if like is None else x.to(dtype=like.dtype, device=like.device)
	if isinstance(x, np.ndarray):
		t = torch.from_numpy(x)  # CPUゼロコピー
		return t if like is None else t.to(dtype=like.dtype, device=like.device)
	raise TypeError('inputs must be numpy.ndarray or torch.Tensor')


def to_numpy(*xs):
	"""入力を numpy.ndarray に変換するユーティリティ関数。
	複数入力を受け取り、1つだけの場合は ndarray を、複数の場合は ndarray タプルを返す。
	None 入力は None のまま返す。
	"""
	if len(xs) == 0:
		raise ValueError('at least one input is required')

	def _conv(x, where: str) -> np.ndarray | None:
		if isinstance(x, torch.Tensor):
			return x.detach().cpu().numpy()
		if isinstance(x, np.ndarray):
			return x
		if x is None:
			return None
		raise TypeError(f'{where} must be torch.Tensor, numpy.ndarray, or None')

	if len(xs) == 1:
		return _conv(xs[0], 'arg1')

	return tuple(_conv(x, f'arg{i + 1}') for i, x in enumerate(xs))
