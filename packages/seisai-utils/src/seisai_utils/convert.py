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


def first_or_self(v):
	"""DataLoader の default collate で list 化された値から 1要素目を取り出す。

	- v が list なら v[0]（空なら None）
	- それ以外なら v をそのまま返す

	用途: batch 内の file_path / key_name 等（str が list[str] になりがち）を
	可視化タイトル等で手軽に扱う。
	"""
	if isinstance(v, list):
		return v[0] if len(v) > 0 else None
	return v
