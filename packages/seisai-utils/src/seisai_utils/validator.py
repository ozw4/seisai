# validators.py
from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
from torch import Tensor

Backend = Literal['numpy', 'torch', 'auto']

# 例示表記（必要に応じて拡張）
_SHAPE_EXAMPLES = {1: '(W,)', 2: '(H,W)', 3: '(C,H,W)', 4: '(B,C,H,W)'}


def _allowed_shapes_text(allowed_ndims: Iterable[int]) -> str:
	items: list[str] = [
		_SHAPE_EXAMPLES.get(int(d), f'(ndim={int(d)})') for d in allowed_ndims
	]
	return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' or ' + items[-1]


# ---------- NumPy ----------
def require_numpy(x, *, name: str = 'x') -> None:
	if not isinstance(x, np.ndarray):
		raise TypeError(f'{name} must be numpy.ndarray')


def require_ndim_numpy(
	x: np.ndarray, *, allowed_ndims: Iterable[int], name: str = 'x'
) -> None:
	allowed = tuple(int(d) for d in allowed_ndims)
	if x.ndim not in allowed:
		raise ValueError(f'{name} must be {_allowed_shapes_text(allowed)}')


def require_non_empty_numpy(x: np.ndarray, *, name: str = 'x') -> None:
	if x.size == 0:
		raise ValueError(f'{name} must be non-empty')


def validate_numpy(
	x,
	*,
	allowed_ndims: Iterable[int] = (1, 2, 3, 4),
	name: str = 'x',
) -> None:
	"""NumPy配列xの型・次元・空配列を検証（成功時は何も返さない）"""
	require_numpy(x, name=name)
	require_ndim_numpy(x, allowed_ndims=allowed_ndims, name=name)
	require_non_empty_numpy(x, name=name)


# ---------- Torch ----------
def require_torch_tensor(x, *, name: str = 'x') -> None:
	if not isinstance(x, torch.Tensor):
		raise TypeError(f'{name} must be torch.Tensor')


def require_ndim_torch(
	x: Tensor, *, allowed_ndims: Iterable[int], name: str = 'x'
) -> None:
	allowed = tuple(int(d) for d in allowed_ndims)
	if int(x.ndim) not in allowed:
		raise ValueError(f'{name} must be {_allowed_shapes_text(allowed)}')


def require_non_empty_torch(x: Tensor, *, name: str = 'x') -> None:
	if int(x.numel()) == 0:
		raise ValueError(f'{name} must be non-empty')


def validate_torch(
	x,
	*,
	allowed_ndims: Iterable[int] = (1, 2, 3, 4),
	name: str = 'x',
) -> None:
	"""Torchテンソルxの型・次元・空配列を検証（成功時は何も返さない）"""
	require_torch_tensor(x, name=name)
	require_ndim_torch(x, allowed_ndims=allowed_ndims, name=name)
	require_non_empty_torch(x, name=name)


# ---------- 汎用（自動/明示バックエンド） ----------
def validate_array(
	x,
	*,
	allowed_ndims: Iterable[int] = (1, 2, 3, 4),
	name: str = 'x',
	backend: Backend = 'auto',
) -> None:
	"""X が numpy.ndarray または torch.Tensor で、
	allowed_ndims のいずれかの次元数かつ非空であることを検証（成功時は何も返さない）。
	backend: "numpy" | "torch" | "auto"
	"""
	if backend == 'numpy':
		validate_numpy(x, allowed_ndims=allowed_ndims, name=name)
		return
	if backend == 'torch':
		validate_torch(x, allowed_ndims=allowed_ndims, name=name)
		return
	# backend == "auto"
	if isinstance(x, np.ndarray):
		validate_numpy(x, allowed_ndims=allowed_ndims, name=name)
		return
	if isinstance(x, torch.Tensor):
		validate_torch(x, allowed_ndims=allowed_ndims, name=name)
		return
	raise TypeError(f'{name} must be numpy.ndarray or torch.Tensor')
