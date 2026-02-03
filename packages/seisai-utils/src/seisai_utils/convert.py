"""Conversion helpers between NumPy arrays and Torch tensors.

This module provides:
- `to_torch`: convert a NumPy array or Torch tensor to a Torch tensor (optionally matching dtype/device).
- `to_numpy`: convert one or more NumPy arrays / Torch tensors / None to NumPy arrays (or None).
- `first_or_self`: return the first element of a list, or the value itself.
"""

from typing import TypeVar

import numpy as np
import torch
from torch import Tensor


def to_torch(x: np.ndarray | Tensor, *, like: Tensor | None = None) -> Tensor:
	"""Convert a NumPy array or Torch tensor to a Torch tensor.

	Parameters
	----------
	x : numpy.ndarray | torch.Tensor
		Input value to convert.
	like : torch.Tensor | None, optional
		If provided, the output tensor is converted to match ``like.dtype`` and
		``like.device``.

	Returns
	-------
	torch.Tensor
		Converted tensor.

	Raises
	------
	TypeError
		If ``x`` is not a ``numpy.ndarray`` or ``torch.Tensor``.

	"""
	if isinstance(x, torch.Tensor):
		return x if like is None else x.to(dtype=like.dtype, device=like.device)
	if isinstance(x, np.ndarray):
		t = torch.from_numpy(x)  # CPUゼロコピー
		return t if like is None else t.to(dtype=like.dtype, device=like.device)
	msg = 'inputs must be numpy.ndarray or torch.Tensor'
	raise TypeError(msg)


def to_numpy(
	*xs: np.ndarray | Tensor | None,
) -> np.ndarray | None | tuple[np.ndarray | None, ...]:
	"""Convert one or more inputs to NumPy arrays.

	Parameters
	----------
	*xs : numpy.ndarray | torch.Tensor | None
		One or more values to convert. Torch tensors are converted via
		``x.detach().cpu().numpy()``. NumPy arrays are returned as-is. ``None`` is
		passed through.

	Returns
	-------
	numpy.ndarray | None | tuple[numpy.ndarray | None, ...]
		If one input is provided, returns the converted value directly; otherwise
		returns a tuple of converted values.

	Raises
	------
	ValueError
		If no inputs are provided.
	TypeError
		If any input is not a ``numpy.ndarray``, ``torch.Tensor``, or ``None``.

	"""
	if len(xs) == 0:
		msg = 'at least one input is required'
		raise ValueError(msg)

	def _conv(x: np.ndarray | Tensor | None, where: str) -> np.ndarray | None:
		if isinstance(x, torch.Tensor):
			return x.detach().cpu().numpy()
		if isinstance(x, np.ndarray):
			return x
		if x is None:
			return None
		msg = f'{where} must be torch.Tensor, numpy.ndarray, or None'
		raise TypeError(msg)

	if len(xs) == 1:
		return _conv(xs[0], 'arg1')

	return tuple(_conv(x, f'arg{i + 1}') for i, x in enumerate(xs))


T = TypeVar('T')


def first_or_self(v: T | list[T]) -> T | None:
	"""Return the first element of a list, or the value itself.

	Parameters
	----------
	v : T | list[T]
		Input value. If a list is provided, returns its first element; if the list
		is empty, returns ``None``.

	Returns
	-------
	T | None
		First element of ``v`` if it is a non-empty list; ``None`` if it is an
		empty list; otherwise returns ``v`` unchanged.

	"""
	if isinstance(v, list):
		return v[0] if len(v) > 0 else None
	return v
