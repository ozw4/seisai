from __future__ import annotations

import inspect
from typing import Protocol, cast

import numpy as np
import torch


class RNGLike(Protocol):
	def random(self, *args, **kwargs):
		pass

	def uniform(self, *args, **kwargs):
		pass

	def integers(self, *args, **kwargs):
		pass


ArrayLike2D = np.ndarray | torch.Tensor
TransformOut2D = ArrayLike2D | tuple[ArrayLike2D, dict]


class Transform2D(Protocol):
	def __call__(
		self,
		x: ArrayLike2D,
		*,
		rng: RNGLike,
		return_meta: bool = False,
	) -> TransformOut2D:
		pass


def validate_transform_rng_meta(
	transform: object,
	*,
	name: str = 'transform',
) -> Transform2D:
	if not callable(transform):
		raise TypeError(f'{name} must be callable, got {type(transform).__name__}')
	try:
		signature = inspect.signature(transform)
	except (TypeError, ValueError) as exc:
		msg = (
			f'{name} must accept rng and return_meta keyword arguments; '
			f'cannot determine signature: {exc}'
		)
		raise TypeError(msg) from exc

	params = signature.parameters
	has_kwargs = any(
		param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
	)
	if has_kwargs:
		return cast(Transform2D, transform)

	missing = [key for key in ('rng', 'return_meta') if key not in params]
	if missing:
		missing_str = ', '.join(missing)
		msg = (
			f'{name} must accept keyword parameters rng and return_meta; '
			f'missing {missing_str} in signature {signature}'
		)
		raise TypeError(msg)

	return cast(Transform2D, transform)
