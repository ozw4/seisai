"""Postprocess composition utilities.

This module provides :class:`PostprocessCompose`, which applies a sequence of
postprocess operations to model logits and aggregates any auxiliary info dicts
returned by those operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping

Tensor = torch.Tensor


class PostprocessCompose:
	"""後処理を順次適用(logits -> logits)。.

	各opは (logits, batch) -> logits | (logits, dict) を返す。
	dict は監視用(ログなど)。集約して返す。.
	"""

	def __init__(self, ops: Iterable) -> None:
		"""Initialize the postprocess pipeline.

		Args:
			ops: Iterable of postprocess callables applied in order.

		"""
		self.ops = tuple(ops)

	def __call__(self, logits: Tensor, batch: Mapping[str, Any]) -> tuple[Tensor, dict]:
		"""Apply postprocess ops sequentially to logits.

		Each op is called as ``op(logits, batch)`` and must return either updated
		logits, or a tuple ``(logits, info)`` where ``info`` is a dict of auxiliary
		monitoring values (e.g., logs/metrics) to be merged and returned.

		Args:
			logits: Model output tensor to be post-processed.
			batch: Input batch mapping passed to each op.

		Returns:
			A tuple of ``(logits, aux)`` where ``aux`` is a dict aggregated from the
			ops that return ``(logits, info)``.

		"""
		y = logits
		aux: dict[str, Any] = {}
		for op in self.ops:
			out = op(y, batch)
			if isinstance(out, tuple):
				y, info = out
				aux.update(info)
			else:
				y = out
		return y, aux
