from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

Tensor = torch.Tensor


class PostprocessCompose:
	"""後処理を順次適用（logits -> logits）。各opは (logits, batch) -> logits | (logits, dict) を返す。
	dict は監視用（ログなど）。集約して返す。
	"""

	def __init__(self, ops: Iterable):
		self.ops = tuple(ops)

	def __call__(self, logits: Tensor, batch: Mapping[str, Any]) -> tuple[Tensor, dict]:
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
