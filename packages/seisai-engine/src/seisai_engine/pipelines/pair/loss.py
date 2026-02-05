from __future__ import annotations

from seisai_engine.loss.pixelwise_loss import build_criterion as _build_criterion

__all__ = ['build_criterion']


def build_criterion(loss_kind: str):
	kind = str(loss_kind).lower()
	if kind not in ('l1', 'mse'):
		raise ValueError('train.loss_kind must be "l1" or "mse"')
	return _build_criterion(kind)
