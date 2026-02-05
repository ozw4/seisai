from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .config import PairModelCfg

__all__ = ['save_checkpoint', 'load_checkpoint']


def save_checkpoint(
	ckpt_path: str | Path,
	model: torch.nn.Module,
	model_cfg: PairModelCfg,
	epoch: int,
	global_step: int,
	optimizer: torch.optim.Optimizer | None = None,
) -> None:
	ckpt = {
		'model_state_dict': model.state_dict(),
		'model_cfg': asdict(model_cfg),
		'epoch': int(epoch),
		'global_step': int(global_step),
	}
	if optimizer is not None:
		ckpt['optimizer_state_dict'] = optimizer.state_dict()

	out_path = Path(ckpt_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(ckpt, out_path)


def load_checkpoint(ckpt_path: str | Path) -> dict:
	ckpt = torch.load(Path(ckpt_path), map_location='cpu')
	if not isinstance(ckpt, dict):
		raise ValueError('checkpoint must be a dict')
	if 'model_state_dict' not in ckpt:
		raise ValueError('checkpoint missing: model_state_dict')
	if 'model_cfg' not in ckpt:
		raise ValueError('checkpoint missing: model_cfg')
	return ckpt
