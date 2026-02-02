"""Checkpoint loading utilities for PyTorch models.

This module provides helpers to load a checkpoint file into a ``torch.nn.Module``,
including support for common nested state-dict keys (e.g., ``model_ema``,
``state_dict``, ``model``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
	from pathlib import Path


def load_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
	"""Load a PyTorch checkpoint into a model.

	The function loads a checkpoint file and attempts to find a nested state dict
	under common keys (e.g., ``model_ema``, ``state_dict``, ``model``); if none are
	found, it assumes the loaded object is itself a state dict.

	Parameters
	----------
	model : torch.nn.Module
		Model to load parameters into.
	path : str | pathlib.Path
		Path to the checkpoint file.

	Notes
	-----
	Loads weights on CPU and uses ``strict=False``; missing/unexpected keys are
	printed to stdout.

	"""
	state = torch.load(str(path), map_location='cpu', weights_only=False)
	cand_keys = ['model_ema', 'state_dict', 'model']
	for k in cand_keys:
		if k in state and isinstance(state[k], dict):
			missing, unexpected = model.load_state_dict(state[k], strict=False)
			print(
				f"[ckpt] loaded via key='{k}': missing={len(missing)} unexpected={len(unexpected)}"
			)
			return
	missing, unexpected = model.load_state_dict(state, strict=False)
	print(
		f'[ckpt] loaded raw dict: missing={len(missing)} unexpected={len(unexpected)}'
	)
