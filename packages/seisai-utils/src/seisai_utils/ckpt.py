from __future__ import annotations

from pathlib import Path

import torch


def load_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
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
