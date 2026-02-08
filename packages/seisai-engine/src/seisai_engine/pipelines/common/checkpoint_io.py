from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

__all__ = ['load_checkpoint', 'save_checkpoint']

_REQUIRED_KEYS: dict[str, type] = {
    'version': int,
    'pipeline': str,
    'epoch': int,
    'global_step': int,
    'model_state_dict': dict,
    'model_sig': dict,
}


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_checkpoint(ckpt: dict) -> None:
    if not isinstance(ckpt, dict):
        msg = 'checkpoint must be dict'
        raise TypeError(msg)

    for key, expected_type in _REQUIRED_KEYS.items():
        if key not in ckpt:
            raise KeyError(f'checkpoint missing: {key}')
        value = ckpt[key]
        if expected_type is int:
            if not _is_strict_int(value):
                raise TypeError(f'checkpoint {key} must be int')
        elif not isinstance(value, expected_type):
            raise TypeError(f'checkpoint {key} must be {expected_type.__name__}')

    if ckpt['version'] != 1:
        msg = 'checkpoint version must be 1'
        raise ValueError(msg)

    pipeline = ckpt['pipeline']
    if not pipeline.strip():
        msg = 'checkpoint pipeline must be non-empty'
        raise ValueError(msg)


def save_checkpoint(path: str | Path, ckpt: dict) -> None:
    _validate_checkpoint(ckpt)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)


def load_checkpoint(path: str | Path) -> dict:
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    if not isinstance(ckpt, dict):
        msg = 'checkpoint must be dict'
        raise TypeError(msg)

    _validate_checkpoint(ckpt)
    return ckpt
