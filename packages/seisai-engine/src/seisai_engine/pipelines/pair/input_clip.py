from __future__ import annotations

import math
from typing import Any

import torch
from torch.utils.data import Dataset

__all__ = [
    'maybe_soft_clip_pair_input',
    'maybe_wrap_pair_input_soft_clip_dataset',
    'resolve_pair_input_soft_clip_abs',
    'soft_clip_tanh',
]


def _coerce_clip_abs(
    value: object,
    *,
    type_message: str,
    value_message: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(type_message)
    clip_abs = float(value)
    if not math.isfinite(clip_abs):
        raise ValueError(value_message)
    if clip_abs <= 0.0:
        raise ValueError(value_message)
    return clip_abs


def resolve_pair_input_soft_clip_abs(cfg: dict[str, Any]) -> float | None:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    pair_cfg = cfg.get('pair')
    if pair_cfg is None:
        return None
    if not isinstance(pair_cfg, dict):
        msg = 'pair must be dict'
        raise TypeError(msg)
    if 'input_soft_clip_abs' not in pair_cfg:
        return None

    clip_abs_raw = pair_cfg['input_soft_clip_abs']
    if clip_abs_raw is None:
        return None
    return _coerce_clip_abs(
        clip_abs_raw,
        type_message='pair.input_soft_clip_abs must be float | null',
        value_message='pair.input_soft_clip_abs must be > 0 when provided',
    )


def soft_clip_tanh(x: torch.Tensor, clip_abs: float) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        msg = 'x must be torch.Tensor'
        raise TypeError(msg)
    clip_abs_float = _coerce_clip_abs(
        clip_abs,
        type_message='clip_abs must be float > 0',
        value_message='clip_abs must be > 0',
    )

    return torch.tanh(x / clip_abs_float) * clip_abs_float


def maybe_soft_clip_pair_input(
    x: torch.Tensor,
    *,
    clip_abs: float | None,
) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        msg = 'x must be torch.Tensor'
        raise TypeError(msg)
    if clip_abs is None:
        return x
    return soft_clip_tanh(x, clip_abs)


class _PairInputSoftClipDataset(Dataset):
    def __init__(self, base_dataset: Any, *, clip_abs: float) -> None:
        self._base_dataset = base_dataset
        self._clip_abs = _coerce_clip_abs(
            clip_abs,
            type_message='clip_abs must be float > 0',
            value_message='clip_abs must be > 0',
        )

    def __len__(self) -> int:
        return int(len(self._base_dataset))

    def __getitem__(self, index: Any) -> dict[str, Any]:
        sample = self._base_dataset[index]
        if not isinstance(sample, dict):
            msg = 'pair dataset sample must be dict'
            raise TypeError(msg)
        if 'input' not in sample:
            msg = "pair dataset sample must contain 'input'"
            raise KeyError(msg)
        x_in = sample['input']
        if not isinstance(x_in, torch.Tensor):
            msg = "pair dataset sample['input'] must be torch.Tensor"
            raise TypeError(msg)

        clipped_sample = dict(sample)
        clipped_sample['input'] = maybe_soft_clip_pair_input(
            x_in,
            clip_abs=self._clip_abs,
        )
        return clipped_sample

    def close(self) -> None:
        close_fn = getattr(self._base_dataset, 'close', None)
        if not callable(close_fn):
            msg = 'base_dataset must provide callable close()'
            raise TypeError(msg)
        close_fn()

    @property
    def _rng(self) -> Any:
        return getattr(self._base_dataset, '_rng')

    @_rng.setter
    def _rng(self, value: Any) -> None:
        setattr(self._base_dataset, '_rng', value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_dataset, name)


def maybe_wrap_pair_input_soft_clip_dataset(
    base_dataset: Any,
    *,
    clip_abs: float | None,
) -> Any:
    if clip_abs is None:
        return base_dataset
    return _PairInputSoftClipDataset(base_dataset, clip_abs=clip_abs)
