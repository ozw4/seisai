from __future__ import annotations

import math

import torch

__all__ = [
    'maybe_soft_clip_pair_input',
    'reconstruct_pair_prediction',
    'soft_clip_tanh',
]


def _coerce_clip_abs(clip_abs: object) -> float:
    if isinstance(clip_abs, bool) or not isinstance(clip_abs, (int, float)):
        msg = 'clip_abs must be float > 0'
        raise TypeError(msg)
    clip_abs_float = float(clip_abs)
    if not math.isfinite(clip_abs_float) or clip_abs_float <= 0.0:
        msg = 'clip_abs must be > 0'
        raise ValueError(msg)
    return clip_abs_float


def soft_clip_tanh(x: torch.Tensor, clip_abs: float) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        msg = 'x must be torch.Tensor'
        raise TypeError(msg)
    clip_abs_float = _coerce_clip_abs(clip_abs)
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


def reconstruct_pair_prediction(
    pred_raw: torch.Tensor,
    x_in: torch.Tensor,
    *,
    residual_learning: bool,
) -> torch.Tensor:
    if not isinstance(pred_raw, torch.Tensor):
        msg = 'pred_raw must be torch.Tensor'
        raise TypeError(msg)
    if not residual_learning:
        return pred_raw

    if not isinstance(x_in, torch.Tensor):
        msg = 'x_in must be torch.Tensor'
        raise TypeError(msg)
    if pred_raw.shape != x_in.shape:
        msg = f'pred_raw.shape {tuple(pred_raw.shape)} != x_in.shape {tuple(x_in.shape)}'
        raise ValueError(msg)
    if pred_raw.dtype != x_in.dtype:
        msg = f'pred_raw.dtype {pred_raw.dtype} != x_in.dtype {x_in.dtype}'
        raise TypeError(msg)
    if pred_raw.device != x_in.device:
        msg = f'pred_raw.device {pred_raw.device} != x_in.device {x_in.device}'
        raise ValueError(msg)
    return x_in + pred_raw
