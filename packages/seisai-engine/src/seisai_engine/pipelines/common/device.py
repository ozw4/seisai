from __future__ import annotations

import torch

__all__ = ['resolve_device']


def resolve_device(device_str: str | None) -> torch.device:
    """Resolve a device string into a torch.device with validation."""
    if device_str is None:
        normalized = ''
    else:
        normalized = str(device_str).strip().lower()

    if normalized in ('', 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if normalized == 'cpu':
        return torch.device('cpu')

    if normalized == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('train.device="cuda" requested but CUDA is not available')
        return torch.device('cuda')

    if normalized.startswith('cuda:'):
        idx_str = normalized.split(':', 1)[1].strip()
        if idx_str == '':
            raise ValueError('train.device="cuda:" is invalid; expected cuda:N')
        try:
            idx = int(idx_str)
        except ValueError as exc:
            raise ValueError(
                f'train.device="cuda:{idx_str}" is invalid; expected cuda:N'
            ) from exc
        if idx < 0:
            raise ValueError(
                f'train.device="cuda:{idx}" is invalid; expected non-negative index'
            )
        if not torch.cuda.is_available():
            raise ValueError(
                f'train.device="cuda:{idx}" requested but CUDA is not available'
            )
        count = torch.cuda.device_count()
        if idx >= count:
            raise ValueError(
                f'train.device="cuda:{idx}" is out of range; device_count={count}'
            )
        return torch.device(f'cuda:{idx}')

    raise ValueError(
        f'train.device="{device_str}" is invalid; expected auto|cpu|cuda|cuda:N'
    )
