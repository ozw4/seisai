from __future__ import annotations

import torch
from seisai_utils.config import optional_bool, optional_str, require_dict


def _normalize_opt_name(name: str) -> str:
    opt = str(name).strip().lower()
    if not opt:
        raise ValueError('optimizer.name must be non-empty')
    return opt


def _load_optimizer_kwargs(opt_cfg: dict) -> dict[str, object]:
    kwargs = opt_cfg.get('kwargs', {})
    if kwargs is None:
        return {}
    if not isinstance(kwargs, dict):
        raise TypeError('optimizer.kwargs must be dict')

    if 'lr' in kwargs:
        raise ValueError('optimizer.kwargs must not include "lr" (use train.lr)')
    if 'weight_decay' in kwargs:
        raise ValueError(
            'optimizer.kwargs must not include "weight_decay" (use train.weight_decay)'
        )

    # YAML often encodes betas as a list; timm / torch expect tuple.
    if 'betas' in kwargs and isinstance(kwargs['betas'], list):
        betas = kwargs['betas']
        if len(betas) != 2:
            raise ValueError('optimizer.kwargs.betas must be [float, float]')
        if not isinstance(betas[0], (int, float)) or not isinstance(betas[1], (int, float)):
            raise TypeError('optimizer.kwargs.betas must be [float, float]')
        kwargs = dict(kwargs)
        kwargs['betas'] = (float(betas[0]), float(betas[1]))

    return kwargs


def build_optimizer(
    cfg: dict,
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build an optimizer from config.

    Backward-compatible behavior:
    - If cfg has no "optimizer" section, defaults to torch.optim.AdamW.
    - If cfg has an "optimizer" section, uses timm's optimizer factory.

    Expected config shape (optional):
    optimizer:
      name: adamw | lion | sgd | ...
      filter_bias_and_bn: false
      kwargs: { ... optimizer-specific kwargs ... }
    """

    opt_cfg_obj = cfg.get('optimizer', None)
    if opt_cfg_obj is None:
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

    if not isinstance(opt_cfg_obj, dict):
        raise TypeError('optimizer must be dict')
    opt_cfg = require_dict(cfg, 'optimizer')
    opt_name = _normalize_opt_name(optional_str(opt_cfg, 'name', 'adamw'))
    filter_bias_and_bn = optional_bool(opt_cfg, 'filter_bias_and_bn', default=False)
    opt_kwargs = _load_optimizer_kwargs(opt_cfg)

    from timm.optim import create_optimizer_v2

    return create_optimizer_v2(
        model,
        opt=str(opt_name),
        lr=float(lr),
        weight_decay=float(weight_decay),
        filter_bias_and_bn=bool(filter_bias_and_bn),
        **opt_kwargs,
    )
