from __future__ import annotations

from typing import Any

from seisai_utils.config import require_dict
from seisai_utils.validator import require_positive_float

__all__ = ['resolve_standardize_eps', 'resolve_tta_requested']


def resolve_standardize_eps(cfg: dict[str, Any]) -> float:
    infer_cfg = require_dict(cfg, 'infer')
    eps_raw = infer_cfg.get('standardize_eps')
    if eps_raw is not None:
        return require_positive_float(eps_raw, name='infer.standardize_eps')

    transform_cfg = cfg.get('transform')
    if transform_cfg is not None:
        if not isinstance(transform_cfg, dict):
            msg = 'transform must be dict'
            raise TypeError(msg)
        if 'standardize_eps' in transform_cfg:
            return require_positive_float(
                transform_cfg['standardize_eps'],
                name='transform.standardize_eps',
            )

    return 1.0e-8


def resolve_tta_requested(cfg: dict[str, Any]) -> list[Any]:
    tta_obj = cfg.get('tta', [])
    if tta_obj is None:
        return []
    if not isinstance(tta_obj, list):
        msg = 'tta must be list or null'
        raise TypeError(msg)
    return list(tta_obj)
