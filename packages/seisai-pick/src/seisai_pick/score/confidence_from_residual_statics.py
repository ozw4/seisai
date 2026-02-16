from __future__ import annotations

import numpy as np
import torch
from seisai_utils.convert import to_bool_mask_torch, to_numpy, to_torch
from seisai_utils.validator import (
    require_all_finite,
    require_all_numpy,
    require_boolint_array,
    require_float_array,
    require_same_shape_and_backend,
    validate_array,
)
from torch import Tensor


def _as_2d_bh(x: Tensor) -> tuple[Tensor, bool]:
    if x.ndim == 1:
        return x.unsqueeze(0), True
    return x, False


@torch.no_grad()
def trace_confidence_from_residual_statics(
    delta_pick: Tensor | np.ndarray,
    cmax: Tensor | np.ndarray,
    valid: Tensor | np.ndarray,
    *,
    c_th: float = 0.5,
    max_lag: float = 8.0,
) -> Tensor | np.ndarray:
    """Residual statics の出力から(トレース毎)の confidence を算出。

    score = clip((cmax - c_th) / (1 - c_th), 0..1) * exp(-(abs(delta)/max_lag)^2)
    """
    if not (0.0 <= c_th < 1.0):
        msg = f'c_th must be in [0,1), got {c_th}'
        raise ValueError(msg)
    if max_lag < 0.0:
        msg = f'max_lag must be >= 0, got {max_lag}'
        raise ValueError(msg)

    all_numpy = require_all_numpy(delta_pick, cmax, valid)

    d = to_torch(delta_pick).to(dtype=torch.float32)
    c = to_torch(cmax, like=d).to(dtype=torch.float32)
    v = to_torch(valid, like=d)
    v = to_bool_mask_torch(v, like=d)
    validate_array(
        d, allowed_ndims=(1, 2), name='delta_pick', backend='torch', shape_hint='(B,H)'
    )
    validate_array(
        c, allowed_ndims=(1, 2), name='cmax', backend='torch', shape_hint='(B,H)'
    )
    validate_array(
        v, allowed_ndims=(1, 2), name='valid', backend='torch', shape_hint='(B,H)'
    )
    require_float_array(d, name='delta_pick', backend='torch')
    require_float_array(c, name='cmax', backend='torch')
    require_boolint_array(v, name='valid', backend='torch')
    require_all_finite(d, name='delta_pick', backend='torch')
    require_all_finite(c, name='cmax', backend='torch')

    d, sq = _as_2d_bh(d)
    c, _ = _as_2d_bh(c)
    v, _ = _as_2d_bh(v)

    require_same_shape_and_backend(
        d, c, name_a='delta_pick', name_b='cmax', backend='torch'
    )
    require_same_shape_and_backend(
        d, v, name_a='delta_pick', name_b='valid', backend='torch'
    )

    denom = max(1.0 - float(c_th), 1e-6)
    conf_c = ((c - float(c_th)) / float(denom)).clamp(0.0, 1.0)
    conf_c = conf_c * v.to(dtype=torch.float32)

    if float(max_lag) > 0.0:
        conf_l = torch.exp(-((d.abs() / float(max_lag)) ** 2))
    else:
        conf_l = torch.ones_like(conf_c)

    out = (conf_c * conf_l).to(dtype=torch.float32)

    if sq:
        out = out.squeeze(0)

    return to_numpy(out) if all_numpy else out
