from __future__ import annotations

import numpy as np
import torch
from seisai_utils.convert import to_numpy, to_torch
from seisai_utils.validator import (
    require_all_finite,
    require_all_numpy,
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
def select_pick_by_confidence(
    pick0_i: Tensor | np.ndarray,
    pick1_i: Tensor | np.ndarray,
    conf0: Tensor | np.ndarray,
    conf1: Tensor | np.ndarray,
    *,
    keep_th: float,
) -> tuple[
    Tensor | np.ndarray,
    Tensor | np.ndarray,
    Tensor | np.ndarray,
    Tensor | np.ndarray,
]:
    """2つの候補(pick0/pick1)から confidence で採用候補を選ぶ。"""
    if keep_th < 0.0:
        msg = f'keep_th must be >= 0, got {keep_th}'
        raise ValueError(msg)

    all_numpy = require_all_numpy(pick0_i, pick1_i, conf0, conf1)

    p0 = to_torch(pick0_i)
    p1 = to_torch(pick1_i, like=p0)
    c0 = to_torch(conf0).to(device=p0.device, dtype=torch.float32)
    c1 = to_torch(conf1).to(device=p0.device, dtype=torch.float32)

    validate_array(
        p0, allowed_ndims=(1, 2), name='pick0_i', backend='torch', shape_hint='(B,H)'
    )
    validate_array(
        p1, allowed_ndims=(1, 2), name='pick1_i', backend='torch', shape_hint='(B,H)'
    )
    validate_array(
        c0, allowed_ndims=(1, 2), name='conf0', backend='torch', shape_hint='(B,H)'
    )
    validate_array(
        c1, allowed_ndims=(1, 2), name='conf1', backend='torch', shape_hint='(B,H)'
    )

    require_float_array(c0, name='conf0', backend='torch')
    require_float_array(c1, name='conf1', backend='torch')
    require_all_finite(c0, name='conf0', backend='torch')
    require_all_finite(c1, name='conf1', backend='torch')

    p0, sq = _as_2d_bh(p0)
    p1, _ = _as_2d_bh(p1)
    c0, _ = _as_2d_bh(c0)
    c1, _ = _as_2d_bh(c1)

    require_same_shape_and_backend(
        p0, p1, name_a='pick0_i', name_b='pick1_i', backend='torch'
    )
    require_same_shape_and_backend(
        p0, c0, name_a='pick0_i', name_b='conf0', backend='torch'
    )
    require_same_shape_and_backend(
        p0, c1, name_a='pick0_i', name_b='conf1', backend='torch'
    )

    keep_is_p1 = c1 >= c0
    conf_keep = torch.where(keep_is_p1, c1, c0).to(dtype=torch.float32)
    pick_keep = torch.where(keep_is_p1, p1, p0)

    keep_mask = conf_keep >= float(keep_th)
    pick_keep = torch.where(keep_mask, pick_keep, torch.zeros_like(pick_keep))
    pick_keep = pick_keep.to(dtype=torch.int32)

    if sq:
        pick_keep = pick_keep.squeeze(0)
        conf_keep = conf_keep.squeeze(0)
        keep_is_p1 = keep_is_p1.squeeze(0)
        keep_mask = keep_mask.squeeze(0)

    if all_numpy:
        return (
            to_numpy(pick_keep),
            to_numpy(conf_keep),
            to_numpy(keep_is_p1),
            to_numpy(keep_mask),
        )
    return pick_keep, conf_keep, keep_is_p1, keep_mask
