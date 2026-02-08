from __future__ import annotations

import math

import numpy as np
import torch
from seisai_utils.convert import to_numpy, to_torch
from seisai_utils.validator import (
    require_all_finite,
    require_all_numpy,
    require_float_array,
    validate_array,
)
from torch import Tensor


@torch.no_grad()
def trace_confidence_from_prob(
    prob: Tensor | np.ndarray,  # (B,H,W) after softmax
    floor: float = 0.2,  # 最低重み(自己強化の回避用)
    power: float = 0.5,  # 緩和(0.5 = sqrt)
    eps: float = 1e-9,  # 数値下限(log(0)回避)
) -> Tensor | np.ndarray:
    """エントロピーで (B,H) の自信度を算出。
    すべてNumPy入力→NumPyで返却。それ以外→Torchで返却。内部計算はTorch(CPU)。
    """
    # 返却形態の決定(単一引数でも利用可)
    all_numpy = require_all_numpy(prob)

    # Torchへ正規化(dtype/deviceはprob基準)
    t_prob = to_torch(prob)

    # 入力検証(Torchバックエンド)
    validate_array(
        t_prob, allowed_ndims=(3,), name='prob', backend='torch', shape_hint='(B,H,W)'
    )
    require_float_array(t_prob, name='prob', backend='torch')
    require_all_finite(t_prob, name='prob', backend='torch')

    # スカラー検証
    assert 0.0 <= floor <= 1.0, 'floor must be in [0,1]'
    assert power > 0.0, 'power must be > 0'
    assert eps > 0.0, 'eps must be > 0'

    W = int(t_prob.size(-1))
    assert W > 1, 'W must be > 1'

    # エントロピー → 自信度 (0..1)
    p = t_prob.clamp_min(eps)
    H = -(p * p.log()).sum(dim=-1)  # (B,H)
    Hnorm = H / float(math.log(W))  # 0..1 に正規化
    w = (1.0 - Hnorm).clamp(0.0, 1.0)  # 尖り=自信度
    out = w.clamp_min(floor) ** power  # floor適用→緩和

    return to_numpy(out) if all_numpy else out
