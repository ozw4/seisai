from __future__ import annotations

import math
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    import numpy as np


@torch.no_grad()
def trace_confidence_from_prob(
    prob: Tensor | np.ndarray,  # (B,H,W) after softmax
    floor: float = 0.2,  # 最低重み(自己強化の回避用)
    power: float = 0.5,  # 緩和(0.5 = sqrt)
    eps: float = 1e-9,  # 数値下限(log(0)回避)
) -> Tensor | np.ndarray:
    """エントロピーで (B,H) の自信度を算出。
    すべてNumPy入力→NumPyで返却。それ以外→Torchで返却。内部計算はTorch(CPU)。.
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


@torch.no_grad()
def trace_confidence_from_prob_local_window(
    prob: Tensor | np.ndarray,  # (H,W) or (B,H,W) after softmax
    picks_i: Tensor | np.ndarray,  # (H,) or (B,H) sample index
    *,
    half_win: int = 20,
    m0: float = 0.2,
    eps: float = 1e-12,
) -> Tensor | np.ndarray:
    """局所窓(prob[pick±half_win])の形状から confidence を作る。

    各トレースごとに、pick 周辺の確率を正規化して
    - エントロピー(尖り)
    - top2 マージン
    を組み合わせる。

    conf = (1 - H_norm) * clip(margin / m0, 0, 1)

    - pick<=0 または pick>=W は 0
    - 窓の確率和が 0 の場合も 0
    - すべてNumPy入力→NumPyで返却。それ以外→Torchで返却。
    """
    if half_win < 0:
        msg = f'half_win must be >= 0, got {half_win}'
        raise ValueError(msg)
    if m0 <= 0.0:
        msg = f'm0 must be > 0, got {m0}'
        raise ValueError(msg)
    if eps <= 0.0:
        msg = f'eps must be > 0, got {eps}'
        raise ValueError(msg)

    all_numpy = require_all_numpy(prob, picks_i)

    t_prob = to_torch(prob).to(dtype=torch.float32)
    validate_array(
        t_prob,
        allowed_ndims=(2, 3),
        name='prob',
        backend='torch',
        shape_hint='(B,H,W) or (H,W)',
    )
    require_float_array(t_prob, name='prob', backend='torch')
    require_all_finite(t_prob, name='prob', backend='torch')

    squeezed = False
    if int(t_prob.ndim) == 2:
        t_prob = t_prob.unsqueeze(0)
        squeezed = True

    B, H, W = (int(t_prob.size(0)), int(t_prob.size(1)), int(t_prob.size(2)))
    if W <= 0:
        msg = f'W must be positive, got {W}'
        raise ValueError(msg)

    t_picks = to_torch(picks_i, like=t_prob)
    validate_array(
        t_picks,
        allowed_ndims=(1, 2),
        name='picks_i',
        backend='torch',
        shape_hint='(B,H) or (H,)',
    )
    require_all_finite(t_picks, name='picks_i', backend='torch')

    if int(t_picks.ndim) == 1:
        t_picks = t_picks.unsqueeze(0)
    if int(t_picks.size(0)) == 1 and B > 1:
        t_picks = t_picks.expand(B, -1)
    require_same_shape_and_backend(
        t_prob[:, :, 0],
        t_picks,
        name_a='prob[:, :, 0]',
        name_b='picks_i',
        backend='torch',
    )

    centers = t_picks.to(dtype=torch.int64)
    valid_center = (centers > 0) & (centers < int(W))

    L = int(2 * half_win + 1)
    offs = torch.arange(-int(half_win), int(half_win) + 1, device=t_prob.device)
    idx = centers.unsqueeze(-1) + offs.view(1, 1, L)
    in_range = (idx >= 0) & (idx < int(W))
    m = in_range & valid_center.unsqueeze(-1)

    idx = idx.clamp(0, int(W) - 1).to(dtype=torch.int64)
    pwin = t_prob.gather(dim=-1, index=idx)
    pwin = torch.where(m, pwin, torch.zeros_like(pwin))

    s = pwin.sum(dim=-1)  # (B,H)
    has_mass = s > 0.0
    den = s.clamp_min(float(eps))
    w = pwin / den.unsqueeze(-1)

    n_eff = m.sum(dim=-1).to(dtype=torch.float32)
    h = -(w * (w.clamp_min(float(eps))).log()).sum(dim=-1)
    h_norm = torch.where(n_eff > 1.0, h / n_eff.log(), torch.zeros_like(h))
    h_norm = h_norm.clamp(0.0, 1.0)

    if L >= 2:
        top2 = torch.topk(w, k=2, dim=-1).values  # (B,H,2)
        p1 = top2[..., 0]
        p2 = top2[..., 1]
        margin = torch.where(n_eff > 1.0, p1 - p2, p1)
    else:
        p1 = w.squeeze(-1)
        margin = p1

    conf = (1.0 - h_norm) * (margin / float(m0)).clamp(0.0, 1.0)
    conf = torch.where(has_mass & valid_center, conf, torch.zeros_like(conf))

    if squeezed:
        conf = conf.squeeze(0)

    return to_numpy(conf) if all_numpy else conf
