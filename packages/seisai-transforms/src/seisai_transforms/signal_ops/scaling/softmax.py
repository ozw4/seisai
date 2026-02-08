import numpy as np
import torch
from seisai_utils.validator import validate_array
from torch import Tensor


def softmax_per_trace_np(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """トレース毎(最後の軸=W)にソフトマックスを適用する NumPy 実装。
    (W,), (H,W), (C,H,W), (B,C,H,W) を受け付ける。.
    """
    validate_array(x, allowed_ndims=(1, 2, 3, 4), name='x')
    xf = x.astype(np.float32, copy=False)
    m = np.max(xf, axis=-1, keepdims=True)
    ex = np.exp(xf - m)
    denom = ex.sum(axis=-1, keepdims=True)
    denom = np.maximum(denom, float(eps))
    return ex / denom


def softmax_per_trace_torch(x: Tensor, eps: float = 1e-10) -> Tensor:
    """トレース毎(最後の軸=W)にソフトマックスを適用する Torch 実装。
    (W,), (H,W), (C,H,W), (B,C,H,W) を受け付け、W は最後の軸と仮定。.
    """
    validate_array(x, allowed_ndims=(1, 2, 3, 4), name='x', backend='torch')
    xf: Tensor = x.to(dtype=torch.float32)
    m: Tensor = xf.amax(dim=-1, keepdim=True)
    ex: Tensor = torch.exp(xf - m)
    denom: Tensor = ex.sum(dim=-1, keepdim=True)
    denom = torch.clamp(denom, min=float(eps))
    return ex / denom


class PerTraceSoftmax:
    """トレース方向(最後の軸=W)でソフトマックス正規化する。
    - NumPy: (W,), (H,W), (C,H,W), (B,C,H,W)
    - Torch: (W,), (H,W), (C,H,W), (B,C,H,W)(CPU/GPU 両対応).
    """

    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = float(eps)

    def __call__(
        self,
        x,
        rng: np.random.Generator | None = None,
        return_meta: bool = False,
    ):
        """x: np.ndarray または torch.Tensor(CPU/GPU)
        rng はインターフェース維持のためのダミー(未使用)。
        return_meta=True の場合は (y, {}) を返す。.
        """
        validate_array(
            x,
            allowed_ndims=(1, 2, 3, 4),
            name='x',
            backend='auto',
        )

        if isinstance(x, np.ndarray):
            y = softmax_per_trace_np(x, eps=self.eps)
        elif isinstance(x, Tensor):
            y = softmax_per_trace_torch(x, eps=self.eps)
        else:
            msg = f'x must be numpy.ndarray or torch.Tensor, got {type(x)}'
            raise TypeError(msg)

        if return_meta:
            return y, {}

        return y
