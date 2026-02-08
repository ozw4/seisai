from collections.abc import Callable, Mapping
from typing import Any, Literal

import torch


class PixelwiseLoss:
    """ピクセル単位の損失を計算し、任意のマスクで集約する。
    IF: loss = PixelwiseLoss(criterion)(pred, batch, reduction='mean')
      - pred: (B,C,H,W)
      - batch['target']: (B,C,H,W)
      - 任意: batch['mask_bool']: (B,H) もしくは (B,C,H,W) の bool(True=採用)
    """

    def __init__(self, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.criterion = criterion

    def __call__(
        self,
        pred: torch.Tensor,
        batch: Mapping[str, Any],
        *,
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
    ) -> torch.Tensor:
        assert isinstance(pred, torch.Tensor) and pred.ndim == 4, (
            'pred: (B,C,H,W) tensor expected'
        )
        assert 'target' in batch, "batch['target'] is required"
        target = batch['target']
        assert isinstance(target, torch.Tensor) and target.shape == pred.shape, (
            'target must have same shape as pred'
        )
        if target.dtype != pred.dtype:
            target = target.to(dtype=pred.dtype)
        if target.device != pred.device:
            target = target.to(device=pred.device, non_blocking=True)

        values = self.criterion(pred, target)  # (B,C,H,W) を返す関数を想定
        assert isinstance(values, torch.Tensor) and values.shape == pred.shape, (
            'criterion must return (B,C,H,W)'
        )

        mask_bool = batch.get('mask_bool', None)
        if mask_bool is None:
            if reduction == 'none':
                return values
            if reduction == 'sum':
                return values.sum()
            return values.mean()

        # mask あり
        assert mask_bool.dtype == torch.bool, 'mask_bool must be bool tensor'
        B, C, H, W = pred.shape
        if mask_bool.ndim == 2:
            assert mask_bool.shape == (B, H), 'mask_bool (B,H) expected'
            mask_bool = mask_bool[:, None, :, None].expand(B, C, H, W)
        else:
            assert mask_bool.ndim == 4 and mask_bool.shape == (B, C, H, W), (
                'mask_bool (B,C,H,W) expected'
            )

        sel = mask_bool
        if reduction == 'none':
            # 仕様: マスクあり + none → 1D にフラット
            sel_vals = values[sel]
            assert sel_vals.numel() > 0, 'mask has no True'
            return sel_vals
        # 集約
        sel_vals = values[sel]
        assert sel_vals.numel() > 0, 'mask has no True'
        if reduction == 'sum':
            return sel_vals.sum()
        return sel_vals.mean()


# 代表的な criterion 実装例
def l2_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target) ** 2


def l1_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs()


def huber_map(
    pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    diff = pred - target
    abs_diff = diff.abs()
    squared_loss = 0.5 * diff**2
    huber_loss = torch.where(
        abs_diff <= delta, squared_loss, delta * (abs_diff - 0.5 * delta)
    )
    return huber_loss


def build_criterion(
    kind: Literal['l1', 'mse', 'huber'] = 'l1', *, reduction='mean', huber_delta=1.0
):
    if kind == 'l1':
        map_fn = l1_map
    elif kind == 'mse':
        map_fn = l2_map
    elif kind == 'huber':
        map_fn = lambda p, t: huber_map(p, t, delta=huber_delta)
    else:
        raise ValueError(kind)

    pw = PixelwiseLoss(map_fn)

    def _criterion(pred, target, batch):
        b = {'target': target}
        if 'mask_bool' in batch:
            b['mask_bool'] = batch['mask_bool']
        return pw(pred, b, reduction=reduction)

    return _criterion
