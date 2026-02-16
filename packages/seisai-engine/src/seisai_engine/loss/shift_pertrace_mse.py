from collections.abc import Mapping
from typing import Any, Literal

import torch


def _mask_bool_to_trace_mask(
    mask_bool: torch.Tensor,  # (B,H) or (B,C,H,W) of bool, True=採用
    pred: torch.Tensor,  # (B,C,H,W)
    *,
    ch_reduce: Literal['all', 'any'] = 'all',
) -> torch.Tensor:  # -> (B,H) bool
    assert isinstance(mask_bool, torch.Tensor) and mask_bool.dtype == torch.bool, (
        'mask_bool must be bool Tensor'
    )
    B, C, H, W = pred.shape
    if mask_bool.ndim == 2:
        assert mask_bool.shape == (B, H), 'mask_bool (B,H) expected'
        return mask_bool.to(device=pred.device, non_blocking=True)

    assert mask_bool.ndim == 4 and mask_bool.shape == (B, C, H, W), (
        'mask_bool must be (B,H) or (B,C,H,W)'
    )
    w_all = mask_bool.all(dim=-1)  # (B,C,H)
    w_any = mask_bool.any(dim=-1)  # (B,C,H)
    assert torch.equal(w_all, w_any), 'mask_bool must be uniform across W per (b,c,h)'
    m_ch = w_all
    m_bh = m_ch.all(dim=1) if ch_reduce == 'all' else m_ch.any(dim=1)  # (B,H)
    return m_bh.to(device=pred.device, non_blocking=True)


def _shift_robust_l2_pertrace_vec(
    pred: torch.Tensor,  # (B, C, H, W)  ※外側で dtype/device/shape 検証・整合は済み
    gt: torch.Tensor,  # (B, C, H, W)
    max_shift: int,
) -> torch.Tensor:
    """W軸の±max_shift の範囲で per-(b,h) の MSE を最小化した値を返す。"""
    return _shift_robust_pertrace_vec(pred, gt, max_shift=max_shift, metric='l2')


def _shift_robust_l1_pertrace_vec(
    pred: torch.Tensor,  # (B, C, H, W)  ※外側で dtype/device/shape 検証・整合は済み
    gt: torch.Tensor,  # (B, C, H, W)
    max_shift: int,
) -> torch.Tensor:
    """W軸の±max_shift の範囲で per-(b,h) の MAE を最小化した値を返す。"""
    return _shift_robust_pertrace_vec(pred, gt, max_shift=max_shift, metric='l1')


def _shift_robust_pertrace_vec(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    max_shift: int,
    metric: Literal['l1', 'l2'],
) -> torch.Tensor:
    """W軸の±max_shiftで shift-robust per-trace 損失を返す。返り値: (B,H)."""
    B, C, H, W = pred.shape
    S = int(max_shift)
    if S < 0:
        raise ValueError(f'max_shift must be >= 0 (got {S})')
    if S >= W:
        raise ValueError(f'max_shift must be < W (got max_shift={S}, W={W})')
    if metric not in ('l1', 'l2'):
        raise ValueError(f'metric must be "l1" or "l2" (got {metric!r})')

    K = 2 * S + 1
    minW = W - S
    device = pred.device

    # 各シフトの比較開始位置を構築(gather で抽出)
    s_offsets = torch.arange(-S, S + 1, device=device)  # (K,)
    start_pred = torch.clamp(s_offsets, min=0)  # (K,)
    start_gt = torch.clamp(-s_offsets, min=0)  # (K,)
    base_idx = torch.arange(minW, device=device)  # (minW,)

    idx_pred = start_pred[:, None] + base_idx[None, :]  # (K,minW)
    idx_gt = start_gt[:, None] + base_idx[None, :]  # (K,minW)

    idxp = idx_pred.view(K, 1, 1, 1, minW).expand(-1, B, C, H, -1)
    idxg = idx_gt.view(K, 1, 1, 1, minW).expand(-1, B, C, H, -1)

    pred_g = (
        pred.unsqueeze(0).expand(K, -1, -1, -1, -1).gather(dim=-1, index=idxp)
    )  # (K,B,C,H,minW)
    gt_g = (
        gt.unsqueeze(0).expand(K, -1, -1, -1, -1).gather(dim=-1, index=idxg)
    )  # (K,B,C,H,minW)

    if metric == 'l2':
        diff = (pred_g - gt_g) ** 2
    else:
        diff = (pred_g - gt_g).abs()
    loss_kbh = diff.mean(dim=(2, 4))  # (K,B,H)
    return loss_kbh.min(dim=0).values  # (B,H)


def _validate_max_shift(max_shift: Any) -> int:
    if isinstance(max_shift, bool) or not isinstance(max_shift, int):
        raise TypeError('max_shift must be int')
    if max_shift < 0:
        raise ValueError('max_shift must be >= 0')
    return int(max_shift)


class _ShiftRobustPerTraceBase:
    def __init__(
        self,
        max_shift: int = 8,
        *,
        ch_reduce: Literal['all', 'any'] = 'all',
    ) -> None:
        self.max_shift = _validate_max_shift(max_shift)
        self.ch_reduce = ch_reduce

    def _per_trace_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

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
        gt = batch['target']
        assert isinstance(gt, torch.Tensor) and gt.shape == pred.shape, (
            'target must have same shape as pred'
        )
        if gt.dtype != pred.dtype:
            gt = gt.to(dtype=pred.dtype)
        if gt.device != pred.device:
            gt = gt.to(device=pred.device, non_blocking=True)

        trace_mask = batch.get('trace_mask', None)
        if trace_mask is None:
            mask_bool = batch.get('mask_bool', None)
            assert mask_bool is not None, (
                "either batch['trace_mask'] or batch['mask_bool'] is required"
            )
            trace_mask = _mask_bool_to_trace_mask(
                mask_bool, pred, ch_reduce=self.ch_reduce
            )

        if trace_mask.device != pred.device:
            trace_mask = trace_mask.to(device=pred.device, non_blocking=True)

        assert trace_mask.dtype == torch.bool and trace_mask.ndim == 2, (
            'trace_mask: (B,H) bool expected'
        )
        B, _C, H, _W = pred.shape
        assert trace_mask.shape == (B, H), 'trace_mask must be (B,H)'
        if self.max_shift >= _W:
            raise ValueError(
                f'max_shift must be < W (got max_shift={self.max_shift}, W={_W})'
            )

        per_trace = self._per_trace_loss(pred, gt)  # (B,H)
        sel_vals = per_trace[trace_mask]
        assert sel_vals.numel() > 0, 'no traces selected'

        if reduction == 'none':
            return sel_vals.reshape(-1)
        if reduction == 'sum':
            return sel_vals.sum()
        return sel_vals.mean()


class ShiftRobustPerTraceMSE(_ShiftRobustPerTraceBase):
    """W 軸のずれに頑健な per-trace MSE。
    IF: loss = ShiftRobustPerTraceMSE(max_shift)(pred, batch, reduction='mean')
      - pred: (B,C,H,W)
      - batch['target']: (B,C,H,W)
      - 優先: batch.get('trace_mask'): (B,H) bool
      - 代替: batch.get('mask_bool'): (B,H) or (B,C,H,W) → (B,H) に正規化.
    """

    def _per_trace_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return _shift_robust_l2_pertrace_vec(pred, gt, self.max_shift)


class ShiftRobustPerTraceL1(_ShiftRobustPerTraceBase):
    """W 軸のずれに頑健な per-trace MAE。"""

    def _per_trace_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return _shift_robust_l1_pertrace_vec(pred, gt, self.max_shift)
