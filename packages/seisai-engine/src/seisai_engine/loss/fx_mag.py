from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch


class FxMagPerTraceMSE:
    """f-x 空間の |FFT| (振幅スペクトル) の per-trace MSE。.

    pred/target の W 軸に対して rFFT を取り、|FFT| (または log|FFT|) を比較する。
    返り値は trace 次元(H)で集約した per-trace の誤差を trace_mask で選択し、reduction する。

    IF: loss = FxMagPerTraceMSE(...)(pred, batch, reduction='mean')
      - pred: (B,C,H,W)
      - batch['target']: (B,C,H,W)
      - 優先: batch.get('trace_mask'): (B,H) bool
      - 代替: batch.get('mask_bool'): (B,H) or (B,H,W) or (B,C,H,W) の bool(True=採用)
    """

    def __init__(
        self,
        *,
        use_log: bool = True,
        eps: float = 1e-6,
        f_lo: int = 0,
        f_hi: int | None = None,
    ) -> None:
        if eps <= 0.0:
            raise ValueError('eps must be > 0.0')
        if f_lo < 0:
            raise ValueError('f_lo must be >= 0')
        if f_hi is not None and f_hi <= f_lo:
            raise ValueError('f_hi must be > f_lo')

        self.use_log = bool(use_log)
        self.eps = float(eps)

    def __call__(
        self,
        pred: torch.Tensor,
        batch: Mapping[str, Any],
        *,
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
    ) -> torch.Tensor:
        if not isinstance(pred, torch.Tensor):
            raise TypeError('pred: torch.Tensor expected')
        if pred.ndim != 4:
            raise ValueError('pred: (B,C,H,W) tensor expected')

        if 'target' not in batch:
            raise KeyError("batch['target'] is required")
        gt = batch['target']
        if not isinstance(gt, torch.Tensor):
            raise TypeError("batch['target'] must be a torch.Tensor")
        if gt.shape != pred.shape:
            raise ValueError('target must have same shape as pred')

        if gt.dtype != pred.dtype:
            gt = gt.to(dtype=pred.dtype)
        if gt.device != pred.device:
            gt = gt.to(device=pred.device, non_blocking=True)

        trace_mask = batch.get('trace_mask', None)
        if trace_mask is None:
            mask_bool = batch.get('mask_bool', None)
            if mask_bool is None:
                raise KeyError(
                    "either batch['trace_mask'] or batch['mask_bool'] is required"
                )
            if not isinstance(mask_bool, torch.Tensor):
                raise TypeError("batch['mask_bool'] must be a torch.Tensor")
            if mask_bool.dtype != torch.bool:
                raise TypeError("batch['mask_bool'] must be a bool tensor")

            B, C, H, W = pred.shape
            if mask_bool.ndim == 2:
                if mask_bool.shape != (B, H):
                    raise ValueError("batch['mask_bool'] shape must be (B,H)")
                trace_mask = mask_bool
            elif mask_bool.ndim == 3:
                if mask_bool.shape != (B, H, W):
                    raise ValueError("batch['mask_bool'] shape must be (B,H,W)")
                trace_mask = mask_bool.any(dim=-1)
            else:
                if mask_bool.ndim != 4 or mask_bool.shape != (B, C, H, W):
                    raise ValueError("batch['mask_bool'] shape must be (B,C,H,W)")
                w_all = mask_bool.all(dim=-1)  # (B,C,H)
                w_any = mask_bool.any(dim=-1)  # (B,C,H)
                if not torch.equal(w_all, w_any):
                    raise ValueError(
                        "batch['mask_bool'] must be consistent over W for each trace"
                    )
                trace_mask = w_all.all(dim=1)
        elif not isinstance(trace_mask, torch.Tensor):
            raise TypeError("batch['trace_mask'] must be a torch.Tensor")

        if trace_mask.device != pred.device:
            trace_mask = trace_mask.to(device=pred.device, non_blocking=True)

        if trace_mask.dtype != torch.bool:
            raise TypeError("batch['trace_mask'] must be a bool tensor")
        if trace_mask.ndim != 2:
            raise ValueError("batch['trace_mask'] shape must be (B,H)")

        B, _C, H, W = pred.shape
        if trace_mask.shape != (B, H):
            raise ValueError("batch['trace_mask'] shape must be (B,H)")

        # (B,C,H,F)
        pred_f = torch.fft.rfft(pred, dim=-1)
        gt_f = torch.fft.rfft(gt, dim=-1)
        pred_mag = pred_f.abs()
        gt_mag = gt_f.abs()

        if self.f_lo > 0 or self.f_hi is not None:
            F = pred_mag.shape[-1]
            lo = self.f_lo
            hi = F if self.f_hi is None else self.f_hi
            if not (0 <= lo < hi <= F):
                raise ValueError(
                    f'frequency slice out of range: lo={lo}, hi={hi}, F={F}'
                )
            pred_mag = pred_mag[..., lo:hi]
            gt_mag = gt_mag[..., lo:hi]

        if self.use_log:
            pred_mag = torch.log(pred_mag + self.eps)
            gt_mag = torch.log(gt_mag + self.eps)

        diff2 = (pred_mag - gt_mag) ** 2  # (B,C,H,F')
        per_trace = diff2.mean(dim=(1, 3))  # (B,H)

        sel_vals = per_trace[trace_mask]
        if sel_vals.numel() <= 0:
            msg = 'no traces selected'
            raise ValueError(msg)

        if reduction == 'none':
            return sel_vals.reshape(-1)
        if reduction == 'sum':
            return sel_vals.sum()
        return sel_vals.mean()
        if reduction == 'sum':
            return sel_vals.sum()
        return sel_vals.mean()
