from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import torch
import torch.nn.functional as F
from seisai_pick.score.confidence_from_prob import trace_confidence_from_prob
from seisai_utils.config import optional_bool, require_int

from seisai_engine.infer.runner import TiledWConfig, infer_batch_tiled_w
from seisai_engine.pipelines.common import InferEpochResult

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

__all__ = [
    'CoarseInferBatchResult',
    'build_tiled_w_cfg',
    'logits_to_prob',
    'prob_to_pick_confidence',
    'run_infer_batch',
    'run_infer_epoch',
]


@dataclass(frozen=True)
class CoarseInferBatchResult:
    logits: torch.Tensor
    prob: torch.Tensor
    pick_idx: torch.Tensor
    confidence: torch.Tensor


def build_tiled_w_cfg(tile_cfg: Mapping[str, Any]) -> TiledWConfig:
    if not isinstance(tile_cfg, Mapping):
        raise TypeError('tile_cfg must be mapping')
    return TiledWConfig(
        tile_w=int(require_int(tile_cfg, 'tile_w')),
        overlap_w=int(require_int(tile_cfg, 'overlap_w')),
        tiles_per_batch=int(require_int(tile_cfg, 'tiles_per_batch')),
        amp=bool(optional_bool(tile_cfg, 'amp', default=True)),
        use_tqdm=bool(optional_bool(tile_cfg, 'use_tqdm', default=False)),
    )


def _validate_logits(logits: torch.Tensor, *, label: str) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or int(logits.ndim) != 4:
        raise ValueError(f'{label}: logits must be torch.Tensor with shape (B,1,H,W)')
    if int(logits.shape[1]) != 1:
        raise ValueError(f'{label}: coarse logits channel dim must be 1, got {int(logits.shape[1])}')
    if int(logits.shape[0]) <= 0 or int(logits.shape[2]) <= 0 or int(logits.shape[3]) <= 0:
        raise ValueError(f'{label}: invalid logits shape {tuple(logits.shape)}')
    return logits


def _validate_prob(prob: torch.Tensor, *, label: str) -> torch.Tensor:
    if not isinstance(prob, torch.Tensor) or int(prob.ndim) != 4:
        raise ValueError(f'{label}: prob must be torch.Tensor with shape (B,1,H,W)')
    if int(prob.shape[1]) != 1:
        raise ValueError(f'{label}: coarse probability channel dim must be 1, got {int(prob.shape[1])}')
    if int(prob.shape[3]) <= 1:
        raise ValueError(f'{label}: W must be > 1 for confidence estimation, got {int(prob.shape[3])}')
    return prob


def _normalize_trace_valid(
    trace_valid: torch.Tensor | None,
    *,
    b: int,
    h: int,
    device: torch.device,
) -> torch.Tensor | None:
    if trace_valid is None:
        return None
    tv = trace_valid if isinstance(trace_valid, torch.Tensor) else torch.as_tensor(trace_valid)
    if tv.dtype != torch.bool:
        tv = tv.to(dtype=torch.bool)
    if int(tv.ndim) != 2:
        raise ValueError(f'trace_valid must have shape (B,H), got {tuple(tv.shape)}')
    if tuple(tv.shape) != (int(b), int(h)):
        raise ValueError(f'trace_valid shape mismatch: {tuple(tv.shape)} vs ({int(b)},{int(h)})')
    if tv.device != device:
        tv = tv.to(device=device, non_blocking=True)
    return tv


def _infer_logits_tiled_w(
    *,
    model: torch.nn.Module,
    x_bchw: torch.Tensor,
    tiled_cfg: TiledWConfig,
) -> torch.Tensor:
    if not isinstance(x_bchw, torch.Tensor) or int(x_bchw.ndim) != 4:
        raise ValueError(f'x_bchw must be torch.Tensor with shape (B,C,H,W), got {type(x_bchw)} {getattr(x_bchw, "shape", None)}')

    raw_w = int(x_bchw.shape[-1])
    if raw_w <= 0:
        raise ValueError('x_bchw W must be positive')

    x_work = x_bchw
    if raw_w < int(tiled_cfg.tile_w):
        pad_w = int(tiled_cfg.tile_w) - int(raw_w)
        x_work = F.pad(x_work, (0, pad_w))

    logits = infer_batch_tiled_w(model, x_work, cfg=tiled_cfg)
    logits = _validate_logits(logits, label='infer_batch_tiled_w')
    if int(logits.shape[0]) != int(x_bchw.shape[0]) or int(logits.shape[2]) != int(x_bchw.shape[2]):
        raise ValueError(
            f'tiled infer shape mismatch: input {tuple(x_bchw.shape)} vs logits {tuple(logits.shape)}'
        )
    if int(logits.shape[-1]) < int(raw_w):
        raise ValueError(f'tiled infer returned W={int(logits.shape[-1])} < input W={int(raw_w)}')
    if int(logits.shape[-1]) != int(raw_w):
        logits = logits[..., :raw_w]
    return logits


def logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
    logits_tensor = _validate_logits(logits, label='logits_to_prob')
    return torch.softmax(logits_tensor, dim=-1)


def prob_to_pick_confidence(
    prob: torch.Tensor,
    *,
    trace_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    prob_tensor = _validate_prob(prob, label='prob_to_pick_confidence')
    b = int(prob_tensor.shape[0])
    h = int(prob_tensor.shape[2])
    device = prob_tensor.device
    trace_valid_tensor = _normalize_trace_valid(
        trace_valid,
        b=b,
        h=h,
        device=device,
    )

    prob_bhw = prob_tensor[:, 0]
    pick_idx = prob_bhw.argmax(dim=-1).to(dtype=torch.int64)
    confidence = trace_confidence_from_prob(prob_bhw)
    if not isinstance(confidence, torch.Tensor):
        confidence = torch.as_tensor(confidence)
    confidence = confidence.to(device=device, dtype=torch.float32)

    if trace_valid_tensor is not None:
        invalid = ~trace_valid_tensor
        pick_idx = torch.where(invalid, torch.full_like(pick_idx, -1), pick_idx)
        confidence = torch.where(invalid, torch.zeros_like(confidence), confidence)

    return pick_idx, confidence


def run_infer_batch(
    *,
    model: torch.nn.Module,
    x_bchw: torch.Tensor,
    tiled_cfg: TiledWConfig,
    trace_valid: torch.Tensor | None = None,
) -> CoarseInferBatchResult:
    logits = _infer_logits_tiled_w(
        model=model,
        x_bchw=x_bchw,
        tiled_cfg=tiled_cfg,
    )
    prob = logits_to_prob(logits)
    b = int(prob.shape[0])
    h = int(prob.shape[2])
    trace_valid_tensor = _normalize_trace_valid(
        trace_valid,
        b=b,
        h=h,
        device=prob.device,
    )
    pick_idx, confidence = prob_to_pick_confidence(
        prob,
        trace_valid=trace_valid_tensor,
    )
    if trace_valid_tensor is not None:
        prob = torch.where(
            trace_valid_tensor[:, None, :, None],
            prob,
            torch.zeros_like(prob),
        )
    return CoarseInferBatchResult(
        logits=logits,
        prob=prob,
        pick_idx=pick_idx,
        confidence=confidence,
    )


def _batch_eval_metrics(
    *,
    pick_idx: torch.Tensor,
    confidence: torch.Tensor,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if 'fb_idx_view' not in batch:
        raise KeyError("batch['fb_idx_view'] is required for coarse eval metrics")
    if 'label_valid' not in batch:
        raise KeyError("batch['label_valid'] is required for coarse eval metrics")

    gt = batch['fb_idx_view']
    valid = batch['label_valid']
    if not isinstance(gt, torch.Tensor):
        gt = torch.as_tensor(gt)
    if not isinstance(valid, torch.Tensor):
        valid = torch.as_tensor(valid)
    if gt.dtype != torch.int64:
        gt = gt.to(dtype=torch.int64)
    if valid.dtype != torch.bool:
        valid = valid.to(dtype=torch.bool)

    pred_cpu = pick_idx.detach().cpu()
    conf_cpu = confidence.detach().cpu()
    gt_cpu = gt.detach().cpu()
    valid_cpu = valid.detach().cpu()
    if tuple(pred_cpu.shape) != tuple(gt_cpu.shape) or tuple(pred_cpu.shape) != tuple(valid_cpu.shape):
        raise ValueError(
            f'coarse eval metric shape mismatch: pred={tuple(pred_cpu.shape)} gt={tuple(gt_cpu.shape)} valid={tuple(valid_cpu.shape)}'
        )
    if not torch.any(valid_cpu):
        return (
            torch.empty((0,), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
        )

    abs_err = (pred_cpu[valid_cpu] - gt_cpu[valid_cpu]).abs().to(dtype=torch.float32)
    sq_err = abs_err.square()
    conf_sel = conf_cpu[valid_cpu].to(dtype=torch.float32)
    return abs_err, sq_err, conf_sel


def run_infer_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion,
    tiled_cfg: TiledWConfig,
    vis_out_dir: str,
    vis_n: int,
    max_batches: int,
) -> InferEpochResult:
    del vis_n

    non_blocking = bool(device.type == 'cuda')
    infer_loss_sum = 0.0
    infer_samples = 0
    abs_err_chunks: list[torch.Tensor] = []
    sq_err_chunks: list[torch.Tensor] = []
    conf_chunks: list[torch.Tensor] = []

    Path(vis_out_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= int(max_batches):
                break
            if not isinstance(batch, dict):
                raise TypeError('loader must yield dict batches for coarse eval')
            if 'input' not in batch or 'target' not in batch:
                raise KeyError('coarse eval batch must contain input and target')

            x_in = batch['input'].to(device=device, non_blocking=non_blocking)
            x_tg = batch['target'].to(device=device, non_blocking=non_blocking)
            trace_valid = batch.get('trace_valid')
            batch_result = run_infer_batch(
                model=model,
                x_bchw=x_in,
                tiled_cfg=tiled_cfg,
                trace_valid=trace_valid,
            )
            loss = criterion(batch_result.logits, x_tg, batch)

            bsize = int(x_in.shape[0])
            infer_loss_sum += float(loss.detach().item()) * bsize
            infer_samples += bsize

            abs_err, sq_err, conf_sel = _batch_eval_metrics(
                pick_idx=batch_result.pick_idx,
                confidence=batch_result.confidence,
                batch=batch,
            )
            if int(abs_err.numel()) > 0:
                abs_err_chunks.append(abs_err)
                sq_err_chunks.append(sq_err)
                conf_chunks.append(conf_sel)

    if infer_samples <= 0:
        raise RuntimeError('no inference samples were processed')
    if not abs_err_chunks:
        raise RuntimeError('no valid coarse labels were available during eval inference')

    abs_err_all = torch.cat(abs_err_chunks, dim=0)
    sq_err_all = torch.cat(sq_err_chunks, dim=0)
    conf_all = torch.cat(conf_chunks, dim=0)
    metrics = {
        'pick_mae': float(abs_err_all.mean().item()),
        'pick_rmse': float(math.sqrt(float(sq_err_all.mean().item()))),
        'confidence_mean': float(conf_all.mean().item()),
    }
    return InferEpochResult(
        loss=infer_loss_sum / float(infer_samples),
        metrics=metrics,
    )
