from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from seisai_pick.score.confidence_from_prob import trace_confidence_from_prob

from seisai_engine.pipelines.common import InferEpochResult

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

__all__ = [
    'FineInferBatchResult',
    'local_pick_to_raw_pick',
    'logits_to_local_prob',
    'local_prob_to_pick_confidence',
    'run_infer_batch',
    'run_infer_epoch',
]


@dataclass(frozen=True)
class FineInferBatchResult:
    logits: torch.Tensor
    local_prob: torch.Tensor
    local_pick_idx: torch.Tensor
    raw_pick_idx: torch.Tensor
    confidence: torch.Tensor


def _validate_input(x_bchw: torch.Tensor, *, label: str) -> torch.Tensor:
    if not isinstance(x_bchw, torch.Tensor) or int(x_bchw.ndim) != 4:
        raise ValueError(f'{label}: input must be torch.Tensor with shape (B,1,1,W_local)')
    if int(x_bchw.shape[1]) != 1:
        raise ValueError(f'{label}: fine input channel dim must be 1, got {int(x_bchw.shape[1])}')
    if int(x_bchw.shape[2]) != 1:
        raise ValueError(f'{label}: fine input trace axis must be 1, got {int(x_bchw.shape[2])}')
    if int(x_bchw.shape[0]) <= 0 or int(x_bchw.shape[3]) <= 0:
        raise ValueError(f'{label}: invalid fine input shape {tuple(x_bchw.shape)}')
    return x_bchw


def _validate_logits(
    logits: torch.Tensor,
    *,
    input_shape: tuple[int, ...],
    label: str,
) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or int(logits.ndim) != 4:
        raise ValueError(f'{label}: logits must be torch.Tensor with shape (B,1,1,W_local)')
    if int(logits.shape[1]) != 1:
        raise ValueError(f'{label}: fine logits channel dim must be 1, got {int(logits.shape[1])}')
    if int(logits.shape[2]) != 1:
        raise ValueError(f'{label}: fine logits trace axis must be 1, got {int(logits.shape[2])}')
    expected = (int(input_shape[0]), 1, 1, int(input_shape[3]))
    if tuple(logits.shape) != expected:
        raise ValueError(f'{label}: logits shape {tuple(logits.shape)} must match input geometry {expected}')
    return logits


def _validate_local_prob(prob: torch.Tensor, *, label: str) -> torch.Tensor:
    if not isinstance(prob, torch.Tensor) or int(prob.ndim) != 4:
        raise ValueError(f'{label}: local_prob must be torch.Tensor with shape (B,1,1,W_local)')
    if int(prob.shape[1]) != 1:
        raise ValueError(f'{label}: fine probability channel dim must be 1, got {int(prob.shape[1])}')
    if int(prob.shape[2]) != 1:
        raise ValueError(f'{label}: fine probability trace axis must be 1, got {int(prob.shape[2])}')
    if int(prob.shape[3]) <= 1:
        raise ValueError(
            f'{label}: local window length must be > 1 for confidence estimation, got {int(prob.shape[3])}'
        )
    return prob


def _normalize_trace_valid(
    trace_valid: torch.Tensor | None,
    *,
    b: int,
    device: torch.device,
) -> torch.Tensor | None:
    if trace_valid is None:
        return None
    tv = trace_valid if isinstance(trace_valid, torch.Tensor) else torch.as_tensor(trace_valid)
    if int(tv.ndim) == 1:
        tv = tv.reshape(-1, 1)
    if tv.dtype != torch.bool:
        tv = tv.to(dtype=torch.bool)
    if tuple(tv.shape) != (int(b), 1):
        raise ValueError(f'trace_valid shape mismatch: {tuple(tv.shape)} vs ({int(b)},1)')
    if tv.device != device:
        tv = tv.to(device=device, non_blocking=True)
    return tv


def _normalize_raw_sample_idx_local(
    raw_sample_idx_local: torch.Tensor | Any,
    *,
    b: int,
    w_local: int,
    device: torch.device,
) -> torch.Tensor:
    idx = (
        raw_sample_idx_local
        if isinstance(raw_sample_idx_local, torch.Tensor)
        else torch.as_tensor(raw_sample_idx_local)
    )
    if int(idx.ndim) == 1:
        idx = idx.reshape(1, -1)
    if tuple(idx.shape) != (int(b), int(w_local)):
        raise ValueError(
            f'raw_sample_idx_local shape mismatch: {tuple(idx.shape)} vs ({int(b)},{int(w_local)})'
        )
    if idx.dtype != torch.int64:
        idx = idx.to(dtype=torch.int64)
    if torch.any(idx < -1):
        raise ValueError('raw_sample_idx_local must contain only -1 or valid raw indices')
    if idx.device != device:
        idx = idx.to(device=device, non_blocking=True)
    return idx


def _to_tensor_b1(value: torch.Tensor | Any, *, name: str, dtype: torch.dtype) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if int(tensor.ndim) == 1:
        tensor = tensor.reshape(-1, 1)
    if int(tensor.ndim) != 2 or int(tensor.shape[1]) != 1:
        raise ValueError(f'{name} must have shape (B,1), got {tuple(tensor.shape)}')
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor


def logits_to_local_prob(logits: torch.Tensor) -> torch.Tensor:
    logits_tensor = _validate_logits(
        logits,
        input_shape=tuple(logits.shape),
        label='logits_to_local_prob',
    )
    return torch.softmax(logits_tensor, dim=-1)


def local_prob_to_pick_confidence(
    local_prob: torch.Tensor,
    *,
    trace_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    prob_tensor = _validate_local_prob(local_prob, label='local_prob_to_pick_confidence')
    b = int(prob_tensor.shape[0])
    device = prob_tensor.device
    trace_valid_tensor = _normalize_trace_valid(
        trace_valid,
        b=b,
        device=device,
    )

    prob_bhw = prob_tensor[:, 0]
    local_pick_idx = prob_bhw.argmax(dim=-1).to(dtype=torch.int64)
    confidence = trace_confidence_from_prob(prob_bhw)
    if not isinstance(confidence, torch.Tensor):
        confidence = torch.as_tensor(confidence)
    confidence = confidence.to(device=device, dtype=torch.float32)

    if trace_valid_tensor is not None:
        invalid = ~trace_valid_tensor
        local_pick_idx = torch.where(invalid, torch.full_like(local_pick_idx, -1), local_pick_idx)
        confidence = torch.where(invalid, torch.zeros_like(confidence), confidence)

    return local_pick_idx, confidence


def local_pick_to_raw_pick(
    local_pick_idx: torch.Tensor | Any,
    *,
    raw_sample_idx_local: torch.Tensor | Any,
    trace_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    local_pick_idx_tensor = _to_tensor_b1(
        local_pick_idx,
        name='local_pick_idx',
        dtype=torch.int64,
    )
    b = int(local_pick_idx_tensor.shape[0])
    device = local_pick_idx_tensor.device

    raw_idx = raw_sample_idx_local if isinstance(raw_sample_idx_local, torch.Tensor) else torch.as_tensor(raw_sample_idx_local)
    if int(raw_idx.ndim) == 1:
        raw_idx = raw_idx.reshape(1, -1)
    if int(raw_idx.ndim) != 2:
        raise ValueError(f'raw_sample_idx_local must have shape (B,W_local), got {tuple(raw_idx.shape)}')
    w_local = int(raw_idx.shape[1])
    raw_sample_idx_local_tensor = _normalize_raw_sample_idx_local(
        raw_idx,
        b=b,
        w_local=w_local,
        device=device,
    )
    trace_valid_tensor = _normalize_trace_valid(trace_valid, b=b, device=device)
    valid = (
        trace_valid_tensor
        if trace_valid_tensor is not None
        else torch.ones((b, 1), dtype=torch.bool, device=device)
    )

    if torch.any(valid & ((local_pick_idx_tensor < 0) | (local_pick_idx_tensor >= int(w_local)))):
        raise ValueError(
            f'local_pick_idx must satisfy 0 <= idx < {int(w_local)} for valid fine windows'
        )

    safe_pick_idx = torch.where(valid, local_pick_idx_tensor, torch.zeros_like(local_pick_idx_tensor))
    raw_pick_idx = raw_sample_idx_local_tensor.gather(dim=1, index=safe_pick_idx)
    if torch.any(valid & (raw_pick_idx == -1)):
        raise ValueError('raw_sample_idx_local[local_pick_idx] must not be -1 for valid fine windows')

    return torch.where(valid, raw_pick_idx, torch.full_like(raw_pick_idx, -1))


def run_infer_batch(
    *,
    model: torch.nn.Module,
    x_bchw: torch.Tensor,
    raw_sample_idx_local: torch.Tensor | Any,
    trace_valid: torch.Tensor | None = None,
) -> FineInferBatchResult:
    x_input = _validate_input(x_bchw, label='run_infer_batch')
    logits = model(x_input)
    logits = _validate_logits(
        logits,
        input_shape=tuple(x_input.shape),
        label='run_infer_batch',
    )
    local_prob = logits_to_local_prob(logits)
    b = int(local_prob.shape[0])
    trace_valid_tensor = _normalize_trace_valid(
        trace_valid,
        b=b,
        device=local_prob.device,
    )
    local_pick_idx, confidence = local_prob_to_pick_confidence(
        local_prob,
        trace_valid=trace_valid_tensor,
    )
    raw_pick_idx = local_pick_to_raw_pick(
        local_pick_idx,
        raw_sample_idx_local=raw_sample_idx_local,
        trace_valid=trace_valid_tensor,
    )
    if trace_valid_tensor is not None:
        local_prob = torch.where(
            trace_valid_tensor[:, None, :, None],
            local_prob,
            torch.zeros_like(local_prob),
        )
    return FineInferBatchResult(
        logits=logits,
        local_prob=local_prob,
        local_pick_idx=local_pick_idx,
        raw_pick_idx=raw_pick_idx,
        confidence=confidence,
    )


def _batch_eval_metrics(
    *,
    local_pick_idx: torch.Tensor,
    raw_pick_idx: torch.Tensor,
    confidence: torch.Tensor,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if 'local_pick_idx' not in batch:
        raise KeyError("batch['local_pick_idx'] is required for fine eval metrics")
    if 'raw_pick_idx' not in batch:
        raise KeyError("batch['raw_pick_idx'] is required for fine eval metrics")
    if 'label_valid' not in batch:
        raise KeyError("batch['label_valid'] is required for fine eval metrics")

    gt_local = _to_tensor_b1(batch['local_pick_idx'], name='batch.local_pick_idx', dtype=torch.int64)
    gt_raw = _to_tensor_b1(batch['raw_pick_idx'], name='batch.raw_pick_idx', dtype=torch.int64)
    valid = _to_tensor_b1(batch['label_valid'], name='batch.label_valid', dtype=torch.bool)

    pred_local_cpu = _to_tensor_b1(local_pick_idx.detach().cpu(), name='pred.local_pick_idx', dtype=torch.int64)
    pred_raw_cpu = _to_tensor_b1(raw_pick_idx.detach().cpu(), name='pred.raw_pick_idx', dtype=torch.int64)
    conf_cpu = _to_tensor_b1(confidence.detach().cpu(), name='pred.confidence', dtype=torch.float32)
    gt_local_cpu = gt_local.detach().cpu()
    gt_raw_cpu = gt_raw.detach().cpu()
    valid_cpu = valid.detach().cpu()

    if not torch.any(valid_cpu):
        empty = torch.empty((0,), dtype=torch.float32)
        return empty, empty, empty, empty, empty

    local_abs = (pred_local_cpu[valid_cpu] - gt_local_cpu[valid_cpu]).abs().to(dtype=torch.float32)
    raw_abs = (pred_raw_cpu[valid_cpu] - gt_raw_cpu[valid_cpu]).abs().to(dtype=torch.float32)
    conf_sel = conf_cpu[valid_cpu].to(dtype=torch.float32)
    return local_abs, local_abs.square(), raw_abs, raw_abs.square(), conf_sel


def run_infer_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion,
    vis_out_dir: str,
    vis_n: int,
    max_batches: int,
) -> InferEpochResult:
    del vis_n

    non_blocking = bool(device.type == 'cuda')
    infer_loss_sum = 0.0
    infer_samples = 0
    local_abs_chunks: list[torch.Tensor] = []
    local_sq_chunks: list[torch.Tensor] = []
    raw_abs_chunks: list[torch.Tensor] = []
    raw_sq_chunks: list[torch.Tensor] = []
    conf_chunks: list[torch.Tensor] = []

    Path(vis_out_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= int(max_batches):
                break
            if not isinstance(batch, dict):
                raise TypeError('loader must yield dict batches for fine eval')
            for key in ('input', 'target', 'raw_sample_idx_local'):
                if key not in batch:
                    raise KeyError(f'fine eval batch must contain {key!r}')

            x_in = batch['input'].to(device=device, non_blocking=non_blocking)
            x_tg = batch['target'].to(device=device, non_blocking=non_blocking)
            batch_result = run_infer_batch(
                model=model,
                x_bchw=x_in,
                raw_sample_idx_local=batch['raw_sample_idx_local'],
                trace_valid=batch.get('trace_valid'),
            )
            loss = criterion(batch_result.logits, x_tg, batch)

            bsize = int(x_in.shape[0])
            infer_loss_sum += float(loss.detach().item()) * bsize
            infer_samples += bsize

            local_abs, local_sq, raw_abs, raw_sq, conf_sel = _batch_eval_metrics(
                local_pick_idx=batch_result.local_pick_idx,
                raw_pick_idx=batch_result.raw_pick_idx,
                confidence=batch_result.confidence,
                batch=batch,
            )
            if int(local_abs.numel()) > 0:
                local_abs_chunks.append(local_abs)
                local_sq_chunks.append(local_sq)
                raw_abs_chunks.append(raw_abs)
                raw_sq_chunks.append(raw_sq)
                conf_chunks.append(conf_sel)

    if infer_samples <= 0:
        raise RuntimeError('no fine inference samples were processed')
    if not local_abs_chunks:
        raise RuntimeError('no valid fine labels were available during eval inference')

    local_abs_all = torch.cat(local_abs_chunks, dim=0)
    local_sq_all = torch.cat(local_sq_chunks, dim=0)
    raw_abs_all = torch.cat(raw_abs_chunks, dim=0)
    raw_sq_all = torch.cat(raw_sq_chunks, dim=0)
    conf_all = torch.cat(conf_chunks, dim=0)
    metrics = {
        'local_pick_mae': float(local_abs_all.mean().item()),
        'local_pick_rmse': float(math.sqrt(float(local_sq_all.mean().item()))),
        'raw_pick_mae': float(raw_abs_all.mean().item()),
        'raw_pick_rmse': float(math.sqrt(float(raw_sq_all.mean().item()))),
        'confidence_mean': float(conf_all.mean().item()),
    }
    return InferEpochResult(
        loss=infer_loss_sum / float(infer_samples),
        metrics=metrics,
    )
