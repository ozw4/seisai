from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
from seisai_utils.logging import MetricLogger, SmoothedValue
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import DataLoader

__all__ = ['after_step', 'setup_amp', 'train_one_epoch', 'train_step']


def setup_amp(
    device: torch.device,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[Callable[[], Any], torch.cuda.amp.GradScaler | None]:
    do_amp = bool(use_amp and device.type == 'cuda')
    # 新API(将来の非推奨回避): torch.amp.autocast('cuda', enabled=True/False)
    autocast_ctx = (lambda: torch.amp.autocast('cuda')) if do_amp else nullcontext
    local_scaler = (
        scaler
        if (do_amp and scaler is not None)
        else (torch.cuda.amp.GradScaler() if do_amp else None)
    )
    return autocast_ctx, local_scaler


def train_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: Callable[..., torch.Tensor],
    *,
    device: torch.device,
    autocast_ctx: Callable[[], Any],
    local_scaler: torch.cuda.amp.GradScaler | None,
    gradient_accumulation_steps: int,
    max_norm: float | None,
    batch_index: int,
) -> tuple[torch.Tensor, bool, int]:
    if not isinstance(batch, dict) or ('input' not in batch or 'target' not in batch):
        msg = "batch must be a dict containing 'input' and 'target'"
        raise KeyError(msg)

    x = batch['input'].to(device, non_blocking=True)
    y = batch['target'].to(device, non_blocking=True)

    with autocast_ctx():
        pred = model(x)
        loss = criterion(pred, y, batch)

    loss_to_backprop = (
        loss / gradient_accumulation_steps if gradient_accumulation_steps > 1 else loss
    )
    if local_scaler is not None:
        local_scaler.scale(loss_to_backprop).backward()
    else:
        loss_to_backprop.backward()

    batch_size = int(x.shape[0])

    did_step = False
    if (batch_index + 1) % gradient_accumulation_steps == 0:
        if local_scaler is not None:
            local_scaler.unscale_(optimizer)
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        if local_scaler is not None:
            local_scaler.step(optimizer)
            local_scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        did_step = True

    return loss, did_step, batch_size


def after_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    meter: MetricLogger,
    step: int,
    lr_scheduler: Any | None = None,
    ema: Any | None = None,
    on_step: Callable[[int, dict[str, float]], None] | None = None,
) -> int:
    if lr_scheduler is not None:
        lr_scheduler.step()

    step += 1
    lr0 = float(optimizer.param_groups[0].get('lr', 0.0))
    meter.update(loss=float(loss.detach().item()), lr=lr0)

    if ema is not None:
        ema.update(model)

    if on_step is not None:
        on_step(step, {'loss': float(meter.meters['loss'].global_avg), 'lr': lr0})

    return step


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[..., torch.Tensor],
    *,
    device: torch.device,
    lr_scheduler: Any | None = None,
    gradient_accumulation_steps: int = 1,
    max_norm: float | None = 1.0,
    use_amp: bool = True,
    scaler: torch.cuda.amp.GradScaler | None = None,
    ema: Any | None = None,  # expects .update(model)
    step_offset: int = 0,
    print_freq: int = 50,
    on_step: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, float]:
    if not isinstance(device, torch.device):
        msg = 'device must be a torch.device'
        raise TypeError(msg)
    if gradient_accumulation_steps <= 0:
        msg = 'gradient_accumulation_steps must be > 0'
        raise ValueError(msg)

    model.train()
    meter = MetricLogger(delimiter='\t')
    meter.add_meter('loss', SmoothedValue(fmt='{median:.4E} ({global_avg:.4E})'))
    meter.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.4E}'))

    total_samples = 0
    step = step_offset

    autocast_ctx, local_scaler = setup_amp(device, use_amp, scaler)

    optimizer.zero_grad(set_to_none=True)

    saw_any_batch = False
    last_loss: torch.Tensor | None = None
    last_i = -1
    for i, batch in enumerate(meter.log_every(dataloader, print_freq, header='Train')):
        saw_any_batch = True
        loss, did_step, batch_size = train_step(
            model,
            batch,
            optimizer,
            criterion,
            device=device,
            autocast_ctx=autocast_ctx,
            local_scaler=local_scaler,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_norm=max_norm,
            batch_index=i,
        )

        last_loss = loss
        last_i = i
        total_samples += batch_size

        if did_step:
            step = after_step(
                model=model,
                optimizer=optimizer,
                loss=loss,
                meter=meter,
                step=step,
                lr_scheduler=lr_scheduler,
                ema=ema,
                on_step=on_step,
            )

    # 端数(accumの余り)があれば最後に一度だけstepする
    if saw_any_batch and (last_i + 1) % gradient_accumulation_steps != 0:
        if local_scaler is not None:
            local_scaler.unscale_(optimizer)
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        if local_scaler is not None:
            local_scaler.step(optimizer)
            local_scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if last_loss is None:
            msg = 'internal error: last_loss is None'
            raise RuntimeError(msg)
        step = after_step(
            model=model,
            optimizer=optimizer,
            loss=last_loss,
            meter=meter,
            step=step,
            lr_scheduler=lr_scheduler,
            ema=ema,
            on_step=on_step,
        )

    meter.synchronize_between_processes()

    if not saw_any_batch:
        msg = 'dataloader yielded no batches'
        raise ValueError(msg)

    return {
        'loss': float(meter.meters['loss'].global_avg),
        'steps': float(step - step_offset),
        'samples': float(total_samples),
    }
