from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from seisai_utils.config import optional_value, require_float, require_int, require_value

from seisai_engine.scheduler import WarmupCosineScheduler

Interval = Literal['step', 'epoch']


@dataclass(frozen=True)
class LRSchedulerSpec:
    scheduler: Any
    interval: Interval
    name: str
    monitor: str | None = None


def load_lr_scheduler_cfg(cfg: dict) -> dict | None:
    """Load optional scheduler config.

    Expected shape:

        scheduler:
          type: <str>
          interval: step|epoch   # optional depending on scheduler
          ... type-specific fields ...

    Missing/None => no scheduler.
    """

    if 'scheduler' not in cfg:
        return None
    sched = cfg['scheduler']
    if sched is None:
        return None
    if not isinstance(sched, dict):
        msg = 'config.scheduler must be dict or null'
        raise TypeError(msg)
    return sched


def _require_interval(sched_cfg: dict, default: Interval) -> Interval:
    interval = optional_value(
        sched_cfg,
        'interval',
        default,
        str,
        type_message='config.scheduler.interval must be str',
        coerce_default=True,
    )
    if interval not in ('step', 'epoch'):
        msg = 'config.scheduler.interval must be "step" or "epoch"'
        raise ValueError(msg)
    return interval  # type: ignore[return-value]


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg_root: dict,
    *,
    steps_per_epoch: int,
    epochs: int,
) -> LRSchedulerSpec | None:
    """Build a LR scheduler from config.

    Parameters
    ----------
    optimizer:
        Optimizer to schedule.
    cfg_root:
        Root config dict (must include optional "scheduler" section).
    steps_per_epoch:
        Number of optimizer steps per epoch.
    epochs:
        Total epochs.
    """

    if steps_per_epoch <= 0:
        msg = 'steps_per_epoch must be positive'
        raise ValueError(msg)
    if epochs <= 0:
        msg = 'epochs must be positive'
        raise ValueError(msg)

    sched_cfg = load_lr_scheduler_cfg(cfg_root)
    if sched_cfg is None:
        return None

    sched_type = require_value(
        sched_cfg,
        'type',
        str,
        type_message='config.scheduler.type must be str',
    )
    sched_type = str(sched_type)

    total_steps = int(steps_per_epoch) * int(epochs)
    if total_steps <= 0:
        msg = 'total_steps must be positive'
        raise ValueError(msg)

    if sched_type == 'warmup_cosine':
        interval = _require_interval(sched_cfg, 'step')
        if interval != 'step':
            msg = 'config.scheduler.interval must be "step" for warmup_cosine'
            raise ValueError(msg)
        warmup_steps = require_int(sched_cfg, 'warmup_steps')
        eta_min = optional_value(
            sched_cfg,
            'eta_min',
            0.0,
            (int, float),
            type_message='config.scheduler.eta_min must be float',
            coerce=float,
            coerce_default=True,
        )
        if warmup_steps < 0:
            msg = 'config.scheduler.warmup_steps must be >= 0'
            raise ValueError(msg)
        if warmup_steps > total_steps:
            msg = 'config.scheduler.warmup_steps must be <= total_steps'
            raise ValueError(msg)
        sched = WarmupCosineScheduler(
            optimizer,
            warmup_steps=int(warmup_steps),
            total_steps=int(total_steps),
            eta_min=float(eta_min),
        )
        return LRSchedulerSpec(
            scheduler=sched,
            interval=interval,
            name='warmup_cosine',
        )

    if sched_type == 'step_lr':
        interval = _require_interval(sched_cfg, 'epoch')
        step_size = require_int(sched_cfg, 'step_size')
        gamma = require_float(sched_cfg, 'gamma')
        if step_size <= 0:
            msg = 'config.scheduler.step_size must be positive'
            raise ValueError(msg)
        if gamma <= 0:
            msg = 'config.scheduler.gamma must be > 0'
            raise ValueError(msg)
        sched = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(step_size),
            gamma=float(gamma),
        )
        return LRSchedulerSpec(
            scheduler=sched,
            interval=interval,
            name='step_lr',
        )

    if sched_type == 'multistep_lr':
        interval = _require_interval(sched_cfg, 'epoch')
        milestones = require_value(
            sched_cfg,
            'milestones',
            list,
            type_message='config.scheduler.milestones must be list[int]',
        )
        if not all(isinstance(x, int) for x in milestones):
            msg = 'config.scheduler.milestones must be list[int]'
            raise TypeError(msg)
        if len(milestones) == 0:
            msg = 'config.scheduler.milestones must be non-empty'
            raise ValueError(msg)
        gamma = require_float(sched_cfg, 'gamma')
        if gamma <= 0:
            msg = 'config.scheduler.gamma must be > 0'
            raise ValueError(msg)
        sched = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(x) for x in milestones],
            gamma=float(gamma),
        )
        return LRSchedulerSpec(
            scheduler=sched,
            interval=interval,
            name='multistep_lr',
        )

    if sched_type == 'exponential_lr':
        interval = _require_interval(sched_cfg, 'epoch')
        gamma = require_float(sched_cfg, 'gamma')
        if gamma <= 0:
            msg = 'config.scheduler.gamma must be > 0'
            raise ValueError(msg)
        sched = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(gamma),
        )
        return LRSchedulerSpec(
            scheduler=sched,
            interval=interval,
            name='exponential_lr',
        )

    if sched_type == 'cosine_annealing':
        interval = _require_interval(sched_cfg, 'epoch')
        default_t_max = int(epochs) if interval == 'epoch' else int(total_steps)
        t_max = optional_value(
            sched_cfg,
            't_max',
            default_t_max,
            int,
            type_message='config.scheduler.t_max must be int',
            coerce=int,
            coerce_default=True,
        )
        eta_min = optional_value(
            sched_cfg,
            'eta_min',
            0.0,
            (int, float),
            type_message='config.scheduler.eta_min must be float',
            coerce=float,
            coerce_default=True,
        )
        if t_max <= 0:
            msg = 'config.scheduler.t_max must be positive'
            raise ValueError(msg)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(t_max),
            eta_min=float(eta_min),
        )
        return LRSchedulerSpec(
            scheduler=sched,
            interval=interval,
            name='cosine_annealing',
        )

    if sched_type == 'cosine_warm_restarts':
        interval = _require_interval(sched_cfg, 'epoch')
        t0 = require_int(sched_cfg, 't_0')
        t_mult = optional_value(
            sched_cfg,
            't_mult',
            1,
            int,
            type_message='config.scheduler.t_mult must be int',
            coerce=int,
            coerce_default=True,
        )
        eta_min = optional_value(
            sched_cfg,
            'eta_min',
            0.0,
            (int, float),
            type_message='config.scheduler.eta_min must be float',
            coerce=float,
            coerce_default=True,
        )
        if t0 <= 0:
            msg = 'config.scheduler.t_0 must be positive'
            raise ValueError(msg)
        if t_mult <= 0:
            msg = 'config.scheduler.t_mult must be positive'
            raise ValueError(msg)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(t0),
            T_mult=int(t_mult),
            eta_min=float(eta_min),
        )
        return LRSchedulerSpec(
            scheduler=sched,
            interval=interval,
            name='cosine_warm_restarts',
        )

    if sched_type == 'one_cycle':
        interval = _require_interval(sched_cfg, 'step')
        if interval != 'step':
            msg = 'config.scheduler.interval must be "step" for one_cycle'
            raise ValueError(msg)
        max_lr = require_float(sched_cfg, 'max_lr')
        pct_start = optional_value(
            sched_cfg,
            'pct_start',
            0.3,
            (int, float),
            type_message='config.scheduler.pct_start must be float',
            coerce=float,
            coerce_default=True,
        )
        div_factor = optional_value(
            sched_cfg,
            'div_factor',
            25.0,
            (int, float),
            type_message='config.scheduler.div_factor must be float',
            coerce=float,
            coerce_default=True,
        )
        final_div_factor = optional_value(
            sched_cfg,
            'final_div_factor',
            1e4,
            (int, float),
            type_message='config.scheduler.final_div_factor must be float',
            coerce=float,
            coerce_default=True,
        )
        anneal_strategy = optional_value(
            sched_cfg,
            'anneal_strategy',
            'cos',
            str,
            type_message='config.scheduler.anneal_strategy must be str',
            coerce_default=True,
        )
        if anneal_strategy not in ('cos', 'linear'):
            msg = 'config.scheduler.anneal_strategy must be "cos" or "linear"'
            raise ValueError(msg)
        if not (0.0 < float(pct_start) < 1.0):
            msg = 'config.scheduler.pct_start must be in (0, 1)'
            raise ValueError(msg)
        if div_factor <= 0:
            msg = 'config.scheduler.div_factor must be > 0'
            raise ValueError(msg)
        if final_div_factor <= 0:
            msg = 'config.scheduler.final_div_factor must be > 0'
            raise ValueError(msg)

        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(max_lr),
            total_steps=int(total_steps),
            pct_start=float(pct_start),
            anneal_strategy=str(anneal_strategy),
            div_factor=float(div_factor),
            final_div_factor=float(final_div_factor),
        )
        return LRSchedulerSpec(
            scheduler=sched,
            interval=interval,
            name='one_cycle',
        )

    if sched_type == 'reduce_on_plateau':
        interval = _require_interval(sched_cfg, 'epoch')
        if interval != 'epoch':
            msg = 'config.scheduler.interval must be "epoch" for reduce_on_plateau'
            raise ValueError(msg)
        monitor = optional_value(
            sched_cfg,
            'monitor',
            'infer_loss',
            str,
            type_message='config.scheduler.monitor must be str',
            coerce_default=True,
        )
        mode = optional_value(
            sched_cfg,
            'mode',
            'min',
            str,
            type_message='config.scheduler.mode must be str',
            coerce_default=True,
        )
        if mode not in ('min', 'max'):
            msg = 'config.scheduler.mode must be "min" or "max"'
            raise ValueError(msg)
        factor = optional_value(
            sched_cfg,
            'factor',
            0.1,
            (int, float),
            type_message='config.scheduler.factor must be float',
            coerce=float,
            coerce_default=True,
        )
        patience = optional_value(
            sched_cfg,
            'patience',
            10,
            int,
            type_message='config.scheduler.patience must be int',
            coerce=int,
            coerce_default=True,
        )
        threshold = optional_value(
            sched_cfg,
            'threshold',
            1e-4,
            (int, float),
            type_message='config.scheduler.threshold must be float',
            coerce=float,
            coerce_default=True,
        )
        min_lr = optional_value(
            sched_cfg,
            'min_lr',
            0.0,
            (int, float),
            type_message='config.scheduler.min_lr must be float',
            coerce=float,
            coerce_default=True,
        )
        if factor <= 0 or factor >= 1:
            msg = 'config.scheduler.factor must be in (0, 1)'
            raise ValueError(msg)
        if patience < 0:
            msg = 'config.scheduler.patience must be >= 0'
            raise ValueError(msg)
        if threshold < 0:
            msg = 'config.scheduler.threshold must be >= 0'
            raise ValueError(msg)

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(mode),
            factor=float(factor),
            patience=int(patience),
            threshold=float(threshold),
            min_lr=float(min_lr),
        )
        return LRSchedulerSpec(
            scheduler=sched,
            interval=interval,
            name='reduce_on_plateau',
            monitor=str(monitor),
        )

    known = [
        'warmup_cosine',
        'step_lr',
        'multistep_lr',
        'exponential_lr',
        'cosine_annealing',
        'cosine_warm_restarts',
        'one_cycle',
        'reduce_on_plateau',
    ]
    msg = f'unknown scheduler type: {sched_type} (known: {", ".join(known)})'
    raise ValueError(msg)
