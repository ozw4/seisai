from __future__ import annotations

from dataclasses import dataclass

import torch

from .ema import ModelEMA


@dataclass(frozen=True)
class EmaConfig:
    decay: float = 0.999
    start_step: int = 0
    update_every: int = 1
    use_for_infer: bool = True
    device: torch.device | None = None

    def __post_init__(self) -> None:
        decay = float(self.decay)
        if not (0.0 < decay < 1.0):
            raise ValueError('ema.decay must be in (0, 1)')
        if int(self.start_step) < 0:
            raise ValueError('ema.start_step must be >= 0')
        if int(self.update_every) <= 0:
            raise ValueError('ema.update_every must be >= 1')


class EmaController:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: EmaConfig,
        *,
        initial_step: int = 0,
    ) -> None:
        if not isinstance(initial_step, int) or initial_step < 0:
            raise ValueError('initial_step must be int >= 0')
        self.cfg = cfg
        self._step = int(initial_step)
        self.ema = ModelEMA(model, decay=float(cfg.decay), device=cfg.device)

    def set_step(self, step: int) -> None:
        if not isinstance(step, int) or step < 0:
            raise ValueError('step must be int >= 0')
        self._step = int(step)

    @property
    def step(self) -> int:
        return self._step

    @property
    def module(self) -> torch.nn.Module:
        return self.ema.module

    def update(self, model: torch.nn.Module) -> None:
        self._step += 1
        if self._step < int(self.cfg.start_step):
            self.ema.set(model)
            return
        if int(self.cfg.update_every) != 1:
            if (
                (self._step - int(self.cfg.start_step))
                % int(self.cfg.update_every)
                != 0
            ):
                return
        self.ema.update(model)

    def state_dict_cpu(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.ema.module.state_dict().items()}
