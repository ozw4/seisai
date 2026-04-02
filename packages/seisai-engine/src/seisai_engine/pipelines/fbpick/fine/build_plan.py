from __future__ import annotations

import numpy as np
from seisai_dataset import BuildPlan, InputOnlyPlan
from seisai_dataset.builder.builder import IdentitySignal, SelectStack
from seisai_dataset.builder.fb_local_ops import FBLocalGaussMap

from .config import (
    FineInferConfig,
    FineInputCfg,
    FineTargetCfg,
    FineTrainConfig,
)

__all__ = [
    'build_input_only_plan',
    'build_input_only_plan_from_config',
    'build_plan',
    'build_plan_from_config',
]


def _require_input_cfg(cfg: FineInputCfg) -> FineInputCfg:
    if not isinstance(cfg, FineInputCfg):
        msg = 'input_cfg must be FineInputCfg'
        raise TypeError(msg)
    if cfg.input_key != 'input':
        msg = 'config.input.input_key must be "input" in Phase 5'
        raise ValueError(msg)
    if cfg.use_offset_channel:
        msg = 'config.input.use_offset_channel must be false in Phase 5'
        raise ValueError(msg)
    if cfg.use_relative_time_channel:
        msg = 'config.input.use_relative_time_channel must be false in Phase 5'
        raise ValueError(msg)
    if len(cfg.stack_keys) != 1:
        msg = 'fine input stack must contain exactly 1 key in Phase 5'
        raise ValueError(msg)
    return cfg


def _require_target_cfg(cfg: FineTargetCfg) -> FineTargetCfg:
    if not isinstance(cfg, FineTargetCfg):
        msg = 'target_cfg must be FineTargetCfg'
        raise TypeError(msg)
    if float(cfg.sigma) <= 0.0:
        msg = 'config.target.sigma must be positive'
        raise ValueError(msg)
    if cfg.local_pick_idx_key != 'local_pick_idx':
        msg = 'config.target.local_pick_idx_key must be "local_pick_idx" in Phase 5'
        raise ValueError(msg)
    if cfg.target_key != 'target':
        msg = 'config.target.target_key must be "target" in Phase 5'
        raise ValueError(msg)
    return cfg


def _build_wave_ops(input_cfg: FineInputCfg) -> list:
    cfg = _require_input_cfg(input_cfg)
    return [
        IdentitySignal(
            src='x_view_local',
            dst=cfg.amplitude_key,
            copy=False,
        ),
    ]


def _build_input_stack(input_cfg: FineInputCfg) -> SelectStack:
    cfg = _require_input_cfg(input_cfg)
    return SelectStack(
        keys=list(cfg.stack_keys),
        dst=cfg.input_key,
        dtype=np.float32,
        to_torch=True,
    )


def build_plan(*, input_cfg: FineInputCfg, target_cfg: FineTargetCfg) -> BuildPlan:
    input_cfg_checked = _require_input_cfg(input_cfg)
    target_cfg_checked = _require_target_cfg(target_cfg)
    return BuildPlan(
        wave_ops=_build_wave_ops(input_cfg_checked),
        label_ops=[
            FBLocalGaussMap(
                dst=target_cfg_checked.probability_key,
                sigma=float(target_cfg_checked.sigma),
                src=target_cfg_checked.local_pick_idx_key,
                valid_key='label_valid',
            ),
        ],
        input_stack=_build_input_stack(input_cfg_checked),
        target_stack=SelectStack(
            keys=[target_cfg_checked.probability_key],
            dst=target_cfg_checked.target_key,
            dtype=np.float32,
            to_torch=True,
        ),
    )


def build_input_only_plan(*, input_cfg: FineInputCfg) -> InputOnlyPlan:
    input_cfg_checked = _require_input_cfg(input_cfg)
    return InputOnlyPlan(
        wave_ops=_build_wave_ops(input_cfg_checked),
        label_ops=[],
        input_stack=_build_input_stack(input_cfg_checked),
    )


def build_plan_from_config(cfg: FineTrainConfig) -> BuildPlan:
    if not isinstance(cfg, FineTrainConfig):
        msg = 'cfg must be FineTrainConfig'
        raise TypeError(msg)
    return build_plan(input_cfg=cfg.input, target_cfg=cfg.target)


def build_input_only_plan_from_config(cfg: FineTrainConfig | FineInferConfig) -> InputOnlyPlan:
    if not isinstance(cfg, (FineTrainConfig, FineInferConfig)):
        msg = 'cfg must be FineTrainConfig or FineInferConfig'
        raise TypeError(msg)
    return build_input_only_plan(input_cfg=cfg.input)
