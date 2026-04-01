from __future__ import annotations

import numpy as np
from seisai_dataset import BuildPlan, InputOnlyPlan
from seisai_dataset.builder.builder import (
    FBGaussMap,
    MakeOffsetChannel,
    MakeTimeChannel,
    SelectStack,
)

from .config import CoarseInputCfg, CoarseTargetCfg, CoarseTrainConfig

__all__ = [
    'build_input_only_plan',
    'build_input_only_plan_from_config',
    'build_plan',
    'build_plan_from_config',
]


def _require_input_cfg(cfg: CoarseInputCfg) -> CoarseInputCfg:
    if not isinstance(cfg, CoarseInputCfg):
        msg = 'input_cfg must be CoarseInputCfg'
        raise TypeError(msg)
    if cfg.amplitude_key != 'x_view':
        msg = 'config.input.amplitude_key must be "x_view" in Phase 2'
        raise ValueError(msg)
    if not cfg.use_offset_channel:
        msg = 'config.input.use_offset_channel must be true in Phase 2'
        raise ValueError(msg)
    if not cfg.use_time_channel:
        msg = 'config.input.use_time_channel must be true in Phase 2'
        raise ValueError(msg)
    if cfg.offset_mode != 'abs':
        msg = 'config.input.offset_mode must be "abs" in Phase 2'
        raise ValueError(msg)
    if len(cfg.stack_keys) != 3:
        msg = 'coarse input stack must contain exactly 3 keys'
        raise ValueError(msg)
    return cfg


def _require_target_cfg(cfg: CoarseTargetCfg) -> CoarseTargetCfg:
    if not isinstance(cfg, CoarseTargetCfg):
        msg = 'target_cfg must be CoarseTargetCfg'
        raise TypeError(msg)
    if float(cfg.sigma) <= 0.0:
        msg = 'config.target.sigma must be positive'
        raise ValueError(msg)
    return cfg


def _build_wave_ops(input_cfg: CoarseInputCfg) -> list:
    cfg = _require_input_cfg(input_cfg)
    return [
        MakeOffsetChannel(
            dst=cfg.abs_offset_key,
            normalize=bool(cfg.offset_normalize),
            mode='abs',
        ),
        MakeTimeChannel(dst=cfg.absolute_time_key),
    ]


def _build_input_stack(input_cfg: CoarseInputCfg) -> SelectStack:
    cfg = _require_input_cfg(input_cfg)
    return SelectStack(
        keys=list(cfg.stack_keys),
        dst=cfg.input_key,
        dtype=np.float32,
        to_torch=True,
    )


def build_plan(*, input_cfg: CoarseInputCfg, target_cfg: CoarseTargetCfg) -> BuildPlan:
    input_cfg_checked = _require_input_cfg(input_cfg)
    target_cfg_checked = _require_target_cfg(target_cfg)
    return BuildPlan(
        wave_ops=_build_wave_ops(input_cfg_checked),
        label_ops=[
            FBGaussMap(
                dst=target_cfg_checked.probability_key,
                sigma=float(target_cfg_checked.sigma),
                src=target_cfg_checked.fb_index_key,
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


def build_input_only_plan(*, input_cfg: CoarseInputCfg) -> InputOnlyPlan:
    input_cfg_checked = _require_input_cfg(input_cfg)
    return InputOnlyPlan(
        wave_ops=_build_wave_ops(input_cfg_checked),
        label_ops=[],
        input_stack=_build_input_stack(input_cfg_checked),
    )


def build_plan_from_config(cfg: CoarseTrainConfig) -> BuildPlan:
    if not isinstance(cfg, CoarseTrainConfig):
        msg = 'cfg must be CoarseTrainConfig'
        raise TypeError(msg)
    return build_plan(input_cfg=cfg.input, target_cfg=cfg.target)


def build_input_only_plan_from_config(cfg: CoarseTrainConfig) -> InputOnlyPlan:
    if not isinstance(cfg, CoarseTrainConfig):
        msg = 'cfg must be CoarseTrainConfig'
        raise TypeError(msg)
    return build_input_only_plan(input_cfg=cfg.input)
