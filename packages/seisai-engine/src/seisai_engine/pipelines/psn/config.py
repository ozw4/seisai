from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from seisai_utils.config import (
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_value,
)

from seisai_engine.pipelines.common.config_loaders import load_common_train_config

if TYPE_CHECKING:
    from seisai_engine.pipelines.common.config_schema import CommonTrainConfig

__all__ = ['PsnTrainConfig', 'load_psn_train_config']


@dataclass(frozen=True)
class PsnTrainCfg:
    lr: float
    subset_traces: int


@dataclass(frozen=True)
class PsnInferCfg:
    subset_traces: int


@dataclass(frozen=True)
class PsnModelCfg:
    backbone: str
    pretrained: bool
    in_chans: int
    out_chans: int


@dataclass(frozen=True)
class PsnCkptCfg:
    save_best_only: bool
    metric: str
    mode: str


@dataclass(frozen=True)
class PsnTrainConfig:
    common: CommonTrainConfig
    train: PsnTrainCfg
    infer: PsnInferCfg
    model: PsnModelCfg
    ckpt: PsnCkptCfg


def load_psn_train_config(cfg: dict) -> PsnTrainConfig:
    common = load_common_train_config(cfg)

    train_cfg = require_dict(cfg, 'train')
    infer_cfg = require_dict(cfg, 'infer')
    ckpt_cfg = require_dict(cfg, 'ckpt')
    model_cfg = require_dict(cfg, 'model')

    lr = require_float(train_cfg, 'lr')
    train_subset_traces = require_int(train_cfg, 'subset_traces')
    infer_subset_traces = require_int(infer_cfg, 'subset_traces')

    save_best_only = require_bool(ckpt_cfg, 'save_best_only')
    metric = require_value(
        ckpt_cfg,
        'metric',
        str,
        type_message='config.ckpt.metric must be str',
    )
    mode = require_value(
        ckpt_cfg,
        'mode',
        str,
        type_message='config.ckpt.mode must be str',
    )

    backbone = require_value(
        model_cfg,
        'backbone',
        str,
        type_message='config.model.backbone must be str',
    )
    pretrained = require_bool(model_cfg, 'pretrained')
    in_chans = require_int(model_cfg, 'in_chans')
    out_chans = require_int(model_cfg, 'out_chans')

    return PsnTrainConfig(
        common=common,
        train=PsnTrainCfg(
            lr=float(lr),
            subset_traces=int(train_subset_traces),
        ),
        infer=PsnInferCfg(
            subset_traces=int(infer_subset_traces),
        ),
        model=PsnModelCfg(
            backbone=str(backbone),
            pretrained=bool(pretrained),
            in_chans=int(in_chans),
            out_chans=int(out_chans),
        ),
        ckpt=PsnCkptCfg(
            save_best_only=bool(save_best_only),
            metric=str(metric),
            mode=str(mode),
        ),
    )
