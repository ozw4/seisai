from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from seisai_utils.config import (
    optional_str,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_value,
)

from seisai_engine.loss import composite
from seisai_engine.pipelines.common.config_loaders import load_common_train_config
from seisai_engine.pipelines.common.encdec2d_cfg import build_encdec2d_kwargs

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
    stage_strides: list[tuple[int, int]] | None
    extra_stages: int
    extra_stage_strides: list[tuple[int, int]] | None
    extra_stage_channels: tuple[int, ...] | None
    extra_stage_use_bn: bool
    pre_stages: int
    pre_stage_strides: list[tuple[int, int]] | None
    pre_stage_kernels: tuple[int, ...] | None
    pre_stage_channels: tuple[int, ...] | None
    pre_stage_use_bn: bool
    pre_stage_antialias: bool
    pre_stage_aa_taps: int
    pre_stage_aa_pad_mode: str
    decoder_channels: tuple[int, ...]
    decoder_scales: tuple[int, ...]
    upsample_mode: str
    attention_type: str | None
    intermediate_conv: bool


@dataclass(frozen=True)
class PsnCkptCfg:
    save_best_only: bool
    metric: str
    mode: str


@dataclass(frozen=True)
class PsnTrainConfig:
    common: CommonTrainConfig
    train: PsnTrainCfg
    loss_specs_train: tuple[composite.LossSpec, ...]
    loss_specs_eval: tuple[composite.LossSpec, ...]
    infer: PsnInferCfg
    model: PsnModelCfg
    ckpt: PsnCkptCfg


def _load_psn_loss_specs(
    cfg: dict, *, train_cfg: dict
) -> tuple[tuple[composite.LossSpec, ...], tuple[composite.LossSpec, ...]]:
    train_loss_scope = optional_str(train_cfg, 'loss_scope', 'all')
    train_loss_specs = composite.parse_loss_specs(
        train_cfg.get('losses', None),
        default_scope=train_loss_scope,
        label='train.losses',
        scope_label='train.loss_scope',
    )

    eval_cfg = cfg.get('eval')
    if eval_cfg is None:
        eval_loss_specs = train_loss_specs
    else:
        if not isinstance(eval_cfg, dict):
            raise TypeError('eval must be dict')
        eval_losses = eval_cfg.get('losses', None)
        if eval_losses is None:
            eval_loss_specs = train_loss_specs
        else:
            eval_loss_scope = optional_str(eval_cfg, 'loss_scope', train_loss_scope)
            eval_loss_specs = composite.parse_loss_specs(
                eval_losses,
                default_scope=eval_loss_scope,
                label='eval.losses',
                scope_label='eval.loss_scope',
            )

    return tuple(train_loss_specs), tuple(eval_loss_specs)


def load_psn_train_config(cfg: dict) -> PsnTrainConfig:
    common = load_common_train_config(cfg)

    train_cfg = require_dict(cfg, 'train')
    infer_cfg = require_dict(cfg, 'infer')
    ckpt_cfg = require_dict(cfg, 'ckpt')
    model_cfg = require_dict(cfg, 'model')

    lr = require_float(train_cfg, 'lr')
    train_subset_traces = require_int(train_cfg, 'subset_traces')
    infer_subset_traces = require_int(infer_cfg, 'subset_traces')
    loss_specs_train, loss_specs_eval = _load_psn_loss_specs(cfg, train_cfg=train_cfg)

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

    in_chans = require_int(model_cfg, 'in_chans')
    out_chans = require_int(model_cfg, 'out_chans')
    if int(in_chans) != 1:
        msg = 'model.in_chans must be 1 (waveform only)'
        raise ValueError(msg)
    if int(out_chans) != 3:
        msg = 'model.out_chans must be 3 (P/S/Noise)'
        raise ValueError(msg)

    return PsnTrainConfig(
        common=common,
        train=PsnTrainCfg(
            lr=float(lr),
            subset_traces=int(train_subset_traces),
        ),
        loss_specs_train=loss_specs_train,
        loss_specs_eval=loss_specs_eval,
        infer=PsnInferCfg(
            subset_traces=int(infer_subset_traces),
        ),
        model=PsnModelCfg(
            **build_encdec2d_kwargs(
                model_cfg,
                in_chans=int(in_chans),
                out_chans=int(out_chans),
            )
        ),
        ckpt=PsnCkptCfg(
            save_best_only=bool(save_best_only),
            metric=str(metric),
            mode=str(mode),
        ),
    )
