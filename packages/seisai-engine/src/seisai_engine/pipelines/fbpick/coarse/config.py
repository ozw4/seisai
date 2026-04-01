from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from seisai_utils.config import (
    optional_bool,
    optional_str,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_value,
)

from seisai_engine.pipelines.common.config_io import resolve_relpath
from seisai_engine.pipelines.common.config_loaders import load_common_train_config
from seisai_engine.pipelines.common.encdec2d_cfg import build_encdec2d_kwargs
from seisai_engine.pipelines.fbpick.common.config import (
    FbpickCommonConfig,
    load_fbpick_common_config,
)

if TYPE_CHECKING:
    from seisai_engine.pipelines.common.config_schema import CommonTrainConfig

__all__ = [
    'CoarseCkptCfg',
    'CoarseEvalCfg',
    'CoarseInputCfg',
    'CoarseInferCfg',
    'CoarseModelCfg',
    'CoarseTargetCfg',
    'CoarseTrainCfg',
    'CoarseTrainConfig',
    'load_coarse_train_config',
]


def _require_non_empty_str(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        msg = f'{label} must be str'
        raise TypeError(msg)
    value_str = value.strip()
    if value_str == '':
        msg = f'{label} must not be empty'
        raise ValueError(msg)
    return value_str


def _validate_model_channels(*, in_chans: int, out_chans: int, label_prefix: str) -> None:
    if int(in_chans) != 3:
        msg = (
            f'{label_prefix}.in_chans must be 3 '
            '(amplitude / abs offset / absolute time)'
        )
        raise ValueError(msg)
    if int(out_chans) != 1:
        msg = f'{label_prefix}.out_chans must be 1 (coarse first-break probability)'
        raise ValueError(msg)


def _resolve_common_train_cfg(
    cfg: dict,
    *,
    base_dir: str | Path | None,
) -> dict:
    if base_dir is None:
        return cfg
    paths_cfg = require_dict(cfg, 'paths')
    out_dir_raw = require_value(
        paths_cfg,
        'out_dir',
        str,
        type_message='config.paths.out_dir must be str',
    )
    resolved_paths = dict(paths_cfg)
    resolved_paths['out_dir'] = resolve_relpath(base_dir, out_dir_raw)
    resolved_cfg = dict(cfg)
    resolved_cfg['paths'] = resolved_paths
    return resolved_cfg


@dataclass(frozen=True)
class CoarseTrainCfg:
    lr: float
    subset_traces: int
    use_label_valid_mask: bool

    def __post_init__(self) -> None:
        if float(self.lr) <= 0.0:
            msg = 'config.train.lr must be positive'
            raise ValueError(msg)
        if int(self.subset_traces) <= 0:
            msg = 'config.train.subset_traces must be positive'
            raise ValueError(msg)


@dataclass(frozen=True)
class CoarseEvalCfg:
    use_label_valid_mask: bool


@dataclass(frozen=True)
class CoarseInferCfg:
    subset_traces: int

    def __post_init__(self) -> None:
        if int(self.subset_traces) <= 0:
            msg = 'config.infer.subset_traces must be positive'
            raise ValueError(msg)


@dataclass(frozen=True)
class CoarseInputCfg:
    amplitude_key: str
    abs_offset_key: str
    absolute_time_key: str
    input_key: str
    use_offset_channel: bool
    use_time_channel: bool
    offset_mode: str
    offset_normalize: bool

    def __post_init__(self) -> None:
        amplitude_key = _require_non_empty_str(
            self.amplitude_key,
            label='config.input.amplitude_key',
        )
        abs_offset_key = _require_non_empty_str(
            self.abs_offset_key,
            label='config.input.abs_offset_key',
        )
        absolute_time_key = _require_non_empty_str(
            self.absolute_time_key,
            label='config.input.absolute_time_key',
        )
        _require_non_empty_str(self.input_key, label='config.input.input_key')

        if amplitude_key != 'x_view':
            msg = 'config.input.amplitude_key must be "x_view" in Phase 2'
            raise ValueError(msg)
        if len({amplitude_key, abs_offset_key, absolute_time_key}) != 3:
            msg = 'config.input amplitude/abs_offset/absolute_time keys must be distinct'
            raise ValueError(msg)
        if not bool(self.use_offset_channel):
            msg = 'config.input.use_offset_channel must be true in Phase 2'
            raise ValueError(msg)
        if not bool(self.use_time_channel):
            msg = 'config.input.use_time_channel must be true in Phase 2'
            raise ValueError(msg)
        if str(self.offset_mode) != 'abs':
            msg = 'config.input.offset_mode must be "abs" in Phase 2'
            raise ValueError(msg)

    @property
    def stack_keys(self) -> tuple[str, str, str]:
        return (
            str(self.amplitude_key),
            str(self.abs_offset_key),
            str(self.absolute_time_key),
        )


@dataclass(frozen=True)
class CoarseTargetCfg:
    sigma: float
    fb_index_key: str
    probability_key: str
    target_key: str

    def __post_init__(self) -> None:
        if float(self.sigma) <= 0.0:
            msg = 'config.target.sigma must be positive'
            raise ValueError(msg)
        _require_non_empty_str(
            self.fb_index_key,
            label='config.target.fb_index_key',
        )
        _require_non_empty_str(
            self.probability_key,
            label='config.target.probability_key',
        )
        _require_non_empty_str(
            self.target_key,
            label='config.target.target_key',
        )


@dataclass(frozen=True)
class CoarseModelCfg:
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
    disable_prestage_skip_indices: tuple[int, ...]
    disable_backbone_skip_indices: tuple[int, ...]
    decoder_channels: tuple[int, ...]
    decoder_scales: tuple[int, ...]
    upsample_mode: str
    attention_type: str | None
    intermediate_conv: bool


@dataclass(frozen=True)
class CoarseCkptCfg:
    save_best_only: bool
    metric: str
    mode: str


@dataclass(frozen=True)
class CoarseTrainConfig:
    common: CommonTrainConfig
    fbpick: FbpickCommonConfig
    train: CoarseTrainCfg
    eval: CoarseEvalCfg
    infer: CoarseInferCfg
    input: CoarseInputCfg
    target: CoarseTargetCfg
    model: CoarseModelCfg
    ckpt: CoarseCkptCfg

    def __post_init__(self) -> None:
        _validate_model_channels(
            in_chans=int(self.model.in_chans),
            out_chans=int(self.model.out_chans),
            label_prefix='config.model',
        )
        if len(self.input.stack_keys) != int(self.model.in_chans):
            msg = 'config.model.in_chans must match the fixed 3-channel coarse input stack'
            raise ValueError(msg)


def load_coarse_train_config(
    cfg: dict,
    *,
    base_dir: str | Path | None = None,
) -> CoarseTrainConfig:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    common = load_common_train_config(
        _resolve_common_train_cfg(cfg, base_dir=base_dir)
    )
    fbpick = load_fbpick_common_config(cfg, base_dir=base_dir)

    train_cfg = require_dict(cfg, 'train')
    infer_cfg = require_dict(cfg, 'infer')
    ckpt_cfg = require_dict(cfg, 'ckpt')
    model_cfg = require_dict(cfg, 'model')

    input_cfg = cfg.get('input', {})
    if not isinstance(input_cfg, dict):
        msg = 'config.input must be dict'
        raise TypeError(msg)
    target_cfg = require_dict(cfg, 'target')

    lr = require_float(train_cfg, 'lr')
    train_subset_traces = require_int(train_cfg, 'subset_traces')
    train_use_label_valid_mask = optional_bool(
        train_cfg,
        'use_label_valid_mask',
        default=True,
    )

    infer_subset_traces = require_int(infer_cfg, 'subset_traces')

    eval_cfg = cfg.get('eval')
    if eval_cfg is None:
        eval_use_label_valid_mask = bool(train_use_label_valid_mask)
    else:
        if not isinstance(eval_cfg, dict):
            msg = 'config.eval must be dict'
            raise TypeError(msg)
        eval_use_label_valid_mask = optional_bool(
            eval_cfg,
            'use_label_valid_mask',
            default=bool(train_use_label_valid_mask),
        )

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
    _validate_model_channels(
        in_chans=int(in_chans),
        out_chans=int(out_chans),
        label_prefix='config.model',
    )

    return CoarseTrainConfig(
        common=common,
        fbpick=fbpick,
        train=CoarseTrainCfg(
            lr=float(lr),
            subset_traces=int(train_subset_traces),
            use_label_valid_mask=bool(train_use_label_valid_mask),
        ),
        eval=CoarseEvalCfg(
            use_label_valid_mask=bool(eval_use_label_valid_mask),
        ),
        infer=CoarseInferCfg(
            subset_traces=int(infer_subset_traces),
        ),
        input=CoarseInputCfg(
            amplitude_key=optional_str(input_cfg, 'amplitude_key', 'x_view'),
            abs_offset_key=optional_str(input_cfg, 'abs_offset_key', 'x_offset_abs_ch'),
            absolute_time_key=optional_str(
                input_cfg,
                'absolute_time_key',
                'x_time_ch',
            ),
            input_key=optional_str(input_cfg, 'input_key', 'input'),
            use_offset_channel=optional_bool(
                input_cfg,
                'use_offset_channel',
                default=True,
            ),
            use_time_channel=optional_bool(
                input_cfg,
                'use_time_channel',
                default=True,
            ),
            offset_mode=optional_str(input_cfg, 'offset_mode', 'abs'),
            offset_normalize=optional_bool(
                input_cfg,
                'offset_normalize',
                default=True,
            ),
        ),
        target=CoarseTargetCfg(
            sigma=float(require_float(target_cfg, 'sigma')),
            fb_index_key=optional_str(target_cfg, 'fb_index_key', 'fb_idx_view'),
            probability_key=optional_str(
                target_cfg,
                'probability_key',
                'y_coarse_fb_prob',
            ),
            target_key=optional_str(target_cfg, 'target_key', 'target'),
        ),
        model=CoarseModelCfg(
            **build_encdec2d_kwargs(
                model_cfg,
                in_chans=int(in_chans),
                out_chans=int(out_chans),
            )
        ),
        ckpt=CoarseCkptCfg(
            save_best_only=bool(save_best_only),
            metric=str(metric),
            mode=str(mode),
        ),
    )
