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
    FbpickPathsCfg,
    load_fbpick_common_config,
)
from seisai_engine.pipelines.fbpick.common.io import resolve_artifact_paths

if TYPE_CHECKING:
    from seisai_engine.pipelines.common.config_schema import CommonTrainConfig

__all__ = [
    'FineCkptCfg',
    'FineCoarseArtifactCfg',
    'FineEvalCfg',
    'FineInferCfg',
    'FineInferConfig',
    'FineInputCfg',
    'FineModelCfg',
    'FineTargetCfg',
    'FineTrainCfg',
    'FineTrainConfig',
    'FineWindowCfg',
    'load_fine_infer_config',
    'load_fine_train_config',
    'resolve_default_coarse_artifact_paths',
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


def _resolve_optional_path(
    *,
    base_dir: str | Path | None,
    value: str | None,
    label: str,
) -> str | None:
    if value is None:
        return None
    path_str = _require_non_empty_str(value, label=label)
    if base_dir is None:
        return str(Path(path_str).expanduser())
    return resolve_relpath(base_dir, path_str)


def _validate_model_channels(*, in_chans: int, out_chans: int, label_prefix: str) -> None:
    if int(in_chans) != 1:
        msg = f'{label_prefix}.in_chans must be 1 (fine v1 amplitude only)'
        raise ValueError(msg)
    if int(out_chans) != 1:
        msg = f'{label_prefix}.out_chans must be 1 (fine local probability map)'
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


def resolve_default_coarse_artifact_paths(paths_cfg: FbpickPathsCfg) -> tuple[str, str]:
    if not isinstance(paths_cfg, FbpickPathsCfg):
        msg = 'paths_cfg must be FbpickPathsCfg'
        raise TypeError(msg)
    artifact_paths = resolve_artifact_paths(paths_cfg, stage='coarse')
    return str(artifact_paths.npz_path), str(artifact_paths.meta_path)


@dataclass(frozen=True)
class FineTrainCfg:
    lr: float
    use_label_valid_mask: bool

    def __post_init__(self) -> None:
        if float(self.lr) <= 0.0:
            msg = 'config.train.lr must be positive'
            raise ValueError(msg)


@dataclass(frozen=True)
class FineEvalCfg:
    use_label_valid_mask: bool


@dataclass(frozen=True)
class FineInferCfg:
    ckpt_path: str
    device: str
    allow_unsafe_override: bool

    def __post_init__(self) -> None:
        _require_non_empty_str(self.ckpt_path, label='config.infer.ckpt_path')
        _require_non_empty_str(self.device, label='config.infer.device')


@dataclass(frozen=True)
class FineInputCfg:
    amplitude_key: str
    input_key: str
    use_offset_channel: bool
    use_relative_time_channel: bool
    offset_key: str
    relative_time_key: str

    def __post_init__(self) -> None:
        amplitude_key = _require_non_empty_str(
            self.amplitude_key,
            label='config.input.amplitude_key',
        )
        input_key = _require_non_empty_str(
            self.input_key,
            label='config.input.input_key',
        )
        offset_key = _require_non_empty_str(
            self.offset_key,
            label='config.input.offset_key',
        )
        relative_time_key = _require_non_empty_str(
            self.relative_time_key,
            label='config.input.relative_time_key',
        )

        if input_key != 'input':
            msg = 'config.input.input_key must be "input" in Phase 5'
            raise ValueError(msg)
        if len({amplitude_key, input_key, offset_key, relative_time_key}) != 4:
            msg = 'config.input amplitude/input/offset/time keys must be distinct'
            raise ValueError(msg)
        if bool(self.use_offset_channel):
            msg = 'config.input.use_offset_channel must be false in Phase 5'
            raise ValueError(msg)
        if bool(self.use_relative_time_channel):
            msg = 'config.input.use_relative_time_channel must be false in Phase 5'
            raise ValueError(msg)

    @property
    def stack_keys(self) -> tuple[str]:
        return (str(self.amplitude_key),)


@dataclass(frozen=True)
class FineTargetCfg:
    sigma: float
    local_pick_idx_key: str
    probability_key: str
    target_key: str

    def __post_init__(self) -> None:
        if float(self.sigma) <= 0.0:
            msg = 'config.target.sigma must be positive'
            raise ValueError(msg)
        local_pick_idx_key = _require_non_empty_str(
            self.local_pick_idx_key,
            label='config.target.local_pick_idx_key',
        )
        _require_non_empty_str(
            self.probability_key,
            label='config.target.probability_key',
        )
        target_key = _require_non_empty_str(
            self.target_key,
            label='config.target.target_key',
        )
        if local_pick_idx_key != 'local_pick_idx':
            msg = 'config.target.local_pick_idx_key must be "local_pick_idx" in Phase 5'
            raise ValueError(msg)
        if target_key != 'target':
            msg = 'config.target.target_key must be "target" in Phase 5'
            raise ValueError(msg)


@dataclass(frozen=True)
class FineWindowCfg:
    local_window_len: int

    def __post_init__(self) -> None:
        if int(self.local_window_len) <= 0:
            msg = 'config.window.local_window_len must be positive'
            raise ValueError(msg)


@dataclass(frozen=True)
class FineCoarseArtifactCfg:
    artifact_npz_path: str
    artifact_meta_path: str

    def __post_init__(self) -> None:
        _require_non_empty_str(
            self.artifact_npz_path,
            label='config.coarse_seed.artifact_npz_path',
        )
        _require_non_empty_str(
            self.artifact_meta_path,
            label='config.coarse_seed.artifact_meta_path',
        )


@dataclass(frozen=True)
class FineModelCfg:
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
class FineCkptCfg:
    save_best_only: bool
    metric: str
    mode: str


@dataclass(frozen=True)
class FineTrainConfig:
    common: CommonTrainConfig
    fbpick: FbpickCommonConfig
    train: FineTrainCfg
    eval: FineEvalCfg
    input: FineInputCfg
    target: FineTargetCfg
    window: FineWindowCfg
    coarse_seed: FineCoarseArtifactCfg
    model: FineModelCfg
    ckpt: FineCkptCfg

    def __post_init__(self) -> None:
        _validate_model_channels(
            in_chans=int(self.model.in_chans),
            out_chans=int(self.model.out_chans),
            label_prefix='config.model',
        )
        if len(self.input.stack_keys) != int(self.model.in_chans):
            msg = 'config.model.in_chans must match the fixed 1-channel fine input stack'
            raise ValueError(msg)


@dataclass(frozen=True)
class FineInferConfig:
    fbpick: FbpickCommonConfig
    infer: FineInferCfg
    input: FineInputCfg
    target: FineTargetCfg
    window: FineWindowCfg
    coarse_seed: FineCoarseArtifactCfg
    model: FineModelCfg

    def __post_init__(self) -> None:
        _validate_model_channels(
            in_chans=int(self.model.in_chans),
            out_chans=int(self.model.out_chans),
            label_prefix='config.model',
        )
        if len(self.input.stack_keys) != int(self.model.in_chans):
            msg = 'config.model.in_chans must match the fixed 1-channel fine input stack'
            raise ValueError(msg)


def _load_input_cfg(input_cfg: dict) -> FineInputCfg:
    if not isinstance(input_cfg, dict):
        msg = 'config.input must be dict'
        raise TypeError(msg)
    return FineInputCfg(
        amplitude_key=optional_str(input_cfg, 'amplitude_key', 'x_local_amp'),
        input_key=optional_str(input_cfg, 'input_key', 'input'),
        use_offset_channel=optional_bool(
            input_cfg,
            'use_offset_channel',
            default=False,
        ),
        use_relative_time_channel=optional_bool(
            input_cfg,
            'use_relative_time_channel',
            default=False,
        ),
        offset_key=optional_str(input_cfg, 'offset_key', 'offsets_view_local'),
        relative_time_key=optional_str(
            input_cfg,
            'relative_time_key',
            'time_view_local',
        ),
    )


def _load_target_cfg(target_cfg: dict) -> FineTargetCfg:
    if not isinstance(target_cfg, dict):
        msg = 'config.target must be dict'
        raise TypeError(msg)
    return FineTargetCfg(
        sigma=float(require_float(target_cfg, 'sigma')),
        local_pick_idx_key=optional_str(
            target_cfg,
            'local_pick_idx_key',
            'local_pick_idx',
        ),
        probability_key=optional_str(
            target_cfg,
            'probability_key',
            'y_fine_local_prob',
        ),
        target_key=optional_str(target_cfg, 'target_key', 'target'),
    )


def _load_window_cfg(window_cfg: dict) -> FineWindowCfg:
    if not isinstance(window_cfg, dict):
        msg = 'config.window must be dict'
        raise TypeError(msg)
    return FineWindowCfg(
        local_window_len=int(require_int(window_cfg, 'local_window_len')),
    )


def _load_coarse_seed_cfg(
    *,
    coarse_seed_cfg: dict,
    base_dir: str | Path | None,
    paths_cfg: FbpickPathsCfg,
) -> FineCoarseArtifactCfg:
    if not isinstance(coarse_seed_cfg, dict):
        msg = 'config.coarse_seed must be dict'
        raise TypeError(msg)

    default_npz_path, default_meta_path = resolve_default_coarse_artifact_paths(paths_cfg)
    npz_path = _resolve_optional_path(
        base_dir=base_dir,
        value=coarse_seed_cfg.get('artifact_npz_path'),
        label='config.coarse_seed.artifact_npz_path',
    )
    meta_path = _resolve_optional_path(
        base_dir=base_dir,
        value=coarse_seed_cfg.get('artifact_meta_path'),
        label='config.coarse_seed.artifact_meta_path',
    )
    return FineCoarseArtifactCfg(
        artifact_npz_path=default_npz_path if npz_path is None else npz_path,
        artifact_meta_path=default_meta_path if meta_path is None else meta_path,
    )


def _load_model_cfg(model_cfg: dict) -> FineModelCfg:
    if not isinstance(model_cfg, dict):
        msg = 'config.model must be dict'
        raise TypeError(msg)
    in_chans = require_int(model_cfg, 'in_chans')
    out_chans = require_int(model_cfg, 'out_chans')
    _validate_model_channels(
        in_chans=int(in_chans),
        out_chans=int(out_chans),
        label_prefix='config.model',
    )
    return FineModelCfg(
        **build_encdec2d_kwargs(
            model_cfg,
            in_chans=int(in_chans),
            out_chans=int(out_chans),
        )
    )


def load_fine_train_config(
    cfg: dict,
    *,
    base_dir: str | Path | None = None,
) -> FineTrainConfig:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    common = load_common_train_config(
        _resolve_common_train_cfg(cfg, base_dir=base_dir)
    )
    fbpick = load_fbpick_common_config(cfg, base_dir=base_dir)

    train_cfg = require_dict(cfg, 'train')
    target_cfg = require_dict(cfg, 'target')
    window_cfg = require_dict(cfg, 'window')
    ckpt_cfg = require_dict(cfg, 'ckpt')
    model_cfg = require_dict(cfg, 'model')

    input_cfg = cfg.get('input', {})
    coarse_seed_cfg = cfg.get('coarse_seed', {})
    if not isinstance(input_cfg, dict):
        msg = 'config.input must be dict'
        raise TypeError(msg)
    if not isinstance(coarse_seed_cfg, dict):
        msg = 'config.coarse_seed must be dict'
        raise TypeError(msg)

    lr = require_float(train_cfg, 'lr')
    train_use_label_valid_mask = optional_bool(
        train_cfg,
        'use_label_valid_mask',
        default=True,
    )

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

    return FineTrainConfig(
        common=common,
        fbpick=fbpick,
        train=FineTrainCfg(
            lr=float(lr),
            use_label_valid_mask=bool(train_use_label_valid_mask),
        ),
        eval=FineEvalCfg(
            use_label_valid_mask=bool(eval_use_label_valid_mask),
        ),
        input=_load_input_cfg(input_cfg),
        target=_load_target_cfg(target_cfg),
        window=_load_window_cfg(window_cfg),
        coarse_seed=_load_coarse_seed_cfg(
            coarse_seed_cfg=coarse_seed_cfg,
            base_dir=base_dir,
            paths_cfg=fbpick.paths,
        ),
        model=_load_model_cfg(model_cfg),
        ckpt=FineCkptCfg(
            save_best_only=bool(save_best_only),
            metric=str(metric),
            mode=str(mode),
        ),
    )


def load_fine_infer_config(
    cfg: dict,
    *,
    base_dir: str | Path | None = None,
) -> FineInferConfig:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    fbpick = load_fbpick_common_config(cfg, base_dir=base_dir)

    infer_cfg = require_dict(cfg, 'infer')
    target_cfg = require_dict(cfg, 'target')
    window_cfg = require_dict(cfg, 'window')
    model_cfg = require_dict(cfg, 'model')

    input_cfg = cfg.get('input', {})
    coarse_seed_cfg = cfg.get('coarse_seed', {})
    if not isinstance(input_cfg, dict):
        msg = 'config.input must be dict'
        raise TypeError(msg)
    if not isinstance(coarse_seed_cfg, dict):
        msg = 'config.coarse_seed must be dict'
        raise TypeError(msg)

    ckpt_path_raw = require_value(
        infer_cfg,
        'ckpt_path',
        str,
        type_message='config.infer.ckpt_path must be str',
    )
    device = optional_str(infer_cfg, 'device', 'auto')
    allow_unsafe_override = optional_bool(
        infer_cfg,
        'allow_unsafe_override',
        default=False,
    )

    return FineInferConfig(
        fbpick=fbpick,
        infer=FineInferCfg(
            ckpt_path=_resolve_optional_path(
                base_dir=base_dir,
                value=ckpt_path_raw,
                label='config.infer.ckpt_path',
            )
            or '',
            device=str(device),
            allow_unsafe_override=bool(allow_unsafe_override),
        ),
        input=_load_input_cfg(input_cfg),
        target=_load_target_cfg(target_cfg),
        window=_load_window_cfg(window_cfg),
        coarse_seed=_load_coarse_seed_cfg(
            coarse_seed_cfg=coarse_seed_cfg,
            base_dir=base_dir,
            paths_cfg=fbpick.paths,
        ),
        model=_load_model_cfg(model_cfg),
    )
