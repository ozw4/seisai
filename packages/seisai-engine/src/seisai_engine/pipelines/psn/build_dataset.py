from __future__ import annotations

from seisai_dataset import (
    FirstBreakGate,
    FirstBreakGateConfig,
    SegyGatherPhasePipelineDataset,
)
from seisai_transforms.augment import (
    DeterministicCropOrPad,
    PerTraceStandardize,
    RandomCropOrPad,
    ViewCompose,
)
from seisai_utils.config import (
    optional_bool,
    optional_float,
    optional_int,
    optional_str,
    require_dict,
    require_int,
    require_list_str,
)

from seisai_engine.pipelines.common.augment import build_train_augment_ops
from seisai_engine.pipelines.common.validate_files import validate_files_exist
from seisai_engine.pipelines.common.validate_primary_keys import validate_primary_keys

from .build_plan import build_plan

__all__ = ['build_dataset', 'build_infer_transform', 'build_train_transform']


def _format_key(section: str, key: str) -> str:
    return f'{section}.{key}'


def _raise_if_deprecated_time_len_keys(*, cfg: dict) -> None:
    train_cfg = cfg.get('train')
    if isinstance(train_cfg, dict) and 'time_len' in train_cfg:
        msg = (
            f'deprecated key: {_format_key("train", "time_len")}; '
            f'use {_format_key("transform", "time_len")}'
        )
        raise ValueError(msg)
    transform_cfg = cfg.get('transform')
    if isinstance(transform_cfg, dict) and 'target_len' in transform_cfg:
        msg = (
            f'deprecated key: {_format_key("transform", "target_len")}; '
            f'use {_format_key("transform", "time_len")}'
        )
        raise ValueError(msg)


def _resolve_time_len(cfg: dict) -> tuple[dict, int]:
    _raise_if_deprecated_time_len_keys(cfg=cfg)
    transform_cfg = require_dict(cfg, 'transform')
    return transform_cfg, int(require_int(transform_cfg, 'time_len'))


def _build_fbgate(fbgate_cfg: dict | None) -> FirstBreakGate:
    if fbgate_cfg is None:
        return FirstBreakGate(
            FirstBreakGateConfig(
                apply_on='off',
                min_pick_ratio=0.0,
                verbose=False,
            )
        )
    if not isinstance(fbgate_cfg, dict):
        msg = 'fbgate must be dict'
        raise TypeError(msg)

    apply_on = optional_str(fbgate_cfg, 'apply_on', 'off').lower()
    if apply_on == 'on':
        apply_on = 'any'
    if apply_on not in ('any', 'super_only', 'off'):
        msg = 'fbgate.apply_on must be "any", "super_only", or "off"'
        raise ValueError(msg)

    min_pick_ratio = optional_float(fbgate_cfg, 'min_pick_ratio', 0.0)
    verbose = optional_bool(fbgate_cfg, 'verbose', default=False)
    return FirstBreakGate(
        FirstBreakGateConfig(
            apply_on=apply_on,
            min_pick_ratio=float(min_pick_ratio),
            verbose=bool(verbose),
        )
    )


def build_train_transform(cfg: dict) -> ViewCompose:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)
    transform_cfg, time_len = _resolve_time_len(cfg)
    augment_cfg = cfg.get('augment')
    standardize_eps = optional_float(transform_cfg, 'standardize_eps', 1.0e-8)
    geom_ops, post_ops = build_train_augment_ops(augment_cfg)
    return ViewCompose(
        [
            *geom_ops,
            RandomCropOrPad(target_len=int(time_len)),
            *post_ops,
            PerTraceStandardize(eps=float(standardize_eps)),
        ]
    )


def build_infer_transform(cfg: dict) -> ViewCompose:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)
    transform_cfg, time_len = _resolve_time_len(cfg)
    standardize_eps = optional_float(transform_cfg, 'standardize_eps', 1.0e-8)
    return ViewCompose(
        [
            DeterministicCropOrPad(target_len=int(time_len)),
            PerTraceStandardize(eps=float(standardize_eps)),
        ]
    )


def build_dataset(
    cfg: dict,
    *,
    transform: ViewCompose,
    segy_endian: str | None = None,
    sampling_overrides: list[dict[str, object] | None] | None = None,
) -> SegyGatherPhasePipelineDataset:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    paths = require_dict(cfg, 'paths')
    ds_cfg = require_dict(cfg, 'dataset')
    train_cfg = require_dict(cfg, 'train')
    fbgate_cfg = cfg.get('fbgate')
    if fbgate_cfg is not None and not isinstance(fbgate_cfg, dict):
        msg = 'fbgate must be dict'
        raise TypeError(msg)

    segy_files = require_list_str(paths, 'segy_files')
    phase_pick_files = require_list_str(paths, 'phase_pick_files')
    if len(segy_files) != len(phase_pick_files):
        msg = 'paths.segy_files and paths.phase_pick_files must have same length'
        raise ValueError(
            msg
        )
    if sampling_overrides is not None and len(sampling_overrides) != len(segy_files):
        msg = 'sampling_overrides length must match paths.segy_files length'
        raise ValueError(msg)
    validate_files_exist(list(segy_files) + list(phase_pick_files))

    max_trials = optional_int(ds_cfg, 'max_trials', 2048)
    use_header_cache = optional_bool(ds_cfg, 'use_header_cache', default=True)
    verbose = optional_bool(ds_cfg, 'verbose', default=True)
    progress = optional_bool(ds_cfg, 'progress', default=bool(verbose))
    include_empty_gathers = optional_bool(
        ds_cfg, 'include_empty_gathers', default=False
    )
    secondary_key_fixed = optional_bool(ds_cfg, 'secondary_key_fixed', default=False)
    primary_keys_list = ds_cfg.get('primary_keys', ['ffid'])
    primary_keys = validate_primary_keys(primary_keys_list)
    waveform_mode = optional_str(ds_cfg, 'waveform_mode', 'eager').lower()
    if waveform_mode not in ('eager', 'mmap'):
        msg = 'dataset.waveform_mode must be "eager" or "mmap"'
        raise ValueError(msg)

    subset_traces = require_int(train_cfg, 'subset_traces')
    psn_sigma = optional_float(train_cfg, 'psn_sigma', 1.5)

    if not callable(transform):
        msg = 'transform must be callable'
        raise TypeError(msg)
    fbgate = _build_fbgate(fbgate_cfg)
    plan = build_plan(psn_sigma=float(psn_sigma))

    dataset_endian = (
        optional_str(ds_cfg, 'train_endian', 'big')
        if segy_endian is None
        else str(segy_endian)
    )
    dataset_endian = dataset_endian.lower()
    if dataset_endian not in ('big', 'little'):
        msg = 'dataset segy_endian must be "big" or "little"'
        raise ValueError(msg)

    return SegyGatherPhasePipelineDataset(
        segy_files=list(segy_files),
        phase_pick_files=list(phase_pick_files),
        transform=transform,
        fbgate=fbgate,
        plan=plan,
        subset_traces=int(subset_traces),
        include_empty_gathers=bool(include_empty_gathers),
        use_header_cache=bool(use_header_cache),
        primary_keys=primary_keys,
        secondary_key_fixed=bool(secondary_key_fixed),
        sampling_overrides=sampling_overrides,
        waveform_mode=str(waveform_mode),
        segy_endian=str(dataset_endian),
        verbose=bool(verbose),
        progress=bool(progress),
        max_trials=int(max_trials),
    )
