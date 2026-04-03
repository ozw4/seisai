from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from seisai_utils.config import (
    optional_bool,
    optional_float,
    optional_int,
    optional_str,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_list_str,
    require_value,
)

from seisai_engine.pipelines.common.config_keys import (
    normalize_endian,
    raise_if_deprecated_time_len_keys,
)
from seisai_engine.pipelines.common.config_loaders import load_common_train_config
from seisai_engine.pipelines.common.encdec2d_cfg import build_encdec2d_kwargs
from seisai_engine.pipelines.common.validate_primary_keys import validate_primary_keys
from seisai_engine.pipelines.fbpick.common import FBPickNormRefs, load_norm_refs_cfg

if TYPE_CHECKING:
    from seisai_engine.pipelines.common.config_schema import CommonTrainConfig

__all__ = [
    'CoarseCkptCfg',
    'CoarseDatasetCfg',
    'CoarseInferConfig',
    'CoarseInferRuntimeCfg',
    'CoarsePaths',
    'CoarseTrainCfg',
    'CoarseTrainConfig',
    'CoarseTrainInferCfg',
    'CoarseTransformCfg',
    'load_coarse_infer_config',
    'load_coarse_train_config',
]


@dataclass(frozen=True)
class CoarsePaths:
    segy_files: tuple[str, ...]
    fb_files: tuple[str, ...] | None
    infer_segy_files: tuple[str, ...] | None
    infer_fb_files: tuple[str, ...] | None
    out_dir: str


@dataclass(frozen=True)
class CoarseDatasetCfg:
    max_trials: int
    use_header_cache: bool
    verbose: bool
    progress: bool
    primary_keys: tuple[str, ...]
    secondary_key_fixed: bool
    waveform_mode: str
    train_endian: str
    infer_endian: str


@dataclass(frozen=True)
class CoarseTransformCfg:
    time_len: int
    standardize_eps: float


@dataclass(frozen=True)
class CoarseTrainCfg:
    lr: float
    weight_decay: float
    subset_traces: int
    fb_sigma_ms: float
    trace_decimate_prob: float
    trace_decimate_stride_range: tuple[int, int]


@dataclass(frozen=True)
class CoarseTrainInferCfg:
    subset_traces: int


@dataclass(frozen=True)
class CoarseInferRuntimeCfg:
    subset_traces: int
    batch_size: int
    num_workers: int
    overlap_h: int
    tile_w: int
    overlap_w: int
    tiles_per_batch: int
    amp: bool
    use_tqdm: bool


@dataclass(frozen=True)
class CoarseCkptCfg:
    save_best_only: bool
    metric: str
    mode: str


@dataclass(frozen=True)
class CoarseTrainConfig:
    common: 'CommonTrainConfig'
    paths: CoarsePaths
    dataset: CoarseDatasetCfg
    transform: CoarseTransformCfg
    norm_refs: FBPickNormRefs
    train: CoarseTrainCfg
    infer: CoarseTrainInferCfg
    model_sig: dict[str, Any]
    ckpt: CoarseCkptCfg


@dataclass(frozen=True)
class CoarseInferConfig:
    paths: CoarsePaths
    dataset: CoarseDatasetCfg
    transform: CoarseTransformCfg
    norm_refs: FBPickNormRefs
    infer: CoarseInferRuntimeCfg
    model_sig: dict[str, Any]


def _parse_trace_decimation_cfg(train_cfg: dict) -> tuple[float, tuple[int, int]]:
    raw = train_cfg.get('trace_decimation')
    if raw is None:
        return 0.0, (1, 1)
    if not isinstance(raw, dict):
        msg = 'train.trace_decimation must be dict'
        raise TypeError(msg)

    prob_raw = raw.get('prob', 0.0)
    if isinstance(prob_raw, bool) or not isinstance(prob_raw, (int, float)):
        msg = 'train.trace_decimation.prob must be float'
        raise TypeError(msg)
    prob = float(prob_raw)
    if (not math.isfinite(prob)) or prob < 0.0 or prob > 1.0:
        msg = 'train.trace_decimation.prob must be finite and within [0, 1]'
        raise ValueError(msg)

    stride_range_raw = raw.get('stride_range', (1, 1))
    if (
        not isinstance(stride_range_raw, (list, tuple))
        or len(stride_range_raw) != 2
    ):
        msg = 'train.trace_decimation.stride_range must be [min_int, max_int]'
        raise TypeError(msg)
    lo_raw, hi_raw = stride_range_raw
    if isinstance(lo_raw, bool) or not isinstance(lo_raw, int):
        msg = 'train.trace_decimation.stride_range[0] must be int'
        raise TypeError(msg)
    if isinstance(hi_raw, bool) or not isinstance(hi_raw, int):
        msg = 'train.trace_decimation.stride_range[1] must be int'
        raise TypeError(msg)
    lo = int(lo_raw)
    hi = int(hi_raw)
    if lo < 1 or hi < lo:
        msg = 'train.trace_decimation.stride_range requires 1 <= min <= max'
        raise ValueError(msg)
    return prob, (lo, hi)


def _load_paths_cfg(cfg: dict, *, allow_missing_infer_pairs: bool) -> CoarsePaths:
    paths = require_dict(cfg, 'paths')
    segy_files = tuple(require_list_str(paths, 'segy_files'))
    fb_files_raw = paths.get('fb_files')
    if fb_files_raw is None:
        fb_files = None
    else:
        if not isinstance(fb_files_raw, list):
            msg = 'paths.fb_files must be list[str] or null'
            raise TypeError(msg)
        fb_files = tuple(require_list_str(paths, 'fb_files'))
        if len(segy_files) != len(fb_files):
            msg = 'paths.segy_files and paths.fb_files must have the same length'
            raise ValueError(msg)

    infer_segy_raw = paths.get('infer_segy_files')
    infer_fb_raw = paths.get('infer_fb_files')
    if infer_segy_raw is None:
        infer_segy_files = None
    else:
        infer_segy_files = tuple(require_list_str(paths, 'infer_segy_files'))
    if infer_fb_raw is None:
        infer_fb_files = None
    else:
        infer_fb_files = tuple(require_list_str(paths, 'infer_fb_files'))

    if not allow_missing_infer_pairs and infer_segy_files is None:
        infer_segy_files = segy_files
        infer_fb_files = fb_files
    if infer_segy_files is not None and infer_fb_files is not None:
        if len(infer_segy_files) != len(infer_fb_files):
            msg = (
                'paths.infer_segy_files and paths.infer_fb_files must have the same '
                'length'
            )
            raise ValueError(msg)

    out_dir = require_value(
        paths,
        'out_dir',
        str,
        type_message='config.paths.out_dir must be str',
    )
    return CoarsePaths(
        segy_files=segy_files,
        fb_files=fb_files,
        infer_segy_files=infer_segy_files,
        infer_fb_files=infer_fb_files,
        out_dir=str(out_dir),
    )


def _load_dataset_cfg(cfg: dict) -> CoarseDatasetCfg:
    ds_cfg = require_dict(cfg, 'dataset')
    primary_keys = validate_primary_keys(ds_cfg.get('primary_keys', ['ffid']))
    train_endian = normalize_endian(
        value=optional_str(ds_cfg, 'train_endian', 'big'),
        key_name='dataset.train_endian',
    )
    infer_endian = normalize_endian(
        value=optional_str(ds_cfg, 'infer_endian', train_endian),
        key_name='dataset.infer_endian',
    )
    waveform_mode = optional_str(ds_cfg, 'waveform_mode', 'eager').strip().lower()
    if waveform_mode not in ('eager', 'mmap'):
        msg = 'dataset.waveform_mode must be "eager" or "mmap"'
        raise ValueError(msg)
    return CoarseDatasetCfg(
        max_trials=int(optional_int(ds_cfg, 'max_trials', 2048)),
        use_header_cache=bool(optional_bool(ds_cfg, 'use_header_cache', default=True)),
        verbose=bool(optional_bool(ds_cfg, 'verbose', default=True)),
        progress=bool(optional_bool(ds_cfg, 'progress', default=True)),
        primary_keys=tuple(primary_keys),
        secondary_key_fixed=bool(
            optional_bool(ds_cfg, 'secondary_key_fixed', default=False)
        ),
        waveform_mode=str(waveform_mode),
        train_endian=str(train_endian),
        infer_endian=str(infer_endian),
    )


def _load_transform_cfg(cfg: dict) -> CoarseTransformCfg:
    raise_if_deprecated_time_len_keys(
        train_cfg=cfg.get('train'),
        transform_cfg=cfg.get('transform'),
    )
    transform_cfg = require_dict(cfg, 'transform')
    time_len = int(require_int(transform_cfg, 'time_len'))
    if time_len <= 0:
        msg = 'transform.time_len must be positive'
        raise ValueError(msg)
    return CoarseTransformCfg(
        time_len=time_len,
        standardize_eps=float(optional_float(transform_cfg, 'standardize_eps', 1.0e-8)),
    )


def _load_model_sig(cfg: dict) -> dict[str, Any]:
    model_cfg = require_dict(cfg, 'model')
    in_chans = int(require_int(model_cfg, 'in_chans'))
    out_chans = int(require_int(model_cfg, 'out_chans'))
    if in_chans != 3:
        msg = 'model.in_chans must be 3 for fbpick coarse'
        raise ValueError(msg)
    if out_chans != 1:
        msg = 'model.out_chans must be 1 for fbpick coarse'
        raise ValueError(msg)
    return build_encdec2d_kwargs(model_cfg, in_chans=in_chans, out_chans=out_chans)


def load_coarse_train_config(cfg: dict) -> CoarseTrainConfig:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    common = load_common_train_config(cfg)
    train_cfg = require_dict(cfg, 'train')
    infer_cfg = require_dict(cfg, 'infer')
    ckpt_cfg = require_dict(cfg, 'ckpt')

    trace_decimate_prob, trace_decimate_stride_range = _parse_trace_decimation_cfg(
        train_cfg
    )
    return CoarseTrainConfig(
        common=common,
        paths=_load_paths_cfg(cfg, allow_missing_infer_pairs=False),
        dataset=_load_dataset_cfg(cfg),
        transform=_load_transform_cfg(cfg),
        norm_refs=load_norm_refs_cfg(cfg),
        train=CoarseTrainCfg(
            lr=float(require_float(train_cfg, 'lr')),
            weight_decay=float(optional_float(train_cfg, 'weight_decay', 0.0)),
            subset_traces=int(require_int(train_cfg, 'subset_traces')),
            fb_sigma_ms=float(optional_float(train_cfg, 'fb_sigma_ms', 10.0)),
            trace_decimate_prob=float(trace_decimate_prob),
            trace_decimate_stride_range=tuple(trace_decimate_stride_range),
        ),
        infer=CoarseTrainInferCfg(
            subset_traces=int(require_int(infer_cfg, 'subset_traces')),
        ),
        model_sig=_load_model_sig(cfg),
        ckpt=CoarseCkptCfg(
            save_best_only=bool(require_bool(ckpt_cfg, 'save_best_only')),
            metric=str(
                require_value(
                    ckpt_cfg,
                    'metric',
                    str,
                    type_message='config.ckpt.metric must be str',
                )
            ),
            mode=str(
                require_value(
                    ckpt_cfg,
                    'mode',
                    str,
                    type_message='config.ckpt.mode must be str',
                )
            ),
        ),
    )


def load_coarse_infer_config(cfg: dict) -> CoarseInferConfig:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    infer_cfg = require_dict(cfg, 'infer')
    subset_traces = int(require_int(infer_cfg, 'subset_traces'))
    overlap_h = int(optional_int(infer_cfg, 'overlap_h', 96))
    if overlap_h < 0 or overlap_h >= subset_traces:
        msg = 'infer.overlap_h must satisfy 0 <= overlap_h < infer.subset_traces'
        raise ValueError(msg)

    return CoarseInferConfig(
        paths=_load_paths_cfg(cfg, allow_missing_infer_pairs=True),
        dataset=_load_dataset_cfg(cfg),
        transform=_load_transform_cfg(cfg),
        norm_refs=load_norm_refs_cfg(cfg),
        infer=CoarseInferRuntimeCfg(
            subset_traces=subset_traces,
            batch_size=int(optional_int(infer_cfg, 'batch_size', 1)),
            num_workers=int(optional_int(infer_cfg, 'num_workers', 0)),
            overlap_h=overlap_h,
            tile_w=int(optional_int(infer_cfg, 'tile_w', 6016)),
            overlap_w=int(optional_int(infer_cfg, 'overlap_w', 1024)),
            tiles_per_batch=int(optional_int(infer_cfg, 'tiles_per_batch', 16)),
            amp=bool(optional_bool(infer_cfg, 'amp', default=False)),
            use_tqdm=bool(optional_bool(infer_cfg, 'use_tqdm', default=False)),
        ),
        model_sig=_load_model_sig(cfg),
    )
