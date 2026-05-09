from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
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
    'COARSE_CKPT_OUTPUT_IDS',
    'COARSE_CKPT_PIPELINE',
    'COARSE_CKPT_SOFTMAX_AXIS',
    'COARSE_IN_CHANS',
    'COARSE_INPUT_CHANNELS',
    'COARSE_INPUT_MODE_GLOBAL_ANCHOR_RESIZE',
    'COARSE_OUT_CHANS',
    'COARSE_TRACE_LEN',
    'COARSE_TIME_LEN',
    'CoarseCkptCfg',
    'CoarseDatasetCfg',
    'CoarseInferConfig',
    'CoarseInferRuntimeCfg',
    'CoarseModeCfg',
    'CoarsePaths',
    'CoarseQCCfg',
    'CoarseTraceAnchorCfg',
    'CoarseTrainCfg',
    'CoarseTrainConfig',
    'CoarseTransformCfg',
    'load_coarse_infer_config',
    'load_coarse_train_config',
]

COARSE_INPUT_MODE_GLOBAL_ANCHOR_RESIZE = 'global_anchor_resize'
COARSE_TRACE_LEN = 256
COARSE_IN_CHANS = 3
COARSE_OUT_CHANS = 1
COARSE_TIME_LEN = 2048
COARSE_INPUT_CHANNELS = ('waveform', 'offset_ch', 'time_ch')
COARSE_CKPT_PIPELINE = 'fbpick'
COARSE_CKPT_OUTPUT_IDS = ('P',)
COARSE_CKPT_SOFTMAX_AXIS = 'time'
INFER_PAIR_MESSAGE = (
    'paths.infer_segy_files and paths.infer_fb_files must be provided together, '
    'or omit both to reuse the training segy/fb pairs'
)
LEGACY_TILED_INFER_KEYS = (
    'subset_traces',
    'overlap_h',
    'tile_w',
    'overlap_w',
    'tiles_per_batch',
)


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
    coord_unit_scale_to_m: float


@dataclass(frozen=True)
class CoarseModeCfg:
    input_mode: str


@dataclass(frozen=True)
class CoarseTransformCfg:
    trace_len: int
    time_len: int
    standardize_eps: float


@dataclass(frozen=True)
class CoarseTraceAnchorCfg:
    gap_ratio: float
    min_gap_m: float | None
    train_mode: str
    infer_mode: str


@dataclass(frozen=True)
class CoarseTrainCfg:
    lr: float
    weight_decay: float
    fb_sigma_ms: float
    trace_decimate_prob: float
    trace_decimate_stride_range: tuple[int, int]


@dataclass(frozen=True)
class CoarseInferRuntimeCfg:
    batch_size: int
    num_workers: int
    amp: bool
    use_tqdm: bool


@dataclass(frozen=True)
class CoarseQCCfg:
    enabled: bool
    max_gathers: int
    out_subdir: str
    plot_anchor_grid: bool
    plot_original_gather: bool
    plot_confidence: bool
    plot_error_if_labels_available: bool
    fine_window_half_samples: int
    max_display_traces: int
    max_display_samples: int
    low_confidence_threshold: float | None
    dpi: int
    clip_percentile: float


@dataclass(frozen=True)
class CoarseCkptCfg:
    save_best_only: bool
    metric: str
    mode: str
    pipeline: str = COARSE_CKPT_PIPELINE
    output_ids: tuple[str, ...] = COARSE_CKPT_OUTPUT_IDS
    softmax_axis: str = COARSE_CKPT_SOFTMAX_AXIS


@dataclass(frozen=True)
class CoarseTrainConfig:
    common: 'CommonTrainConfig'
    coarse: CoarseModeCfg
    paths: CoarsePaths
    dataset: CoarseDatasetCfg
    transform: CoarseTransformCfg
    trace_anchor: CoarseTraceAnchorCfg
    norm_refs: FBPickNormRefs
    train: CoarseTrainCfg
    model_sig: dict[str, Any]
    ckpt: CoarseCkptCfg


@dataclass(frozen=True)
class CoarseInferConfig:
    coarse: CoarseModeCfg
    paths: CoarsePaths
    dataset: CoarseDatasetCfg
    transform: CoarseTransformCfg
    trace_anchor: CoarseTraceAnchorCfg
    norm_refs: FBPickNormRefs
    infer: CoarseInferRuntimeCfg
    qc: CoarseQCCfg
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
    infer_segy_provided = infer_segy_raw is not None
    infer_fb_provided = infer_fb_raw is not None
    if not allow_missing_infer_pairs and infer_segy_provided != infer_fb_provided:
        raise ValueError(INFER_PAIR_MESSAGE)

    if infer_segy_raw is None:
        infer_segy_files = None
    else:
        infer_segy_files = tuple(require_list_str(paths, 'infer_segy_files'))
    if infer_fb_raw is None:
        infer_fb_files = None
    else:
        infer_fb_files = tuple(require_list_str(paths, 'infer_fb_files'))

    if not allow_missing_infer_pairs and not infer_segy_provided:
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
    coord_unit_scale_raw = ds_cfg.get('coord_unit_scale_to_m', 1.0)
    if isinstance(coord_unit_scale_raw, bool) or not isinstance(
        coord_unit_scale_raw,
        (int, float),
    ):
        msg = 'config.dataset.coord_unit_scale_to_m must be float'
        raise TypeError(msg)
    coord_unit_scale_to_m = float(coord_unit_scale_raw)
    if (not math.isfinite(coord_unit_scale_to_m)) or coord_unit_scale_to_m <= 0.0:
        msg = 'dataset.coord_unit_scale_to_m must be finite and > 0'
        raise ValueError(msg)
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
        coord_unit_scale_to_m=coord_unit_scale_to_m,
    )


def _load_coarse_mode_cfg(cfg: dict) -> CoarseModeCfg:
    coarse_cfg = require_dict(cfg, 'coarse')
    input_mode = str(
        require_value(
            coarse_cfg,
            'input_mode',
            str,
            type_message='config.coarse.input_mode must be str',
        )
    )
    if input_mode != COARSE_INPUT_MODE_GLOBAL_ANCHOR_RESIZE:
        msg = (
            'coarse.input_mode must be '
            f'{COARSE_INPUT_MODE_GLOBAL_ANCHOR_RESIZE!r} for fbpick coarse, '
            f'got {input_mode!r}'
        )
        raise ValueError(msg)
    return CoarseModeCfg(input_mode=input_mode)


def _load_transform_cfg(cfg: dict) -> CoarseTransformCfg:
    raise_if_deprecated_time_len_keys(
        train_cfg=cfg.get('train'),
        transform_cfg=cfg.get('transform'),
    )
    transform_cfg = require_dict(cfg, 'transform')
    trace_len = int(require_int(transform_cfg, 'trace_len'))
    if trace_len != COARSE_TRACE_LEN:
        msg = f'transform.trace_len must be {COARSE_TRACE_LEN} for fbpick coarse'
        raise ValueError(msg)
    time_len = int(require_int(transform_cfg, 'time_len'))
    if time_len != COARSE_TIME_LEN:
        msg = f'transform.time_len must be {COARSE_TIME_LEN} for fbpick coarse'
        raise ValueError(msg)
    return CoarseTransformCfg(
        trace_len=trace_len,
        time_len=time_len,
        standardize_eps=float(optional_float(transform_cfg, 'standardize_eps', 1.0e-8)),
    )


def _load_trace_anchor_cfg(cfg: dict) -> CoarseTraceAnchorCfg:
    anchor_cfg = require_dict(cfg, 'trace_anchor')

    gap_ratio_raw = require_value(
        anchor_cfg,
        'gap_ratio',
        (int, float),
        type_message='config.trace_anchor.gap_ratio must be float',
    )
    if isinstance(gap_ratio_raw, bool):
        msg = 'config.trace_anchor.gap_ratio must be float'
        raise TypeError(msg)
    gap_ratio = float(gap_ratio_raw)
    if (not math.isfinite(gap_ratio)) or gap_ratio <= 1.0:
        msg = 'trace_anchor.gap_ratio must be > 1.0'
        raise ValueError(msg)

    min_gap_raw = require_value(
        anchor_cfg,
        'min_gap_m',
        (int, float),
        allow_none=True,
        type_message='config.trace_anchor.min_gap_m must be float or null',
    )
    if min_gap_raw is None:
        min_gap_m = None
    else:
        if isinstance(min_gap_raw, bool):
            msg = 'config.trace_anchor.min_gap_m must be float or null'
            raise TypeError(msg)
        min_gap_m = float(min_gap_raw)
        if (not math.isfinite(min_gap_m)) or min_gap_m <= 0.0:
            msg = 'trace_anchor.min_gap_m must be null or > 0'
            raise ValueError(msg)

    train_mode = str(
        require_value(
            anchor_cfg,
            'train_mode',
            str,
            type_message='config.trace_anchor.train_mode must be str',
        )
    )
    if train_mode != 'random':
        msg = 'trace_anchor.train_mode must be "random"'
        raise ValueError(msg)

    infer_mode = str(
        require_value(
            anchor_cfg,
            'infer_mode',
            str,
            type_message='config.trace_anchor.infer_mode must be str',
        )
    )
    if infer_mode != 'center':
        msg = 'trace_anchor.infer_mode must be "center"'
        raise ValueError(msg)

    return CoarseTraceAnchorCfg(
        gap_ratio=gap_ratio,
        min_gap_m=min_gap_m,
        train_mode=train_mode,
        infer_mode=infer_mode,
    )


def _load_model_sig(cfg: dict) -> dict[str, Any]:
    model_cfg = require_dict(cfg, 'model')
    in_chans = int(require_int(model_cfg, 'in_chans'))
    out_chans = int(require_int(model_cfg, 'out_chans'))
    if in_chans != COARSE_IN_CHANS:
        msg = f'model.in_chans must be {COARSE_IN_CHANS} for fbpick coarse'
        raise ValueError(msg)
    if out_chans != COARSE_OUT_CHANS:
        msg = f'model.out_chans must be {COARSE_OUT_CHANS} for fbpick coarse'
        raise ValueError(msg)
    return build_encdec2d_kwargs(model_cfg, in_chans=in_chans, out_chans=out_chans)


def _load_qc_cfg(cfg: dict) -> CoarseQCCfg:
    raw = cfg.get('qc')
    if raw is None:
        qc_cfg: dict = {}
    else:
        if not isinstance(raw, dict):
            msg = 'config.qc must be dict'
            raise TypeError(msg)
        qc_cfg = raw

    max_gathers = int(optional_int(qc_cfg, 'max_gathers', 16))
    if max_gathers < 0:
        msg = 'qc.max_gathers must be >= 0'
        raise ValueError(msg)

    fine_window_half_samples = int(
        optional_int(qc_cfg, 'fine_window_half_samples', 128)
    )
    if fine_window_half_samples < 0:
        msg = 'qc.fine_window_half_samples must be >= 0'
        raise ValueError(msg)

    max_display_traces = int(optional_int(qc_cfg, 'max_display_traces', 512))
    if max_display_traces <= 0:
        msg = 'qc.max_display_traces must be > 0'
        raise ValueError(msg)

    max_display_samples = int(optional_int(qc_cfg, 'max_display_samples', 4096))
    if max_display_samples <= 0:
        msg = 'qc.max_display_samples must be > 0'
        raise ValueError(msg)

    low_confidence_threshold_raw = qc_cfg.get('low_confidence_threshold')
    if low_confidence_threshold_raw is None:
        low_confidence_threshold = None
    else:
        if isinstance(low_confidence_threshold_raw, bool) or not isinstance(
            low_confidence_threshold_raw,
            (int, float),
        ):
            msg = 'config.qc.low_confidence_threshold must be float or null'
            raise TypeError(msg)
        low_confidence_threshold = float(low_confidence_threshold_raw)
        if (
            not math.isfinite(low_confidence_threshold)
            or low_confidence_threshold < 0.0
            or low_confidence_threshold > 1.0
        ):
            msg = 'qc.low_confidence_threshold must lie in [0, 1]'
            raise ValueError(msg)

    dpi = int(optional_int(qc_cfg, 'dpi', 150))
    if dpi <= 0:
        msg = 'qc.dpi must be > 0'
        raise ValueError(msg)

    clip_percentile = float(optional_float(qc_cfg, 'clip_percentile', 99.0))
    if (
        not math.isfinite(clip_percentile)
        or clip_percentile <= 0.0
        or clip_percentile > 100.0
    ):
        msg = 'qc.clip_percentile must lie in (0, 100]'
        raise ValueError(msg)

    out_subdir = str(optional_str(qc_cfg, 'out_subdir', 'vis/coarse_global_anchor'))
    if not out_subdir.strip():
        msg = 'qc.out_subdir must be non-empty'
        raise ValueError(msg)
    if Path(out_subdir).is_absolute():
        msg = 'qc.out_subdir must be relative to paths.out_dir'
        raise ValueError(msg)

    return CoarseQCCfg(
        enabled=bool(optional_bool(qc_cfg, 'enabled', default=False)),
        max_gathers=max_gathers,
        out_subdir=out_subdir,
        plot_anchor_grid=bool(
            optional_bool(qc_cfg, 'plot_anchor_grid', default=True)
        ),
        plot_original_gather=bool(
            optional_bool(qc_cfg, 'plot_original_gather', default=True)
        ),
        plot_confidence=bool(optional_bool(qc_cfg, 'plot_confidence', default=True)),
        plot_error_if_labels_available=bool(
            optional_bool(qc_cfg, 'plot_error_if_labels_available', default=True)
        ),
        fine_window_half_samples=fine_window_half_samples,
        max_display_traces=max_display_traces,
        max_display_samples=max_display_samples,
        low_confidence_threshold=low_confidence_threshold,
        dpi=dpi,
        clip_percentile=clip_percentile,
    )


def load_coarse_train_config(cfg: dict) -> CoarseTrainConfig:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    common = load_common_train_config(cfg)
    train_cfg = require_dict(cfg, 'train')
    infer_cfg = require_dict(cfg, 'infer')
    ckpt_cfg = require_dict(cfg, 'ckpt')
    if 'subset_traces' in train_cfg:
        msg = (
            'fbpick-coarse global-anchor training uses transform.trace_len; '
            'remove legacy local-crop train.subset_traces'
        )
        raise ValueError(msg)
    legacy_infer_keys = [key for key in LEGACY_TILED_INFER_KEYS if key in infer_cfg]
    if legacy_infer_keys:
        msg = (
            'fbpick-coarse global-anchor training does not use legacy tiled '
            f'infer keys: {legacy_infer_keys!r}'
        )
        raise ValueError(msg)

    trace_decimate_prob, trace_decimate_stride_range = _parse_trace_decimation_cfg(
        train_cfg
    )
    fb_sigma_ms = float(optional_float(train_cfg, 'fb_sigma_ms', 10.0))
    if fb_sigma_ms <= 0.0:
        msg = 'train.fb_sigma_ms must be > 0'
        raise ValueError(msg)

    return CoarseTrainConfig(
        common=common,
        coarse=_load_coarse_mode_cfg(cfg),
        paths=_load_paths_cfg(cfg, allow_missing_infer_pairs=False),
        dataset=_load_dataset_cfg(cfg),
        transform=_load_transform_cfg(cfg),
        trace_anchor=_load_trace_anchor_cfg(cfg),
        norm_refs=load_norm_refs_cfg(cfg),
        train=CoarseTrainCfg(
            lr=float(require_float(train_cfg, 'lr')),
            weight_decay=float(optional_float(train_cfg, 'weight_decay', 0.0)),
            fb_sigma_ms=float(fb_sigma_ms),
            trace_decimate_prob=float(trace_decimate_prob),
            trace_decimate_stride_range=tuple(trace_decimate_stride_range),
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
    legacy_keys = [key for key in LEGACY_TILED_INFER_KEYS if key in infer_cfg]
    if legacy_keys:
        msg = (
            'fbpick-coarse global-anchor inference does not use legacy tiled '
            f'infer keys: {legacy_keys!r}'
        )
        raise ValueError(msg)

    batch_size = int(optional_int(infer_cfg, 'batch_size', 1))
    if batch_size != 1:
        msg = 'global-anchor coarse inference currently requires infer.batch_size == 1'
        raise ValueError(msg)
    dataset = _load_dataset_cfg(cfg)
    if len(dataset.primary_keys) != 1:
        msg = (
            'global-anchor coarse raw inference requires exactly one '
            'dataset.primary_keys entry'
        )
        raise ValueError(msg)

    return CoarseInferConfig(
        coarse=_load_coarse_mode_cfg(cfg),
        paths=_load_paths_cfg(cfg, allow_missing_infer_pairs=True),
        dataset=dataset,
        transform=_load_transform_cfg(cfg),
        trace_anchor=_load_trace_anchor_cfg(cfg),
        norm_refs=load_norm_refs_cfg(cfg),
        infer=CoarseInferRuntimeCfg(
            batch_size=batch_size,
            num_workers=int(optional_int(infer_cfg, 'num_workers', 0)),
            amp=bool(optional_bool(infer_cfg, 'amp', default=False)),
            use_tqdm=bool(optional_bool(infer_cfg, 'use_tqdm', default=False)),
        ),
        qc=_load_qc_cfg(cfg),
        model_sig=_load_model_sig(cfg),
    )
