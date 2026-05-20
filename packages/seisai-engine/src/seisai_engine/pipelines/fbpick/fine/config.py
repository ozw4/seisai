from __future__ import annotations

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
from seisai_engine.tracking.config import TrackingConfig, load_tracking_config

if TYPE_CHECKING:
    from seisai_engine.pipelines.common.config_schema import CommonTrainConfig

__all__ = [
    'FINE_CENTER_INDEX',
    'FINE_CKPT_OUTPUT_IDS',
    'FINE_CKPT_PIPELINE',
    'FINE_CKPT_SOFTMAX_AXIS',
    'FINE_CKPT_STAGE',
    'FINE_IN_CHANS',
    'FINE_OUT_CHANS',
    'FINE_TIME_LEN',
    'FINE_TRACE_LEN',
    'FineCenterAugmentCfg',
    'FineCkptCfg',
    'FineDatasetCfg',
    'FineInferConfig',
    'FineInferRuntimeCfg',
    'FineViewerCfg',
    'FineInitCfg',
    'FinePaths',
    'FineTrainCfg',
    'FineTrainConfig',
    'FineTransformCfg',
    'FineUniformJitterCfg',
    'FineWindowCenterCfg',
    'load_fine_infer_config',
    'load_fine_train_config',
]

FINE_IN_CHANS = 1
FINE_OUT_CHANS = 1
FINE_TRACE_LEN = 128
FINE_TIME_LEN = 256
FINE_CENTER_INDEX = 128
FINE_CKPT_PIPELINE = 'fbpick'
FINE_CKPT_STAGE = 'fine'
FINE_CKPT_OUTPUT_IDS = ('P',)
FINE_CKPT_SOFTMAX_AXIS = 'time'


@dataclass(frozen=True)
class FinePaths:
    segy_files: tuple[str, ...]
    fb_files: tuple[str, ...] | None
    robust_npz_files: tuple[str, ...] | None
    coarse_npz_files: tuple[str, ...] | None
    infer_segy_files: tuple[str, ...] | None
    infer_fb_files: tuple[str, ...] | None
    infer_robust_npz_files: tuple[str, ...] | None
    out_dir: str | None


@dataclass(frozen=True)
class FineDatasetCfg:
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
class FineTransformCfg:
    trace_len: int
    time_len: int
    center_index: int
    standardize_eps: float


@dataclass(frozen=True)
class FineWindowCenterCfg:
    npz_key: str
    fallback_npz_key: str | None


@dataclass(frozen=True)
class FineUniformJitterCfg:
    prob: float
    lo: int
    hi: int


@dataclass(frozen=True)
class FineCenterAugmentCfg:
    enabled: bool
    train_only: bool
    p_no_jitter: float
    uniform_jitter_samples: tuple[FineUniformJitterCfg, ...]
    clip_to_record: bool
    require_fb_inside: bool


@dataclass(frozen=True)
class FineTrainCfg:
    lr: float
    weight_decay: float
    fb_sigma_ms: float
    sigma_samples_min: float
    sigma_samples_max: float


@dataclass(frozen=True)
class FineInferRuntimeCfg:
    batch_size: int
    num_workers: int
    overlap_h: int
    amp: bool
    use_tqdm: bool
    high_conf_threshold: float


@dataclass(frozen=True)
class FineViewerCfg:
    enabled: bool
    save_overview_png: bool
    save_gather_png: bool
    max_gathers_per_file: int
    skip_gather_keys: dict[str, frozenset[int]]
    max_traces_per_gather: int | None
    waveform_norm: str
    dpi: int
    clip_percentile: float
    first_panel_only: bool


@dataclass(frozen=True)
class FineInitCfg:
    coarse_ckpt_path: str | None
    use_coarse_init: bool
    reset_seg_head: bool
    reset_first_bn_stats: bool


@dataclass(frozen=True)
class FineCkptCfg:
    save_best_only: bool
    metric: str
    mode: str
    pipeline: str = FINE_CKPT_PIPELINE
    stage: str = FINE_CKPT_STAGE
    output_ids: tuple[str, ...] = FINE_CKPT_OUTPUT_IDS
    softmax_axis: str = FINE_CKPT_SOFTMAX_AXIS


@dataclass(frozen=True)
class FineTrainConfig:
    common: 'CommonTrainConfig'
    tracking: TrackingConfig
    paths: FinePaths
    dataset: FineDatasetCfg
    transform: FineTransformCfg
    window_center: FineWindowCenterCfg
    center_augment: FineCenterAugmentCfg
    train: FineTrainCfg
    init: FineInitCfg
    model_sig: dict[str, Any]
    ckpt: FineCkptCfg


@dataclass(frozen=True)
class FineInferConfig:
    paths: FinePaths
    dataset: FineDatasetCfg
    transform: FineTransformCfg
    window_center: FineWindowCenterCfg
    infer: FineInferRuntimeCfg
    viewer: FineViewerCfg
    model_sig: dict[str, Any]


def _load_optional_str_list(cfg: dict, key: str) -> tuple[str, ...] | None:
    raw = cfg.get(key)
    if raw is None:
        return None
    if not isinstance(raw, list):
        msg = f'paths.{key} must be list[str] or null'
        raise TypeError(msg)
    return tuple(require_list_str(cfg, key))


def _load_paths_cfg(
    cfg: dict,
    *,
    allow_missing_infer_pairs: bool,
    require_fb_files: bool,
    require_robust_files: bool,
    require_out_dir: bool,
) -> FinePaths:
    paths = require_dict(cfg, 'paths')
    segy_files = tuple(require_list_str(paths, 'segy_files'))
    fb_files = _load_optional_str_list(paths, 'fb_files')
    robust_npz_files = _load_optional_str_list(paths, 'robust_npz_files')
    coarse_npz_files = _load_optional_str_list(paths, 'coarse_npz_files')

    if require_fb_files and fb_files is None:
        msg = 'paths.fb_files is required for fbpick fine training'
        raise ValueError(msg)
    if require_robust_files and robust_npz_files is None:
        msg = 'paths.robust_npz_files is required for fbpick fine'
        raise ValueError(msg)

    if fb_files is not None and len(fb_files) != len(segy_files):
        msg = 'paths.segy_files and paths.fb_files must have the same length'
        raise ValueError(msg)
    if robust_npz_files is not None and coarse_npz_files is not None:
        if (
            len(segy_files) != len(robust_npz_files)
            or len(segy_files) != len(coarse_npz_files)
        ):
            msg = (
                'paths.segy_files, paths.robust_npz_files, and '
                'paths.coarse_npz_files must have the same length'
            )
            raise ValueError(msg)
    elif robust_npz_files is not None and len(robust_npz_files) != len(segy_files):
        msg = 'paths.segy_files and paths.robust_npz_files must have the same length'
        raise ValueError(msg)
    elif coarse_npz_files is not None and len(coarse_npz_files) != len(segy_files):
        msg = (
            'paths.segy_files and paths.coarse_npz_files must have the same length'
        )
        raise ValueError(msg)

    infer_segy_files = _load_optional_str_list(paths, 'infer_segy_files')
    infer_fb_files = _load_optional_str_list(paths, 'infer_fb_files')
    infer_robust_npz_files = _load_optional_str_list(paths, 'infer_robust_npz_files')

    if not allow_missing_infer_pairs:
        if infer_segy_files is None:
            infer_segy_files = segy_files
        if infer_fb_files is None:
            infer_fb_files = fb_files
        if infer_robust_npz_files is None:
            infer_robust_npz_files = robust_npz_files

    if infer_segy_files is not None and infer_fb_files is not None:
        if len(infer_segy_files) != len(infer_fb_files):
            msg = 'paths.infer_segy_files and paths.infer_fb_files must have the same length'
            raise ValueError(msg)
    if infer_segy_files is not None and infer_robust_npz_files is not None:
        if len(infer_segy_files) != len(infer_robust_npz_files):
            msg = (
                'paths.infer_segy_files and paths.infer_robust_npz_files must have '
                'the same length'
            )
            raise ValueError(msg)

    if require_out_dir:
        out_dir_value = require_value(
            paths,
            'out_dir',
            str,
            type_message='config.paths.out_dir must be str',
        )
        out_dir = str(out_dir_value)
    else:
        out_dir_raw = paths.get('out_dir')
        if out_dir_raw is None:
            out_dir = None
        else:
            if not isinstance(out_dir_raw, str):
                msg = 'config.paths.out_dir must be str'
                raise TypeError(msg)
            out_dir = str(out_dir_raw)

    return FinePaths(
        segy_files=segy_files,
        fb_files=fb_files,
        robust_npz_files=robust_npz_files,
        coarse_npz_files=coarse_npz_files,
        infer_segy_files=infer_segy_files,
        infer_fb_files=infer_fb_files,
        infer_robust_npz_files=infer_robust_npz_files,
        out_dir=out_dir,
    )


def _load_dataset_cfg(cfg: dict) -> FineDatasetCfg:
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
    return FineDatasetCfg(
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


def _load_transform_cfg(cfg: dict) -> FineTransformCfg:
    raise_if_deprecated_time_len_keys(
        train_cfg=cfg.get('train'),
        transform_cfg=cfg.get('transform'),
    )
    transform_cfg = require_dict(cfg, 'transform')
    trace_len = int(require_int(transform_cfg, 'trace_len'))
    time_len = int(require_int(transform_cfg, 'time_len'))
    center_index = int(require_int(transform_cfg, 'center_index'))
    if trace_len != FINE_TRACE_LEN:
        msg = f'transform.trace_len must be {FINE_TRACE_LEN} for fbpick fine'
        raise ValueError(msg)
    if time_len != FINE_TIME_LEN:
        msg = f'transform.time_len must be {FINE_TIME_LEN} for fbpick fine'
        raise ValueError(msg)
    if center_index != FINE_CENTER_INDEX:
        msg = f'transform.center_index must be {FINE_CENTER_INDEX} for fbpick fine'
        raise ValueError(msg)
    return FineTransformCfg(
        trace_len=trace_len,
        time_len=time_len,
        center_index=center_index,
        standardize_eps=float(optional_float(transform_cfg, 'standardize_eps', 1.0e-8)),
    )


def _load_window_center_cfg(cfg: dict) -> FineWindowCenterCfg:
    window_cfg = cfg.get('window_center')
    if window_cfg is None:
        window_cfg = {}
    if not isinstance(window_cfg, dict):
        msg = 'window_center must be dict'
        raise TypeError(msg)

    npz_key = optional_str(window_cfg, 'npz_key', 'robust_pick_i').strip()
    if not npz_key:
        msg = 'window_center.npz_key must be non-empty'
        raise ValueError(msg)

    fallback_raw = window_cfg.get('fallback_npz_key')
    if fallback_raw is None:
        fallback_npz_key = None
    else:
        if not isinstance(fallback_raw, str):
            msg = 'window_center.fallback_npz_key must be str or null'
            raise TypeError(msg)
        fallback_npz_key = fallback_raw.strip()
        if not fallback_npz_key:
            msg = 'window_center.fallback_npz_key must be non-empty when provided'
            raise ValueError(msg)

    return FineWindowCenterCfg(
        npz_key=npz_key,
        fallback_npz_key=fallback_npz_key,
    )


def _optional_plain_float(
    cfg: dict,
    key: str,
    default: float,
    *,
    key_path: str,
) -> float:
    if key not in cfg:
        return float(default)
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f'{key_path} must be float'
        raise TypeError(msg)
    return float(value)


def _require_plain_float(cfg: dict, key: str, *, key_path: str) -> float:
    if key not in cfg:
        msg = f'missing config key: {key_path}'
        raise ValueError(msg)
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f'{key_path} must be float'
        raise TypeError(msg)
    return float(value)


def _require_plain_int(cfg: dict, key: str, *, key_path: str) -> int:
    if key not in cfg:
        msg = f'missing config key: {key_path}'
        raise ValueError(msg)
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f'{key_path} must be int'
        raise TypeError(msg)
    return int(value)


def _load_center_augment_cfg(cfg: dict) -> FineCenterAugmentCfg:
    augment_cfg = cfg.get('center_augment')
    if augment_cfg is None:
        augment_cfg = {}
    if not isinstance(augment_cfg, dict):
        msg = 'center_augment must be dict'
        raise TypeError(msg)

    p_no_jitter = _optional_plain_float(
        augment_cfg,
        'p_no_jitter',
        1.0,
        key_path='center_augment.p_no_jitter',
    )
    if p_no_jitter < 0.0 or p_no_jitter > 1.0:
        msg = 'center_augment.p_no_jitter must lie in [0, 1]'
        raise ValueError(msg)

    raw_components = augment_cfg.get('uniform_jitter_samples', [])
    if not isinstance(raw_components, list):
        msg = 'center_augment.uniform_jitter_samples must be list'
        raise TypeError(msg)

    components: list[FineUniformJitterCfg] = []
    for idx, item in enumerate(raw_components):
        if not isinstance(item, dict):
            msg = 'center_augment.uniform_jitter_samples entries must be dict'
            raise TypeError(msg)
        prefix = f'center_augment.uniform_jitter_samples[{idx}]'
        prob = _require_plain_float(item, 'prob', key_path=f'{prefix}.prob')
        if prob < 0.0:
            msg = f'{prefix}.prob must be >= 0'
            raise ValueError(msg)
        lo = _require_plain_int(item, 'lo', key_path=f'{prefix}.lo')
        hi = _require_plain_int(item, 'hi', key_path=f'{prefix}.hi')
        if lo > hi:
            msg = f'{prefix}.lo must be <= hi'
            raise ValueError(msg)
        components.append(FineUniformJitterCfg(prob=prob, lo=lo, hi=hi))

    prob_sum = p_no_jitter + sum(component.prob for component in components)
    if prob_sum <= 0.0:
        msg = 'center_augment probabilities must sum to > 0'
        raise ValueError(msg)

    require_fb_inside = bool(
        optional_bool(augment_cfg, 'require_fb_inside', default=True)
    )
    if not require_fb_inside:
        msg = 'center_augment.require_fb_inside must be true'
        raise ValueError(msg)

    return FineCenterAugmentCfg(
        enabled=bool(optional_bool(augment_cfg, 'enabled', default=False)),
        train_only=bool(optional_bool(augment_cfg, 'train_only', default=True)),
        p_no_jitter=p_no_jitter,
        uniform_jitter_samples=tuple(components),
        clip_to_record=bool(optional_bool(augment_cfg, 'clip_to_record', default=True)),
        require_fb_inside=require_fb_inside,
    )


def _load_viewer_cfg(cfg: dict) -> FineViewerCfg:
    viewer_cfg = cfg.get('viewer')
    if viewer_cfg is None:
        viewer_cfg = {}
    if not isinstance(viewer_cfg, dict):
        msg = 'viewer must be dict'
        raise TypeError(msg)

    dpi = int(optional_int(viewer_cfg, 'dpi', 150))
    if dpi <= 0:
        msg = 'viewer.dpi must be > 0'
        raise ValueError(msg)

    clip_percentile = float(optional_float(viewer_cfg, 'clip_percentile', 99.0))
    if clip_percentile <= 0.0 or clip_percentile > 100.0:
        msg = 'viewer.clip_percentile must lie in (0, 100]'
        raise ValueError(msg)

    max_gathers_per_file = int(optional_int(viewer_cfg, 'max_gathers_per_file', 8))
    if max_gathers_per_file < 0:
        msg = 'viewer.max_gathers_per_file must be >= 0'
        raise ValueError(msg)

    skip_gather_keys_raw = viewer_cfg.get('skip_gather_keys', {})
    if not isinstance(skip_gather_keys_raw, dict) or not all(
        isinstance(key, str) for key in skip_gather_keys_raw
    ):
        msg = 'viewer.skip_gather_keys must be dict[str, list[int]]'
        raise TypeError(msg)
    skip_gather_keys: dict[str, frozenset[int]] = {}
    for primary_key, values in skip_gather_keys_raw.items():
        if not isinstance(values, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in values
        ):
            msg = 'viewer.skip_gather_keys must be dict[str, list[int]]'
            raise TypeError(msg)
        skip_gather_keys[primary_key] = frozenset(int(item) for item in values)

    max_traces_per_gather_raw = viewer_cfg.get('max_traces_per_gather', 10000)
    if max_traces_per_gather_raw is None:
        max_traces_per_gather = None
    elif isinstance(max_traces_per_gather_raw, bool) or not isinstance(
        max_traces_per_gather_raw, int
    ):
        msg = 'viewer.max_traces_per_gather must be int > 0 or null'
        raise TypeError(msg)
    elif int(max_traces_per_gather_raw) <= 0:
        msg = 'viewer.max_traces_per_gather must be int > 0 or null'
        raise ValueError(msg)
    else:
        max_traces_per_gather = int(max_traces_per_gather_raw)

    waveform_norm = optional_str(viewer_cfg, 'waveform_norm', 'global').strip()
    if waveform_norm not in ('global', 'per_trace'):
        msg = 'viewer.waveform_norm must be one of: global, per_trace'
        raise ValueError(msg)

    return FineViewerCfg(
        enabled=bool(optional_bool(viewer_cfg, 'enabled', default=False)),
        save_overview_png=bool(
            optional_bool(viewer_cfg, 'save_overview_png', default=False)
        ),
        save_gather_png=bool(
            optional_bool(viewer_cfg, 'save_gather_png', default=False)
        ),
        max_gathers_per_file=max_gathers_per_file,
        skip_gather_keys=skip_gather_keys,
        max_traces_per_gather=max_traces_per_gather,
        waveform_norm=waveform_norm,
        dpi=dpi,
        clip_percentile=clip_percentile,
        first_panel_only=bool(
            optional_bool(viewer_cfg, 'first_panel_only', default=False)
        ),
    )


def _load_model_sig(cfg: dict) -> dict[str, Any]:
    model_cfg = require_dict(cfg, 'model')
    in_chans = int(require_int(model_cfg, 'in_chans'))
    out_chans = int(require_int(model_cfg, 'out_chans'))
    if in_chans != FINE_IN_CHANS:
        msg = f'model.in_chans must be {FINE_IN_CHANS} for fbpick fine'
        raise ValueError(msg)
    if out_chans != FINE_OUT_CHANS:
        msg = f'model.out_chans must be {FINE_OUT_CHANS} for fbpick fine'
        raise ValueError(msg)
    return build_encdec2d_kwargs(model_cfg, in_chans=in_chans, out_chans=out_chans)


def _load_init_cfg(cfg: dict) -> FineInitCfg:
    init_cfg = cfg.get('init')
    if init_cfg is None:
        init_cfg = {}
    if not isinstance(init_cfg, dict):
        msg = 'init must be dict'
        raise TypeError(msg)

    coarse_ckpt_raw = init_cfg.get('coarse_ckpt_path')
    if coarse_ckpt_raw is None:
        coarse_ckpt_path = None
    else:
        coarse_ckpt_value = require_value(
            init_cfg,
            'coarse_ckpt_path',
            str,
            type_message='config.init.coarse_ckpt_path must be str or null',
        )
        coarse_ckpt_path = str(coarse_ckpt_value).strip()
        if not coarse_ckpt_path:
            msg = 'init.coarse_ckpt_path must be non-empty when provided'
            raise ValueError(msg)

    use_coarse_init = bool(optional_bool(init_cfg, 'use_coarse_init', default=False))
    if use_coarse_init and coarse_ckpt_path is None:
        msg = 'init.coarse_ckpt_path is required when init.use_coarse_init=true'
        raise ValueError(msg)

    return FineInitCfg(
        coarse_ckpt_path=coarse_ckpt_path,
        use_coarse_init=use_coarse_init,
        reset_seg_head=bool(optional_bool(init_cfg, 'reset_seg_head', default=True)),
        reset_first_bn_stats=bool(
            optional_bool(init_cfg, 'reset_first_bn_stats', default=True)
        ),
    )


def load_fine_train_config(
    cfg: dict,
    *,
    base_dir: str | Path | None = None,
) -> FineTrainConfig:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    common = load_common_train_config(cfg)
    train_cfg = require_dict(cfg, 'train')
    ckpt_cfg = require_dict(cfg, 'ckpt')

    fb_sigma_ms = float(optional_float(train_cfg, 'fb_sigma_ms', 3.0))
    if fb_sigma_ms <= 0.0:
        msg = 'train.fb_sigma_ms must be > 0'
        raise ValueError(msg)

    sigma_samples_min = float(optional_float(train_cfg, 'sigma_samples_min', 1.5))
    sigma_samples_max = float(optional_float(train_cfg, 'sigma_samples_max', 12.0))
    if sigma_samples_min <= 0.0:
        msg = 'train.sigma_samples_min must be > 0'
        raise ValueError(msg)
    if sigma_samples_max <= 0.0:
        msg = 'train.sigma_samples_max must be > 0'
        raise ValueError(msg)
    if sigma_samples_max < sigma_samples_min:
        msg = 'train.sigma_samples_max must be >= train.sigma_samples_min'
        raise ValueError(msg)

    tracking_base_dir = Path('.') if base_dir is None else Path(base_dir)

    return FineTrainConfig(
        common=common,
        tracking=load_tracking_config(cfg, tracking_base_dir),
        paths=_load_paths_cfg(
            cfg,
            allow_missing_infer_pairs=False,
            require_fb_files=True,
            require_robust_files=True,
            require_out_dir=True,
        ),
        dataset=_load_dataset_cfg(cfg),
        transform=_load_transform_cfg(cfg),
        window_center=_load_window_center_cfg(cfg),
        center_augment=_load_center_augment_cfg(cfg),
        train=FineTrainCfg(
            lr=float(require_float(train_cfg, 'lr')),
            weight_decay=float(optional_float(train_cfg, 'weight_decay', 0.0)),
            fb_sigma_ms=fb_sigma_ms,
            sigma_samples_min=sigma_samples_min,
            sigma_samples_max=sigma_samples_max,
        ),
        init=_load_init_cfg(cfg),
        model_sig=_load_model_sig(cfg),
        ckpt=FineCkptCfg(
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


def load_fine_infer_config(cfg: dict) -> FineInferConfig:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    infer_cfg = require_dict(cfg, 'infer')
    transform = _load_transform_cfg(cfg)
    overlap_h = int(optional_int(infer_cfg, 'overlap_h', 96))
    if overlap_h < 0 or overlap_h >= transform.trace_len:
        msg = 'infer.overlap_h must satisfy 0 <= overlap_h < transform.trace_len'
        raise ValueError(msg)
    high_conf_threshold = float(optional_float(infer_cfg, 'high_conf_threshold', 0.5))
    if high_conf_threshold < 0.0 or high_conf_threshold > 1.0:
        msg = 'infer.high_conf_threshold must lie in [0, 1]'
        raise ValueError(msg)

    return FineInferConfig(
        paths=_load_paths_cfg(
            cfg,
            allow_missing_infer_pairs=True,
            require_fb_files=False,
            require_robust_files=True,
            require_out_dir=False,
        ),
        dataset=_load_dataset_cfg(cfg),
        transform=transform,
        window_center=_load_window_center_cfg(cfg),
        infer=FineInferRuntimeCfg(
            batch_size=int(optional_int(infer_cfg, 'batch_size', 1)),
            num_workers=int(optional_int(infer_cfg, 'num_workers', 0)),
            overlap_h=overlap_h,
            amp=bool(optional_bool(infer_cfg, 'amp', default=False)),
            use_tqdm=bool(optional_bool(infer_cfg, 'use_tqdm', default=False)),
            high_conf_threshold=high_conf_threshold,
        ),
        viewer=_load_viewer_cfg(cfg),
        model_sig=_load_model_sig(cfg),
    )
