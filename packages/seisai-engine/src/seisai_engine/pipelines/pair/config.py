from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from seisai_utils.config import (
    optional_bool,
    optional_float,
    optional_int,
    optional_str,
    optional_tuple2_float,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_list_str,
    require_value,
)

from seisai_engine.pipelines.common.config_io import (
    load_config,
    resolve_cfg_paths,
    resolve_relpath,
)
from seisai_engine.pipelines.common.listfiles import expand_cfg_listfiles
from seisai_engine.pipelines.common.config_loaders import load_common_train_config
from seisai_engine.pipelines.common.validate_primary_keys import validate_primary_keys

if TYPE_CHECKING:
    from seisai_engine.pipelines.common.config_schema import CommonTrainConfig

__all__ = [
    'PairCkptCfg',
    'PairDatasetCfg',
    'PairInferCfg',
    'PairInferConfig',
    'PairModelCfg',
    'PairPaths',
    'PairTileCfg',
    'PairTrainCfg',
    'PairTrainConfig',
    'PairVisCfg',
    'load_infer_config',
    'load_pair_train_config',
    'load_train_config',
]


@dataclass(frozen=True)
class PairPaths:
    input_segy_files: list[str]
    target_segy_files: list[str]
    out_dir: str


@dataclass(frozen=True)
class PairDatasetCfg:
    max_trials: int
    use_header_cache: bool
    verbose: bool
    progress: bool
    primary_keys: tuple[str, ...]
    secondary_key_fixed: bool
    waveform_mode: str


@dataclass(frozen=True)
class PairTrainCfg:
    batch_size: int
    epochs: int
    lr: float
    subset_traces: int
    time_len: int
    samples_per_epoch: int
    loss_kind: str
    seed: int
    use_amp: bool
    max_norm: float
    num_workers: int


@dataclass(frozen=True)
class PairInferCfg:
    batch_size: int
    max_batches: int
    subset_traces: int
    seed: int
    num_workers: int


@dataclass(frozen=True)
class PairTileCfg:
    tile_h: int
    overlap_h: int
    tiles_per_batch: int
    amp: bool
    use_tqdm: bool


@dataclass(frozen=True)
class PairVisCfg:
    out_subdir: str
    n: int
    cmap: str
    vmin: float
    vmax: float
    transpose_for_trace_time: bool
    per_trace_norm: bool
    per_trace_eps: float
    figsize: tuple[float, float]
    dpi: int


@dataclass(frozen=True)
class PairModelCfg:
    backbone: str
    pretrained: bool
    in_chans: int
    out_chans: int


@dataclass(frozen=True)
class PairCkptCfg:
    save_best_only: bool
    metric: str
    mode: str


@dataclass(frozen=True)
class PairTrainConfig:
    common: CommonTrainConfig
    paths: PairPaths
    infer_paths: PairPaths
    dataset: PairDatasetCfg
    train: PairTrainCfg
    infer: PairInferCfg
    tile: PairTileCfg
    vis: PairVisCfg
    ckpt: PairCkptCfg
    model: PairModelCfg


@dataclass(frozen=True)
class PairInferConfig:
    paths: PairPaths
    dataset: PairDatasetCfg
    infer: PairInferCfg
    tile: PairTileCfg
    vis: PairVisCfg
    ckpt: PairCkptCfg
    model: PairModelCfg


def _load_paths(paths: dict, *, base_dir: Path) -> PairPaths:
    input_segy_files = require_list_str(paths, 'input_segy_files')
    target_segy_files = require_list_str(paths, 'target_segy_files')
    if len(input_segy_files) != len(target_segy_files):
        msg = 'paths.input_segy_files and paths.target_segy_files must have same length'
        raise ValueError(
            msg
        )
    out_dir = require_value(
        paths,
        'out_dir',
        str,
        type_message='config.paths.out_dir must be str',
    )
    out_dir = resolve_relpath(base_dir, out_dir)
    return PairPaths(
        input_segy_files=list(input_segy_files),
        target_segy_files=list(target_segy_files),
        out_dir=str(out_dir),
    )


def _load_dataset_cfg(ds_cfg: dict) -> PairDatasetCfg:
    max_trials = require_int(ds_cfg, 'max_trials')
    use_header_cache = require_bool(ds_cfg, 'use_header_cache')
    verbose = require_bool(ds_cfg, 'verbose')
    progress = optional_bool(ds_cfg, 'progress', default=bool(verbose))
    primary_keys_list = require_list_str(ds_cfg, 'primary_keys')
    primary_keys = validate_primary_keys(primary_keys_list)
    secondary_key_fixed = require_bool(ds_cfg, 'secondary_key_fixed')
    waveform_mode = optional_str(ds_cfg, 'waveform_mode', 'eager').lower()
    if waveform_mode not in ('eager', 'mmap'):
        msg = 'dataset.waveform_mode must be "eager" or "mmap"'
        raise ValueError(msg)
    return PairDatasetCfg(
        max_trials=int(max_trials),
        use_header_cache=bool(use_header_cache),
        verbose=bool(verbose),
        progress=bool(progress),
        primary_keys=primary_keys,
        secondary_key_fixed=bool(secondary_key_fixed),
        waveform_mode=str(waveform_mode),
    )


def _load_model_cfg(model_cfg: dict) -> PairModelCfg:
    backbone = require_value(
        model_cfg,
        'backbone',
        str,
        type_message='config.model.backbone must be str',
    )
    pretrained = require_bool(model_cfg, 'pretrained')
    in_chans = require_int(model_cfg, 'in_chans')
    out_chans = require_int(model_cfg, 'out_chans')
    return PairModelCfg(
        backbone=str(backbone),
        pretrained=bool(pretrained),
        in_chans=int(in_chans),
        out_chans=int(out_chans),
    )


def load_pair_train_config(cfg: dict) -> PairTrainConfig:
    common = load_common_train_config(cfg)

    paths = require_dict(cfg, 'paths')
    input_segy_files = require_list_str(paths, 'input_segy_files')
    target_segy_files = require_list_str(paths, 'target_segy_files')
    infer_input_segy_files = require_list_str(paths, 'infer_input_segy_files')
    infer_target_segy_files = require_list_str(paths, 'infer_target_segy_files')
    if len(input_segy_files) != len(target_segy_files):
        msg = 'paths.input_segy_files and paths.target_segy_files must have same length'
        raise ValueError(
            msg
        )
    if len(infer_input_segy_files) != len(infer_target_segy_files):
        msg = (
            'paths.infer_input_segy_files and paths.infer_target_segy_files '
            'must have same length'
        )
        raise ValueError(msg)

    ds_cfg = require_dict(cfg, 'dataset')
    train_cfg = require_dict(cfg, 'train')
    infer_cfg = require_dict(cfg, 'infer')
    tile_cfg = require_dict(cfg, 'tile')
    vis_cfg = require_dict(cfg, 'vis')
    ckpt_cfg = require_dict(cfg, 'ckpt')
    model_cfg = require_dict(cfg, 'model')

    lr = require_float(train_cfg, 'lr')
    train_subset_traces = require_int(train_cfg, 'subset_traces')
    time_len = require_int(train_cfg, 'time_len')
    loss_kind = require_value(
        train_cfg,
        'loss_kind',
        str,
        type_message='config.train.loss_kind must be str',
    ).lower()

    infer_subset_traces = require_int(infer_cfg, 'subset_traces')

    tile_h = require_int(tile_cfg, 'tile_h')
    overlap_h = require_int(tile_cfg, 'overlap_h')
    tiles_per_batch = require_int(tile_cfg, 'tiles_per_batch')
    amp = require_bool(tile_cfg, 'amp')
    use_tqdm = require_bool(tile_cfg, 'use_tqdm')

    cmap = require_value(
        vis_cfg, 'cmap', str, type_message='config.vis.cmap must be str'
    )
    vmin = require_float(vis_cfg, 'vmin')
    vmax = require_float(vis_cfg, 'vmax')
    transpose_for_trace_time = require_bool(vis_cfg, 'transpose_for_trace_time')
    per_trace_norm = require_bool(vis_cfg, 'per_trace_norm')
    per_trace_eps = require_float(vis_cfg, 'per_trace_eps')
    if 'figsize' not in vis_cfg:
        msg = 'missing config key: figsize'
        raise ValueError(msg)
    figsize = optional_tuple2_float(vis_cfg, 'figsize', (0.0, 0.0))
    dpi = require_int(vis_cfg, 'dpi')

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

    return PairTrainConfig(
        common=common,
        paths=PairPaths(
            input_segy_files=list(input_segy_files),
            target_segy_files=list(target_segy_files),
            out_dir=str(common.output.out_dir),
        ),
        infer_paths=PairPaths(
            input_segy_files=list(infer_input_segy_files),
            target_segy_files=list(infer_target_segy_files),
            out_dir=str(common.output.out_dir),
        ),
        dataset=_load_dataset_cfg(ds_cfg),
        train=PairTrainCfg(
            batch_size=int(common.train.train_batch_size),
            epochs=int(common.train.epochs),
            lr=float(lr),
            subset_traces=int(train_subset_traces),
            time_len=int(time_len),
            samples_per_epoch=int(common.train.samples_per_epoch),
            loss_kind=str(loss_kind),
            seed=int(common.seeds.seed_train),
            use_amp=bool(common.train.use_amp_train),
            max_norm=float(common.train.max_norm),
            num_workers=int(common.train.train_num_workers),
        ),
        infer=PairInferCfg(
            batch_size=int(common.infer.infer_batch_size),
            max_batches=int(common.infer.infer_max_batches),
            subset_traces=int(infer_subset_traces),
            seed=int(common.seeds.seed_infer),
            num_workers=int(common.infer.infer_num_workers),
        ),
        tile=PairTileCfg(
            tile_h=int(tile_h),
            overlap_h=int(overlap_h),
            tiles_per_batch=int(tiles_per_batch),
            amp=bool(amp),
            use_tqdm=bool(use_tqdm),
        ),
        vis=PairVisCfg(
            out_subdir=str(common.output.vis_subdir),
            n=int(common.infer.vis_n),
            cmap=str(cmap),
            vmin=float(vmin),
            vmax=float(vmax),
            transpose_for_trace_time=bool(transpose_for_trace_time),
            per_trace_norm=bool(per_trace_norm),
            per_trace_eps=float(per_trace_eps),
            figsize=figsize,
            dpi=int(dpi),
        ),
        ckpt=PairCkptCfg(
            save_best_only=bool(save_best_only),
            metric=str(metric),
            mode=str(mode),
        ),
        model=_load_model_cfg(model_cfg),
    )


def load_train_config(config_path: str | Path) -> PairTrainConfig:
    cfg = load_config(str(config_path))

    base_dir = Path(config_path).expanduser().resolve().parent
    resolve_cfg_paths(
        cfg,
        base_dir,
        keys=[
            'paths.input_segy_files',
            'paths.target_segy_files',
            'paths.infer_input_segy_files',
            'paths.infer_target_segy_files',
        ],
    )
    expand_cfg_listfiles(
        cfg,
        keys=[
            'paths.input_segy_files',
            'paths.target_segy_files',
            'paths.infer_input_segy_files',
            'paths.infer_target_segy_files',
        ],
    )
    return load_pair_train_config(cfg)


def load_infer_config(config_path: str | Path) -> PairInferConfig:
    cfg = load_config(str(config_path))

    base_dir = Path(config_path).expanduser().resolve().parent
    resolve_cfg_paths(
        cfg,
        base_dir,
        keys=['paths.input_segy_files', 'paths.target_segy_files'],
    )
    expand_cfg_listfiles(
        cfg,
        keys=['paths.input_segy_files', 'paths.target_segy_files'],
    )

    paths = require_dict(cfg, 'paths')
    ds_cfg = require_dict(cfg, 'dataset')
    infer_cfg = require_dict(cfg, 'infer')
    tile_cfg = require_dict(cfg, 'tile')
    vis_cfg = require_dict(cfg, 'vis')
    ckpt_cfg = require_dict(cfg, 'ckpt')
    model_cfg = require_dict(cfg, 'model')

    infer_batch_size = require_int(infer_cfg, 'batch_size')
    max_batches = require_int(infer_cfg, 'max_batches')
    subset_traces = require_int(infer_cfg, 'subset_traces')
    seed = optional_int(infer_cfg, 'seed', 43)
    num_workers = optional_int(infer_cfg, 'num_workers', 0)

    tile_h = require_int(tile_cfg, 'tile_h')
    overlap_h = require_int(tile_cfg, 'overlap_h')
    tiles_per_batch = require_int(tile_cfg, 'tiles_per_batch')
    amp = optional_bool(tile_cfg, 'amp', default=True)
    use_tqdm = optional_bool(tile_cfg, 'use_tqdm', default=False)

    out_subdir = optional_str(vis_cfg, 'out_subdir', 'vis')
    n = require_int(vis_cfg, 'n')
    cmap = optional_str(vis_cfg, 'cmap', 'seismic')
    vmin = optional_float(vis_cfg, 'vmin', -3.0)
    vmax = optional_float(vis_cfg, 'vmax', 3.0)
    transpose_for_trace_time = optional_bool(
        vis_cfg, 'transpose_for_trace_time', default=True
    )
    per_trace_norm = optional_bool(vis_cfg, 'per_trace_norm', default=True)
    per_trace_eps = optional_float(vis_cfg, 'per_trace_eps', 1e-8)
    figsize = optional_tuple2_float(vis_cfg, 'figsize', (20.0, 15.0))
    dpi = optional_int(vis_cfg, 'dpi', 300)

    save_best_only = optional_bool(ckpt_cfg, 'save_best_only', default=True)
    metric = optional_str(ckpt_cfg, 'metric', 'infer_loss')
    mode = optional_str(ckpt_cfg, 'mode', 'min')

    return PairInferConfig(
        paths=_load_paths(paths, base_dir=base_dir),
        dataset=_load_dataset_cfg(ds_cfg),
        infer=PairInferCfg(
            batch_size=int(infer_batch_size),
            max_batches=int(max_batches),
            subset_traces=int(subset_traces),
            seed=int(seed),
            num_workers=int(num_workers),
        ),
        tile=PairTileCfg(
            tile_h=int(tile_h),
            overlap_h=int(overlap_h),
            tiles_per_batch=int(tiles_per_batch),
            amp=bool(amp),
            use_tqdm=bool(use_tqdm),
        ),
        vis=PairVisCfg(
            out_subdir=str(out_subdir),
            n=int(n),
            cmap=str(cmap),
            vmin=float(vmin),
            vmax=float(vmax),
            transpose_for_trace_time=bool(transpose_for_trace_time),
            per_trace_norm=bool(per_trace_norm),
            per_trace_eps=float(per_trace_eps),
            figsize=figsize,
            dpi=int(dpi),
        ),
        ckpt=PairCkptCfg(
            save_best_only=bool(save_best_only),
            metric=str(metric),
            mode=str(mode),
        ),
        model=_load_model_cfg(model_cfg),
    )
