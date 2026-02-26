from __future__ import annotations

from pathlib import Path

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
from seisai_utils.listfiles import expand_cfg_listfiles

from seisai_engine.pipelines.common.config_keys import (
    format_cfg_key,
    normalize_endian,
    raise_if_deprecated_time_len_keys,
)
from seisai_engine.pipelines.common.config_loaders import (
    load_common_train_config,
    parse_train_eval_loss_specs,
)
from seisai_engine.pipelines.common.config_io import load_config, resolve_relpath
from seisai_engine.pipelines.common.encdec2d_cfg import build_encdec2d_kwargs
from seisai_engine.pipelines.common.validate_primary_keys import validate_primary_keys

from .schema import (
    PairCkptCfg,
    PairDatasetCfg,
    PairInferCfg,
    PairInferConfig,
    PairModelCfg,
    PairPaths,
    PairTileCfg,
    PairTrainCfg,
    PairTrainConfig,
    PairTransformCfg,
    PairVisCfg,
)

__all__ = [
    '_load_dataset_cfg',
    'load_infer_config',
    'load_pair_train_config',
    'load_train_config',
]


def _load_paths(paths: dict, *, base_dir: Path) -> PairPaths:
    input_segy_files = require_list_str(paths, 'input_segy_files')
    target_segy_files = require_list_str(paths, 'target_segy_files')
    if len(input_segy_files) != len(target_segy_files):
        msg = 'paths.input_segy_files and paths.target_segy_files must have same length'
        raise ValueError(msg)
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


def _format_key(section: str, key: str) -> str:
    return format_cfg_key(section, key)


def _require_pair_time_len(cfg: dict, *, train_cfg: dict) -> int:
    raise_if_deprecated_time_len_keys(
        train_cfg=train_cfg,
        transform_cfg=cfg.get('transform'),
    )
    transform_cfg = require_dict(cfg, 'transform')
    return int(require_int(transform_cfg, 'time_len'))


def _require_pair_loss_specs(
    cfg: dict, *, train_cfg: dict
):
    if 'loss_kind' in train_cfg:
        msg = (
            f'deprecated key: {_format_key("train", "loss_kind")}; '
            f'use {_format_key("train", "losses")}'
        )
        raise ValueError(msg)
    return parse_train_eval_loss_specs(
        cfg,
        train_cfg=train_cfg,
        default_scope='all',
        scope_key='loss_scope',
        losses_key='losses',
        train_label='train.losses',
        eval_label='eval.losses',
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
    train_input_endian = normalize_endian(
        value=optional_str(ds_cfg, 'train_input_endian', 'big'),
        key_name='dataset.train_input_endian',
    )
    train_target_endian = normalize_endian(
        value=optional_str(ds_cfg, 'train_target_endian', 'big'),
        key_name='dataset.train_target_endian',
    )
    infer_input_endian = normalize_endian(
        value=optional_str(ds_cfg, 'infer_input_endian', 'big'),
        key_name='dataset.infer_input_endian',
    )
    infer_target_endian = normalize_endian(
        value=optional_str(ds_cfg, 'infer_target_endian', 'big'),
        key_name='dataset.infer_target_endian',
    )
    return PairDatasetCfg(
        max_trials=int(max_trials),
        use_header_cache=bool(use_header_cache),
        verbose=bool(verbose),
        progress=bool(progress),
        primary_keys=primary_keys,
        secondary_key_fixed=bool(secondary_key_fixed),
        waveform_mode=str(waveform_mode),
        train_input_endian=str(train_input_endian),
        train_target_endian=str(train_target_endian),
        infer_input_endian=str(infer_input_endian),
        infer_target_endian=str(infer_target_endian),
    )


def _load_model_cfg(model_cfg: dict) -> PairModelCfg:
    in_chans = require_int(model_cfg, 'in_chans')
    out_chans = require_int(model_cfg, 'out_chans')
    encdec_kwargs = build_encdec2d_kwargs(
        model_cfg,
        in_chans=int(in_chans),
        out_chans=int(out_chans),
    )
    return PairModelCfg(**encdec_kwargs)


def load_pair_train_config(cfg: dict) -> PairTrainConfig:
    common = load_common_train_config(cfg)

    paths = require_dict(cfg, 'paths')
    input_segy_files = require_list_str(paths, 'input_segy_files')
    target_segy_files = require_list_str(paths, 'target_segy_files')
    infer_input_segy_files = require_list_str(paths, 'infer_input_segy_files')
    infer_target_segy_files = require_list_str(paths, 'infer_target_segy_files')
    if len(input_segy_files) != len(target_segy_files):
        msg = 'paths.input_segy_files and paths.target_segy_files must have same length'
        raise ValueError(msg)
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
    time_len = _require_pair_time_len(cfg, train_cfg=train_cfg)
    loss_specs_train, loss_specs_eval = _require_pair_loss_specs(
        cfg, train_cfg=train_cfg
    )

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
            samples_per_epoch=int(common.train.samples_per_epoch),
            seed=int(common.seeds.seed_train),
            use_amp=bool(common.train.use_amp_train),
            max_norm=float(common.train.max_norm),
            num_workers=int(common.train.train_num_workers),
        ),
        loss_specs_train=loss_specs_train,
        loss_specs_eval=loss_specs_eval,
        transform=PairTransformCfg(time_len=int(time_len)),
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
