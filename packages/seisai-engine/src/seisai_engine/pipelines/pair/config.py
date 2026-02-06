from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from seisai_engine.pipelines.common.config_io import (
	load_config,
	resolve_cfg_paths,
	resolve_relpath,
)
from seisai_utils.config import (
	optional_bool,
	optional_float,
	optional_int,
	optional_str,
	optional_tuple2_float,
	require_dict,
	require_float,
	require_int,
	require_list_str,
	require_value,
)

__all__ = [
	'PairPaths',
	'PairDatasetCfg',
	'PairTrainCfg',
	'PairInferCfg',
	'PairTileCfg',
	'PairVisCfg',
	'PairCkptCfg',
	'PairModelCfg',
	'PairTrainConfig',
	'PairInferConfig',
	'load_train_config',
	'load_infer_config',
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
	primary_keys: tuple[str, ...]


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
	paths: PairPaths
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


def _validate_primary_keys(primary_keys_list: object) -> tuple[str, ...]:
	if not isinstance(primary_keys_list, list) or not all(
		isinstance(x, str) for x in primary_keys_list
	):
		raise ValueError('dataset.primary_keys must be list[str]')
	return tuple(primary_keys_list)


def _load_paths(paths: dict, *, base_dir: Path) -> PairPaths:
	input_segy_files = require_list_str(paths, 'input_segy_files')
	target_segy_files = require_list_str(paths, 'target_segy_files')
	if len(input_segy_files) != len(target_segy_files):
		raise ValueError(
			'paths.input_segy_files and paths.target_segy_files must have same length'
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
	max_trials = optional_int(ds_cfg, 'max_trials', 2048)
	use_header_cache = optional_bool(ds_cfg, 'use_header_cache', default=True)
	verbose = optional_bool(ds_cfg, 'verbose', default=True)
	primary_keys_list = ds_cfg.get('primary_keys', ['ffid'])
	primary_keys = _validate_primary_keys(primary_keys_list)
	return PairDatasetCfg(
		max_trials=int(max_trials),
		use_header_cache=bool(use_header_cache),
		verbose=bool(verbose),
		primary_keys=primary_keys,
	)


def _load_model_cfg(model_cfg: dict) -> PairModelCfg:
	backbone = optional_str(model_cfg, 'backbone', 'resnet18')
	pretrained = optional_bool(model_cfg, 'pretrained', default=False)
	in_chans = optional_int(model_cfg, 'in_chans', 1)
	out_chans = optional_int(model_cfg, 'out_chans', 1)
	return PairModelCfg(
		backbone=str(backbone),
		pretrained=bool(pretrained),
		in_chans=int(in_chans),
		out_chans=int(out_chans),
	)


def load_train_config(config_path: str | Path) -> PairTrainConfig:
	cfg = load_config(str(config_path))

	base_dir = Path(config_path).expanduser().resolve().parent
	resolve_cfg_paths(
		cfg,
		base_dir,
		keys=['paths.input_segy_files', 'paths.target_segy_files'],
	)

	paths = require_dict(cfg, 'paths')
	ds_cfg = require_dict(cfg, 'dataset')
	train_cfg = require_dict(cfg, 'train')
	infer_cfg = require_dict(cfg, 'infer')
	tile_cfg = require_dict(cfg, 'tile')
	vis_cfg = require_dict(cfg, 'vis')
	ckpt_cfg = require_dict(cfg, 'ckpt')
	model_cfg = require_dict(cfg, 'model')

	train_batch_size = require_int(train_cfg, 'batch_size')
	epochs = require_int(train_cfg, 'epochs')
	lr = require_float(train_cfg, 'lr')
	train_subset_traces = require_int(train_cfg, 'subset_traces')
	time_len = require_int(train_cfg, 'time_len')
	samples_per_epoch = require_int(train_cfg, 'samples_per_epoch')
	loss_kind = optional_str(train_cfg, 'loss_kind', 'l1').lower()
	seed = optional_int(train_cfg, 'seed', 42)
	use_amp = optional_bool(train_cfg, 'use_amp', default=True)
	max_norm = optional_float(train_cfg, 'max_norm', 1.0)
	train_num_workers = optional_int(train_cfg, 'num_workers', 0)

	if loss_kind not in ('l1', 'mse'):
		raise ValueError('train.loss_kind must be "l1" or "mse"')

	infer_batch_size = require_int(infer_cfg, 'batch_size')
	max_batches = require_int(infer_cfg, 'max_batches')
	infer_subset_traces = require_int(infer_cfg, 'subset_traces')
	infer_seed = optional_int(infer_cfg, 'seed', 43)
	infer_num_workers = optional_int(infer_cfg, 'num_workers', 0)

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

	return PairTrainConfig(
		paths=_load_paths(paths, base_dir=base_dir),
		dataset=_load_dataset_cfg(ds_cfg),
		train=PairTrainCfg(
			batch_size=int(train_batch_size),
			epochs=int(epochs),
			lr=float(lr),
			subset_traces=int(train_subset_traces),
			time_len=int(time_len),
			samples_per_epoch=int(samples_per_epoch),
			loss_kind=str(loss_kind),
			seed=int(seed),
			use_amp=bool(use_amp),
			max_norm=float(max_norm),
			num_workers=int(train_num_workers),
		),
		infer=PairInferCfg(
			batch_size=int(infer_batch_size),
			max_batches=int(max_batches),
			subset_traces=int(infer_subset_traces),
			seed=int(infer_seed),
			num_workers=int(infer_num_workers),
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


def load_infer_config(config_path: str | Path) -> PairInferConfig:
	cfg = load_config(str(config_path))

	base_dir = Path(config_path).expanduser().resolve().parent
	resolve_cfg_paths(
		cfg,
		base_dir,
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
