"""Shared helpers for paired SEG-Y training/inference examples.

These helpers are intentionally small and fail-fast.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from seisai_dataset import BuildPlan, SegyGatherPairDataset
from seisai_dataset.builder.builder import IdentitySignal, SelectStack
from seisai_models.models.encdec2d import EncDec2D
from seisai_transforms.augment import PerTraceStandardize, RandomCropOrPad, ViewCompose
from seisai_utils.config import (
	load_config,
	optional_bool,
	optional_float,
	optional_int,
	optional_str,
	optional_tuple2_float,
	require_dict,
	require_float,
	require_int,
	require_list_str,
)


@dataclass(frozen=True)
class PairPaths:
	input_segy_files: list[str]
	target_segy_files: list[str]
	ckpt_path: str


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
	out_dir: str
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
class PairTrainConfig:
	paths: PairPaths
	dataset: PairDatasetCfg
	train: PairTrainCfg
	model: PairModelCfg


@dataclass(frozen=True)
class PairInferConfig:
	paths: PairPaths
	dataset: PairDatasetCfg
	infer: PairInferCfg
	tile: PairTileCfg
	vis: PairVisCfg
	model: PairModelCfg


def build_device() -> torch.device:
	return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_all(seed: int) -> None:
	torch.manual_seed(int(seed))
	_ = np.random.default_rng(int(seed))


def build_plan() -> BuildPlan:
	return BuildPlan(
		wave_ops=[
			IdentitySignal(source_key='x_view_input', dst='x_in', copy=False),
			IdentitySignal(source_key='x_view_target', dst='x_tg', copy=False),
		],
		label_ops=[],
		input_stack=SelectStack(
			keys=['x_in'],
			dst='input',
			dtype=np.float32,
			to_torch=True,
		),
		target_stack=SelectStack(
			keys=['x_tg'],
			dst='target',
			dtype=np.float32,
			to_torch=True,
		),
	)


def build_train_transform(time_len: int, eps: float = 1e-8) -> ViewCompose:
	return ViewCompose(
		[
			RandomCropOrPad(target_len=int(time_len)),
			PerTraceStandardize(eps=float(eps)),
		]
	)


def build_infer_transform(eps: float = 1e-8) -> ViewCompose:
	return ViewCompose([PerTraceStandardize(eps=float(eps))])


def build_pair_dataset(
	paths: PairPaths,
	ds_cfg: PairDatasetCfg,
	transform: ViewCompose,
	plan: BuildPlan,
	subset_traces: int,
	valid: bool,
) -> SegyGatherPairDataset:
	return SegyGatherPairDataset(
		input_segy_files=paths.input_segy_files,
		target_segy_files=paths.target_segy_files,
		transform=transform,
		plan=plan,
		subset_traces=int(subset_traces),
		primary_keys=ds_cfg.primary_keys,
		valid=bool(valid),
		verbose=bool(ds_cfg.verbose),
		max_trials=int(ds_cfg.max_trials),
		use_header_cache=bool(ds_cfg.use_header_cache),
	)


def build_model(cfg: PairModelCfg) -> EncDec2D:
	model = EncDec2D(
		backbone=cfg.backbone,
		in_chans=int(cfg.in_chans),
		out_chans=int(cfg.out_chans),
		pretrained=bool(cfg.pretrained),
	)
	model.use_tta = False
	return model


def save_checkpoint(
	ckpt_path: str | Path,
	model: torch.nn.Module,
	model_cfg: PairModelCfg,
	epoch: int,
	global_step: int,
	optimizer: torch.optim.Optimizer | None = None,
) -> None:
	ckpt = {
		'model_state_dict': model.state_dict(),
		'model_cfg': asdict(model_cfg),
		'epoch': int(epoch),
		'global_step': int(global_step),
	}
	if optimizer is not None:
		ckpt['optimizer_state_dict'] = optimizer.state_dict()

	out_path = Path(ckpt_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(ckpt, out_path)


def load_checkpoint(ckpt_path: str | Path) -> dict:
	ckpt = torch.load(Path(ckpt_path), map_location='cpu')
	if not isinstance(ckpt, dict):
		raise ValueError('checkpoint must be a dict')
	if 'model_state_dict' not in ckpt:
		raise ValueError('checkpoint missing: model_state_dict')
	if 'model_cfg' not in ckpt:
		raise ValueError('checkpoint missing: model_cfg')
	return ckpt


def _load_paths(paths: dict) -> PairPaths:
	input_segy_files = require_list_str(paths, 'input_segy_files')
	target_segy_files = require_list_str(paths, 'target_segy_files')
	if len(input_segy_files) != len(target_segy_files):
		raise ValueError(
			'paths.input_segy_files and paths.target_segy_files must have same length'
		)
	ckpt_path = optional_str(paths, 'ckpt_path', './_pair_ckpt/encdec2d_pair.pth')
	return PairPaths(
		input_segy_files=list(input_segy_files),
		target_segy_files=list(target_segy_files),
		ckpt_path=str(ckpt_path),
	)


def _load_dataset_cfg(ds_cfg: dict) -> PairDatasetCfg:
	max_trials = optional_int(ds_cfg, 'max_trials', 2048)
	use_header_cache = optional_bool(ds_cfg, 'use_header_cache', default=True)
	verbose = optional_bool(ds_cfg, 'verbose', default=True)
	primary_keys_list = ds_cfg.get('primary_keys', ['ffid'])
	if not isinstance(primary_keys_list, list) or not all(
		isinstance(x, str) for x in primary_keys_list
	):
		raise ValueError('dataset.primary_keys must be list[str]')
	primary_keys = tuple(primary_keys_list)
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

	paths = require_dict(cfg, 'paths')
	ds_cfg = require_dict(cfg, 'dataset')
	train_cfg = require_dict(cfg, 'train')
	model_cfg = require_dict(cfg, 'model')

	train_batch_size = require_int(train_cfg, 'batch_size')
	epochs = require_int(train_cfg, 'epochs')
	lr = require_float(train_cfg, 'lr')
	subset_traces = require_int(train_cfg, 'subset_traces')
	time_len = require_int(train_cfg, 'time_len')
	samples_per_epoch = require_int(train_cfg, 'samples_per_epoch')
	loss_kind = optional_str(train_cfg, 'loss_kind', 'l1').lower()
	seed = optional_int(train_cfg, 'seed', 42)
	use_amp = optional_bool(train_cfg, 'use_amp', default=True)
	max_norm = optional_float(train_cfg, 'max_norm', 1.0)
	num_workers = optional_int(train_cfg, 'num_workers', 0)

	if loss_kind not in ('l1', 'mse'):
		raise ValueError('train.loss_kind must be "l1" or "mse"')

	return PairTrainConfig(
		paths=_load_paths(paths),
		dataset=_load_dataset_cfg(ds_cfg),
		train=PairTrainCfg(
			batch_size=int(train_batch_size),
			epochs=int(epochs),
			lr=float(lr),
			subset_traces=int(subset_traces),
			time_len=int(time_len),
			samples_per_epoch=int(samples_per_epoch),
			loss_kind=str(loss_kind),
			seed=int(seed),
			use_amp=bool(use_amp),
			max_norm=float(max_norm),
			num_workers=int(num_workers),
		),
		model=_load_model_cfg(model_cfg),
	)


def load_infer_config(config_path: str | Path) -> PairInferConfig:
	cfg = load_config(str(config_path))

	paths = require_dict(cfg, 'paths')
	ds_cfg = require_dict(cfg, 'dataset')
	infer_cfg = require_dict(cfg, 'infer')
	tile_cfg = require_dict(cfg, 'tile')
	vis_cfg = require_dict(cfg, 'vis')
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

	out_dir = optional_str(vis_cfg, 'out_dir', './_pair_vis')
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

	return PairInferConfig(
		paths=_load_paths(paths),
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
			out_dir=str(out_dir),
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
		model=_load_model_cfg(model_cfg),
	)
