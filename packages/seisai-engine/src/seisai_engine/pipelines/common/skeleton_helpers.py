from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import random
import numpy as np
import torch
from seisai_utils.config import require_dict, require_value
from torch.utils.data import Subset, get_worker_info

from .checkpoint_io import save_checkpoint
from .config_io import load_config, resolve_cfg_paths as _resolve_cfg_paths, resolve_relpath

__all__ = [
	'load_cfg_with_base_dir',
	'resolve_cfg_paths',
	'resolve_out_dir',
	'prepare_output_dirs',
	'set_dataset_rng',
	'make_train_worker_init_fn',
	'ensure_fixed_infer_num_workers',
	'epoch_vis_dir',
	'maybe_save_best_min',
]


def load_cfg_with_base_dir(cfg_path: Path) -> tuple[dict, Path]:
	cfg_path = Path(cfg_path).expanduser().resolve()
	if not cfg_path.is_file():
		raise FileNotFoundError(f'config not found: {cfg_path}')

	cfg = load_config(str(cfg_path))
	if cfg is None:
		raise ValueError(f'config is empty or failed to load: {cfg_path}')
	if not isinstance(cfg, dict):
		raise TypeError('config must be dict')

	base_dir = cfg_path.parent
	return cfg, base_dir


def resolve_cfg_paths(cfg: dict, base_dir: Path, keys: list[str]) -> None:
	_resolve_cfg_paths(cfg, base_dir, keys=keys)


def resolve_out_dir(cfg: dict, base_dir: Path) -> Path:
	paths = require_dict(cfg, 'paths')
	out_dir = require_value(
		paths,
		'out_dir',
		str,
		type_message='config.paths.out_dir must be str',
	)
	out_dir = resolve_relpath(base_dir, out_dir)
	return Path(out_dir)


def prepare_output_dirs(out_dir: Path, vis_subdir: str) -> tuple[Path, Path]:
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	ckpt_dir = out_dir / 'ckpt'
	ckpt_dir.mkdir(parents=True, exist_ok=True)

	vis_root = out_dir / str(vis_subdir)
	vis_root.mkdir(parents=True, exist_ok=True)

	return ckpt_dir, vis_root


def _unwrap_subset(ds: Any) -> Any:
	base = ds
	while isinstance(base, Subset):
		base = base.dataset
	return base


def set_dataset_rng(ds_or_subset: Any, seed: int) -> None:
	base = _unwrap_subset(ds_or_subset)
	if not hasattr(base, '_rng'):
		raise AttributeError(
			f'dataset {type(base).__name__} has no _rng attribute'
		)
	base._rng = np.random.default_rng(int(seed))


def make_train_worker_init_fn(seed_epoch: int) -> Callable[[int], None]:
	seed_epoch = int(seed_epoch)

	def _train_worker_init_fn(worker_id: int) -> None:
		info = get_worker_info()
		if info is None:
			raise RuntimeError('get_worker_info() returned None in worker')

		base = _unwrap_subset(info.dataset)
		seed_worker = int(seed_epoch) + int(worker_id) * 1000

		base._rng = np.random.default_rng(seed_worker)
		random.seed(seed_worker)
		np.random.seed(seed_worker)
		torch.manual_seed(seed_worker)

	return _train_worker_init_fn


def ensure_fixed_infer_num_workers(num_workers: int) -> None:
	if int(num_workers) != 0:
		msg = (
			'infer.num_workers must be 0 to keep fixed inference samples '
			'(set infer.num_workers: 0)'
		)
		raise ValueError(msg)


def epoch_vis_dir(vis_root: Path, epoch: int) -> Path:
	vis_root = Path(vis_root)
	vis_epoch_dir = vis_root / f'epoch_{int(epoch):04d}'
	vis_epoch_dir.mkdir(parents=True, exist_ok=True)
	return vis_epoch_dir


def maybe_save_best_min(
	best: float | None,
	current: float,
	ckpt_path: Path,
	payload: dict,
) -> float:
	current_val = float(current)
	if best is None or current_val < float(best):
		save_checkpoint(ckpt_path, payload)
		return current_val
	return float(best)
