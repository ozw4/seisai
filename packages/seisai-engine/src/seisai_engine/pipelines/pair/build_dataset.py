from __future__ import annotations

from pathlib import Path

from seisai_dataset import BuildPlan, SegyGatherPairDataset
from seisai_transforms.augment import PerTraceStandardize, RandomCropOrPad, ViewCompose

from .config import PairDatasetCfg, PairPaths

__all__ = [
	'build_train_transform',
	'build_infer_transform',
	'build_pair_dataset',
]


def build_train_transform(time_len: int, eps: float = 1e-8) -> ViewCompose:
	return ViewCompose(
		[
			RandomCropOrPad(target_len=int(time_len)),
			PerTraceStandardize(eps=float(eps)),
		]
	)


def build_infer_transform(eps: float = 1e-8) -> ViewCompose:
	return ViewCompose([PerTraceStandardize(eps=float(eps))])


def _validate_files(paths: PairPaths) -> None:
	for p in list(paths.input_segy_files) + list(paths.target_segy_files):
		if not Path(p).is_file():
			raise FileNotFoundError(f'file not found: {p}')


def build_pair_dataset(
	paths: PairPaths,
	ds_cfg: PairDatasetCfg,
	transform: ViewCompose,
	plan: BuildPlan,
	subset_traces: int,
	secondary_key_fixed: bool,
) -> SegyGatherPairDataset:
	_validate_files(paths)

	return SegyGatherPairDataset(
		input_segy_files=paths.input_segy_files,
		target_segy_files=paths.target_segy_files,
		transform=transform,
		plan=plan,
		subset_traces=int(subset_traces),
		primary_keys=ds_cfg.primary_keys,
		secondary_key_fixed=bool(secondary_key_fixed),
		verbose=bool(ds_cfg.verbose),
		max_trials=int(ds_cfg.max_trials),
		use_header_cache=bool(ds_cfg.use_header_cache),
	)
