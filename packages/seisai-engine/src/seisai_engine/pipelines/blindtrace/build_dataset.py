from __future__ import annotations

from seisai_dataset import (
	FirstBreakGate,
	FirstBreakGateConfig,
	SegyGatherPipelineDataset,
)
from seisai_transforms.augment import PerTraceStandardize, RandomCropOrPad, ViewCompose

from seisai_engine.pipelines.common.validate_files import validate_files_exist

__all__ = [
	'build_transform',
	'build_fbgate',
	'build_dataset',
	'validate_primary_keys',
]


def build_transform(*, time_len: int, per_trace_standardize: bool) -> ViewCompose:
	ops: list = [RandomCropOrPad(target_len=int(time_len))]
	if per_trace_standardize:
		ops.append(PerTraceStandardize(eps=1e-8))
	return ViewCompose(ops)


def build_fbgate(
	*, apply_on: str, min_pick_ratio: float, verbose: bool
) -> FirstBreakGate:
	ap = str(apply_on).lower()
	if ap == 'on':
		ap = 'any'
	if ap not in ('any', 'super_only', 'off'):
		msg = 'fbgate.apply_on must be "any", "super_only", or "off"'
		raise ValueError(msg)

	cfg = FirstBreakGateConfig(
		apply_on=ap,
		min_pick_ratio=float(min_pick_ratio),
		verbose=bool(verbose),
	)
	return FirstBreakGate(cfg)


def validate_primary_keys(primary_keys_list: object) -> tuple[str, ...]:
	if not isinstance(primary_keys_list, list) or not all(
		isinstance(x, str) for x in primary_keys_list
	):
		msg = 'dataset.primary_keys must be list[str]'
		raise ValueError(msg)
	return tuple(primary_keys_list)


def build_dataset(
	*,
	segy_files: list[str],
	fb_files: list[str],
	transform: ViewCompose,
	fbgate: FirstBreakGate,
	plan,
	subset_traces: int,
	primary_keys: tuple[str, ...],
	secondary_key_fixed: bool,
	verbose: bool,
	max_trials: int,
	use_header_cache: bool,
) -> SegyGatherPipelineDataset:
	validate_files_exist(list(segy_files) + list(fb_files))
	return SegyGatherPipelineDataset(
		segy_files=segy_files,
		fb_files=fb_files,
		transform=transform,
		fbgate=fbgate,
		plan=plan,
		subset_traces=int(subset_traces),
		primary_keys=primary_keys,
		secondary_key_fixed=bool(secondary_key_fixed),
		verbose=bool(verbose),
		max_trials=int(max_trials),
		use_header_cache=bool(use_header_cache),
	)
