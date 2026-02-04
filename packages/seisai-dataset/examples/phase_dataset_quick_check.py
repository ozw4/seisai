# %%
"""Quick check for SegyGatherPhasePipelineDataset.

Requires:
  - seisai-dataset
  - seisai-pick
  - seisai-transforms
  - torch, numpy, segyio

Run:
  python packages/seisai-dataset/examples/phase_dataset_quick_check.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import segyio
from torch.utils.data import DataLoader

from seisai_dataset import (
	BuildPlan,
	FirstBreakGate,
	FirstBreakGateConfig,
	SegyGatherPhasePipelineDataset,
)
from seisai_dataset.builder.builder import IdentitySignal, PhasePSNMap, SelectStack
from seisai_transforms.augment import DeterministicCropOrPad, PerTraceStandardize, ViewCompose


def _write_empty_phase_picks_npz(path: Path, *, n_traces: int) -> None:
	n_traces = int(n_traces)
	if n_traces <= 0:
		raise ValueError(f'n_traces must be > 0, got {n_traces}')
	indptr = np.zeros(n_traces + 1, dtype=np.int64)
	data = np.zeros(0, dtype=np.int64)
	np.savez_compressed(
		path,
		p_indptr=indptr,
		p_data=data,
		s_indptr=indptr.copy(),
		s_data=data.copy(),
	)


def _describe(name: str, v) -> None:
	shape = getattr(v, 'shape', None)
	dtype = getattr(v, 'dtype', None)
	print(f'{name}: type={type(v).__name__}, dtype={dtype}, shape={shape}')


def main() -> None:
	# ====== Edit parameters ======
	segy_files = [
		'/workspace/example_data/merged_F1.sgy',
	]
	phase_pick_files = [
		'/tmp/merged_F1_phase_picks_empty.npz',
	]
	target_len = 2048
	subset_traces = 128
	include_empty_gathers = True
	use_header_cache = False
	seed = 0
	# ============================

	for p in segy_files:
		if not Path(p).exists():
			raise FileNotFoundError(f'SEGY not found: {p}')

	# Create an empty CSR pick file if missing (useful for smoke checks).
	pick_path = Path(phase_pick_files[0])
	if not pick_path.exists():
		with segyio.open(segy_files[0], 'r', ignore_geometry=True) as f:
			n_traces = int(f.tracecount)
		_write_empty_phase_picks_npz(pick_path, n_traces=n_traces)

	transform = ViewCompose([DeterministicCropOrPad(target_len), PerTraceStandardize()])
	fbgate = FirstBreakGate(
		FirstBreakGateConfig(
			apply_on='off',
			min_pick_ratio=0.0,
		)
	)
	plan = BuildPlan(
		wave_ops=[IdentitySignal(src='x_view', dst='x', copy=False)],
		label_ops=[PhasePSNMap(dst='psn_map', sigma=1.5)],
		input_stack=SelectStack(keys='x', dst='input'),
		target_stack=SelectStack(keys='psn_map', dst='target'),
	)

	ds = SegyGatherPhasePipelineDataset(
		segy_files=segy_files,
		phase_pick_files=phase_pick_files,
		transform=transform,
		fbgate=fbgate,
		plan=plan,
		subset_traces=int(subset_traces),
		include_empty_gathers=bool(include_empty_gathers),
		use_header_cache=bool(use_header_cache),
		valid=True,
		verbose=False,
		max_trials=256,
	)
	ds._rng = np.random.default_rng(int(seed))

	try:
		sample = ds[0]
	finally:
		ds.close()

	print('=== Sample keys ===')
	print(sorted(sample.keys()))
	_describe('input', sample['input'])
	_describe('target', sample['target'])
	_describe('trace_valid', sample['trace_valid'])
	_describe('label_valid', sample['label_valid'])
	_describe('fb_idx', sample['fb_idx'])
	_describe('p_idx', sample['p_idx'])
	_describe('s_idx', sample['s_idx'])

	meta = sample['meta']
	print('=== Meta keys ===')
	print(sorted(meta.keys()))
	_describe("meta['fb_idx_view']", meta['fb_idx_view'])
	_describe("meta['p_idx_view']", meta['p_idx_view'])
	_describe("meta['s_idx_view']", meta['s_idx_view'])
	_describe("meta['time_view']", meta['time_view'])

	# Optional: show a single-batch view via default DataLoader collation.
	ds2 = SegyGatherPhasePipelineDataset(
		segy_files=segy_files,
		phase_pick_files=phase_pick_files,
		transform=transform,
		fbgate=fbgate,
		plan=plan,
		subset_traces=int(subset_traces),
		include_empty_gathers=bool(include_empty_gathers),
		use_header_cache=bool(use_header_cache),
		valid=True,
		verbose=False,
		max_trials=256,
	)
	ds2._rng = np.random.default_rng(int(seed))
	try:
		loader = DataLoader(ds2, batch_size=1, num_workers=0)
		batch = next(iter(loader))
	finally:
		ds2.close()

	print('=== Batch keys (batch_size=1) ===')
	print(sorted(batch.keys()))
	_describe('batch[input]', batch['input'])
	_describe('batch[target]', batch['target'])


if __name__ == '__main__':
	main()

