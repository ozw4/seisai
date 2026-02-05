# %%
"""Quick check for NoiseTraceSubsetDataset.

Requires:
  - seisai-dataset
  - seisai-transforms (seisai_transforms)
  - torch, numpy, segyio

Run:
  python packages/seisai-dataset/examples/noise_dataset_quick_check.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from seisai_dataset.config import LoaderConfig, TraceSubsetSamplerConfig
from seisai_dataset.noise_dataset import NoiseTraceSubsetDataset
from seisai_dataset.noise_decider import EventDetectConfig

# transforms(任意)
from seisai_transforms.augment import (
	DeterministicCropOrPad,
	PerTraceStandardize,
	ViewCompose,
)


def main() -> None:
	"""Run a quick sanity check for `NoiseTraceSubsetDataset`.

	This script loads a single sample from the configured SEG-Y file(s),
	applies basic transforms, and prints key metadata and tensor shapes.
	"""
	# ====== 編集パラメータ ======
	segy_files = [
		'/workspace/example_data/merged_F1.sgy',
	]
	target_len = 2048
	subset_traces = 128
	use_cache = False
	cache_dir = './.segy_hdr_cache'
	secondary_key_fixed = True
	seed = 0
	# ============================

	for p in segy_files:
		if not Path(p).exists():
			msg = f'SEGY not found: {p}'
			raise FileNotFoundError(msg)

	# transforms(例)
	transform = ViewCompose([DeterministicCropOrPad(target_len), PerTraceStandardize()])

	loader_cfg = LoaderConfig(pad_traces_to=int(subset_traces))
	sampler_cfg = TraceSubsetSamplerConfig(
		primary_keys=('ffid',),
		use_superwindow=False,
		sw_halfspan=0,
		sw_prob=0.0,
		secondary_key_fixed=bool(secondary_key_fixed),
		subset_traces=int(subset_traces),
	)
	detect_cfg = EventDetectConfig()

	ds = NoiseTraceSubsetDataset(
		segy_files=segy_files,
		loader_cfg=loader_cfg,
		sampler_cfg=sampler_cfg,
		detect_cfg=detect_cfg,
		transform=transform,
		header_cache_dir=cache_dir if use_cache else None,
		secondary_key_fixed=bool(secondary_key_fixed),
		verbose=True,
		seed=int(seed),
	)

	sample = ds[0]
	x = sample['x']
	hw = tuple(x.shape[-2:]) if hasattr(x, 'shape') else ('?', '?')
	idxs = np.asarray(sample['indices'])

	print('=== Sample ===')
	print(f'path: {sample["file_path"]}')
	print(f'indices: shape={idxs.shape}, first5={idxs[:5].tolist()}')
	print(f'x: dtype={getattr(x, "dtype", None)}, shape={hw}')
	print(f'dt_sec: {sample["dt_sec"]}')
	print(f'primary_key: {sample["primary_key"]}')
	print(f'primary_unique: {sample["primary_unique"]}')


if __name__ == '__main__':
	main()
