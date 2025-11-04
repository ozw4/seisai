# %%
# packages/seisai-dataset/src/seisai_dataset/noise_dataset.py
from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import segyio
import torch

# transforms（任意）
from seisai_transforms.augment import (
	DeterministicCropOrPad,
	PerTraceStandardize,
	ViewCompose,
)
from torch.utils.data import Dataset

# seisai-dataset
from seisai_dataset.config import LoaderConfig, TraceSubsetSamplerConfig
from seisai_dataset.file_info import build_file_info

# 共通の判定ロジック（ここに一本化）
from seisai_dataset.noise_decider import EventDetectConfig, decide_noise
from seisai_dataset.trace_subset_preproc import TraceSubsetLoader
from seisai_dataset.trace_subset_sampler import TraceSubsetSampler


class NoiseTraceSubsetDataset(Dataset):
	"""SEG-Y から TraceSubset を抽出 → （任意）Transform → decide_noise() でイベント除外し、
	ノイズのみ返す Dataset。
	"""

	def __init__(
		self,
		segy_files: Iterable[str],
		loader_cfg: LoaderConfig,
		sampler_cfg: TraceSubsetSamplerConfig,
		detect_cfg: EventDetectConfig = EventDetectConfig(),
		transform: object
		| None = None,  # 例: ViewCompose([DeterministicCropOrPad(...), ...])
		*,
		ffid_byte=segyio.TraceField.FieldRecord,
		chno_byte=segyio.TraceField.TraceNumber,
		cmp_byte=segyio.TraceField.CDP,
		header_cache_dir: str | None = None,
		max_retries: int = 1000,
		valid: bool = False,
		verbose: bool = False,
		seed: int | None = None,
	) -> None:
		super().__init__()
		self.segy_files = [str(p) for p in segy_files]
		if len(self.segy_files) == 0:
			raise ValueError('segy_files is empty')

		self.loader_cfg = loader_cfg
		self.sampler_cfg = sampler_cfg
		self.detect_cfg = detect_cfg

		self.ffid_byte = ffid_byte
		self.chno_byte = chno_byte
		self.cmp_byte = cmp_byte

		self.header_cache_dir = header_cache_dir
		self.max_retries = int(max_retries)
		self.valid = bool(valid)
		self.verbose = bool(verbose)
		self.transform = transform
		self._rng = np.random.default_rng(
			seed if seed is not None else np.random.SeedSequence().entropy
		)

		# components
		self.sampler = TraceSubsetSampler(self.sampler_cfg)
		self.loader = TraceSubsetLoader(self.loader_cfg)

		# file infos（supergatherは使わない）
		self.file_infos: list[dict] = [
			build_file_info(
				p,
				ffid_byte=self.ffid_byte,
				chno_byte=self.chno_byte,
				cmp_byte=self.cmp_byte,
				header_cache_dir=self.header_cache_dir,
				include_centroids=False,
			)
			for p in self.segy_files
		]

	def __len__(self) -> int:
		return 10**6

	def __getitem__(self, _=None) -> dict:
		# tries: 1..self.max_retries まで試行。成功で return、全滅なら例外。
		for _tries in range(1, self.max_retries + 1):
			# --- 1) ファイル選択 ---
			fidx = int(self._rng.integers(0, len(self.file_infos)))
			info = self.file_infos[fidx]
			mmap = info['mmap']
			dt_sec = float(info['dt_sec'])

			# --- 2) TraceSubset 抽出 ---
			d = self.sampler.draw(
				info,
				py_random=random.Random(int(self._rng.integers(0, 2**31 - 1))),
			)
			key_name: str = d['key_name']
			indices: np.ndarray = d['indices']
			primary_unique: str = d.get('primary_unique', '')

			# --- 3) ロード (H,T) ---
			x = self.loader.load(mmap, indices).astype(np.float32, copy=False)

			# --- 4) Transform（任意・最終長合わせ/標準化など）---
			meta = {}
			if self.transform is not None:
				out = self.transform(x)
				if isinstance(out, tuple) and len(out) == 2:
					x, meta = out
				else:
					x = out
				if not isinstance(x, np.ndarray) or x.ndim != 2:
					raise ValueError(
						'transform must return 2D numpy array or (array, meta)'
					)

			# --- 5) 共通判定ロジック（noise_decider）---
			dec = decide_noise(x, dt_sec, self.detect_cfg)
			if not dec.is_noise:
				if self.verbose:
					print(f'Rejected by {dec.reason}')
				continue  # イベント含み→やり直し

			# --- 6) ノイズサンプルを返す ---
			start_t = int(meta.get('start', 0))
			return {
				'x': torch.from_numpy(x.copy()),
				'file_path': info['path'],
				'indices': indices,
				'dt_sec': dt_sec,
				'primary_key': key_name,
				'primary_unique': primary_unique,
				'start_t': start_t,
				'is_noise': True,
			}

		# ここまで到達＝ max_retries 回すべて「イベント含み」で棄却された
		raise RuntimeError('Failed to sample noise-only TraceSubset within max_retries')

	def close(self) -> None:
		for info in self.file_infos:
			segy_obj = info.get('segy_obj')
			if segy_obj is not None:
				segy_obj.close()
		self.file_infos.clear()

	def __del__(self) -> None:
		self.close()


# -------------------- quick check --------------------
if __name__ == '__main__':
	from pathlib import Path

	# ====== 編集パラメータ ======
	segy_files = [
		'/workspace/example_data/merged_F1.sgy',
	]
	target_len = 2048
	subset_traces = 128
	use_cache = False
	cache_dir = './.segy_hdr_cache'
	valid = True
	seed = 0
	# ============================

	for p in segy_files:
		if not Path(p).exists():
			raise FileNotFoundError(f'SEGY not found: {p}')

	# transforms（例）
	transform = ViewCompose([DeterministicCropOrPad(target_len), PerTraceStandardize()])

	loader_cfg = LoaderConfig(pad_traces_to=int(subset_traces))
	sampler_cfg = TraceSubsetSamplerConfig(
		primary_keys=('ffid',),
		use_superwindow=False,
		sw_halfspan=0,
		sw_prob=0.0,
		valid=bool(valid),
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
		valid=bool(valid),
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
	print(f'start_t: {sample["start_t"]}')
	print(f'is_noise: {sample["is_noise"]}')
