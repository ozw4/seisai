# %%
import contextlib
import random

import numpy as np
import segyio
import torch
from torch.utils.data import Dataset

from .builder.builder import BuildPlan
from .config import LoaderConfig, TraceSubsetSamplerConfig
from .file_info import build_file_info
from .trace_subset_preproc import TraceSubsetLoader
from .trace_subset_sampler import TraceSubsetSampler


class SegyGatherPairDataset(Dataset):
	"""SEG-Y 対応ペアから同期 transform で input/target を生成する Dataset."""

	def __init__(
		self,
		input_segy_files: list[str],
		target_segy_files: list[str],
		transform,
		plan: BuildPlan,
		*,
		ffid_byte=segyio.TraceField.FieldRecord,
		chno_byte=segyio.TraceField.TraceNumber,
		cmp_byte=segyio.TraceField.CDP,
		primary_keys: tuple[str, ...] | None = None,
		primary_key_weights: tuple[float, ...] | None = None,
		use_superwindow: bool = False,
		sw_halfspan: int = 0,
		sw_prob: float = 0.3,
		use_header_cache: bool = True,
		header_cache_dir: str | None = None,
		subset_traces: int = 128,
		valid: bool = False,
		verbose: bool = False,
		max_trials: int = 2048,
	) -> None:
		if len(input_segy_files) == 0 or len(target_segy_files) == 0:
			raise ValueError('input_segy_files / target_segy_files は空であってはならない')
		if len(input_segy_files) != len(target_segy_files):
			raise ValueError('input_segy_files と target_segy_files の長さが一致していません')

		self.input_segy_files = list(input_segy_files)
		self.target_segy_files = list(target_segy_files)
		self.transform = transform
		self.plan = plan

		self.ffid_byte = ffid_byte
		self.chno_byte = chno_byte
		self.cmp_byte = cmp_byte

		self.primary_keys = tuple(primary_keys) if primary_keys else None
		self.primary_key_weights = (
			tuple(primary_key_weights) if primary_key_weights else None
		)

		self.use_superwindow = bool(use_superwindow)
		self.sw_halfspan = int(sw_halfspan)
		self.sw_prob = float(sw_prob)

		self.use_header_cache = bool(use_header_cache)
		self.header_cache_dir = header_cache_dir

		self.valid = bool(valid)
		self.verbose = bool(verbose)
		self.max_trials = int(max_trials)

		self._rng = np.random.default_rng()

		self.subsetloader = TraceSubsetLoader(
			LoaderConfig(pad_traces_to=int(subset_traces))
		)
		self.sampler = TraceSubsetSampler(
			TraceSubsetSamplerConfig(
				primary_keys=self.primary_keys,
				primary_key_weights=self.primary_key_weights,
				use_superwindow=self.use_superwindow,
				sw_halfspan=self.sw_halfspan,
				sw_prob=self.sw_prob,
				valid=self.valid,
				subset_traces=int(subset_traces),
			)
		)

		self.file_infos: list[dict] = []
		for input_path, target_path in zip(
			self.input_segy_files, self.target_segy_files, strict=True
		):
			input_info = build_file_info(
				input_path,
				ffid_byte=self.ffid_byte,
				chno_byte=self.chno_byte,
				cmp_byte=self.cmp_byte,
				header_cache_dir=self.header_cache_dir,
				use_header_cache=self.use_header_cache,
				include_centroids=True,
			)

			target_obj = segyio.open(target_path, 'r', ignore_geometry=True)
			target_mmap = target_obj.trace.raw[:]
			target_n_samples = (
				int(target_obj.samples.size) if target_obj.samples is not None else 0
			)
			target_n_traces = int(target_obj.tracecount)
			target_dt_us = int(target_obj.bin[segyio.BinField.Interval])
			target_dt_sec = target_dt_us * 1e-6

			if input_info['n_samples'] != target_n_samples:
				input_obj = input_info.get('segy_obj')
				if input_obj is not None:
					input_obj.close()
				target_obj.close()
				raise ValueError(
					'nsamples mismatch: '
					f'{input_path}={input_info["n_samples"]}, '
					f'{target_path}={target_n_samples}'
				)
			if input_info['n_traces'] != target_n_traces:
				input_obj = input_info.get('segy_obj')
				if input_obj is not None:
					input_obj.close()
				target_obj.close()
				raise ValueError(
					'trace count mismatch: '
					f'{input_path}={input_info["n_traces"]}, '
					f'{target_path}={target_n_traces}'
				)
			if not np.isclose(input_info['dt_sec'], target_dt_sec, rtol=0.0, atol=1e-12):
				input_obj = input_info.get('segy_obj')
				if input_obj is not None:
					input_obj.close()
				target_obj.close()
				raise ValueError(
					'dt mismatch: '
					f'{input_path}={input_info["dt_sec"]}, '
					f'{target_path}={target_dt_sec}'
				)

			self.file_infos.append(
				{
					'input_info': input_info,
					'target_path': str(target_path),
					'target_mmap': target_mmap,
					'target_segy_obj': target_obj,
					'target_n_samples': target_n_samples,
					'target_n_traces': target_n_traces,
					'target_dt_sec': float(target_dt_sec),
				}
			)

	def close(self) -> None:
		for info in self.file_infos:
			input_info = info.get('input_info', {})
			segy_obj = input_info.get('segy_obj')
			if segy_obj is not None:
				with contextlib.suppress(Exception):
					segy_obj.close()
			target_obj = info.get('target_segy_obj')
			if target_obj is not None:
				with contextlib.suppress(Exception):
					target_obj.close()
		self.file_infos.clear()

	def __del__(self) -> None:
		self.close()

	def __len__(self) -> int:
		return 1024

	def __getitem__(self, _=None) -> dict:
		for _attempt in range(self.max_trials):
			pair_idx = int(self._rng.integers(0, len(self.file_infos)))
			info = self.file_infos[pair_idx]
			input_info = info['input_info']

			seed = int(self._rng.integers(0, 2**31 - 1))
			sample = self.sampler.draw(input_info, py_random=random.Random(seed))
			indices = np.asarray(sample['indices'], dtype=np.int64)
			if indices.size == 0:
				continue

			x_in = self.subsetloader.load(input_info['mmap'], indices)
			x_tg = self.subsetloader.load(info['target_mmap'], indices)

			H = int(x_in.shape[0])
			H0 = int(indices.size)
			if H0 > H:
				raise ValueError(f'indices length {H0} > loaded H {H}')

			offsets = input_info['offsets'][indices].astype(np.float32, copy=False)
			if H > H0:
				pad = H - H0
				offsets = np.concatenate(
					[offsets, np.zeros(pad, dtype=np.float32)],
					axis=0,
				)
				indices = np.concatenate(
					[indices.astype(np.int64, copy=False), -np.ones(pad, dtype=np.int64)],
					axis=0,
				)
			else:
				indices = indices.astype(np.int64, copy=False)

			seed = int(self._rng.integers(0, 2**31 - 1))
			rng_in = np.random.default_rng(seed)
			rng_tg = np.random.default_rng(seed)
			out_in = self.transform(x_in, rng=rng_in, return_meta=True)
			out_tg = self.transform(x_tg, rng=rng_tg, return_meta=True)

			x_view_input, meta = out_in if isinstance(out_in, tuple) else (out_in, {})
			x_view_target, _meta_tg = (
				out_tg if isinstance(out_tg, tuple) else (out_tg, {})
			)

			if not isinstance(x_view_input, np.ndarray) or x_view_input.ndim != 2:
				raise ValueError(
					'transform(input) は 2D numpy または (2D, meta) を返す必要があります'
				)
			if not isinstance(x_view_target, np.ndarray) or x_view_target.ndim != 2:
				raise ValueError(
					'transform(target) は 2D numpy または (2D, meta) を返す必要があります'
				)
			if x_view_input.shape != x_view_target.shape:
				raise ValueError(
					'input/target transform shape mismatch: '
					f'{x_view_input.shape} vs {x_view_target.shape}'
				)

			dt_sec = float(input_info['dt_sec'])
			sample_for_plan = {
				'x_view_input': x_view_input,
				'x_view_target': x_view_target,
				'meta': meta,
				'dt_sec': dt_sec,
				'offsets': offsets,
				'file_path_input': input_info['path'],
				'file_path_target': info['target_path'],
				'indices': indices,
				'key_name': sample['key_name'],
				'secondary_key': sample['secondary_key'],
				'primary_unique': sample['primary_unique'],
				'did_super': bool(sample['did_super']),
			}

			self.plan.run(sample_for_plan, rng=self._rng)
			if 'input' not in sample_for_plan or 'target' not in sample_for_plan:
				raise KeyError("plan must populate 'input' and 'target'")

			return {
				'input': sample_for_plan['input'],
				'target': sample_for_plan['target'],
				'meta': meta,
				'dt_sec': torch.tensor(dt_sec, dtype=torch.float32),
				'offsets': torch.from_numpy(offsets),
				'file_path_input': input_info['path'],
				'file_path_target': info['target_path'],
				'indices': indices,
				'key_name': sample['key_name'],
				'secondary_key': sample['secondary_key'],
				'primary_unique': sample['primary_unique'],
				'did_superwindow': bool(sample['did_super']) if self.verbose else None,
			}

		raise RuntimeError(
			f'failed to draw a valid sample within max_trials={self.max_trials}; '
			f'files={len(self.file_infos)}'
		)
