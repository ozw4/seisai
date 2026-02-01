# %%
import contextlib
import numpy as np
import segyio
from torch.utils.data import Dataset

from .builder.builder import BuildPlan
from .config import LoaderConfig, TraceSubsetSamplerConfig
from .file_info import PairFileInfo, build_file_info_dataclass
from .sample_flow import SampleFlow
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
		primary_keys: tuple[str, ...] | None = ('ffid',),
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
			raise ValueError(
				'input_segy_files / target_segy_files は空であってはならない'
			)
		if len(input_segy_files) != len(target_segy_files):
			raise ValueError(
				'input_segy_files と target_segy_files の長さが一致していません'
			)

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
		self.sample_flow = SampleFlow(transform, plan)

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

		self.file_infos: list[PairFileInfo] = []
		for input_path, target_path in zip(
			self.input_segy_files, self.target_segy_files, strict=True
		):
			input_info = build_file_info_dataclass(
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

			if input_info.n_samples != target_n_samples:
				if input_info.segy_obj is not None:
					input_info.segy_obj.close()
				target_obj.close()
				raise ValueError(
					'nsamples mismatch: '
					f'{input_path}={input_info.n_samples}, '
					f'{target_path}={target_n_samples}'
				)
			if input_info.n_traces != target_n_traces:
				if input_info.segy_obj is not None:
					input_info.segy_obj.close()
				target_obj.close()
				raise ValueError(
					'trace count mismatch: '
					f'{input_path}={input_info.n_traces}, '
					f'{target_path}={target_n_traces}'
				)
			if not np.isclose(input_info.dt_sec, target_dt_sec, rtol=0.0, atol=1e-12):
				if input_info.segy_obj is not None:
					input_info.segy_obj.close()
				target_obj.close()
				raise ValueError(
					'dt mismatch: '
					f'{input_path}={input_info.dt_sec}, '
					f'{target_path}={target_dt_sec}'
				)

			self.file_infos.append(
				PairFileInfo(
					input_info=input_info,
					target_path=str(target_path),
					target_mmap=target_mmap,
					target_segy_obj=target_obj,
					target_n_samples=target_n_samples,
					target_n_traces=target_n_traces,
					target_dt_sec=float(target_dt_sec),
				)
			)

	def close(self) -> None:
		for info in self.file_infos:
			if info.input_info.segy_obj is not None:
				with contextlib.suppress(Exception):
					info.input_info.segy_obj.close()
			if info.target_segy_obj is not None:
				with contextlib.suppress(Exception):
					info.target_segy_obj.close()
		self.file_infos.clear()

	def __del__(self) -> None:
		self.close()

	def __len__(self) -> int:
		return 1024

	def __getitem__(self, _=None) -> dict:
		for _attempt in range(self.max_trials):
			pair_idx = int(self._rng.integers(0, len(self.file_infos)))
			info = self.file_infos[pair_idx]
			input_info = info.input_info

			sample = self.sample_flow.draw_sample(
				input_info,
				self._rng,
				sampler=self.sampler,
			)
			indices = sample['indices']
			if indices.size == 0:
				continue

			x_in = self.subsetloader.load(input_info.mmap, indices)
			x_tg = self.subsetloader.load(info.target_mmap, indices)

			H = int(x_in.shape[0])
			offsets = input_info.offsets[indices].astype(np.float32, copy=False)
			indices, offsets, _fb_subset, _trace_valid, _pad = (
				self.sample_flow.pad_indices_offsets_fb(
					indices=indices,
					offsets=offsets,
					fb_subset=None,
					H=H,
				)
			)

			seed = int(self._rng.integers(0, 2**31 - 1))
			rng_in = np.random.default_rng(seed)
			rng_tg = np.random.default_rng(seed)
			x_view_input, meta = self.sample_flow.apply_transform(
				x_in,
				rng_in,
				name='input',
			)
			x_view_target, _meta_tg = self.sample_flow.apply_transform(
				x_tg,
				rng_tg,
				name='target',
			)
			if x_view_input.shape != x_view_target.shape:
				raise ValueError(
					'input/target transform shape mismatch: '
					f'{x_view_input.shape} vs {x_view_target.shape}'
				)
			did_superwindow = bool(sample['did_super'])

			dt_sec = float(input_info.dt_sec)
			sample_for_plan = self.sample_flow.build_plan_input_base(
				meta=meta,
				dt_sec=dt_sec,
				offsets=offsets,
				indices=indices,
				key_name=sample['key_name'],
				secondary_key=sample['secondary_key'],
				primary_unique=sample['primary_unique'],
				extra={
					'x_view_input': x_view_input,
					'x_view_target': x_view_target,
					'file_path_input': input_info.path,
					'file_path_target': info.target_path,
					'did_superwindow': did_superwindow,
				},
			)

			self.sample_flow.run_plan(sample_for_plan, rng=self._rng)

			return self.sample_flow.build_output_base(
				sample_for_plan,
				meta=meta,
				dt_sec=dt_sec,
				offsets=offsets,
				indices=indices,
				key_name=sample['key_name'],
				secondary_key=sample['secondary_key'],
				primary_unique=sample['primary_unique'],
				extra={
					'file_path_input': input_info.path,
					'file_path_target': info.target_path,
					'did_superwindow': did_superwindow,
				},
			)

		raise RuntimeError(
			f'failed to draw a valid sample within max_trials={self.max_trials}; '
			f'files={len(self.file_infos)}'
		)
