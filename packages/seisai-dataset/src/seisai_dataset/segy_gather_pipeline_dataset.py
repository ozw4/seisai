"""SEG-Y gather dataset pipeline.

This module provides a `torch.utils.data.Dataset` implementation that:
- loads SEG-Y gathers and corresponding first-break (FB) picks,
- samples a subset of traces (optionally with superwindow logic),
- applies a user-provided transform and projects metadata to the view,
- applies first-break quality gates (min-pick and FBLC),
- optionally builds model inputs/targets via a `BuildPlan`.
"""

# %%
import contextlib

import numpy as np
import segyio
import torch
from seisai_transforms.view_projection import (
	project_fb_idx_view,
	project_offsets_view,
	project_time_view,
)
from torch.utils.data import Dataset

from .builder.builder import (
	BuildPlan,
)
from .config import LoaderConfig, TraceSubsetSamplerConfig
from .file_info import FileInfo, build_file_info_dataclass
from .gate_fblc import FirstBreakGate
from .sample_flow import SampleFlow
from .trace_subset_preproc import TraceSubsetLoader
from .trace_subset_sampler import TraceSubsetSampler


class SampleTransformer:
	"""Load a subset of SEG-Y traces and apply the configured transform.

	This helper encapsulates:
	- loading a fixed-size trace subset via `TraceSubsetLoader` (with padding as needed),
	- aligning indices/offsets/first-break picks to the loaded height (H),
	- applying the user transform and projecting metadata to the transformed view.

	Parameters
	----------
	subsetloader : TraceSubsetLoader
		Loader used to read and (optionally) pad the trace subset.
	transform
		Callable applied to the loaded gather. Expected signature:
		`transform(x: np.ndarray, rng: np.random.Generator, return_meta: bool=True)
		-> np.ndarray | tuple[np.ndarray, dict]`.

	Notes
	-----
	`load_transform` returns `(x_view, meta, offsets, fb_subset, indices, trace_valid)`,
	where `meta` is augmented with view-projected fields like `fb_idx_view`,
	`offsets_view`, and `time_view`.

	"""

	def __init__(
		self,
		subsetloader: TraceSubsetLoader,
		transform,
	) -> None:
		"""Initialize the transformer with a subset loader and transform callable."""
		self.subsetloader = subsetloader
		self.transform = transform

	def load_transform(
		self,
		info: FileInfo,
		indices: np.ndarray,
		fb_subset: np.ndarray,
		rng: np.random.Generator,
	) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		mmap = info.mmap

		# 波形読み出し(subset_traces に満たない場合は loader が H 方向にパッドする)
		x = self.subsetloader.load(mmap, indices)  # (H,W0)
		H = int(x.shape[0])
		W0 = int(x.shape[1])

		# ここで offsets/fb/indices を H に合わせる (仕様)
		offsets = info.offsets[indices].astype(np.float32, copy=False)
		indices, offsets, fb_subset, trace_valid, _pad = (
			SampleFlow.pad_indices_offsets_fb(
				indices=indices,
				offsets=offsets,
				fb_subset=fb_subset,
				H=H,
			)
		)

		# 変換 (Crop/Pad / TimeStretch 等)
		out = self.transform(x, rng=rng, return_meta=True)
		x_view, meta = out if isinstance(out, tuple) else (out, {})
		if not isinstance(meta, dict):
			msg = f'transform meta must be dict, got {type(meta).__name__}'
			raise TypeError(msg)
		if not isinstance(x_view, np.ndarray) or x_view.ndim != 2:
			msg = 'transform は 2D numpy または (2D, meta) を返す必要があります'
			raise ValueError(msg)

		Hv, W = x_view.shape
		if Hv != H:
			msg = f'transform must keep H: got Hv={Hv}, expected H={H}'
			raise ValueError(msg)

		t_raw = np.arange(W0, dtype=np.float32) * float(info.dt_sec)

		meta['trace_valid'] = trace_valid
		meta['fb_idx_view'] = project_fb_idx_view(fb_subset, H, W, meta)
		meta['offsets_view'] = project_offsets_view(offsets, H, meta)
		meta['time_view'] = project_time_view(t_raw, H, W, meta)

		return x_view, meta, offsets, fb_subset, indices, trace_valid


class GateEvaluator:
	"""Evaluate first-break quality gates for a gather.

	Attributes
	----------
	fbgate : FirstBreakGate
		Gate implementation providing min-pick and FBLC checks.
	verbose : bool
		Whether to print rejection details.
	last_reject : str | None
		Most recent rejection reason ("min_pick" or "fblc").

	"""

	def __init__(self, fbgate: FirstBreakGate, *, verbose: bool = False) -> None:
		"""Initialize the gate evaluator with a gate implementation and verbosity."""
		self.fbgate = fbgate
		self.verbose = bool(verbose)
		self.last_reject: str | None = None

	def min_pick_accept(self, fb_subset: np.ndarray) -> bool:
		"""Check whether the gather passes the minimum pick-count quality gate.

		Parameters
		----------
		fb_subset : np.ndarray
			First-break pick indices for the currently sampled trace subset.

		Returns
		-------
		bool
			True if the minimum pick criterion is satisfied; otherwise False, with
			`last_reject` set to "min_pick".

		"""
		ok_pick, _, _ = self.fbgate.min_pick_accept(fb_subset)
		if not ok_pick:
			self.last_reject = 'min_pick'
			return False
		self.last_reject = None
		return True

	def apply_gates(self, meta: dict, *, did_super: bool, info: FileInfo) -> bool:
		"""Apply post-transform first-break quality gates to a gather.

		This currently evaluates the FBLC (first-break linearity/consistency) gate
		using first-break indices projected into the transformed view.

		Parameters
		----------
		meta : dict
			Metadata dictionary produced/augmented during transformation; must
			include `fb_idx_view` and may include `factor` for effective dt scaling.
		did_super : bool
			Whether the sample was drawn using the superwindow logic.
		info : FileInfo
			File-level metadata (e.g., `dt_sec`, `path`) for the current gather.

		Returns
		-------
		bool
			True if the gather passes all gates; otherwise False, with `last_reject`
			set to the rejection reason (currently "fblc").

		"""
		# FBLC gate (After transform)
		factor = float(meta.get('factor', 1.0))
		dt_eff_sec = info.dt_sec / max(factor, 1e-9)
		meta['dt_eff_sec'] = float(dt_eff_sec)
		ok_fblc, p_ms, valid_pairs = self.fbgate.fblc_accept(
			meta['fb_idx_view'], dt_eff_sec=dt_eff_sec, did_super=did_super
		)
		if not ok_fblc:
			if self.verbose:
				print(
					f'Rejecting gather {info.path} key={meta.get("key_name", "")}:{meta.get("primary_unique", "")} '
					f'(FBLC gate; pairs={valid_pairs}, p_ms={p_ms})'
				)
			self.last_reject = 'fblc'
			return False

		self.last_reject = None
		return True


class SegyGatherPipelineDataset(Dataset):
	"""SEG-Y ギャザー読み込み → サンプリング → 変換) → FBLC ゲート → (任意) BuildPlanで入出力生成.

	期待する transform:  x(H,W) -> x_view  もしくは  (x_view, meta)
	- meta は少なくとも { 'hflip':bool, 'factor':float, 'start':int, 'did_space':bool, 'factor_h':float } の任意サブセット
	期待する fbgate: FirstBreakGate (min_pick_accept と fblc_accept を持つ)
	期待する plan:  BuildPlan (任意)。与えれば sample に 'input' / 'target' などを組み立てる
	"""

	def __init__(
		self,
		segy_files: list[str],
		fb_files: list[str],
		transform,
		fbgate: FirstBreakGate,
		plan: BuildPlan,
		*,
		ffid_byte: int = segyio.TraceField.FieldRecord,
		chno_byte: int = segyio.TraceField.TraceNumber,
		cmp_byte: int = segyio.TraceField.CDP,
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
		sample_transformer: SampleTransformer | None = None,
		gate_evaluator: GateEvaluator | None = None,
	) -> None:
		if len(segy_files) == 0 or len(fb_files) == 0:
			msg = 'segy_files / fb_files は空であってはならない'
			raise ValueError(msg)
		if len(segy_files) != len(fb_files):
			msg = 'segy_files と fb_files の長さが一致していません'
			raise ValueError(msg)

		self.segy_files = list(segy_files)
		self.fb_files = list(fb_files)
		self.transform = transform
		self.fbgate = fbgate
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

		self._rng = np.random.default_rng()
		self.max_trials = int(max_trials)
		self.sample_flow = SampleFlow(transform, plan)

		# components
		if sample_transformer is None:
			subsetloader = TraceSubsetLoader(
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

		if sample_transformer is None:
			sample_transformer = SampleTransformer(subsetloader, self.transform)
		if gate_evaluator is None:
			gate_evaluator = GateEvaluator(self.fbgate, verbose=self.verbose)

		self.sample_transformer = sample_transformer
		self.gate_evaluator = gate_evaluator

		# ファイルごとのインデックス辞書等を構築
		self.file_infos: list[FileInfo] = []
		for segy_path, fb_path in zip(self.segy_files, self.fb_files, strict=True):
			info = build_file_info_dataclass(
				segy_path,
				ffid_byte=self.ffid_byte,
				chno_byte=self.chno_byte,
				cmp_byte=self.cmp_byte,
				header_cache_dir=self.header_cache_dir,
				use_header_cache=self.use_header_cache,
				include_centroids=True,  # or False
			)
			fb = np.load(fb_path)
			info.fb = fb
			self.file_infos.append(info)

	def close(self) -> None:
		"""Close any open SEG-Y file handles and release cached file metadata.

		This method is safe to call multiple times and suppresses exceptions raised
		while closing individual SEG-Y objects.
		"""
		for info in self.file_infos:
			if info.segy_obj is not None:
				with contextlib.suppress(Exception):
					info.segy_obj.close()
		self.file_infos.clear()

	def __del__(self) -> None:
		"""Finalize the dataset by closing any open SEG-Y resources."""
		self.close()

	def __len__(self) -> int:
		"""Return the nominal dataset length for iteration purposes."""
		return 1024

	def __getitem__(self, _: int | None = None) -> dict:
		"""Return a single sampled and processed gather from the dataset."""
		rej_pick = 0
		rej_fblc = 0
		for _attempt in range(self.max_trials):
			fidx = int(self._rng.integers(0, len(self.file_infos)))
			info = self.file_infos[fidx]
			sample = self.sample_flow.draw_sample(
				info,
				self._rng,
				sampler=self.sampler,
			)
			indices = sample['indices']
			did_super = sample['did_super']

			fb_subset = info.fb[indices]
			if not self.gate_evaluator.min_pick_accept(fb_subset):
				rej_pick += 1
				continue

			x_view, meta, offsets, fb_subset, indices_pad, trace_valid = (
				self.sample_transformer.load_transform(
					info, indices, fb_subset, self._rng
				)
			)
			sample['indices'] = indices_pad
			sample['trace_valid'] = trace_valid
			meta['key_name'] = sample['key_name']
			meta['primary_unique'] = sample['primary_unique']
			sample['x_view'] = x_view

			if not self.gate_evaluator.apply_gates(
				meta, did_super=did_super, info=info
			):
				if self.gate_evaluator.last_reject == 'fblc':
					rej_fblc += 1
				continue

			dt_eff_sec = float(meta.get('dt_eff_sec', info.dt_sec))
			sample_for_plan = self.sample_flow.build_plan_input_base(
				meta=meta,
				dt_sec=dt_eff_sec,
				offsets=offsets,
				indices=sample['indices'],
				key_name=sample['key_name'],
				secondary_key=sample['secondary_key'],
				primary_unique=sample['primary_unique'],
				extra={
					'x_view': sample['x_view'],
					'fb_idx': fb_subset,
					'file_path': info.path,
					'trace_valid': sample.get('trace_valid'),
				},
			)
			self.sample_flow.run_plan(sample_for_plan, rng=self._rng)

			out = self.sample_flow.build_output_base(
				sample_for_plan,
				meta=meta,
				dt_sec=dt_eff_sec,
				offsets=offsets,
				indices=sample['indices'],
				key_name=sample['key_name'],
				secondary_key=sample['secondary_key'],
				primary_unique=sample['primary_unique'],
				extra={
					'fb_idx': torch.from_numpy(fb_subset),
					'file_path': info.path,
					'did_superwindow': sample['did_super'],
				},
			)
			trace_valid = sample.get('trace_valid')
			if trace_valid is not None:
				out['trace_valid'] = torch.from_numpy(trace_valid)
			mask_bool = sample_for_plan.get('mask_bool')
			if mask_bool is not None:
				out['mask_bool'] = mask_bool

			return out

		msg = (
			f'failed to draw a valid sample within max_trials={self.max_trials}; '
			f'rejections: min_pick={rej_pick}, fblc={rej_fblc}, '
			f'files={len(self.file_infos)}'
		)
		raise RuntimeError(msg)
