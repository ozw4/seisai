"""SEG-Y gather dataset pipeline (phase pick CSR variant).

This dataset mirrors `SegyGatherPipelineDataset` but reads phase picks from CSR
`.npz` files and makes them available to label producers (e.g. `PhasePSNMap`)
without breaking the existing FB-only pipeline.

Key points:
- Keeps the existing load -> sample -> transform -> gates -> BuildPlan flow.
- Uses P-first as the legacy `fb_idx` (compatibility).
- Passes subset/padded CSR picks to the plan (not returned in output).
- Returns fixed-size convenience keys: `p_idx`, `s_idx`, `label_valid` (and `fb_idx`).
"""

import contextlib

import numpy as np
import segyio
import torch
from seisai_transforms.view_projection import (
	project_fb_idx_view,
)
from torch.utils.data import Dataset

from .builder.builder import BuildPlan
from .config import LoaderConfig, TraceSubsetSamplerConfig
from .file_info import FileInfo, build_file_info_dataclass
from .gate_fblc import FirstBreakGate
from .phase_pick_io import load_phase_pick_csr_npz, subset_pad_first_invalidate
from .sample_flow import SampleFlow
from .segy_gather_pipeline_dataset import GateEvaluator, SampleTransformer
from .trace_subset_preproc import TraceSubsetLoader
from .trace_subset_sampler import TraceSubsetSampler


class SegyGatherPhasePipelineDataset(Dataset):
	"""SEG-Y gather dataset that consumes phase picks in CSR (.npz) format.

	This dataset is designed to coexist with `SegyGatherPipelineDataset` without
	changing its behavior. It uses the same sampler/transform/gate conventions.

	Additional inputs
	-----------------
	phase_pick_files : list[str]
		CSR `.npz` files aligned 1:1 with `segy_files`.
	include_empty_gathers : bool
		If False, rejects samples where both P and S picks are absent in the sampled
		trace subset (after CSR invalidation). If True, returns such samples and
		skips FB-based quality gates for those empty samples.
	"""

	def __init__(
		self,
		segy_files: list[str],
		phase_pick_files: list[str],
		transform,
		fbgate: FirstBreakGate,
		plan: BuildPlan,
		*,
		include_empty_gathers: bool = False,
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
		if len(segy_files) == 0 or len(phase_pick_files) == 0:
			msg = 'segy_files / phase_pick_files は空であってはならない'
			raise ValueError(msg)
		if len(segy_files) != len(phase_pick_files):
			msg = 'segy_files と phase_pick_files の長さが一致していません'
			raise ValueError(msg)

		self.segy_files = list(segy_files)
		self.phase_pick_files = list(phase_pick_files)
		self.transform = transform
		self.fbgate = fbgate
		self.plan = plan

		self.include_empty_gathers = bool(include_empty_gathers)
		self.subset_traces = int(subset_traces)

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
				LoaderConfig(pad_traces_to=int(self.subset_traces))
			)
		self.sampler = TraceSubsetSampler(
			TraceSubsetSamplerConfig(
				primary_keys=self.primary_keys,
				primary_key_weights=self.primary_key_weights,
				use_superwindow=self.use_superwindow,
				sw_halfspan=self.sw_halfspan,
				sw_prob=self.sw_prob,
				valid=self.valid,
				subset_traces=int(self.subset_traces),
			)
		)

		if sample_transformer is None:
			sample_transformer = SampleTransformer(subsetloader, self.transform)
		if gate_evaluator is None:
			gate_evaluator = GateEvaluator(self.fbgate, verbose=self.verbose)

		self.sample_transformer = sample_transformer
		self.gate_evaluator = gate_evaluator

		# Build per-file metadata and attach CSR arrays.
		self.file_infos: list[FileInfo] = []
		for segy_path, pick_path in zip(
			self.segy_files, self.phase_pick_files, strict=True
		):
			info = build_file_info_dataclass(
				segy_path,
				ffid_byte=self.ffid_byte,
				chno_byte=self.chno_byte,
				cmp_byte=self.cmp_byte,
				header_cache_dir=self.header_cache_dir,
				use_header_cache=self.use_header_cache,
				include_centroids=True,
			)

			picks = load_phase_pick_csr_npz(pick_path)
			if int(picks.n_traces) != int(info.n_traces):
				if info.segy_obj is not None:
					info.segy_obj.close()
				msg = (
					f'phase picks n_traces mismatch: {pick_path}={picks.n_traces}, '
					f'{segy_path}={info.n_traces}'
				)
				raise ValueError(msg)

			info.p_indptr = picks.p_indptr
			info.p_data = picks.p_data
			info.s_indptr = picks.s_indptr
			info.s_data = picks.s_data
			self.file_infos.append(info)

	def close(self) -> None:
		"""Close any open SEG-Y file handles and release cached file metadata."""
		for info in self.file_infos:
			if info.segy_obj is not None:
				with contextlib.suppress(Exception):
					info.segy_obj.close()
		self.file_infos.clear()

	def __del__(self) -> None:
		"""Ensure any open SEG-Y resources are closed when the object is collected."""
		self.close()

	def __len__(self) -> int:
		"""Return the nominal dataset length used for randomized sampling."""
		return 1024

	def __getitem__(self, _: int | None = None) -> dict:
		"""Draw and return a single training/evaluation sample from the dataset.

		This method performs a randomized retry loop (up to ``self.max_trials``) to
		sample a gather subset from a randomly chosen file, apply padding/subsetting,
		optionally reject invalid/undesired samples, run the configured processing
		plan, and finally build the output dictionary consumed by downstream code.

		High-level steps
		----------------
		1. Randomly select a file (``FileInfo``) and draw a sample via
			``self.sample_flow.draw_sample`` (provides trace ``indices`` and metadata).
		2. Subset/pad phase-pick CSR data (P/S first-arrival picks) into the selected
			gather window.
		3. Optionally reject:
			- empty gather subsets (no P/S picks) when ``include_empty_gathers=False``
			- samples failing minimum-pick acceptance
			- samples failing additional "FB legacy" gates (e.g., FBL/C gates)
		4. Load/transform the seismic window via
			``self.sample_transformer.load_transform`` and validate ``x_view`` shape:
			- Accept (H, W) or channels-first (C, H, W)
			- Reject channels-last (H, W, C)
			- Require ``H == self.subset_traces``
		5. Populate view-space pick indices in ``meta`` (P as ``p_idx_view``, S as
			``s_idx_view``), respecting per-trace validity (invalid traces set to -1).
		6. Build plan input, run the plan (``self.sample_flow.run_plan``), and require
			that it produces ``label_valid`` of shape (H,).
		7. Build and return the final output dictionary.

		Parameters
		----------
		_ : int | None, optional
			Ignored. Present to satisfy a Dataset-style indexing signature.

		Returns
		-------
		dict
			A sample dictionary containing at least:
			- ``x_view``: transformed window (2D or 3D channels-first tensor/array)
			- ``fb_idx``/``p_idx``: P-first picks as a torch tensor
			- ``s_idx``: S-first picks as a torch tensor (invalid traces set to -1)
			- ``trace_valid``: per-trace validity mask as a torch tensor (H,)
			- ``label_valid``: per-trace label validity mask as a torch tensor (H,)
			Plus metadata keys added by the plan and the output builder.

		Raises
		------
		RuntimeError
			If required CSR pick arrays are not attached to ``FileInfo`` or if no valid
			sample can be drawn within ``max_trials``.
		ValueError
			If ``x_view`` has an unsupported dimensionality/layout, if the transformed
			height does not match ``subset_traces``, or if pick/label shapes mismatch.
		KeyError
			If the processing plan does not populate ``label_valid``.

		"""
		rej_empty = 0
		rej_pick = 0
		rej_fblc = 0
		for _attempt in range(self.max_trials):
			fidx = int(self._rng.integers(0, len(self.file_infos)))
			info = self.file_infos[fidx]

			if (
				info.p_indptr is None
				or info.p_data is None
				or info.s_indptr is None
				or info.s_data is None
			):
				raise RuntimeError('phase pick CSR is not attached to FileInfo')

			sample = self.sample_flow.draw_sample(
				info,
				self._rng,
				sampler=self.sampler,
			)
			indices = sample['indices']
			did_super = sample['did_super']
			H0 = int(indices.size)
			if H0 == 0:
				continue

			win = subset_pad_first_invalidate(
				p_indptr=info.p_indptr,
				p_data=info.p_data,
				s_indptr=info.s_indptr,
				s_data=info.s_data,
				indices=indices,
				subset_traces=int(self.subset_traces),
			)

			p_first = win.p_first[:H0]
			s_first = win.s_first[:H0]
			is_empty = (not np.any(p_first > 0)) and (not np.any(s_first > 0))

			# Reject empty gather subsets unless explicitly allowed.
			if (not self.include_empty_gathers) and is_empty:
				rej_empty += 1
				continue

			# Legacy gates operate on P-first picks (fb semantics).
			# If empty-gathers are explicitly allowed, skip FB gates for empty samples.
			apply_fb_gates = not (self.include_empty_gathers and is_empty)
			if apply_fb_gates and (not self.gate_evaluator.min_pick_accept(p_first)):
				rej_pick += 1
				continue

			x_view, meta, offsets, fb_subset, indices_pad, trace_valid = (
				self.sample_transformer.load_transform(
					info, indices, p_first, self._rng
				)
			)
			# Accept 2D (H,W) and 3D channels-first (C,H,W). Reject channels-last (H,W,C).
			if not hasattr(x_view, 'shape') or not hasattr(x_view, 'ndim'):
				msg = f'x_view must have shape/ndim, got {type(x_view).__name__}'
				raise ValueError(msg)
			ndim = int(x_view.ndim)
			shape = tuple(int(s) for s in x_view.shape)
			if ndim == 2:
				H, W = shape[0], shape[1]
			elif ndim == 3:
				st = int(self.subset_traces)
				if shape[1] == st:
					# channels-first: (C,H,W)
					H, W = shape[1], shape[2]
				elif shape[0] == st:
					# channels-last: (H,W,C) is not supported
					msg = f'x_view must be channels-first (C,H,W); got channels-last shape={shape}'
					raise ValueError(msg)
				else:
					msg = (
						f'x_view 3D must be channels-first (C,H,W) with H==subset_traces at axis=1; '
						f'got shape={shape}, subset_traces={st}'
					)
					raise ValueError(msg)
			else:
				msg = f'x_view must be 2D or 3D, got shape={shape}'
				raise ValueError(msg)
			if int(self.subset_traces) != H:
				msg = f'loader/transform must keep H=subset_traces: got H={H}, subset_traces={self.subset_traces}'
				raise ValueError(msg)

			sample['indices'] = indices_pad
			sample['trace_valid'] = trace_valid
			meta['trace_valid'] = trace_valid
			meta['key_name'] = sample['key_name']
			meta['primary_unique'] = sample['primary_unique']
			sample['x_view'] = x_view

			# Add phase-first picks in view space.
			meta['p_idx_view'] = meta['fb_idx_view'].copy()

			s_idx = win.s_first.astype(np.int64, copy=True)
			if s_idx.shape != (H,):
				msg = f's_first shape mismatch: {s_idx.shape} != ({H},)'
				raise ValueError(msg)
			s_idx[~trace_valid] = -1
			meta['s_idx_view'] = project_fb_idx_view(s_idx, H, W, meta)

			if apply_fb_gates:
				if not self.gate_evaluator.apply_gates(
					meta, did_super=did_super, info=info
				):
					if self.gate_evaluator.last_reject == 'fblc':
						rej_fblc += 1
					continue
			else:
				# Keep meta contract consistent even when gates are skipped.
				factor = float(meta.get('factor', 1.0))
				meta['dt_eff_sec'] = float(info.dt_sec / max(factor, 1e-9))

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
					# fb compatibility: P-first is exposed as fb_idx
					'fb_idx': fb_subset,
					'file_path': info.path,
					'trace_valid': sample.get('trace_valid'),
					# CSR picks for label producers (do not return in output)
					'p_indptr': win.p_indptr,
					'p_data': win.p_data,
					's_indptr': win.s_indptr,
					's_data': win.s_data,
				},
			)
			self.sample_flow.run_plan(sample_for_plan, rng=self._rng)

			# label_valid must be produced by the plan (e.g., PhasePSNMap).
			if 'label_valid' not in sample_for_plan:
				msg = "plan must populate 'label_valid' for SegyGatherPhasePipelineDataset"
				raise KeyError(msg)
			label_valid = sample_for_plan['label_valid']

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
					'p_idx': torch.from_numpy(fb_subset),
					's_idx': torch.from_numpy(s_idx),
					'file_path': info.path,
					'did_superwindow': sample['did_super'],
				},
			)

			out['trace_valid'] = torch.from_numpy(trace_valid)

			# Optional: mask_bool (used by masking ops)
			mask_bool = sample_for_plan.get('mask_bool')
			if mask_bool is not None:
				out['mask_bool'] = mask_bool

			# label_valid: (H,) bool
			if isinstance(label_valid, torch.Tensor):
				out['label_valid'] = label_valid
			else:
				lv = np.asarray(label_valid, dtype=np.bool_)
				if lv.shape != (H,):
					msg = f'label_valid shape {lv.shape} != ({H},)'
					raise ValueError(msg)
				out['label_valid'] = torch.from_numpy(lv)

			return out

		msg = (
			f'failed to draw a valid sample within max_trials={self.max_trials}; '
			f'rejections: empty={rej_empty}, min_pick={rej_pick}, fblc={rej_fblc}, '
			f'files={len(self.file_infos)}'
		)
		raise RuntimeError(msg)
