"""Dataset build plans and feature/label producers for SeisAI.

This module defines small, composable operations that transform an in-memory
`sample` dict (waveforms, metadata, labels) into model-ready tensors, and plan
executors that run these operations in sequence.

Main components:
- Wave producers: IdentitySignal, MaskedSignal, MakeTimeChannel, MakeOffsetChannel
- Label producers: FBGaussMap
- Stack/selection utilities: SelectStack
- Pipeline executors: BasePlan, BuildPlan, InputOnlyPlan
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from seisai_pick.gaussian_prob import gaussian_probs1d_np, gaussian_pulse1d_np
from seisai_transforms.view_projection import project_pick_csr_view

if TYPE_CHECKING:
	from collections.abc import Iterable

	from numpy.typing import ArrayLike, DTypeLike


def _to_numpy(x: ArrayLike, dtype: DTypeLike | None = None) -> np.ndarray:
	if isinstance(x, torch.Tensor):
		x = x.detach().cpu().numpy()
	return np.asarray(x, dtype=dtype)


# ---------- Wave producers(波形から作る派生物) ----------
class IdentitySignal:
	"""Copy or reference a waveform-like entry from `sample` into another key.

	This operator reads a waveform-like value (typically a NumPy array or Torch tensor)
	from `sample[self.src]` and writes it to `sample[self.dst]`.

	Parameters
	----------
	src : str, default 'x_view'
		Source key to read from `sample` (ignored if `source_key` is provided).
	dst : str, default 'x_id'
		Destination key to write into `sample`.
	copy : bool, default False
		If True, copy/clone the source value; if False, store a reference.
	source_key : str | None, default None
		Optional override for the source key; takes precedence over `src`.

	Notes
	-----
	- When `copy=True`, supported types are `numpy.ndarray` and `torch.Tensor`.
	- The operator mutates `sample` in-place and returns None.

	"""

	def __init__(
		self,
		src: str = 'x_view',
		dst: str = 'x_id',
		*,
		copy: bool = False,
		source_key: str | None = None,
	) -> None:
		"""Initialize an IdentitySignal operator.

		Parameters
		----------
		src : str, default 'x_view'
			Source key to read from `sample` (ignored if `source_key` is provided).
		dst : str, default 'x_id'
			Destination key to write into `sample`.
		copy : bool, default False
			If True, copy/clone the source value; if False, store a reference.
		source_key : str | None, default None
			Optional override for the source key; takes precedence over `src`.

		"""
		self.src = source_key if source_key is not None else src
		self.dst = dst
		self.copy = copy

	def __call__(
		self, sample: dict[str, Any], rng: np.random.Generator | None = None
	) -> None:
		"""Copy or reference a waveform-like entry from `sample` into `sample[self.dst]`.

		Parameters
		----------
		sample : dict[str, Any]
			Sample dictionary containing at least `self.src`.
		rng : Any, optional
			Unused; accepted for pipeline compatibility.

		Raises
		------
		KeyError
			If `self.src` is missing from `sample`.
		TypeError
			If `copy=True` and the source value type is unsupported.

		"""
		if self.src not in sample:
			msg = f'missing sample key: {self.src}'
			raise KeyError(msg)
		x = sample[self.src]
		if not self.copy:
			sample[self.dst] = x
			return
		if isinstance(x, np.ndarray):
			sample[self.dst] = x.copy()
			return
		if isinstance(x, torch.Tensor):
			sample[self.dst] = x.clone()
			return
		msg = f'unsupported type for copy: {type(x).__name__}'
		raise TypeError(msg)


class MaskedSignal:
	"""Apply a generated boolean mask to a waveform and store both result and mask.

	This operator uses a provided MaskGenerator to mask `sample[src]`, writes the
	masked waveform to `sample[dst]`, and stores the boolean mask in
	`sample[mask_key]`.

	Parameters
	----------
	generator
		MaskGenerator instance used to apply masking to the input waveform.
	src : str, default 'x_view'
		Input key in `sample` containing the waveform array/tensor.
	dst : str, default 'x_masked'
		Output key in `sample` where the masked waveform will be stored.
	mask_key : str, default 'mask_bool'
		Key in `sample` where the generated boolean mask will be stored.
	mode : {'replace', 'add'} | None, optional
		Optional mode hint kept for compatibility/validation; the actual masking mode
		is controlled by the generator itself.

	"""

	def __init__(
		self,
		generator,  # MaskGenerator インスタンス
		*,
		src: str = 'x_view',
		dst: str = 'x_masked',
		mask_key: str = 'mask_bool',
		mode: Literal['replace', 'add'] | None = None,  # ← 任意化(整合チェック用)
	) -> None:
		"""Initialize a masking operator that applies a MaskGenerator to a waveform.

		Parameters
		----------
		generator
			MaskGenerator instance used to apply masking to the input.
		src : str, default 'x_view'
			Input key in `sample` containing the waveform array/tensor.
		dst : str, default 'x_masked'
			Output key in `sample` where the masked waveform will be stored.
		mask_key : str, default 'mask_bool'
			Key in `sample` where the generated boolean mask will be stored.
		mode : {'replace', 'add'} | None, optional
			Optional mode hint kept for compatibility/validation; the actual masking mode
			is controlled by the generator itself.

		"""
		self.gen = generator
		self.src = src
		self.dst = dst
		self.mask_key = mask_key
		self.mode = mode

	def __call__(
		self,
		sample: dict[str, Any],
		rng: np.random.Generator | None = None,
	) -> None:
		"""Apply a pixel-wise mask to the input waveform and store results in `sample`.

		Parameters
		----------
		sample : dict[str, Any]
			Sample dictionary containing at least `self.src`.
		rng : Any, optional
			Random generator passed to the underlying mask generator; if None, a new
			NumPy default generator is created.

		Raises
		------
		KeyError
			If `self.src` is missing from `sample`.

		"""
		r = rng or np.random.default_rng()
		x = sample[self.src]
		# MaskGenerator.apply は mode 引数を受けないため渡さない(生成器に保持されている)
		xm, m = self.gen.apply(x, rng=r, mask=None, return_mask=True)
		sample[self.dst] = xm
		sample[self.mask_key] = m


class MakeTimeChannel:
	"""Create a per-trace time channel from metadata.

	This operator reads `sample['meta']['time_view']` (shape: (W,)) and expands it to a
	2D array (H, W) to match `sample['x_view']`, then stores it in `sample[self.dst]`.

	Parameters
	----------
	dst : str, default 'time_ch'
		Destination key to store the generated time channel.

	Notes
	-----
	- Expects `sample['x_view']` with shape (H, W).
	- Expects `sample['meta']['time_view']` with shape (W,).

	"""

	def __init__(self, dst: str = 'time_ch') -> None:
		"""Initialize the time-channel producer.

		Parameters
		----------
		dst : str, default 'time_ch'
			Destination key to store the generated time channel in `sample`.

		"""
		self.dst = dst

	def __call__(
		self, sample: dict[str, Any], rng: np.random.Generator | None = None
	) -> None:
		"""Create and store a per-trace time channel aligned to `x_view`.

		This reads `sample['meta']['time_view']` (W,) and expands it to (H, W) to match
		`sample['x_view']` (H, W), storing the result in `sample[self.dst]`.

		Parameters
		----------
		sample : dict[str, Any]
			Sample dictionary containing `x_view` and `meta['time_view']`.
		rng : np.random.Generator | None, optional
			Unused; accepted for pipeline compatibility.

		Raises
		------
		KeyError
			If `x_view` or `meta['time_view']` is missing from `sample`.

		"""
		H, _ = sample['x_view'].shape
		t = sample['meta']['time_view'].astype(np.float32)
		sample[self.dst] = np.repeat(t[None, :], H, axis=0)


class MakeOffsetChannel:
	"""Create a per-trace offset channel from metadata.

	This operator reads `sample['meta']['offsets_view']` (shape: (H,)) and expands it
	to a 2D array (H, W) to match `sample['x_view']`, then stores it in `sample[self.dst]`.

	Parameters
	----------
	dst : str, default 'offset_ch'
		Destination key to store the generated offset channel.
	normalize : bool, default True
		If True, z-normalize offsets using only valid traces (`trace_valid`) and force
		invalid traces to 0.

	Notes
	-----
	- Expects `sample['x_view']` with shape (H, W).
	- Expects `sample['meta']['offsets_view']` with shape (H,).
	- Expects `sample['meta']['trace_valid']` with shape (H,).

	"""

	def __init__(self, dst: str = 'offset_ch', *, normalize: bool = True) -> None:
		"""Initialize the offset-channel producer.

		Parameters
		----------
		dst : str, default 'offset_ch'
			Destination key to store the generated offset channel in `sample`.
		normalize : bool, default True
			If True, z-normalize offsets using only valid traces (`trace_valid`) and force
			invalid traces to 0.

		"""
		self.dst, self.normalize = dst, normalize

	def __call__(
		self, sample: dict[str, Any], rng: np.random.Generator | None = None
	) -> None:
		"""Create and store a per-trace offset channel aligned to `x_view`.

		This reads `sample['meta']['offsets_view']` (H,) and expands it to (H, W) to
		match `sample['x_view']` (H, W). If `normalize=True`, offsets are z-normalized
		using only valid traces (`sample['meta']['trace_valid']`) and invalid traces
		are forced to 0.

		Parameters
		----------
		sample : dict[str, Any]
			Sample dictionary containing `x_view` and `meta` entries used by this
			operator.
		rng : np.random.Generator | None, optional
			Unused; accepted for pipeline compatibility.

		Raises
		------
		ValueError
			If `trace_valid` length does not match `offsets_view` length.

		"""
		off = sample['meta']['offsets_view'].astype(np.float32)
		trace_valid = sample['meta']['trace_valid']
		if trace_valid.shape[0] != off.shape[0]:
			msg = f'trace_valid length {trace_valid.shape[0]} != offsets_view length {off.shape[0]}'
			raise ValueError(msg)

		if self.normalize:
			valid = trace_valid.astype(np.bool_, copy=False)
			off_z = np.zeros_like(
				off, dtype=np.float32
			)  # invalid traces are forced to 0
			if np.any(valid):
				m = off[valid].mean()
				s = off[valid].std() + 1e-6
				off_z[valid] = (off[valid] - m) / s
			off = off_z

		_, W = sample['x_view'].shape
		sample[self.dst] = np.repeat(off[:, None], W, axis=1)


# ---------- Label producers(ラベルから作る派生物) ----------


class FBGaussMap:
	"""Create a per-trace 2D Gaussian label map from first-break indices.

	This producer converts first-break pick indices (one per trace) into a dense label
	map with shape (H, W), where each valid trace contains a 1D Gaussian distribution
	along the time/sample axis and invalid traces are all zeros.

	Parameters
	----------
	dst : str, default 'fb_map'
		Destination key in `sample` where the output map is stored.
	sigma : float, default 1.5
		Standard deviation of the Gaussian (in bins); must be positive.
	src : str, default 'fb_idx_view'
		Source key inside `sample['meta']` containing first-break indices.

	Outputs
	-------
	sample[dst] : numpy.ndarray
		Float32 array of shape (H, W).

	"""

	def __init__(
		self, dst: str = 'fb_map', sigma: float = 1.5, src: str = 'fb_idx_view'
	) -> None:
		"""Initialize a Gaussian label-map producer from per-trace first-break indices.

		Parameters
		----------
		dst : str, default 'fb_map'
			Destination key in `sample` where the output map is stored.
		sigma : float, default 1.5
			Standard deviation of the Gaussian (in bins); must be positive.
		src : str, default 'fb_idx_view'
			Source key inside `sample['meta']` containing first-break indices.

		"""
		if float(sigma) <= 0.0:
			msg = 'sigma must be positive'
			raise ValueError(msg)
		self.dst = dst
		self.src = src
		self.sigma = float(sigma)

	def __call__(
		self, sample: dict[str, Any], rng: np.random.Generator | None = None
	) -> None:
		"""Create and store a 2D Gaussian label map from per-trace first-break indices.

		This reads `sample['x_view']` to determine the output shape (H, W) and reads
		first-break indices from `sample['meta'][self.src]` (length H). For valid traces
		(where fb_idx > 0), it writes a per-trace 1D Gaussian distribution over W bins;
		invalid traces are all zeros.

		Parameters
		----------
		sample : dict[str, Any]
			Sample dictionary containing `x_view` and `meta[self.src]`.
		rng : np.random.Generator | None, optional
			Unused; accepted for pipeline compatibility.

		Raises
		------
		KeyError
			If `x_view` is missing from `sample` or `self.src` is missing from `sample['meta']`.
		ValueError
			If the length of `sample['meta'][self.src]` does not match the number of traces H.

		Side Effects
		------------
		Writes `sample[self.dst]` as a float32 array of shape (H, W).

		"""
		if 'x_view' not in sample:
			msg = "missing 'x_view'"
			raise KeyError(msg)
		if self.src not in sample['meta']:
			msg = f"missing '{self.src}' (use ProjectToView before FBGaussMap)"
			raise KeyError(msg)

		x_view = _to_numpy(sample['x_view'])
		H, W = x_view.shape

		fb = np.asarray(sample['meta'][self.src], dtype=np.int64)
		if fb.shape[0] != H:
			msg = f'{self.src} length {fb.shape[0]} != H {H}'
			raise ValueError(msg)

		valid = fb > 0
		y = np.zeros((H, W), dtype=np.float32)
		if np.any(valid):
			mu = fb[valid].astype(np.float32)  # ビンindex中心
			g = gaussian_probs1d_np(mu, self.sigma, W)  # (Nv, W)
			y[valid] = g
		sample[self.dst] = y


# ---------- Label producers(phase picks) ----------
class PhasePSNMap:
	"""Create a 3-class probability target map for P/S/Noise from CSR picks.

	This label producer:
	- projects CSR picks into view space using `project_pick_csr_view`
	- creates per-pick peak-normalized Gaussian pulses (peak=1) using `gaussian_pulse1d_np`
	- merges multiple picks per trace with a saturation-friendly rule:
	`p = 1 - Π(1 - g_i)` (applied sequentially)
	- builds Noise as `noise = max(1 - p - s, 0)` and renormalizes to ensure
	per-pixel `P+S+Noise == 1`

	Outputs
	-------
	sample[dst]
		Float32 array of shape (3, H, W) in channel order [P, S, Noise].
	sample[label_valid_dst]
		Bool array of shape (H,), True only when:
		- meta['trace_valid'][t] is True (if present), and
		- the trace has at least one valid P or S pick after view projection.
	"""

	def __init__(
		self,
		*,
		dst: str = 'psn_map',
		sigma: float = 1.5,
		p_indptr: str = 'p_indptr',
		p_data: str = 'p_data',
		s_indptr: str = 's_indptr',
		s_data: str = 's_data',
		label_valid_dst: str = 'label_valid',
	) -> None:
		if float(sigma) <= 0.0:
			msg = 'sigma must be positive'
			raise ValueError(msg)
		self.dst = dst
		self.sigma = float(sigma)
		self.p_indptr = p_indptr
		self.p_data = p_data
		self.s_indptr = s_indptr
		self.s_data = s_data
		self.label_valid_dst = label_valid_dst

	def __call__(
		self, sample: dict[str, Any], rng: np.random.Generator | None = None
	) -> None:
		if 'x_view' not in sample:
			msg = "missing 'x_view'"
			raise KeyError(msg)
		if 'meta' not in sample or not isinstance(sample['meta'], dict):
			msg = "missing 'meta' dict"
			raise KeyError(msg)

		x_view = _to_numpy(sample['x_view'])
		if x_view.ndim not in (2, 3):
			msg = f'x_view must be 2D or 3D, got shape={x_view.shape} (ndim={x_view.ndim})'
			raise ValueError(msg)
		H, W = x_view.shape[-2:]
		meta = sample['meta']

		for k in (self.p_indptr, self.p_data, self.s_indptr, self.s_data):
			if k not in sample:
				msg = f'missing sample key: {k}'
				raise KeyError(msg)

		p_ip = _to_numpy(sample[self.p_indptr])
		p_d = _to_numpy(sample[self.p_data])
		s_ip = _to_numpy(sample[self.s_indptr])
		s_d = _to_numpy(sample[self.s_data])

		p_ip_v, p_d_v = project_pick_csr_view(p_ip, p_d, H=H, W=W, meta=meta)
		s_ip_v, s_d_v = project_pick_csr_view(s_ip, s_d, H=H, W=W, meta=meta)

		p_has = np.diff(p_ip_v) > 0
		s_has = np.diff(s_ip_v) > 0
		has_pick = p_has | s_has

		trace_valid = meta.get('trace_valid', None)
		if trace_valid is None:
			tv = np.ones(H, dtype=np.bool_)
		else:
			tv = np.asarray(trace_valid, dtype=np.bool_)
			if tv.shape != (H,):
				msg = f"meta['trace_valid'] shape {tv.shape} != ({H},)"
				raise ValueError(msg)

		label_valid = tv & has_pick

		# --- build P/S maps ---
		p_map = np.zeros((H, W), dtype=np.float32)
		s_map = np.zeros((H, W), dtype=np.float32)
		one = np.float32(1.0)

		for t in range(H):
			# P
			st = int(p_ip_v[t])
			en = int(p_ip_v[t + 1])
			if st != en:
				mu = p_d_v[st:en]
				g = gaussian_pulse1d_np(mu=mu, sigma_bins=self.sigma, W=W)  # (K,W)
				p_map[t] = one - np.prod(one - g, axis=0, dtype=np.float32)

			# S
			st = int(s_ip_v[t])
			en = int(s_ip_v[t + 1])
			if st != en:
				mu = s_d_v[st:en]
				g = gaussian_pulse1d_np(mu=mu, sigma_bins=self.sigma, W=W)  # (K,W)
				s_map[t] = one - np.prod(one - g, axis=0, dtype=np.float32)

		noise = (one - p_map - s_map).astype(np.float32, copy=False)
		np.maximum(noise, 0.0, out=noise)

		den = (p_map + s_map + noise).astype(np.float32, copy=False)
		zero = den <= 0.0
		if np.any(zero):
			# Degenerate pixels: force Noise=1.
			p_map[zero] = 0.0
			s_map[zero] = 0.0
			noise[zero] = 1.0
			den[zero] = 1.0

		p_map = (p_map / den).astype(np.float32, copy=False)
		s_map = (s_map / den).astype(np.float32, copy=False)
		noise = (noise / den).astype(np.float32, copy=False)

		sample[self.dst] = np.stack([p_map, s_map, noise], axis=0).astype(
			np.float32, copy=False
		)
		sample[self.label_valid_dst] = label_valid


# ---------- 共通セレクタ(入力にもターゲットにも使う) ----------
class SelectStack:
	"""Stack one or more 2D/3D arrays from `sample` into a single (C, H, W) output.

	Parameters
	----------
	keys
		One key or an iterable of keys to read from `sample`.
	dst : str
		Destination key to store the stacked result into `sample`.
	dtype
		NumPy dtype used when converting inputs via `_to_numpy`.
	to_torch : bool, default True
		If True, store output as a `torch.Tensor`; otherwise store a NumPy array.

	Notes
	-----
	- Each input must be either (H, W) or (C, H, W); 2D inputs are promoted to (1, H, W).
	- All inputs must share the same (H, W).
	- The operator mutates `sample` in-place and returns None.

	"""

	def __init__(self, keys, dst: str, dtype=np.float32, to_torch: bool = True) -> None:
		"""Initialize a stacker that concatenates one or more arrays into (C, H, W).

		Parameters
		----------
		keys
			One key or an iterable of keys to read from `sample`.
		dst : str
			Destination key to store the stacked result into `sample`.
		dtype
			NumPy dtype used when converting inputs via `_to_numpy`.
		to_torch : bool, default True
			If True, store output as a `torch.Tensor`; otherwise store a NumPy array.

		"""
		self.keys = list(keys) if isinstance(keys, (list, tuple)) else [keys]
		self.dst = dst
		self.dtype = dtype
		self.to_torch = to_torch

	def __call__(
		self, sample: dict[str, Any], rng: np.random.Generator | None = None
	) -> None:
		"""Stack 2D/3D arrays from `sample` into a single (C, H, W) tensor/array.

		Reads each entry in `self.keys` from `sample`, converts it to NumPy with
		`self.dtype`, ensures it is either (H, W) or (C, H, W), and concatenates
		along the channel axis into a final (C, H, W) output stored at `sample[self.dst]`.

		Parameters
		----------
		sample : dict[str, Any]
			Sample dictionary containing all keys listed in `self.keys`.
		rng : Any, optional
			Unused; accepted for pipeline compatibility.

		Raises
		------
		ValueError
			If an input is not 2D/3D or if shapes are inconsistent across keys.

		"""
		mats = []
		H = W = None
		for k in self.keys:
			v = sample[k]
			a = _to_numpy(v, dtype=self.dtype)
			if a.ndim == 2:
				a = a[None, ...]  # (1,H,W)
			elif a.ndim != 3:
				msg = f'{k}: expected 2D/3D, got shape {a.shape}'
				raise ValueError(msg)
			if H is None:
				_, H, W = a.shape
			if a.shape[1] != H or a.shape[2] != W:
				msg = f'{k}: shape mismatch {a.shape} vs (*,{H},{W})'
				raise ValueError(msg)
			mats.append(a)
		out = np.concatenate(mats, axis=0)  # (C,H,W)
		sample[self.dst] = torch.from_numpy(out) if self.to_torch else out


# ---------- パイプライン実行器 ----------
class BasePlan:
	"""Executable pipeline that mutates a `sample` dict into model-ready inputs.

	A `BasePlan` applies a sequence of waveform-derived operators (`wave_ops`) and
	label-derived operators (`label_ops`) to a `sample` in-place, then uses
	`input_stack` to assemble the final model input tensor/array under
	`sample['input']`.

	Attributes
	----------
	wave_ops : list
		Operators applied to waveform-derived features.
	label_ops : list
		Operators applied to label-derived features/targets.
	input_stack : SelectStack
		Stacking operator used to build `sample['input']`.

	Methods
	-------
	run(sample: dict[str, Any], rng: np.random.Generator | None = None) -> None
		Run all operators in order and build `sample['input']`.

	"""

	def __init__(
		self,
		wave_ops: Iterable,
		label_ops: Iterable,
		input_stack: SelectStack,
	) -> None:
		"""Initialize a dataset build plan with wave/label operators and an input stacker.

		Parameters
		----------
		wave_ops : Iterable
			Operators applied to waveform-derived features (mutate `sample` in-place).
		label_ops : Iterable
			Operators applied to label-derived features/targets (mutate `sample` in-place).
		input_stack : SelectStack
			Stacking operator used to assemble `sample['input']`.

		"""
		self.wave_ops = list(wave_ops)
		self.label_ops = list(label_ops)
		self.input_stack = input_stack

	def run(
		self, sample: dict[str, Any], rng: np.random.Generator | None = None
	) -> None:
		"""Run the pipeline on a single sample and build the model input.

		Applies all operators in `wave_ops` and `label_ops` in order (mutating `sample`
		in-place), then assembles the final model input under `sample['input']` using
		`input_stack`.

		Parameters
		----------
		sample : dict[str, Any]
			Sample dictionary that will be mutated in-place.
		rng : np.random.Generator | None, optional
			Random generator forwarded to operators; if None, operators may create
			their own generators as needed.

		Raises
		------
		Exception
			Any exception raised by an operator in `wave_ops`, `label_ops`, or
			`input_stack` is propagated.

		"""
		for op in self.wave_ops:
			op(sample, rng)
		for op in self.label_ops:
			op(sample, rng)
		self.input_stack(sample, rng)


class BuildPlan(BasePlan):
	"""Pipeline plan that builds both model inputs and training targets.

	This plan:
	- applies `wave_ops` then `label_ops` to mutate `sample` in-place
	- builds `sample['input']` using `input_stack`
	- builds the target tensor(s) using `target_stack` (e.g., `sample['target']`)

	Parameters
	----------
	wave_ops : Iterable
		Operators that derive/transform waveform-based features.
	label_ops : Iterable
		Operators that derive label-based features/maps.
	input_stack : SelectStack
		Stacking operator used to assemble the model input tensor/array.
	target_stack : SelectStack
		Stacking operator used to assemble the training target tensor/array.

	Methods
	-------
	run(sample: dict[str, Any], rng: np.random.Generator | None = None) -> None
		Execute the plan on a single sample.

	"""

	def __init__(
		self,
		wave_ops: Iterable,
		label_ops: Iterable,
		input_stack: SelectStack,
		target_stack: SelectStack,
	) -> None:
		"""Initialize a build plan that constructs both model inputs and training targets.

		Parameters
		----------
		wave_ops : Iterable
			Operators applied to waveform-derived features (mutate `sample` in-place).
		label_ops : Iterable
			Operators applied to label-derived features/targets (mutate `sample` in-place).
		input_stack : SelectStack
			Stacking operator used to assemble `sample['input']`.
		target_stack : SelectStack
			Stacking operator used to assemble the training target tensor/array.

		"""
		super().__init__(wave_ops, label_ops, input_stack)
		self.target_stack = target_stack

	def run(
		self, sample: dict[str, Any], rng: np.random.Generator | None = None
	) -> None:
		"""Run the build plan on a single sample and construct both inputs and targets.

		This runs the base pipeline (wave ops, label ops, and input stacking) and then
		uses `target_stack` to assemble the training target tensor/array (e.g.
		`sample['target']`).

		Parameters
		----------
		sample : dict[str, Any]
			Sample dictionary that will be mutated in-place.
		rng : np.random.Generator | None, optional
			Random generator forwarded to operators; if None, operators may create
			their own generators as needed.

		Raises
		------
		Exception
			Any exception raised by an operator in the pipeline or by `target_stack`
			is propagated.

		"""
		super().run(sample, rng)
		self.target_stack(sample, rng)


class InputOnlyPlan(BasePlan):
	"""Pipeline plan that builds only model inputs (inference-oriented).

	This plan behaves like `BasePlan` but does not construct any training targets.
	It can optionally keep label-derived operators (e.g., for producing auxiliary
	features at inference time) while still omitting any target stacking.

	Parameters
	----------
	wave_ops : Iterable
		Operators applied to waveform-derived features (mutate `sample` in-place).
	label_ops : Iterable
		Operators applied to label-derived features (optional for inference).
	input_stack : SelectStack
		Stacking operator used to assemble `sample['input']`.

	Methods
	-------
	from_build_plan(plan: BuildPlan, *, include_label_ops: bool = False) -> InputOnlyPlan
		Create an inference-only plan from a `BuildPlan`.

	"""

	def __init__(
		self,
		wave_ops: Iterable,
		label_ops: Iterable,
		input_stack: SelectStack,
	) -> None:
		"""Initialize an inference-only plan that builds only model inputs.

		Parameters
		----------
		wave_ops : Iterable
			Operators applied to waveform-derived features (mutate `sample` in-place).
		label_ops : Iterable
			Operators applied to label-derived features (optional for inference).
		input_stack : SelectStack
			Stacking operator used to assemble `sample['input']`.

		"""
		super().__init__(wave_ops, label_ops, input_stack)
		self.target_stack = None

	@classmethod
	def from_build_plan(
		cls,
		plan: BuildPlan,
		*,
		include_label_ops: bool = False,
	) -> InputOnlyPlan:
		"""Create an inference-only plan from a full BuildPlan.

		Parameters
		----------
		plan : BuildPlan
			Source plan to copy operators from.
		include_label_ops : bool, default False
			If True, keep `plan.label_ops` in the resulting plan; if False, drop label
			ops (typical for inference).

		Returns
		-------
		InputOnlyPlan
			A plan that builds only the input tensor (and optionally label-derived
			features) without constructing any targets.

		"""
		label_ops = plan.label_ops if include_label_ops else []
		return cls(plan.wave_ops, label_ops, plan.input_stack)
