import random

import numpy as np
import torch


class SampleFlow:
	def __init__(self, transform, plan) -> None:
		self.transform = transform
		self.plan = plan

	def draw_sample(
		self,
		info: dict,
		rng: np.random.Generator,
		*,
		sampler,
	) -> dict:
		if sampler is None:
			raise ValueError('sampler must be provided')
		seed = int(rng.integers(0, 2**31 - 1))
		sample = sampler.draw(info, py_random=random.Random(seed))
		return {
			'indices': np.asarray(sample['indices'], dtype=np.int64),
			'key_name': sample['key_name'],
			'secondary_key': sample['secondary_key'],
			'did_super': bool(sample['did_super']),
			'primary_unique': sample['primary_unique'],
		}

	def pad_indices_offsets(
		self,
		indices: np.ndarray,
		offsets: np.ndarray,
		H: int,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
		H0 = int(indices.size)
		if H0 > H:
			raise ValueError(f'indices length {H0} > loaded H {H}')

		trace_valid = np.zeros(H, dtype=np.bool_)
		trace_valid[:H0] = True

		pad = H - H0
		if pad > 0:
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

		return indices, offsets, trace_valid, pad

	def apply_transform(
		self,
		x: np.ndarray,
		rng: np.random.Generator,
		*,
		name: str,
	) -> tuple[np.ndarray, dict]:
		out = self.transform(x, rng=rng, return_meta=True)
		x_view, meta = out if isinstance(out, tuple) else (out, {})
		if not isinstance(x_view, np.ndarray) or x_view.ndim != 2:
			raise ValueError(
				f'transform({name}) は 2D numpy または (2D, meta) を返す必要があります'
			)
		if not isinstance(meta, dict):
			raise ValueError(f'transform({name}) meta must be dict, got {type(meta).__name__}')
		return x_view, meta

	def build_plan_input_base(
		self,
		*,
		meta: dict,
		dt_sec: float,
		offsets: np.ndarray,
		indices: np.ndarray,
		key_name: str,
		secondary_key: str,
		primary_unique: str,
		extra: dict | None = None,
	) -> dict:
		sample_for_plan = {
			'meta': meta,
			'dt_sec': float(dt_sec),
			'offsets': offsets,
			'indices': indices,
			'key_name': key_name,
			'secondary_key': secondary_key,
			'primary_unique': primary_unique,
		}
		if extra:
			sample_for_plan.update(extra)
		return sample_for_plan

	def run_plan(
		self,
		sample_for_plan: dict,
		*,
		rng: np.random.Generator,
		require_target: bool = True,
	) -> dict:
		self.plan.run(sample_for_plan, rng=rng)
		if 'input' not in sample_for_plan:
			raise KeyError("plan must populate 'input'")
		if require_target and 'target' not in sample_for_plan:
			raise KeyError("plan must populate 'target'")
		return sample_for_plan

	def build_output_base(
		self,
		sample_for_plan: dict,
		*,
		meta: dict,
		dt_sec: float,
		offsets: np.ndarray,
		indices: np.ndarray,
		key_name: str,
		secondary_key: str,
		primary_unique: str,
		extra: dict | None = None,
	) -> dict:
		out: dict = {
			'input': sample_for_plan['input'],
			'meta': meta,
			'dt_sec': torch.tensor(float(dt_sec), dtype=torch.float32),
			'offsets': torch.from_numpy(offsets),
			'indices': indices,
			'key_name': key_name,
			'secondary_key': secondary_key,
			'primary_unique': primary_unique,
		}
		if 'target' in sample_for_plan:
			out['target'] = sample_for_plan['target']
		if extra:
			out.update(extra)
		return out
