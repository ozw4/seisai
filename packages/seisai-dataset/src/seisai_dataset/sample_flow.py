"""Sampling and preprocessing flow utilities for SeisAI datasets.

This module provides the `SampleFlow` class, which:
- draws samples from a provided sampler with a deterministic seed;
- pads indices/offsets (optionally with an fb_subset) to a fixed length;
- applies a transform that returns a 2D NumPy array (and optional metadata);
- runs a plan that populates `input` (and optionally `target`) for training.
"""

import random

import numpy as np
import torch

from .file_info import FileInfo
from .transform_flow_utils import (
    apply_transform_2d_with_meta,
    pad_indices_offsets_fb as _pad_indices_offsets_fb,
)


class SampleFlow:
    """Sampling and preprocessing orchestrator for SeisAI datasets.

    This class ties together a sampler, a transform, and a plan:
    - `draw_sample` draws a deterministic sample using an RNG-derived seed.
    - `pad_indices_offsets` / `pad_indices_offsets_fb` pad indices/offsets (and optional
      `fb_subset`) to a fixed length and return a validity mask.
    - `apply_transform` applies the configured transform and validates its output.
    - `run_plan` executes the configured plan to populate `input` (and optionally `target`).

    Parameters
    ----------
    transform
            Callable that accepts `(x, rng=..., return_meta=True)` and returns either a 2D numpy
            array or `(2D numpy array, meta dict)`.
    plan
            Object providing `run(sample_for_plan, rng=...)` that populates `sample_for_plan`.

    """

    def __init__(self, transform, plan) -> None:
        self.transform = transform
        self.plan = plan

    def draw_sample(
        self,
        info: dict | FileInfo,
        rng: np.random.Generator,
        *,
        sampler,
    ) -> dict:
        if sampler is None:
            msg = 'sampler must be provided'
            raise ValueError(msg)
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
        indices, offsets, _fb_subset, trace_valid, pad = self.pad_indices_offsets_fb(
            indices=indices,
            offsets=offsets,
            fb_subset=None,
            H=H,
        )
        return indices, offsets, trace_valid, pad

    @staticmethod
    def pad_indices_offsets_fb(
        *,
        indices: np.ndarray,
        offsets: np.ndarray,
        fb_subset: np.ndarray | None,
        H: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, int]:
        return _pad_indices_offsets_fb(
            indices=indices,
            offsets=offsets,
            fb_subset=fb_subset,
            H=H,
        )

    def apply_transform(
        self,
        x: np.ndarray,
        rng: np.random.Generator,
        *,
        name: str,
    ) -> tuple[np.ndarray, dict]:
        return self.apply_transform_with(self.transform, x, rng, name=name)

    def apply_transform_with(
        self,
        transform,
        x: np.ndarray,
        rng: np.random.Generator,
        *,
        name: str,
    ) -> tuple[np.ndarray, dict]:
        return apply_transform_2d_with_meta(
            transform,
            x,
            rng,
            msg_bad_out=f'transform({name}) は 2D numpy または (2D, meta) を返す必要があります',
            msg_bad_meta=f'transform({name}) meta must be dict, got {{type}}',
            exc_bad_out=ValueError,
            exc_bad_meta=ValueError,
        )

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
            msg = "plan must populate 'input'"
            raise KeyError(msg)
        if require_target and 'target' not in sample_for_plan:
            msg = "plan must populate 'target'"
            raise KeyError(msg)
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
