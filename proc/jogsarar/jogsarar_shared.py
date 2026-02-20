#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def valid_pick_mask(
    picks: np.ndarray,
    *,
    n_samples: int | None = None,
    zero_is_invalid: bool = True,
) -> np.ndarray:
    """Return a boolean mask for valid picks.

    Rules:
    - finite
    - >0 if zero_is_invalid else >=0
    - < n_samples if provided
    """
    pk = np.asarray(picks)
    m = np.isfinite(pk)
    if bool(zero_is_invalid):
        m &= pk > 0
    else:
        m &= pk >= 0
    if n_samples is not None:
        ns = int(n_samples)
        if ns <= 0:
            msg = f'n_samples must be positive, got {n_samples}'
            raise ValueError(msg)
        m &= pk < ns
    return m


def build_pick_aligned_window(
    wave_hw: np.ndarray,
    picks: np.ndarray,
    pre: int,
    post: int,
    fill: float = 0.0,
) -> np.ndarray:
    """Extract per-trace windows aligned on pick indices.

    Args:
      wave_hw: (H, W)
      picks: (H,) pick indices. Non-finite or <=0 picks are treated as invalid.
      pre/post: number of samples before/after the pick (post is exclusive).
      fill: fill value for out-of-range samples.
    """
    wave = np.asarray(wave_hw, dtype=np.float32)
    pk = np.asarray(picks)

    if wave.ndim != 2:
        msg = f'wave_hw must be 2D (H,W), got {wave.shape}'
        raise ValueError(msg)
    if pk.ndim != 1 or pk.shape[0] != wave.shape[0]:
        msg = f'picks must be 1D length H={wave.shape[0]}, got {pk.shape}'
        raise ValueError(msg)
    if int(pre) < 0 or int(post) <= 0:
        msg = f'pre must be >=0 and post must be >0, got pre={pre}, post={post}'
        raise ValueError(msg)

    h, w = wave.shape
    length = int(pre + post)
    out = np.full((h, length), np.float32(fill), dtype=np.float32)

    for i in range(h):
        p = float(pk[i])
        if (not np.isfinite(p)) or p <= 0.0:
            continue

        c = int(np.rint(p))
        src_l = c - int(pre)
        src_r = c + int(post)

        ov_l = max(0, src_l)
        ov_r = min(w, src_r)
        if ov_l >= ov_r:
            continue

        dst_l = ov_l - src_l
        dst_r = dst_l + (ov_r - ov_l)
        out[i, dst_l:dst_r] = wave[i, ov_l:ov_r]

    return out


def build_groups_by_key(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Group trace indices by a 1D integer key array.

    Returns:
      uniq: unique keys sorted ascending
      inv:  per-element group id (0..len(uniq)-1)
      groups: list of index arrays (each group is sorted in original order)
    """
    v = np.asarray(values)
    if v.ndim != 1:
        msg = f'values must be 1D, got {v.shape}'
        raise ValueError(msg)

    uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind='mergesort')
    splits = np.cumsum(counts)[:-1]
    groups = [np.asarray(g, dtype=np.int64) for g in np.split(order, splits)]
    return uniq.astype(v.dtype, copy=False), inv.astype(np.int64, copy=False), groups


def build_key_to_indices(values: np.ndarray) -> dict[int, np.ndarray]:
    """Return dict[key] -> indices for each unique key in values."""
    uniq, _, groups = build_groups_by_key(np.asarray(values))
    return {
        int(k): np.asarray(g, dtype=np.int64)
        for k, g in zip(uniq.tolist(), groups, strict=False)
    }


@dataclass(frozen=True)
class TilePerTraceStandardize:
    """Tile transform: per-trace standardization (same signature as seisai tile_transform)."""

    eps_std: float = 1e-8

    @torch.no_grad()
    def __call__(self, patch: torch.Tensor, *, return_meta: bool = False):
        from seisai_transforms.signal_ops.scaling.standardize import (
            standardize_per_trace_torch,
        )

        out = standardize_per_trace_torch(patch, eps=float(self.eps_std))
        return (out, {}) if return_meta else out
