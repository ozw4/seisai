#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def find_segy_files(
    root: Path | str,
    *,
    exts: Iterable[str] = ('.sgy', '.segy'),
    recursive: bool = True,
) -> list[Path]:
    """Find SEG-Y files under root.

    Args:
        root: directory path.
        exts: file extensions (case-insensitive). ".sgy", ".segy" etc.
        recursive: if True use rglob, else glob only in root.

    Returns:
        Sorted list of paths.

    """
    r = Path(root)
    if not r.exists() or not r.is_dir():
        msg = f'root must be an existing directory: {r}'
        raise FileNotFoundError(msg)

    exts_l = []
    for e in exts:
        ee = str(e).lower()
        if not ee.startswith('.'):
            ee = '.' + ee
        exts_l.append(ee)
    exts_t = tuple(exts_l)

    if bool(recursive):
        cand = r.rglob('*')
    else:
        cand = r.glob('*')

    out: list[Path] = []
    for p in cand:
        if p.is_file() and p.suffix.lower() in exts_t:
            out.append(p)

    out.sort()
    return out


def require_npz_key(z: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    """Load a required key from npz, raising a clear KeyError if missing."""
    if key not in z.files:
        msg = f'npz missing key={key!r}. available={sorted(z.files)}'
        raise KeyError(msg)
    return np.asarray(z[key])


def read_trace_field(
    src,
    field,
    *,
    dtype,
    name: str = 'trace_field',
) -> np.ndarray:
    """Read a SEG-Y trace header field for all traces.

    This is a thin wrapper around segyio's attributes().

    Args:
        src: segyio.SegyFile
        field: segyio.TraceField.* or int
        dtype: numpy dtype for output
        name: label used in error messages

    Returns:
        (n_traces,) array

    """
    n_tr = int(src.tracecount)
    v = np.asarray(src.attributes(field)[:], dtype=dtype)
    if v.ndim != 1 or v.shape[0] != n_tr:
        msg = f'{name} must be (n_traces,), got {v.shape}, n_traces={n_tr}'
        raise ValueError(msg)
    return v


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


def build_groups_by_key(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
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
