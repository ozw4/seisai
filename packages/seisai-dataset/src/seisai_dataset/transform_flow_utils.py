"""Shared transform/padding helpers for dataset flow implementations."""

from __future__ import annotations

import numpy as np
from seisai_transforms.view_projection import (
    project_fb_idx_view,
    project_offsets_view,
    project_time_view,
)


def apply_transform_2d_with_meta(
    transform,
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    msg_bad_out: str,
    msg_bad_meta: str,
    exc_bad_out: type[Exception],
    exc_bad_meta: type[Exception],
    allow_non_dict_meta: bool = False,
) -> tuple[np.ndarray, dict]:
    """Apply transform and normalize output to ``(2D ndarray, meta dict)``."""
    out = transform(x, rng=rng, return_meta=True)
    x_view, meta = out if isinstance(out, tuple) else (out, {})

    if not isinstance(x_view, np.ndarray) or x_view.ndim != 2:
        raise exc_bad_out(msg_bad_out)

    if not isinstance(meta, dict):
        if allow_non_dict_meta:
            meta = {}
        else:
            typename = type(meta).__name__
            raise exc_bad_meta(msg_bad_meta.format(type=typename))

    return x_view, meta


def add_view_projection_meta(
    meta: dict,
    *,
    trace_valid: np.ndarray,
    fb_idx: np.ndarray,
    offsets: np.ndarray,
    dt_sec: float,
    W0: int,
    H: int,
    W: int,
) -> dict:
    """Populate view-space projection fields used by build plans."""
    t_raw = np.arange(W0, dtype=np.float32) * float(dt_sec)

    meta['trace_valid'] = trace_valid
    meta['fb_idx_view'] = project_fb_idx_view(fb_idx, int(H), int(W), meta)
    meta['offsets_view'] = project_offsets_view(offsets, int(H), meta)
    meta['time_view'] = project_time_view(t_raw, int(H), int(W), meta)
    return meta


def pad_indices_offsets_fb(
    *,
    indices: np.ndarray,
    offsets: np.ndarray,
    fb_subset: np.ndarray | None,
    H: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, int]:
    """Pad indices/offsets/(optional) fb subset to ``H`` and return validity mask."""
    H0 = int(indices.size)
    if H0 > H:
        msg = f'indices length {H0} > loaded H {H}'
        raise ValueError(msg)
    if fb_subset is not None and int(fb_subset.size) != H0:
        msg = f'fb_subset length {fb_subset.size} != indices length {H0}'
        raise ValueError(msg)

    trace_valid = np.zeros(H, dtype=np.bool_)
    trace_valid[:H0] = True

    indices = indices.astype(np.int64, copy=False)
    offsets = offsets.astype(np.float32, copy=False)
    if fb_subset is not None:
        fb_subset = np.asarray(fb_subset, dtype=np.int64)

    pad = H - H0
    if pad > 0:
        offsets = np.concatenate(
            [offsets, np.zeros(pad, dtype=np.float32)],
            axis=0,
        )
        indices = np.concatenate(
            [indices, -np.ones(pad, dtype=np.int64)],
            axis=0,
        )
        if fb_subset is not None:
            fb_subset = np.concatenate(
                [fb_subset, -np.ones(pad, dtype=np.int64)],
                axis=0,
            )

    return indices, offsets, fb_subset, trace_valid, pad
