"""Local-window builder ops for fine first-break datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
from seisai_pick.gaussian_prob import gaussian_probs1d_np

from .builder import _to_numpy

__all__ = ['FBLocalGaussMap']


class FBLocalGaussMap:
    """Create a local-window Gaussian label map from local pick indices.

    This producer mirrors :class:`seisai_dataset.builder.builder.FBGaussMap`, but
    its source is the local-window pick index stored directly on ``sample``. The
    output contract is explicit:

    - input ``sample['x_view']`` must be a 2D array with shape ``(H, W_local)``
    - input ``sample[src]`` must be a 1D array with shape ``(H,)`` and dtype
      coercible to ``int64``
    - output ``sample[dst]`` is a ``float32`` array with shape ``(H, W_local)``

    Invalid labels must be encoded as ``-1``. Local indices are on the local
    window axis, so ``0`` is a valid pick location.
    """

    def __init__(
        self,
        dst: str = 'fb_local_map',
        sigma: float = 1.5,
        src: str = 'local_pick_idx',
        valid_key: str | None = 'label_valid',
    ) -> None:
        if float(sigma) <= 0.0:
            msg = 'sigma must be positive'
            raise ValueError(msg)
        self.dst = dst
        self.sigma = float(sigma)
        self.src = src
        self.valid_key = valid_key

    def __call__(
        self,
        sample: dict[str, Any],
        rng: np.random.Generator | None = None,
    ) -> None:
        del rng
        if 'x_view' not in sample:
            msg = "missing 'x_view'"
            raise KeyError(msg)
        if self.src not in sample:
            msg = f"missing '{self.src}'"
            raise KeyError(msg)

        x_view = _to_numpy(sample['x_view'])
        if x_view.ndim != 2:
            msg = f"x_view must be 2D with shape (H,W), got {tuple(x_view.shape)}"
            raise ValueError(msg)
        h, w_local = x_view.shape
        if h <= 0 or w_local <= 0:
            msg = f'x_view must have positive shape, got {tuple(x_view.shape)}'
            raise ValueError(msg)

        local_pick_idx = np.asarray(sample[self.src], dtype=np.int64)
        if local_pick_idx.ndim == 0:
            local_pick_idx = local_pick_idx.reshape(1)
        if local_pick_idx.shape != (h,):
            msg = f'{self.src} shape {local_pick_idx.shape} != ({h},)'
            raise ValueError(msg)
        if np.any(local_pick_idx < -1):
            msg = f'{self.src} must contain only -1 or valid local indices'
            raise ValueError(msg)

        if self.valid_key is None:
            label_valid = local_pick_idx >= 0
        else:
            if self.valid_key not in sample:
                msg = f"missing '{self.valid_key}'"
                raise KeyError(msg)
            label_valid = np.asarray(sample[self.valid_key], dtype=np.bool_)
            if label_valid.ndim == 0:
                label_valid = label_valid.reshape(1)
            if label_valid.shape != (h,):
                msg = f'{self.valid_key} shape {label_valid.shape} != ({h},)'
                raise ValueError(msg)
            if np.any(label_valid & (local_pick_idx < 0)):
                msg = f'{self.src} must be >= 0 when {self.valid_key} is true'
                raise ValueError(msg)
            if np.any((~label_valid) & (local_pick_idx != -1)):
                msg = f'{self.src} must be -1 when {self.valid_key} is false'
                raise ValueError(msg)

        if np.any(local_pick_idx[label_valid] >= int(w_local)):
            bad = int(local_pick_idx[label_valid].max(initial=-1))
            msg = (
                f'{self.src} contains out-of-range local index {bad}; '
                f'expected < W_local={w_local}'
            )
            raise ValueError(msg)

        y = np.zeros((h, w_local), dtype=np.float32)
        if np.any(label_valid):
            mu = local_pick_idx[label_valid].astype(np.float32, copy=False)
            y[label_valid] = gaussian_probs1d_np(mu, self.sigma, w_local).astype(
                np.float32,
                copy=False,
            )
        sample[self.dst] = y
