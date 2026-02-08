# packages/seisai-transforms/src/seisai_transforms/augment.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeAlias

import numpy as np
from seisai_utils.validator import validate_array
from torch import Tensor

from .config import FreqAugConfig, SpaceAugConfig, TimeAugConfig
from .kernels import _apply_freq_augment, _spatial_stretch, _time_stretch_poly
from .signal_ops.scaling.standardize import (
    standardize_per_trace_np,
    standardize_per_trace_torch,
)


class RandomFreqFilter:
    def __init__(self, cfg: FreqAugConfig = FreqAugConfig()):
        self.cfg = cfg

    def __call__(
        self,
        x_hw: np.ndarray,
        rng: np.random.Generator | None = None,
        return_meta: bool = False,
    ):
        p = float(self.cfg.prob or 0.0)
        # ← ここが大事：prob=0 は無条件で No-Op！
        if p <= 0.0:
            return (x_hw, {}) if return_meta else x_hw

        r = rng.random() if rng is not None else np.random.random()
        if r >= p:
            return (x_hw, {}) if return_meta else x_hw

        y = _apply_freq_augment(
            x_hw,
            self.cfg.kinds,
            self.cfg.band,
            self.cfg.width,
            self.cfg.roll,
            self.cfg.restandardize,
            rng=rng,  # ★ rng を下に渡す(再現性◯)
        )
        return (y, {}) if return_meta else y


class RandomTimeStretch:
    def __init__(self, cfg: TimeAugConfig = TimeAugConfig()):
        self.cfg = cfg

    def __call__(
        self,
        x_hw: np.ndarray,
        rng: np.random.Generator | None = None,
        return_meta: bool = False,
    ):
        p = max(0.0, min(1.0, float(self.cfg.prob)))
        if p <= 0.0:
            meta = {'factor': 1.0}
            return (x_hw, meta) if return_meta else x_hw

        r = rng or np.random.default_rng()
        if r.random() >= p:
            meta = {'factor': 1.0}
            return (x_hw, meta) if return_meta else x_hw

        f = float(r.uniform(*self.cfg.factor_range))
        y = _time_stretch_poly(x_hw, f)
        meta = {'factor': f}
        return (y, meta) if return_meta else y


class RandomSpatialStretchSameH:
    def __init__(self, cfg: SpaceAugConfig = SpaceAugConfig()):
        self.cfg = cfg

    def __call__(
        self,
        x_hw: np.ndarray,
        rng: np.random.Generator | None = None,
        return_meta: bool = False,
    ):
        p = float(self.cfg.prob or 0.0)
        r = rng or np.random.default_rng()
        if p <= 0.0 or r.random() >= p:
            return (x_hw, {'factor_h': 1.0}) if return_meta else x_hw

        # 適用
        f = float(r.uniform(*self.cfg.factor_range))
        y = _spatial_stretch(x_hw, f)
        return (y, {'factor_h': f}) if return_meta else y


class RandomHFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = float(prob)

    def __call__(
        self,
        x_hw: np.ndarray,
        rng: np.random.Generator | None = None,
        return_meta=False,
    ):
        r = rng or np.random.default_rng()
        if r.random() < self.prob:
            y = x_hw[::-1, :].copy()
            meta = {'hflip': True}
            return (y, meta) if return_meta else y
        meta = {'hflip': False}
        return (x_hw, meta) if return_meta else x_hw


class RandomCropOrPad:
    def __init__(self, target_len: int):
        self.L = int(target_len)

    def __call__(
        self,
        x_hw: np.ndarray,
        rng: np.random.Generator | None = None,
        return_meta=False,
    ):
        r = rng or np.random.default_rng()
        H, W = x_hw.shape
        if self.L == W:
            meta = {'start': 0}
            return (x_hw, meta) if return_meta else x_hw
        if self.L < W:
            start = int(r.integers(0, W - self.L + 1))
            y = x_hw[:, start : start + self.L]
            meta = {'start': start}
            return (y, meta) if return_meta else y
        pad = self.L - W
        y = np.pad(x_hw, ((0, 0), (0, pad)), mode='constant')
        meta = {'start': 0}
        return (y, meta) if return_meta else y


class DeterministicCropOrPad:
    """常に決定論"""

    def __init__(self, target_len: int):
        if target_len <= 0:
            msg = 'target_len must be positive'
            raise ValueError(msg)
        self.L = int(target_len)

    def __call__(
        self,
        x_hw: np.ndarray,
        rng: np.random.Generator | None = None,
        return_meta=False,
    ):
        H, W = x_hw.shape
        if self.L == W:
            meta = {'start': 0}
            return (x_hw, meta) if return_meta else x_hw
        if self.L < W:
            start = (W - self.L) // 2
            y = x_hw[:, start : start + self.L]
            meta = {'start': int(start)}
            return (y, meta) if return_meta else y
        # W < L -> symmetric pad
        total = self.L - W
        y = np.pad(x_hw, ((0, 0), (0, total)), mode='constant')
        meta = {'start': 0}
        return (y, meta) if return_meta else y


class PerTraceStandardize:
    """トレース方向(最後の軸=W)で平均0・分散1に標準化する。
    - NumPy:  (W,), (H,W), (C,H,W), (B,C,H,W)
    - Torch:  (W,), (H,W), (C,H,W), (B,C,H,W)(CPU/GPU両対応)
    """

    def __init__(self, eps: float = 1e-10):
        self.eps = float(eps)

    def __call__(
        self,
        x,
        rng: np.random.Generator | None = None,
        return_meta: bool = False,
    ):
        """x: np.ndarray または torch.Tensor(CPU/GPU)
        rng はインターフェース維持のためのダミー(未使用)。
        return_meta=True の場合は (y, {}) を返す。
        """
        # backend='auto' で NumPy / Torch 両方を検証
        validate_array(
            x,
            allowed_ndims=(1, 2, 3, 4),
            name='x',
            backend='auto',
        )

        if isinstance(x, np.ndarray):
            y = standardize_per_trace_np(x, eps=self.eps)
        elif isinstance(x, Tensor):
            y = standardize_per_trace_torch(x, eps=self.eps)
        else:
            raise TypeError(f'x must be numpy.ndarray or torch.Tensor, got {type(x)}')

        if return_meta:
            # ViewCompose 互換のため空 dict を返す
            return y, {}

        return y


ArrayLike: TypeAlias = np.ndarray | Tensor


class RNGLike(Protocol):
    def random(self, *args, **kwargs): ...
    def uniform(self, *args, **kwargs): ...
    def integers(self, *args, **kwargs): ...


class ViewCompose:
    def __init__(self, ops: Iterable):
        self.ops = list(ops)

    def __call__(
        self,
        x_hw: ArrayLike,
        rng: RNGLike | None = None,
        return_meta: bool = False,
    ) -> ArrayLike | tuple[ArrayLike, dict]:
        r = rng or np.random.default_rng()
        meta: dict = {}
        for op in self.ops:
            try:
                y = op(x_hw, r, return_meta=True)
            except TypeError:
                y = op(x_hw, r)
            if isinstance(y, tuple):
                x_hw, mu = y
                if isinstance(mu, dict) and mu:
                    meta.update(mu)
            else:
                x_hw = y
        return (x_hw, meta) if return_meta else x_hw
