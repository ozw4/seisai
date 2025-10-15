# packages/seisai-transforms/src/seisai_transforms/augment.py
from __future__ import annotations
import numpy as np
from .kernels import (
    _apply_freq_augment, _time_stretch_poly, _spatial_stretch_sameH
)
from .config import FreqAugConfig, TimeAugConfig, SpaceAugConfig


class RandomFreqFilter:
    def __init__(self, cfg: FreqAugConfig = FreqAugConfig()):
        self.cfg = cfg

    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        r = rng or np.random.default_rng()
        if self.cfg.prob > 0.0 and r.random() >= self.cfg.prob:
            return x_hw
        # NOTE: kernels 側は内部で random/np.random を使っているので、
        # 完全再現性が必要なら kernels も rng 受け取りに差し替えると良い。
        return _apply_freq_augment(
            x_hw,
            self.cfg.kinds,
            self.cfg.band,
            self.cfg.width,
            self.cfg.roll,
            self.cfg.restandardize,
        )


class RandomTimeStretch:
    def __init__(self, cfg: TimeAugConfig = TimeAugConfig()):
        self.cfg = cfg

    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        r = rng or np.random.default_rng()
        if self.cfg.prob > 0.0 and r.random() >= self.cfg.prob:
            return x_hw
        f = float(r.uniform(*self.cfg.factor_range))
        L = self.cfg.target_len or x_hw.shape[1]
        return _time_stretch_poly(x_hw, f, L)


class RandomSpatialStretchSameH:
    def __init__(self, cfg: SpaceAugConfig = SpaceAugConfig()):
        self.cfg = cfg

    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        r = rng or np.random.default_rng()
        if self.cfg.prob > 0.0 and r.random() >= self.cfg.prob:
            return x_hw
        f = float(r.uniform(*self.cfg.factor_range))
        return _spatial_stretch_sameH(x_hw, f)


class RandomCropOrPad:
    def __init__(self, target_len: int):
        self.L = int(target_len)

    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        H, W = x_hw.shape
        if self.L == W:
            return x_hw
        if self.L < W:
            r = rng or np.random.default_rng()
            start = int(r.integers(0, W - self.L + 1))
            return x_hw[:, start:start + self.L]
        pad = self.L - W
        return np.pad(x_hw, ((0, 0), (0, pad)), mode='constant')


class Compose:
    def __init__(self, ops):
        self.ops = list(ops)

    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        r = rng or np.random.default_rng()
        for op in self.ops:
            x_hw = op(x_hw, r)
        return x_hw

