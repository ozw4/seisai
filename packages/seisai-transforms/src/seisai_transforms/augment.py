# packages/seisai-transforms/src/seisai_transforms/augment.py
from __future__ import annotations
import numpy as np
from .kernels import (
    _apply_freq_augment, _time_stretch_poly, _spatial_stretch_sameH
)
from .config import FreqAugConfig, TimeAugConfig, SpaceAugConfig
from .signal_ops import standardize_per_trace

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

    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None, return_meta: bool=False):
        r = rng or np.random.default_rng()
        if self.cfg.prob > 0.0 and r.random() >= self.cfg.prob:
            return (x_hw, {"factor": 1.0}) if return_meta else x_hw
        f = float(r.uniform(*self.cfg.factor_range))
        L = self.cfg.target_len or x_hw.shape[1]
        y = _time_stretch_poly(x_hw, f, L)
        return (y, {"factor": f}) if return_meta else y

class RandomSpatialStretchSameH:
    def __init__(self, cfg: SpaceAugConfig = SpaceAugConfig()):
        self.cfg = cfg

    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None, return_meta: bool=False):
        r = rng or np.random.default_rng()
        if self.cfg.prob > 0.0 and r.random() >= self.cfg.prob:
            return (x_hw, {"did_space": False, "factor_h": 1.0}) if return_meta else x_hw
        f = float(r.uniform(*self.cfg.factor_range))
        y = _spatial_stretch_sameH(x_hw, f)
        return (y, {"did_space": True, "factor_h": f}) if return_meta else y

class RandomHFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = float(prob)
    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None, return_meta=False):
        r = rng or np.random.default_rng()
        if r.random() < self.prob:
            y = x_hw[::-1, :].copy()
            meta = {"hflip": True}
            return (y, meta) if return_meta else y
        meta = {"hflip": False}
        return (x_hw, meta) if return_meta else x_hw

class RandomCropOrPad:
    def __init__(self, target_len: int):
        self.L = int(target_len)
    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None, return_meta=False):
        r = rng or np.random.default_rng()
        H, W = x_hw.shape
        if self.L == W:
            meta = {"start": 0}
            return (x_hw, meta) if return_meta else x_hw
        if self.L < W:
            start = int(r.integers(0, W - self.L + 1))
            y = x_hw[:, start:start+self.L]
            meta = {"start": start}
            return (y, meta) if return_meta else y
        pad = self.L - W
        y = np.pad(x_hw, ((0,0),(0,pad)), mode='constant')
        meta = {"start": 0}
        return (y, meta) if return_meta else y

class PerTraceStandardize:
    def __init__(self, eps: float = 1e-10):
        self.eps = float(eps)
    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        return standardize_per_trace(x_hw, eps=self.eps)

class ViewCompose:
    def __init__(self, ops):
        self.ops = list(ops)
    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None, return_meta: bool=False):
        r = rng or np.random.default_rng()
        meta = {}
        for op in self.ops:
            # 各 op が return_meta をサポートしていれば meta を収集
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

