# packages/seisai-transforms/src/seisai_transforms/augment.py
from __future__ import annotations
import numpy as np
from typing import Sequence
from .kernels import (  # ← ここに今の _xxx をそのまま移すイメージ
    _apply_freq_augment, _time_stretch_poly, _spatial_stretch_sameH, _fit_time_len_np
)

class RandomFreqFilter:
    def __init__(self,
        kinds: Sequence[str] = ("bandpass","lowpass","highpass"),
        band: tuple[float,float] = (0.01, 0.99),
        width: tuple[float,float] = (0.05, 0.35),
        roll: float = 0.02,
        restandardize: bool = True,
    ):
        self.kinds, self.band, self.width = tuple(kinds), band, width
        self.roll, self.restandardize = roll, restandardize

    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        # rngは今の実装に渡すなら、kernels側をrng対応に薄く改修してもOK
        return _apply_freq_augment(x_hw, self.kinds, self.band, self.width, self.roll, self.restandardize)

class RandomTimeStretch:
    def __init__(self, factor_range=(0.9, 1.1), target_len: int | None = None):
        self.fr, self.target_len = factor_range, target_len

    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        r = rng or np.random.default_rng()
        f = float(r.uniform(self.fr[0], self.fr[1]))
        tl = self.target_len or x_hw.shape[1]
        return _time_stretch_poly(x_hw, f, tl)

class RandomSpatialStretchSameH:
    def __init__(self, factor_range=(0.9, 1.1)):
        self.fr = factor_range

    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        r = rng or np.random.default_rng()
        f = float(r.uniform(self.fr[0], self.fr[1]))
        return _spatial_stretch_sameH(x_hw, f)

class RandomCropOrPad:
    def __init__(self, target_len: int):
        self.L = int(target_len)

    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        # kernels側を rng 受け取りにしてもOK。とりあえず現状の _fit_time_len_np を使用。
        return _fit_time_len_np(x_hw, self.L)

class Compose:
    def __init__(self, ops): self.ops = ops
    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        r = rng or np.random.default_rng()
        for op in self.ops: x_hw = op(x_hw, r)
        return x_hw
