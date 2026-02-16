# packages/seisai-transforms/src/seisai_transforms/augment.py
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np
from seisai_utils.validator import validate_array
from torch import Tensor

from .config import FreqAugConfig, SpaceAugConfig, TimeAugConfig
from .kernels import _apply_freq_augment, _spatial_stretch, _time_stretch_poly
from .signal_ops.scaling.standardize import (
    standardize_per_trace_np,
    standardize_per_trace_torch,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


class RandomFreqFilter:
    def __init__(self, cfg: FreqAugConfig = FreqAugConfig()) -> None:
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
    def __init__(self, cfg: TimeAugConfig = TimeAugConfig()) -> None:
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
    def __init__(self, cfg: SpaceAugConfig = SpaceAugConfig()) -> None:
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
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = float(prob)

    def __call__(
        self,
        x_hw: np.ndarray,
        rng: np.random.Generator | None = None,
        return_meta: bool = False,
    ):
        p = float(self.prob)
        if p <= 0.0:
            meta = {'hflip': False}
            return (x_hw, meta) if return_meta else x_hw

        r = rng or np.random.default_rng()
        if r.random() < p:
            y = x_hw[::-1, :].copy()
            meta = {'hflip': True}
            return (y, meta) if return_meta else y

        meta = {'hflip': False}
        return (x_hw, meta) if return_meta else x_hw


class RandomPolarityFlip:
    def __init__(self, prob: float = 0.0) -> None:
        self.prob = float(prob)

    def __call__(
        self,
        x_hw: np.ndarray,
        rng: np.random.Generator | None = None,
        return_meta: bool = False,
    ):
        p = float(self.prob)
        if p <= 0.0:
            meta = {'polarity_flip': False}
            return (x_hw, meta) if return_meta else x_hw

        r = rng or np.random.default_rng()
        if r.random() < p:
            y = -x_hw
            meta = {'polarity_flip': True}
            return (y, meta) if return_meta else y

        meta = {'polarity_flip': False}
        return (x_hw, meta) if return_meta else x_hw


class RandomCropOrPad:
    def __init__(self, target_len: int) -> None:
        self.L = int(target_len)

    def __call__(
        self,
        x_hw: np.ndarray,
        rng: np.random.Generator | None = None,
        return_meta=False,
    ):
        r = rng or np.random.default_rng()
        _H, W = x_hw.shape
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
    """常に決定論."""

    def __init__(self, target_len: int) -> None:
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
        _H, W = x_hw.shape
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
    - Torch:  (W,), (H,W), (C,H,W), (B,C,H,W)(CPU/GPU両対応).
    """

    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = float(eps)

    def __call__(
        self,
        x,
        rng: np.random.Generator | None = None,
        return_meta: bool = False,
    ):
        """x: np.ndarray または torch.Tensor(CPU/GPU)
        rng はインターフェース維持のためのダミー(未使用)。
        return_meta=True の場合は (y, {}) を返す。.
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
            msg = f'x must be numpy.ndarray or torch.Tensor, got {type(x)}'
            raise TypeError(msg)

        if return_meta:
            # ViewCompose 互換のため空 dict を返す
            return y, {}

        return y


ArrayLike: TypeAlias = np.ndarray | Tensor


class RNGLike(Protocol):
    def random(self, *args, **kwargs) -> None: ...
    def uniform(self, *args, **kwargs) -> None: ...
    def integers(self, *args, **kwargs) -> None: ...


class ViewCompose:
    def __init__(self, ops: Iterable) -> None:
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


class RandomSparseTraceTimeShift:
    """Randomly time-shift a small subset of traces with zero padding.

    Intended to simulate sparse site-effect-like local misalignment:
    only a few traces per gather shift by a few samples.

    Meta:
      - meta_key (default: 'trace_tshift_view') stores int16 array (H,)
        containing per-trace shift in view sample units.
    """

    def __init__(
        self,
        *,
        p_apply: float = 0.3,
        p_trace: float = 0.02,
        min_abs_shift: int = 1,
        max_abs_shift: int = 3,
        force_one: bool = True,
        ignore_zero: bool = True,
        fill: float = 0.0,
        meta_key: str = 'trace_tshift_view',
    ):
        if not (0.0 <= float(p_apply) <= 1.0):
            raise ValueError(f'p_apply must be in [0,1], got {p_apply}')
        if not (0.0 <= float(p_trace) <= 1.0):
            raise ValueError(f'p_trace must be in [0,1], got {p_trace}')

        min_s = int(min_abs_shift)
        max_s = int(max_abs_shift)
        if min_s <= 0:
            raise ValueError(f'min_abs_shift must be positive, got {min_abs_shift}')
        if max_s < min_s:
            raise ValueError(
                f'max_abs_shift must be >= min_abs_shift, got {max_abs_shift} < {min_abs_shift}'
            )

        self.p_apply = float(p_apply)
        self.p_trace = float(p_trace)
        self.min_abs_shift = min_s
        self.max_abs_shift = max_s
        self.force_one = bool(force_one)
        self.ignore_zero = bool(ignore_zero)
        self.fill = float(fill)
        self.meta_key = str(meta_key)

    def __call__(self, x_hw: np.ndarray, rng: np.random.Generator, return_meta=False):
        if float(rng.random()) > self.p_apply:
            meta = {self.meta_key: np.zeros((int(x_hw.shape[0]),), dtype=np.int16)}
            return (x_hw, meta) if return_meta else x_hw

        x_hw = np.asarray(x_hw)
        if x_hw.ndim != 2:
            raise ValueError(f'x_hw must be 2D (H,W), got {x_hw.shape}')

        H, W = int(x_hw.shape[0]), int(x_hw.shape[1])

        if self.ignore_zero:
            nz = np.max(np.abs(x_hw), axis=1) > 0.0
            eligible = np.flatnonzero(nz).astype(np.int64, copy=False)
        else:
            eligible = np.arange(H, dtype=np.int64)

        tshift = np.zeros((H,), dtype=np.int16)
        if eligible.size == 0:
            meta = {self.meta_key: tshift}
            return (x_hw, meta) if return_meta else x_hw

        sel = rng.random(eligible.size) < self.p_trace
        if self.force_one and (not np.any(sel)):
            j = int(rng.integers(0, eligible.size))
            sel[j] = True

        idx_shift = eligible[sel]
        if idx_shift.size == 0:
            meta = {self.meta_key: tshift}
            return (x_hw, meta) if return_meta else x_hw

        out = np.array(x_hw, copy=True)

        for i in idx_shift:
            sign = -1 if float(rng.random()) < 0.5 else 1
            mag = int(rng.integers(self.min_abs_shift, self.max_abs_shift + 1))
            s = int(sign * mag)
            tshift[int(i)] = s

            if s > 0:
                out[int(i), :s] = self.fill
                out[int(i), s:] = x_hw[int(i), : W - s]
            else:
                ss = -s
                out[int(i), : W - ss] = x_hw[int(i), ss:]
                out[int(i), W - ss :] = self.fill

        meta = {self.meta_key: tshift}
        return (out, meta) if return_meta else out
