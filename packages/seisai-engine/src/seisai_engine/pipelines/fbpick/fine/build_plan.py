from __future__ import annotations

import numpy as np
from seisai_dataset import BuildPlan
from seisai_dataset.builder import FBGaussMapMs, IdentitySignal, SelectStack

__all__ = ['build_plan']


def build_plan(
    *,
    sigma_ms: float,
    sigma_samples_min: float,
    sigma_samples_max: float,
) -> BuildPlan:
    return BuildPlan(
        wave_ops=[
            IdentitySignal(src='x_view', dst='x_wave', copy=False),
        ],
        label_ops=[
            FBGaussMapMs(
                dst='y_fb_map',
                src='fb_idx_view',
                sigma_ms=float(sigma_ms),
                sigma_samples_min=float(sigma_samples_min),
                sigma_samples_max=float(sigma_samples_max),
            )
        ],
        input_stack=SelectStack(
            keys=['x_wave'],
            dst='input',
            dtype=np.float32,
            to_torch=True,
        ),
        target_stack=SelectStack(
            keys=['y_fb_map'],
            dst='target',
            dtype=np.float32,
            to_torch=True,
        ),
    )
