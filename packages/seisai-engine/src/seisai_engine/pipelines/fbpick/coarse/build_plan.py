from __future__ import annotations

import numpy as np
from seisai_dataset import BuildPlan
from seisai_dataset.builder import (
    FBGaussMapMs,
    IdentitySignal,
    MakeOffsetChannel,
    MakeTimeChannel,
    NormalizeOffsetByConst,
    NormalizeTimeByConst,
    SelectStack,
)

__all__ = ['build_plan']


def build_plan(
    *,
    sigma_ms: float,
    time_ref_sec: float,
    offset_ref_m: float,
) -> BuildPlan:
    return BuildPlan(
        wave_ops=[
            IdentitySignal(src='x_view', dst='x_wave', copy=False),
            MakeOffsetChannel(dst='offset_ch_raw', normalize=False),
            NormalizeOffsetByConst(
                src='offset_ch_raw',
                dst='offset_ch',
                ref_m=float(offset_ref_m),
                use_abs=True,
                clip_lo=0.0,
                clip_hi=1.5,
            ),
            MakeTimeChannel(dst='time_ch_raw'),
            NormalizeTimeByConst(
                src='time_ch_raw',
                dst='time_ch',
                ref_sec=float(time_ref_sec),
                clip_lo=0.0,
                clip_hi=1.5,
            ),
        ],
        label_ops=[
            FBGaussMapMs(
                dst='y_fb_map',
                src='fb_idx_view',
                sigma_ms=float(sigma_ms),
                sigma_samples_min=4.0,
                sigma_samples_max=40.0,
            )
        ],
        input_stack=SelectStack(
            keys=['x_wave', 'offset_ch', 'time_ch'],
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
