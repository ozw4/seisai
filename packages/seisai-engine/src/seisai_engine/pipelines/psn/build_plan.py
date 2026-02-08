from __future__ import annotations

import numpy as np
from seisai_dataset import BuildPlan
from seisai_dataset.builder.builder import IdentitySignal, PhasePSNMap, SelectStack


def build_plan(*, psn_sigma: float) -> BuildPlan:
    return BuildPlan(
        wave_ops=[
            IdentitySignal(src='x_view', dst='x_wave', copy=False),
        ],
        label_ops=[
            PhasePSNMap(dst='y_psn_map', sigma=float(psn_sigma)),
        ],
        input_stack=SelectStack(
            keys=['x_wave'],
            dst='input',
            dtype=np.float32,
            to_torch=True,
        ),
        target_stack=SelectStack(
            keys=['y_psn_map'],
            dst='target',
            dtype=np.float32,
            to_torch=True,
        ),
    )
