from __future__ import annotations

import numpy as np
from seisai_dataset import BuildPlan
from seisai_dataset.builder.builder import IdentitySignal, SelectStack


def build_plan() -> BuildPlan:
    return BuildPlan(
        wave_ops=[
            IdentitySignal(source_key='x_view_input', dst='x_wave', copy=False),
            IdentitySignal(source_key='x_view_target', dst='y_wave', copy=False),
        ],
        label_ops=[],
        input_stack=SelectStack(
            keys=['x_wave'],
            dst='input',
            dtype=np.float32,
            to_torch=True,
        ),
        target_stack=SelectStack(
            keys=['y_wave'],
            dst='target',
            dtype=np.float32,
            to_torch=True,
        ),
    )
