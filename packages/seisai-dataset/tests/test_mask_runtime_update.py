import os
from pathlib import Path

import numpy as np
import pytest
from seisai_dataset import (
    BuildPlan,
    FirstBreakGate,
    FirstBreakGateConfig,
    SegyGatherPipelineDataset,
)
from seisai_dataset.builder.builder import IdentitySignal, MaskedSignal, SelectStack
from seisai_transforms.augment import (
    DeterministicCropOrPad,
    PerTraceStandardize,
    ViewCompose,
)
from seisai_transforms.masking import MaskGenerator

# まずは「あなたの手元で何も設定しなくても動く」デフォルト
_DEFAULT_SEGY = Path(
    '/home/dcuser/data/ActiveSeisField/aso19-2/input_TRCTAB_ml_fbpick_Aso19-2_wolmo.sgy'
)
_DEFAULT_FBNP = Path('/home/dcuser/data/ActiveSeisField/aso19-2/fb_Aso19-2.npy')

# env var は任意(必要な人だけ上書き)
SEGY = Path(os.environ.get('FBP_TEST_SEGY', str(_DEFAULT_SEGY)))
FBNP = Path(os.environ.get('FBP_TEST_FB', str(_DEFAULT_FBNP)))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (SEGY.exists() and FBNP.exists()),
        reason=(
            'Integration test data not found. '
            f'Expected SEG-Y at {SEGY} and FB at {FBNP}. '
            'Optionally set FBP_TEST_SEGY / FBP_TEST_FB to override.'
        ),
    ),
]


def _build_ds(mask_ratio: float):
    transform = ViewCompose([PerTraceStandardize(), DeterministicCropOrPad(2048)])

    fbgate = FirstBreakGate(
        FirstBreakGateConfig(
            apply_on='off',  # FBLC gate 無効(テストが安定)
            min_pick_ratio=0.0,  # min_pick も無効
        )
    )

    gen = MaskGenerator.traces(
        ratio=float(mask_ratio), width=1, mode='replace', noise_std=1.0
    )
    mask_op = MaskedSignal(gen, src='x_view', dst='x_masked', mask_key='mask_bool')

    plan = BuildPlan(
        wave_ops=[
            IdentitySignal(src='x_view', dst='x_orig', copy=True),
            mask_op,
        ],
        label_ops=[],
        input_stack=SelectStack(keys='x_masked', dst='input'),
        target_stack=SelectStack(keys='x_orig', dst='target'),
    )

    ds = SegyGatherPipelineDataset(
        segy_files=[str(SEGY)],
        fb_files=[str(FBNP)],
        transform=transform,
        fbgate=fbgate,
        plan=plan,
        use_superwindow=False,
        secondary_key_fixed=True,
        verbose=False,
    )
    return ds, mask_op


def _masked_traces(mask_bool: np.ndarray) -> int:
    m = np.asarray(mask_bool)
    return int(m[:, 0].sum())


def test_mask_ratio_runtime_update_changes_mask_count():
    ds, mask_op = _build_ds(mask_ratio=0.0)

    a = ds[None]
    H = int(a['input'].shape[1])
    assert _masked_traces(a['mask_bool']) == 0

    mask_op.gen = MaskGenerator.traces(
        ratio=0.5, width=1, mode='replace', noise_std=1.0
    )

    b = ds[None]
    assert _masked_traces(b['mask_bool']) == int(round(0.5 * H))

    ds.close()


def test_mask_mode_and_noise_runtime_update_effect():
    ds, mask_op = _build_ds(mask_ratio=0.5)

    x1 = ds[None]
    H = int(x1['input'].shape[1])
    assert _masked_traces(x1['mask_bool']) == int(round(0.5 * H))

    mask_op.gen.mode = 'add'
    mask_op.gen.noise_std = 0.0

    x2 = ds[None]
    assert _masked_traces(x2['mask_bool']) == int(round(0.5 * H))

    np.testing.assert_allclose(
        x2['input'].cpu().numpy(),
        x2['target'].cpu().numpy(),
        atol=0.0,
        rtol=0.0,
    )

    ds.close()
