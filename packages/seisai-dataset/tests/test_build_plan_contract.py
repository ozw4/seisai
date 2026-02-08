import numpy as np
import pytest
import torch
from seisai_dataset import BuildPlan
from seisai_dataset.builder.builder import IdentitySignal, MaskedSignal, SelectStack
from seisai_transforms.masking import MaskGenerator


def _make_sample(H: int = 8, W: int = 32) -> dict:
    rng = np.random.default_rng(0)
    x_view = rng.normal(size=(H, W)).astype(np.float32)

    meta = {
        'time_view': (np.arange(W, dtype=np.float32) * 0.004).astype(np.float32),
        'offsets_view': rng.uniform(-1000.0, 1000.0, size=H).astype(np.float32),
        'fb_idx_view': rng.integers(1, W, size=H, dtype=np.int64),  # valid (>0)
        'dt_eff_sec': 0.004,
        'trace_valid': np.ones(H, dtype=np.bool_),
    }

    return {
        'x_view': x_view,
        'meta': meta,
    }


def _make_plan(mask_ratio: float = 0.25) -> BuildPlan:
    gen = MaskGenerator.traces(
        ratio=float(mask_ratio),
        width=1,
        mode='replace',
        noise_std=1.0,
    )
    mask_op = MaskedSignal(gen, src='x_view', dst='x_masked', mask_key='mask_bool')

    return BuildPlan(
        wave_ops=[
            IdentitySignal(src='x_view', dst='x_orig', copy=True),
            mask_op,
        ],
        label_ops=[],
        input_stack=SelectStack(keys='x_masked', dst='input'),
        target_stack=SelectStack(keys='x_orig', dst='target'),
    )


def test_build_plan_produces_required_keys_and_shapes() -> None:
    sample = _make_sample(H=10, W=64)
    plan = _make_plan(mask_ratio=0.25)

    plan.run(sample, rng=np.random.default_rng(1))

    assert 'input' in sample
    assert 'target' in sample

    x = sample['input']
    y = sample['target']

    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float32
    assert x.ndim == 3

    assert isinstance(y, torch.Tensor)
    assert y.dtype == torch.float32
    assert y.ndim == 3

    C_in, H, W = x.shape
    C_tgt, H2, W2 = y.shape
    assert C_in >= 1
    assert C_tgt >= 1
    assert H2 == H
    assert W2 == W

    # MaskedSignal contract
    assert 'mask_bool' in sample
    m = sample['mask_bool']
    assert isinstance(m, np.ndarray)
    assert m.dtype == np.bool_
    assert m.shape == (H, W)


def test_select_stack_expands_2d_to_3d() -> None:
    sample = _make_sample(H=6, W=20)

    plan = BuildPlan(
        wave_ops=[IdentitySignal(src='x_view', dst='x_id', copy=True)],
        label_ops=[],
        input_stack=SelectStack(keys='x_id', dst='input'),
        target_stack=SelectStack(keys='x_id', dst='target'),
    )
    plan.run(sample, rng=np.random.default_rng(2))

    x = sample['input']
    y = sample['target']

    assert x.shape == (1, 6, 20)
    assert y.shape == (1, 6, 20)


def test_missing_required_key_raises_keyerror() -> None:
    sample = _make_sample(H=4, W=12)

    plan = BuildPlan(
        wave_ops=[IdentitySignal(src='does_not_exist', dst='x', copy=True)],
        label_ops=[],
        input_stack=SelectStack(keys='x', dst='input'),
        target_stack=SelectStack(keys='x', dst='target'),
    )

    with pytest.raises(KeyError):
        plan.run(sample, rng=np.random.default_rng(3))


def test_select_stack_shape_mismatch_raises_valueerror() -> None:
    rng = np.random.default_rng(0)
    H, W = 5, 16
    sample = {
        'x_a': rng.normal(size=(H, W)).astype(np.float32),
        'x_b': rng.normal(size=(H, W + 1)).astype(np.float32),
        'meta': {
            'time_view': (np.arange(W, dtype=np.float32) * 0.004).astype(np.float32),
            'offsets_view': rng.uniform(-1000.0, 1000.0, size=H).astype(np.float32),
            'fb_idx_view': rng.integers(1, W, size=H, dtype=np.int64),
            'dt_eff_sec': 0.004,
            'trace_valid': np.ones(H, dtype=np.bool_),
        },
    }

    plan = BuildPlan(
        wave_ops=[],
        label_ops=[],
        input_stack=SelectStack(keys=('x_a', 'x_b'), dst='input'),
        target_stack=SelectStack(keys='x_a', dst='target'),
    )

    with pytest.raises(ValueError):
        plan.run(sample, rng=np.random.default_rng(4))
