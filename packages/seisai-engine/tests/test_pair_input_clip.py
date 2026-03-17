from __future__ import annotations

import pytest
import torch

from seisai_engine.pipelines.pair.input_clip import (
    maybe_soft_clip_pair_input,
    resolve_pair_input_soft_clip_abs,
)


def test_resolve_pair_input_soft_clip_abs_defaults_to_none() -> None:
    assert resolve_pair_input_soft_clip_abs({}) is None


def test_resolve_pair_input_soft_clip_abs_accepts_null() -> None:
    assert resolve_pair_input_soft_clip_abs({'pair': {'input_soft_clip_abs': None}}) is None


def test_resolve_pair_input_soft_clip_abs_coerces_numeric_to_float() -> None:
    assert resolve_pair_input_soft_clip_abs({'pair': {'input_soft_clip_abs': 3}}) == 3.0


def test_resolve_pair_input_soft_clip_abs_rejects_bool() -> None:
    with pytest.raises(TypeError, match='pair.input_soft_clip_abs'):
        resolve_pair_input_soft_clip_abs({'pair': {'input_soft_clip_abs': True}})


@pytest.mark.parametrize('bad_value', [0, -1, -0.5])
def test_resolve_pair_input_soft_clip_abs_rejects_non_positive(
    bad_value: float,
) -> None:
    with pytest.raises(ValueError, match='pair.input_soft_clip_abs'):
        resolve_pair_input_soft_clip_abs({'pair': {'input_soft_clip_abs': bad_value}})


def test_maybe_soft_clip_pair_input_returns_input_as_is_when_disabled() -> None:
    x = torch.tensor([-10.0, 0.0, 10.0], dtype=torch.float32)

    out = maybe_soft_clip_pair_input(x, clip_abs=None)

    assert out is x


def test_maybe_soft_clip_pair_input_tanh_bounds_output() -> None:
    x = torch.tensor([-100.0, -5.0, 0.0, 5.0, 100.0], dtype=torch.float32)

    out = maybe_soft_clip_pair_input(x, clip_abs=3.0)

    assert float(out.abs().max()) <= 3.0 + 1.0e-6


def test_maybe_soft_clip_pair_input_tanh_preserves_small_amplitudes() -> None:
    x = torch.tensor([-0.3, 0.0, 0.3], dtype=torch.float32)

    out = maybe_soft_clip_pair_input(x, clip_abs=3.0)

    assert torch.allclose(out, x, atol=2.0e-3, rtol=0.0)
