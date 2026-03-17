from __future__ import annotations

import pytest
import torch

from seisai_engine.pipelines.pair.residual import (
    reconstruct_pair_prediction,
    resolve_pair_residual_learning,
)


def test_resolve_pair_residual_learning_defaults_false() -> None:
    assert resolve_pair_residual_learning({}) is False


def test_reconstruct_pair_prediction_direct_mode_returns_raw() -> None:
    pred_raw = torch.randn(2, 1, 3, 4)
    x_in = torch.randn(2, 1, 3, 4)

    pred = reconstruct_pair_prediction(
        pred_raw,
        x_in,
        residual_learning=False,
    )

    assert pred is pred_raw


def test_reconstruct_pair_prediction_residual_mode_adds_input() -> None:
    pred_raw = torch.full((1, 1, 2, 3), 2.0, dtype=torch.float32)
    x_in = torch.full((1, 1, 2, 3), 5.0, dtype=torch.float32)

    pred = reconstruct_pair_prediction(
        pred_raw,
        x_in,
        residual_learning=True,
    )

    assert torch.equal(pred, x_in + pred_raw)


def test_reconstruct_pair_prediction_rejects_shape_mismatch() -> None:
    pred_raw = torch.zeros((1, 1, 2, 3), dtype=torch.float32)
    x_in = torch.zeros((1, 1, 2, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match='pred_raw.shape'):
        reconstruct_pair_prediction(
            pred_raw,
            x_in,
            residual_learning=True,
        )
