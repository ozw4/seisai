from __future__ import annotations

import pytest
import torch

from seisai_engine.loss import composite
from seisai_engine.pipelines.psn.loss import build_psn_criterion


def _make_loss_specs() -> list[composite.LossSpec]:
    return [
        composite.LossSpec(
            kind='soft_label_ce',
            weight=1.0,
            scope='all',
            params={},
        )
    ]


def _make_logits_and_target() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn(2, 3, 4, 5, dtype=torch.float32)
    target = torch.softmax(torch.randn(2, 3, 4, 5, dtype=torch.float32), dim=1)
    return logits, target


def _make_trace_valid() -> torch.Tensor:
    return torch.tensor(
        [
            [True, True, False, True],
            [True, False, True, True],
        ],
        dtype=torch.bool,
    )


def test_psn_criterion_forward_without_label_valid_when_disabled() -> None:
    criterion = build_psn_criterion(_make_loss_specs(), use_label_valid=False)
    logits, target = _make_logits_and_target()
    batch = {
        'trace_valid': _make_trace_valid(),
    }

    loss = criterion(logits, target, batch)

    assert isinstance(loss, torch.Tensor)
    assert tuple(loss.shape) == ()
    assert bool(torch.isfinite(loss).item())


def test_psn_criterion_forward_without_label_valid_when_enabled_raises() -> None:
    criterion = build_psn_criterion(_make_loss_specs(), use_label_valid=True)
    logits, target = _make_logits_and_target()
    batch = {
        'trace_valid': _make_trace_valid(),
    }

    with pytest.raises(KeyError, match='label_valid'):
        criterion(logits, target, batch)
