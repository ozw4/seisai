from __future__ import annotations

import torch
from torch import nn

from seisai_models.ops import channels


def test_inflate_conv_in_channels_uses_module_weight_hook(monkeypatch) -> None:
    conv = nn.Conv2d(1, 2, kernel_size=1, bias=False)
    hook_weight = torch.full((2, 3, 1, 1), 7.0)
    calls: list[tuple[nn.Conv2d, int, str]] = []

    def make_inflated_weight(
        observed_conv: nn.Conv2d,
        target_in_ch: int,
        init_mode: str,
    ) -> torch.Tensor:
        calls.append((observed_conv, target_in_ch, init_mode))
        return hook_weight

    monkeypatch.setattr(
        channels,
        '_make_inflated_weight',
        make_inflated_weight,
        raising=False,
    )

    changed = channels._inflate_conv_in_channels(
        conv,
        3,
        verbose=False,
        init_mode='duplicate',
        name='conv',
    )

    assert changed is True
    assert calls == [(conv, 3, 'duplicate')]
    assert conv.in_channels == 3
    assert torch.equal(conv.weight, hook_weight)
