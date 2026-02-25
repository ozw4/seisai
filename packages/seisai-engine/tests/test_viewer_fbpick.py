import pytest
import torch
from seisai_engine.infer.ckpt_meta import resolve_output_ids, resolve_softmax_axis
from seisai_engine.viewer.fbpick import (
    _apply_softmax,
    _resolve_channel_index,
)


def test_apply_softmax_channel_normalizes_channel_axis() -> None:
    logits = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[-1.0, 0.0], [1.0, 2.0]],
            [[0.5, 0.5], [0.5, 0.5]],
        ],
        dtype=torch.float32,
    )
    probs = _apply_softmax(
        logits_chw=logits,
        softmax_axis='channel',
        tau=1.0,
        out_chans=3,
    )
    sums = probs.sum(dim=0)
    assert torch.allclose(sums, torch.ones_like(sums))


def test_apply_softmax_time_normalizes_width_axis() -> None:
    logits = torch.tensor(
        [[[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]]],
        dtype=torch.float32,
    )
    probs = _apply_softmax(
        logits_chw=logits,
        softmax_axis='time',
        tau=1.0,
        out_chans=1,
    )
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))


def test_apply_softmax_time_requires_single_output_channel() -> None:
    logits = torch.zeros((2, 3, 4), dtype=torch.float32)
    with pytest.raises(ValueError):
        _apply_softmax(
            logits_chw=logits,
            softmax_axis='time',
            tau=1.0,
            out_chans=2,
        )


def test_resolve_softmax_axis_from_checkpoint_and_defaults() -> None:
    axis_1 = resolve_softmax_axis(
        ckpt={'pipeline': 'fbpick', 'softmax_axis': 'channel'},
        out_chans=4,
        pipeline_name='psn',
    )
    assert axis_1 == 'channel'

    axis_2 = resolve_softmax_axis(
        ckpt={'pipeline': 'psn'},
        out_chans=3,
        pipeline_name='psn',
    )
    assert axis_2 == 'channel'

    axis_3 = resolve_softmax_axis(
        ckpt={'pipeline': 'fbpick'},
        out_chans=1,
        pipeline_name='psn',
    )
    assert axis_3 == 'time'

    with pytest.raises(ValueError):
        resolve_softmax_axis(
            ckpt={'pipeline': 'fbpick'},
            out_chans=2,
            pipeline_name='psn',
        )

    with pytest.raises(ValueError):
        resolve_softmax_axis(
            ckpt={'pipeline': 'fbpick', 'softmax_axis': 'time'},
            out_chans=2,
            pipeline_name='psn',
        )


def test_resolve_output_ids_from_ckpt_and_fallback_rules() -> None:
    output_ids = resolve_output_ids(
        ckpt={'pipeline': 'fbpick', 'output_ids': ['A', 'B']},
        out_chans=2,
        pipeline_name='psn',
    )
    assert output_ids == ('A', 'B')

    assert resolve_output_ids(
        ckpt={'pipeline': 'psn'},
        out_chans=3,
        pipeline_name='psn',
    ) == (
        'P',
        'S',
        'N',
    )
    assert resolve_output_ids(
        ckpt={'pipeline': 'fbpick'},
        out_chans=1,
        pipeline_name='psn',
    ) == ('P',)
    assert resolve_output_ids(
        ckpt={'pipeline': 'fbpick'},
        out_chans=2,
        pipeline_name='psn',
    ) == (
        'ch0',
        'ch1',
    )

    with pytest.raises(ValueError):
        resolve_output_ids(
            ckpt={'pipeline': 'fbpick', 'output_ids': ['P']},
            out_chans=2,
            pipeline_name='psn',
        )


def test_resolve_channel_index_accepts_supported_forms() -> None:
    output_ids = ('P', 'S', 'N')
    assert _resolve_channel_index(None, output_ids=output_ids) == 0
    assert _resolve_channel_index(2, output_ids=output_ids) == 2
    assert _resolve_channel_index('S', output_ids=output_ids) == 1
    assert _resolve_channel_index('ch2', output_ids=output_ids) == 2
    assert _resolve_channel_index(' P ', output_ids=output_ids) == 0
    assert _resolve_channel_index('  ch2  ', output_ids=output_ids) == 2

    with pytest.raises(ValueError):
        _resolve_channel_index('X', output_ids=output_ids)
    with pytest.raises(ValueError):
        _resolve_channel_index('p', output_ids=output_ids)
    with pytest.raises(ValueError):
        _resolve_channel_index('ch7', output_ids=output_ids)
