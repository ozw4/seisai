import pytest
from seisai_engine.infer.ckpt_meta import resolve_output_ids, resolve_softmax_axis


def test_resolve_from_checkpoint_explicit_values() -> None:
    ckpt = {
        'pipeline': 'fbpick',
        'softmax_axis': ' Channel ',
        'output_ids': [' A ', 'B'],
    }
    assert (
        resolve_softmax_axis(
            ckpt=ckpt,
            out_chans=2,
            pipeline_name='psn',
        )
        == 'channel'
    )
    assert resolve_output_ids(
        ckpt=ckpt,
        out_chans=2,
        pipeline_name='psn',
    ) == ('A', 'B')


def test_resolve_defaults_when_pipeline_matches() -> None:
    ckpt = {'pipeline': 'psn'}
    assert (
        resolve_softmax_axis(
            ckpt=ckpt,
            out_chans=3,
            pipeline_name='psn',
        )
        == 'channel'
    )
    assert resolve_output_ids(
        ckpt=ckpt,
        out_chans=3,
        pipeline_name='psn',
    ) == ('P', 'S', 'N')


def test_resolve_defaults_for_single_channel_when_pipeline_mismatch() -> None:
    ckpt = {'pipeline': 'fbpick'}
    assert (
        resolve_softmax_axis(
            ckpt=ckpt,
            out_chans=1,
            pipeline_name='psn',
        )
        == 'time'
    )
    assert resolve_output_ids(
        ckpt=ckpt,
        out_chans=1,
        pipeline_name='psn',
    ) == ('P',)


def test_resolve_invalid_and_ambiguous_cases() -> None:
    with pytest.raises(ValueError, match='softmax_axis is ambiguous'):
        resolve_softmax_axis(
            ckpt={'pipeline': 'fbpick'},
            out_chans=2,
            pipeline_name='psn',
        )

    with pytest.raises(TypeError, match='checkpoint softmax_axis must be str'):
        resolve_softmax_axis(
            ckpt={'pipeline': 'fbpick', 'softmax_axis': 1},
            out_chans=2,
            pipeline_name='psn',
        )

    with pytest.raises(ValueError, match='softmax_axis="time" requires out_chans==1'):
        resolve_softmax_axis(
            ckpt={'pipeline': 'fbpick', 'softmax_axis': 'time'},
            out_chans=2,
            pipeline_name='psn',
        )

    with pytest.raises(
        TypeError, match='checkpoint output_ids must be list\\[str\\] or tuple\\[str, \\.\\.\\.\\]'
    ):
        resolve_output_ids(
            ckpt={'pipeline': 'fbpick', 'output_ids': 'P'},
            out_chans=1,
            pipeline_name='psn',
        )

    with pytest.raises(TypeError, match='checkpoint output_ids\\[1\\] must be non-empty str'):
        resolve_output_ids(
            ckpt={'pipeline': 'fbpick', 'output_ids': ['P', '  ']},
            out_chans=2,
            pipeline_name='psn',
        )

    with pytest.raises(ValueError, match='checkpoint output_ids length 1 != out_chans 2'):
        resolve_output_ids(
            ckpt={'pipeline': 'fbpick', 'output_ids': ['P']},
            out_chans=2,
            pipeline_name='psn',
        )

    with pytest.raises(ValueError, match='checkpoint output_ids must be unique'):
        resolve_output_ids(
            ckpt={'pipeline': 'fbpick', 'output_ids': ['P', 'P']},
            out_chans=2,
            pipeline_name='psn',
        )
