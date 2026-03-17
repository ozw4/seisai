from __future__ import annotations

import pytest
import torch

from seisai_models.models.encdec2d import EncDec2D


def _make_model(**overrides) -> EncDec2D:
    kwargs = {
        'backbone': 'resnet18',
        'in_chans': 1,
        'out_chans': 2,
        'pretrained': False,
        'pre_stages': 2,
        'pre_stage_strides': ((1, 2), (1, 2)),
        'pre_stage_kernels': (3, 3),
        'pre_stage_channels': (8, 8),
        'pre_stage_use_bn': False,
    }
    kwargs.update(overrides)
    return EncDec2D(**kwargs)


def _find_layout_entry(
    model: EncDec2D, *, source: str, source_shallow_index: int
):
    for entry in model._skip_layout:
        if (
            entry.source == source
            and entry.source_shallow_index == source_shallow_index
        ):
            return entry
    raise AssertionError(
        f'missing layout entry for source={source}, source_shallow_index={source_shallow_index}'
    )


@pytest.mark.parametrize(
    'disable_kwargs',
    [
        {},
        {'disable_prestage_skip_indices': [0]},
        {'disable_backbone_skip_indices': [0]},
    ],
)
def test_encdec2d_skip_disable_forward_preserves_output_shape(
    disable_kwargs: dict,
) -> None:
    model = _make_model(**disable_kwargs)
    x = torch.randn(2, 1, 64, 64)

    y = model(x)

    assert y.shape == (2, 2, 64, 64)


def test_encdec2d_skip_layout_converts_shallow_indices_to_decoder_order() -> None:
    model = _make_model()

    backbone_entries = [entry for entry in model._skip_layout if entry.source == 'backbone']
    prestage_entries = [entry for entry in model._skip_layout if entry.source == 'prestage']

    assert [entry.source_shallow_index for entry in backbone_entries] == list(
        range(len(backbone_entries) - 1, -1, -1)
    )
    assert [entry.source_shallow_index for entry in prestage_entries] == list(
        range(len(prestage_entries) - 1, -1, -1)
    )

    backbone_highest = _find_layout_entry(model, source='backbone', source_shallow_index=0)
    prestage_highest = _find_layout_entry(model, source='prestage', source_shallow_index=0)

    assert backbone_highest.decoder_skip_index == len(backbone_entries) - 1
    assert prestage_highest.decoder_skip_index == len(backbone_entries) + len(
        prestage_entries
    ) - 1


@pytest.mark.parametrize(
    ('disable_kwargs', 'source'),
    [
        ({'disable_prestage_skip_indices': [0]}, 'prestage'),
        ({'disable_backbone_skip_indices': [0]}, 'backbone'),
    ],
)
def test_encdec2d_skip_disable_updates_decoder_channels_and_skip_tensors(
    disable_kwargs: dict,
    source: str,
) -> None:
    x = torch.randn(1, 1, 64, 64)
    base_model = _make_model()
    disabled_model = _make_model(**disable_kwargs)
    entry = _find_layout_entry(base_model, source=source, source_shallow_index=0)

    base_in_channels = base_model.decoder.blocks[
        entry.decoder_skip_index
    ].conv1.conv.in_channels
    disabled_in_channels = disabled_model.decoder.blocks[
        entry.decoder_skip_index
    ].conv1.conv.in_channels
    disabled_feats = disabled_model._encode(x)

    assert disabled_model.decoder.skip_channels[entry.decoder_skip_index] == 0
    assert disabled_feats[entry.decoder_skip_index + 1] is None
    assert disabled_in_channels == base_in_channels - entry.channels


def test_encdec2d_rejects_out_of_range_prestage_disable_index() -> None:
    base_model = _make_model()
    prestage_count = len(
        [entry for entry in base_model._skip_layout if entry.source == 'prestage']
    )

    with pytest.raises(
        ValueError,
        match=rf'disable_prestage_skip_indices.*{prestage_count}',
    ):
        _make_model(disable_prestage_skip_indices=[prestage_count])


def test_encdec2d_rejects_out_of_range_backbone_disable_index() -> None:
    base_model = _make_model()
    backbone_count = len(
        [entry for entry in base_model._skip_layout if entry.source == 'backbone']
    )

    with pytest.raises(
        ValueError,
        match=rf'disable_backbone_skip_indices.*{backbone_count}',
    ):
        _make_model(disable_backbone_skip_indices=[backbone_count])
