from torch import nn


def set_first_conv_padding(backbone: nn.Module, padding=(1, 1)) -> None:
    """backboneの最初のConvだけpaddingを上書き."""
    for m in backbone.modules():
        if isinstance(m, nn.Conv2d):
            m.padding = padding

            print(f'Adjusted first Conv2d padding to {padding}')
            break


def override_stage_strides(backbone: nn.Module, stage_strides: list[tuple[int, int]]) -> None:
    """Apply anisotropic strides per stage to the backbone.

    Parameters
    ----------
    backbone: nn.Module
            Encoder backbone whose strided convolutions will be modified.
    stage_strides: list[tuple[int, int]]
            List of strides ``(sH, sW)`` applied in order of appearance to
            convolutions with ``stride>1``.

    Raises
    ------
    ValueError
            If ``stage_strides`` has more elements than the number of
            detected strided convolutions.

    """
    convs: list[nn.Conv2d] = [
        m
        for m in backbone.modules()
        if isinstance(m, nn.Conv2d) and (m.stride[0] > 1 or m.stride[1] > 1)
    ]

    if len(stage_strides) > len(convs):
        msg = 'stage_strides longer than detected stride layers'
        raise ValueError(msg)

    for conv, s in zip(convs, stage_strides, strict=False):
        conv.stride = s
