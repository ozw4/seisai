import torch
import torch.nn.functional as F
from torch import nn


class ConvBnAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding: int = 0,
        stride: int = 1,
        norm_layer: nn.Module = nn.Identity,
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = (
            norm_layer(out_channels) if norm_layer != nn.Identity else nn.Identity()
        )
        self.act = act_layer(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SCSEModule2d(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.Tanh(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Attention2d(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule2d(**params)
        else:
            raise ValueError(f'Attention {name} is not implemented')

    def forward(self, x):
        return self.attention(x)


class DecoderBlock2d(nn.Module):
    """Decoder block with interpolation-based upsampling."""

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer: nn.Module = nn.Identity,
        attention_type: str = None,
        intermediate_conv: bool = False,
        upsample_mode: str = 'bilinear',
        scale_factor: int | tuple[int, int] = 2,
    ):
        super().__init__()
        self.upsample_mode = upsample_mode
        self.scale_factor = scale_factor

        if intermediate_conv:
            k = 3
            c = skip_channels if skip_channels != 0 else in_channels
            self.intermediate_conv = nn.Sequential(
                ConvBnAct2d(c, c, k, k // 2),
                ConvBnAct2d(c, c, k, k // 2),
            )
        else:
            self.intermediate_conv = None

        self.attention1 = Attention2d(
            name=attention_type, in_channels=in_channels + skip_channels
        )
        self.conv1 = ConvBnAct2d(
            in_channels + skip_channels, out_channels, 3, 1, norm_layer=norm_layer
        )
        self.conv2 = ConvBnAct2d(
            out_channels, out_channels, 3, 1, norm_layer=norm_layer
        )
        self.attention2 = Attention2d(name=attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        if skip is not None:
            x = self._interpolate(x, size=skip.shape[-2:])
        else:
            x = self._interpolate(x, scale_factor=self.scale_factor)
        if self.intermediate_conv is not None:
            if skip is not None:
                skip = self.intermediate_conv(skip)
            else:
                x = self.intermediate_conv(x)
        if skip is not None:
            x = self.attention1(torch.cat([x, skip], dim=1))
        x = self.conv2(self.conv1(x))
        return self.attention2(x)

    def _interpolate(self, x, size=None, scale_factor=None):
        kwargs = {}
        if self.upsample_mode in ('bilinear', 'bicubic'):
            kwargs['align_corners'] = False
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=self.upsample_mode,
            **kwargs,
        )
