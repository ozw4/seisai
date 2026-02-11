import torch
import torch.nn.functional as F
from torch import nn


def _build_fir_kernel(taps: int) -> torch.Tensor:
    if taps == 3:
        kernel_1d = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32) / 4.0
    elif taps == 5:
        kernel_1d = (
            torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32) / 16.0
        )
    else:
        raise ValueError(f'Unsupported taps: {taps}. Use 3 or 5.')
    return kernel_1d.view(1, 1, 1, taps)


class FixedFIRBlur2d(nn.Module):
    def __init__(self, taps: int = 3, pad_mode: str = 'zeros') -> None:
        super().__init__()
        if pad_mode != 'zeros':
            raise ValueError(f'Unsupported pad_mode: {pad_mode}. Use "zeros".')
        kernel = _build_fir_kernel(taps)
        self.register_buffer('kernel', kernel)
        self.taps = taps
        self.pad_mode = pad_mode
        self.pad_w = taps // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError('FixedFIRBlur2d expects 4D input (B, C, H, W).')
        c = x.shape[1]
        kernel = self.kernel.to(dtype=x.dtype, device=x.device)
        weight = kernel.repeat(c, 1, 1, 1)
        return F.conv2d(
            x,
            weight,
            bias=None,
            stride=(1, 1),
            padding=(0, self.pad_w),
            groups=c,
        )


class BlurPool2d(nn.Module):
    def __init__(
        self, taps: int = 3, stride_w: int = 2, pad_mode: str = 'zeros'
    ) -> None:
        super().__init__()
        if stride_w not in (1, 2, 4):
            raise ValueError(f'Unsupported stride_w: {stride_w}. Use 1, 2, or 4.')
        if pad_mode != 'zeros':
            raise ValueError(f'Unsupported pad_mode: {pad_mode}. Use "zeros".')
        kernel = _build_fir_kernel(taps)
        self.register_buffer('kernel', kernel)
        self.taps = taps
        self.stride_w = stride_w
        self.pad_mode = pad_mode
        self.pad_w = taps // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError('BlurPool2d expects 4D input (B, C, H, W).')
        c = x.shape[1]
        kernel = self.kernel.to(dtype=x.dtype, device=x.device)
        weight = kernel.repeat(c, 1, 1, 1)
        return F.conv2d(
            x,
            weight,
            bias=None,
            stride=(1, self.stride_w),
            padding=(0, self.pad_w),
            groups=c,
        )
