import torch

from seisai_models.models.encdec2d import EncDec2D
from seisai_models.nn.anti_alias import BlurPool2d


def test_blurpool_shape_w6016() -> None:
    x = torch.randn(2, 4, 32, 6016)
    blur = BlurPool2d(taps=3, stride_w=2)
    out = blur(x)
    assert out.shape == (2, 4, 32, 3008)
    assert blur.kernel.requires_grad is False
    assert all(p is not blur.kernel for p in blur.parameters())


def _make_model(pre_stage_antialias: bool) -> EncDec2D:
    return EncDec2D(
        backbone='resnet18',
        in_chans=1,
        out_chans=2,
        pretrained=False,
        pre_stages=1,
        pre_stage_strides=((1, 2),),
        pre_stage_kernels=(3,),
        pre_stage_channels=(1,),
        pre_stage_use_bn=False,
        pre_stage_antialias=pre_stage_antialias,
        pre_stage_aa_taps=3,
    )


def test_encdec2d_antialias_does_not_break_forward() -> None:
    x = torch.randn(2, 1, 32, 64)
    for flag in (False, True):
        model = _make_model(flag)
        y = model(x)
        assert y.shape == (2, 2, 32, 64)
        feats = model._encode(x)
        assert feats[-1].shape[-2] == 32
        assert feats[-1].shape[-1] == 32
