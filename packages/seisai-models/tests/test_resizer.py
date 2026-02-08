import pytest
import torch
from seisai_models.nn.resizer import Resizer, resize_continuous, resize_discrete


def make_img(b=2, c=3, h=11, w=17, seed=0):
    torch.manual_seed(seed)
    return torch.randn(b, c, h, w)


def make_labels(b=2, c=1, h=13, w=19):
    y = torch.zeros(b, c, h, w, dtype=torch.float32)
    y[:, :, h // 2, :] = 1.0
    y[:, :, :, w // 2] = 2.0
    return y


def test_match_input_continuous_shape() -> None:
    ref = make_img(h=64, w=96)
    y = make_img(h=32, w=48)
    out = resize_continuous(y, policy='match_input', ref=ref)
    assert out.shape[-2:] == ref.shape[-2:]


def test_native_returns_same_object_size() -> None:
    y = make_img(h=40, w=50)
    r = Resizer(policy='native', data_kind='continuous')
    out = r(y, ref=None)
    assert out.shape == y.shape


def test_fixed_size_continuous() -> None:
    y = make_img(h=21, w=35)
    out = resize_continuous(y, policy='fixed', fixed_size=(64, 96))
    assert out.shape[-2:] == (64, 96)


def test_scale_factor_continuous() -> None:
    y = make_img(h=20, w=30)
    out = resize_continuous(y, policy='scale', scale=(2.0, 1.5))
    assert out.shape[-2:] == (40, 45)


def test_discrete_preserves_label_set_on_upsample() -> None:
    y = make_labels(h=15, w=21)
    out = resize_discrete(y, policy='scale', scale=2.0)
    u_in = torch.unique(y)
    u_out = torch.unique(out)
    assert set(u_out.tolist()).issubset(set(u_in.tolist()))


def test_invalid_fixed_params_raise() -> None:
    y = make_img()
    with pytest.raises(AssertionError):
        _ = Resizer(policy='fixed', data_kind='continuous')(y, ref=None)


def test_invalid_scale_params_raise() -> None:
    y = make_img()
    with pytest.raises(AssertionError):
        _ = Resizer(policy='scale', data_kind='continuous')(y, ref=None)
