# tests/test_builders.py
import numpy as np
import torch
import pytest

from seisai_builders.builder import (
    IdentitySignal, MaskedSignal, MakeTimeChannel, MakeOffsetChannel,
    FBGaussMap, SelectStack, BuildPlan
)

class DummyMasker:
    def apply(self, x, py_random=None):
        # 中央W/4 幅をゼロ化するだけのダミー
        H, W = x.shape
        xm = x.copy()
        s = W // 4
        e = s + W // 4
        xm[:, s:e] = 0.0
        idx = np.arange(s, e, dtype=np.int64)
        return xm, idx

def make_sample(H=4, W=16, dt=0.001):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((H, W)).astype(np.float32)
    offsets = np.linspace(100, 400, H, dtype=np.float32)
    fb_idx = np.array([5, -1, 8, 3], dtype=np.int64)
    meta = {"hflip": False, "factor": 1.0, "start": 0}
    return {
        "x_view": x,
        "offsets": offsets,
        "dt_sec": float(dt),
        "fb_idx": fb_idx,
        "meta": meta,
    }

def test_make_time_channel():
    s = make_sample(H=3, W=10, dt=0.002)  # 2ms
    MakeTimeChannel("t")(s)
    t = s["t"]
    assert t.shape == (3, 10)
    assert np.allclose(t[0, :3], [0.0, 0.002, 0.004], atol=1e-7)

def test_make_offset_channel_normalize():
    s = make_sample(H=5, W=8)
    MakeOffsetChannel("off", normalize=True)(s)
    off = s["off"]
    assert off.shape == (5, 8)
    # 各行は同じ値なので、列に沿っても平均0に近い
    m = off[:, 0].mean()
    assert abs(m) < 1e-5

def test_fb_gauss_map_basic():
    s = make_sample(H=3, W=11, dt=0.001)
    FBGaussMap("g", sigma=1.0)(s)
    g = s["g"]
    assert g.shape == (3, 11)
    # 1本目は fb=5 付近が最大
    assert int(np.argmax(g[0])) == 5
    # 2本目は fb=-1 → 全ゼロ
    assert np.allclose(g[1], 0.0)
    # 3本目は fb=8
    assert int(np.argmax(g[2])) == 8

def test_fb_gauss_map_with_meta():
    s = make_sample(H=2, W=15)
    # hflip + factor=2.0 + start=3
    s["meta"] = {"hflip": True, "factor": 2.0, "start": 3}
    s["fb_idx"] = np.array([4, 6], dtype=np.int64)
    # hflip で fb 配列は逆順になる → [6,4] が基準
    FBGaussMap("g", sigma=1.0)(s)
    g = s["g"]
    # 逆順後: H=0 に元の2本目(6)→ factor*6 - start = 12 - 3 = 9
    assert int(np.argmax(g[0])) == 9
    # H=1 に元の1本目(4)→ 8 - 3 = 5
    assert int(np.argmax(g[1])) == 5

def test_select_stack_2d_and_3d():
    s = make_sample(H=3, W=10)
    # 2Dを複数積む
    MakeTimeChannel("t")(s)
    MakeOffsetChannel("o")(s)
    SelectStack(keys=["x_view", "t", "o"], dst="input")(s)
    x = s["input"]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 3, 10)  # C=3

    # 3Dをそのまま & 2Dと連結
    s["feat3d"] = np.random.randn(2, 3, 10).astype(np.float32)
    SelectStack(keys=["feat3d", "t"], dst="input2")(s)
    x2 = s["input2"]
    assert x2.shape == (3, 3, 10)

def test_masked_signal_and_build_plan_end_to_end():
    s = make_sample(H=4, W=16)
    masker = DummyMasker()
    wave_ops = [
        MaskedSignal(masker, src="x_view", dst="x_m"),
        MakeTimeChannel("t"),
        MakeOffsetChannel("o"),
    ]
    label_ops = [FBGaussMap("g", sigma=1.0)]
    plan = BuildPlan(
        wave_ops=wave_ops,
        label_ops=label_ops,
        input_stack=SelectStack(keys=["x_m", "t", "o"], dst="input"),
        target_stack=SelectStack(keys=["g"], dst="target"),
    )
    plan.run(s)
    assert s["input"].shape[0] == 3  # x_m, t, o
    assert s["target"].shape[0] == 1
    # マスクインデックスが入っていること
    assert "mask_indices" in s and s["mask_indices"].ndim == 1
