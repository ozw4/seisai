# test_synthetic_pertrace_shifts.py
import math

import torch
from seisai_engine.loss.shift_pertrace_mse import (
    ShiftRobustPerTraceMSE,
    _shift_robust_l2_pertrace_vec,
)

# ===========================================================


def make_synthetic_stack(
    B: int = 2,
    C: int = 2,
    H: int = 50,
    W: int = 256,
    max_true_shift: int = 4,
    noise_std: float = 0.0,
    seed: int = 0,
):
    """合成データを生成:
    - gt: ランダム波形 (B,C,H,W)
    - pred: 各トレース(b,h)毎に W 方向へランダムシフト([-S, S])+ 微小ノイズ
    - shifts: (B,H) の整数シフト量(pred = roll(gt, shifts))
    """
    assert max_true_shift >= 0 and W - max_true_shift > 0
    g = torch.Generator().manual_seed(seed)
    gt = torch.randn(B, C, H, W, dtype=torch.float32, generator=g)

    if max_true_shift == 0:
        shifts = torch.zeros(B, H, dtype=torch.int64)
    else:
        shifts = torch.randint(
            low=-max_true_shift, high=max_true_shift + 1, size=(B, H), generator=g
        )

    pred = torch.empty_like(gt)
    for b in range(B):
        for h in range(H):
            s = int(shifts[b, h].item())
            pred[b, :, h, :] = torch.roll(gt[b, :, h, :], shifts=s, dims=-1)

    if noise_std > 0.0:
        noise = torch.randn(
            pred.shape, dtype=pred.dtype, device=pred.device, generator=g
        )
        pred.add_(noise * noise_std)
    return pred, gt, shifts


def test_shift_robust_near_zero_on_pertrace_random_shifts():
    B, C, H, W, S_true = 2, 3, 40, 256, 4
    pred, gt, _ = make_synthetic_stack(
        B, C, H, W, max_true_shift=S_true, noise_std=0.0, seed=123
    )

    # 正しいS以上を与えれば、per-traceで完全に整列 → ほぼゼロ
    out_bh = _shift_robust_l2_pertrace_vec(pred, gt, max_shift=S_true)
    assert out_bh.shape == (B, H)
    assert torch.allclose(out_bh, torch.zeros_like(out_bh), atol=1e-6, rtol=0)

    # 平均もゼロ近傍
    out_mean = out_bh.mean()
    assert math.isclose(out_mean.item(), 0.0, rel_tol=0, abs_tol=1e-6)


def test_shift_robust_vs_plain_mse_on_shifted_predictions():
    B, C, H, W, S_true = 1, 2, 30, 128, 3
    pred, gt, _ = make_synthetic_stack(
        B, C, H, W, max_true_shift=S_true, noise_std=0.0, seed=7
    )

    # ずらしたままの MSE は > 0 になる(シフトに頑健ではない)
    plain_mse = ((pred - gt) ** 2).mean()

    # shift-robust は 0 近傍
    robust = _shift_robust_l2_pertrace_vec(pred, gt, max_shift=S_true).mean()

    assert plain_mse.item() > 1e-3
    assert math.isclose(robust.item(), 0.0, rel_tol=0, abs_tol=1e-6)


def test_shift_robust_insufficient_max_shift_yields_positive_loss():
    B, C, H, W, S_true = 1, 2, 20, 128, 5
    pred, gt, _ = make_synthetic_stack(
        B, C, H, W, max_true_shift=S_true, noise_std=0.0, seed=99
    )

    # 与える許容シフトを意図的に小さく(true > allowed)
    S_allowed = 2
    per_trace = _shift_robust_l2_pertrace_vec(pred, gt, max_shift=S_allowed)
    assert (per_trace > 1e-8).any()
    assert per_trace.mean().item() > 0.0


def test_ShiftRobustPerTraceMSE_with_trace_mask_preselection():
    B, C, H, W, S_true = 2, 2, 30, 128, 3
    pred, gt, _ = make_synthetic_stack(
        B, C, H, W, max_true_shift=S_true, noise_std=0.0, seed=2024
    )

    # ランダムに約半分のトレースを採用(True=採用)
    torch.manual_seed(1)
    mask_bh = torch.rand(B, H) > 0.5

    # 参照値: per-trace を出してから (B,H) マスクで平均
    per_trace = _shift_robust_l2_pertrace_vec(pred, gt, max_shift=S_true)
    loss_ref = per_trace[mask_bh].mean()

    # クラス経由(mask_bool=(B,H) を渡す)
    crit = ShiftRobustPerTraceMSE(max_shift=S_true, ch_reduce='all')
    batch = {'target': gt, 'mask_bool': mask_bh}
    loss_cls = crit(pred, batch, reduction='mean')

    assert torch.allclose(loss_cls, loss_ref, atol=1e-6, rtol=0)


def test_ShiftRobustPerTraceMSE_with_pixel_mask_uniform_W():
    B, C, H, W, S_true = 2, 2, 24, 96, 4
    pred, gt, _ = make_synthetic_stack(
        B, C, H, W, max_true_shift=S_true, noise_std=0.0, seed=777
    )

    # (B,H) で採用トレースを作り、pixelマスクへ拡張(W方向一様・ch全True)
    torch.manual_seed(3)
    mask_bh = torch.rand(B, H) > 0.6
    pixel_mask = mask_bh[:, None, :, None].expand(-1, C, -1, W).clone()

    # 参照値
    per_trace = _shift_robust_l2_pertrace_vec(pred, gt, max_shift=S_true)
    loss_ref = per_trace[mask_bh].mean()

    # クラス経由(pixel mask → (B,H) に正規化される前提)
    crit = ShiftRobustPerTraceMSE(max_shift=S_true, ch_reduce='all')
    batch = {'target': gt, 'mask_bool': pixel_mask}
    loss_cls = crit(pred, batch, reduction='mean')

    assert torch.allclose(loss_cls, loss_ref, atol=1e-6, rtol=0)


def test_noise_tolerance_small_noise_yields_small_loss():
    B, C, H, W, S_true = 1, 2, 40, 256, 5
    pred, gt, _ = make_synthetic_stack(
        B, C, H, W, max_true_shift=S_true, noise_std=0.01, seed=314
    )

    # 微小ノイズのみ → 期待損失は ≈ noise_var 程度に比例して小さめ
    loss = _shift_robust_l2_pertrace_vec(pred, gt, max_shift=S_true).mean()
    assert loss.item() < 1e-3


def test_loss_nonincreasing_as_allowed_shift_increases():
    """許容シフト max_shift を増やすほど損失(mean)は非増加(単調減少または等しい)。"""
    B, C, H, W, S_true = 1, 2, 40, 256, 5
    pred, gt, _ = make_synthetic_stack(
        B, C, H, W, max_true_shift=S_true, noise_std=0.01, seed=123
    )

    losses = []
    for S_allowed in range(S_true + 1):
        val = _shift_robust_l2_pertrace_vec(pred, gt, max_shift=S_allowed).mean().item()
        losses.append(val)

    for i in range(1, len(losses)):
        assert losses[i] <= losses[i - 1] + 1e-9, f'non-monotonic at i={i}: {losses}'


def test_sum_loss_increases_when_more_traces_are_selected():
    """True=採用の trace マスクで、選択本数を増やすほど sum 損失は非減少(単調増加または等しい)。"""
    B, C, H, W, S_true = 1, 2, 30, 128, 4
    pred, gt, _ = make_synthetic_stack(
        B, C, H, W, max_true_shift=S_true, noise_std=0.02, seed=777
    )

    # まず per-trace の損失マップ (B,H) を取得
    per_trace = _shift_robust_l2_pertrace_vec(pred, gt, max_shift=S_true)
    assert per_trace.shape == (B, H)

    torch.manual_seed(1)
    order = torch.randperm(H)
    sums = []
    crit = ShiftRobustPerTraceMSE(max_shift=S_true, ch_reduce='all')

    for k in range(1, H + 1):
        mask_bh = torch.zeros(B, H, dtype=torch.bool)
        mask_bh[0, order[:k]] = True  # True=採用
        batch = {'target': gt, 'mask_bool': mask_bh}
        s = crit(pred, batch, reduction='sum').item()
        sums.append(s)

    for i in range(1, len(sums)):
        assert sums[i] >= sums[i - 1] - 1e-9, f'non-monotonic at i={i}: {sums}'
