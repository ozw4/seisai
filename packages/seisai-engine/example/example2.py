# %%
# プロットは3枚のみ：
# (1) No Trend(Raw prob + t_sec(単純argmax))
# (2) IRLS：prior を logits に適用後の fused prob(t_sec_ms_irls を重ね描き)
# (3) RANSAC：同上(t_sec_ms_ransac を重ね描き)
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from seisai_pick.trend._time_pick import _argmax_time_parabolic
from seisai_pick.trend.confidence_from_prob import trace_confidence_from_prob
from seisai_pick.trend.gaussian_prior_from_trend import gaussian_prior_from_trend
from seisai_pick.trend.trend_fit import (
    robust_linear_trend,
    robust_linear_trend_sections_ransac,
)

torch.manual_seed(0)


def demo() -> None:
    # ---- Step1: prob ------------------------------------------------
    B, C, H, W = 1, 1, 256, 512
    offsets = torch.linspace(0.0, 1500.0, H).view(1, -1)  # (B,H) [m]
    fb_idx = torch.zeros(B, H, dtype=torch.long)
    dt_sec = torch.tensor([0.002], dtype=torch.float32)  # 2 ms

    # ダミー logits(真値 v_true に沿う単峰+ノイズ): clampしない
    v_true = torch.tensor([2500.0]).view(B, 1, 1)  # [m/s]
    t = torch.arange(W, dtype=torch.float32).view(1, 1, W) * dt_sec.view(B, 1, 1)
    x = offsets.view(B, H, 1)
    t_center = x / v_true
    peak = torch.exp(-0.5 * ((t - t_center) / 0.010) ** 2)  # σ=10ms
    logits = (10.0 * peak).unsqueeze(1) + 4.7 * torch.randn(B, 1, H, W)

    prob = F.softmax(logits, dim=-1)[:, 0]  # (B,H,W)

    # ---- Step2: confidence -----------------------------------------
    w_conf = trace_confidence_from_prob(prob=prob, floor=0.2, power=0.5)  # (B,H)

    # ---- Step3: t_sec(単純argmax) → トレンド(IRLS/RANSAC) ------
    t_sec = _argmax_time_parabolic(prob, dt_sec)  # (B,H) [s]
    valid = (fb_idx >= 0).to(torch.bool)  # (B,H)

    trend_t_i, trend_s_i, v_i, _, covered_i = robust_linear_trend(
        offsets=offsets.to(prob),
        t_sec=t_sec.to(prob),
        valid=valid,
        w_conf=w_conf,
        section_len=128,
        stride=32,
        huber_c=1.345,
        iters=3,
        vmin=300.0,
        vmax=8000.0,
        sort_offsets=True,
        use_taper=True,
    )
    trend_t_r, trend_s_r, v_r, _, covered_r = robust_linear_trend_sections_ransac(
        offsets=offsets.to(prob),
        t_sec=t_sec.to(prob),
        valid=valid,
        w_conf=w_conf,
        section_len=128,
        stride=32,
        vmin=300.0,
        vmax=8000.0,
        ransac_trials=64,
        ransac_tau=2.0,
        ransac_abs_ms=12.0,
        ransac_pack=16,
        sample_weighted=True,
        refine_irls_iters=1,
        use_inlier_blend=True,
        sort_offsets=True,
    )

    # ---- Prior を作って logits に適用(log空間合成) ---------------
    alpha = 1.0
    prior_log_eps = 1e-4

    prior_i = (
        gaussian_prior_from_trend(
            t_trend_sec=trend_t_i,
            dt_sec=dt_sec,
            W=W,
            sigma_ms=20.0,
            ref_tensor=logits,
            covered_mask=covered_i,
        )
        .nan_to_num(0.0)
        .clamp_(min=0.0)
    )  # (B,H,W)

    prior_r = (
        gaussian_prior_from_trend(
            t_trend_sec=trend_t_r,
            dt_sec=dt_sec,
            W=W,
            sigma_ms=20.0,
            ref_tensor=logits,
            covered_mask=covered_r,
        )
        .nan_to_num(0.0)
        .clamp_(min=0.0)
    )

    def fuse_to_prob(
        logits_bchw: torch.Tensor, prior_bhw: torch.Tensor
    ) -> torch.Tensor:
        log_prior = torch.log(prior_bhw.clamp_min(prior_log_eps)).to(torch.float32)
        fused_logits = (
            logits_bchw[:, 0].to(torch.float32) + alpha * log_prior
        )  # (B,H,W)
        fused_logits = fused_logits.clamp_(-30.0, 30.0).to(logits_bchw.dtype)
        return F.softmax(fused_logits.unsqueeze(1), dim=-1)[:, 0]  # (B,H,W)

    prob_i = fuse_to_prob(logits, prior_i)  # IRLS prior 適用後
    prob_r = fuse_to_prob(logits, prior_r)  # RANSAC prior 適用後

    # ---- t_sec を ms に(No Trend / IRLS / RANSAC) ----------------
    t_sec_ms = (t_sec[0] * 1000.0).cpu().numpy()
    t_sec_ms_irls = (trend_t_i[0] * 1000.0).cpu().numpy()
    t_sec_ms_ransac = (trend_t_r[0] * 1000.0).cpu().numpy()

    # ---- 可視化(3枚のみ)-----------------------------------------
    # カラースケールは Raw/IRLS/RANSAC の確率をまとめて決定
    all_vals = torch.cat([prob[0].flatten(), prob_i[0].flatten(), prob_r[0].flatten()])
    vmax_img = float(np.percentile(all_vals.cpu().numpy(), 96))
    vmin_img = 0.0

    t_ms = (torch.arange(W) * dt_sec[0] * 1000.0).cpu().numpy()
    y_off = offsets[0].cpu().numpy()

    def show(ax, img, title: str):
        im = ax.imshow(
            img.detach().cpu().numpy(),
            origin='lower',
            aspect='auto',
            extent=[float(t_ms[0]), float(t_ms[-1]), float(y_off[0]), float(y_off[-1])],
            vmin=vmin_img,
            vmax=vmax_img,
        )
        ax.set_title(title)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Offset [m]')
        return im

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # (1) No Trend(Raw prob + t_sec(単純argmax))
    im0 = show(axes[0], prob[0], 'No Trend (Raw prob + t_sec[argmax])')
    axes[0].plot(t_sec_ms, y_off, 'w.', ms=6, label='t_sec (argmax)')
    axes[0].legend(loc='upper left')

    # (2) IRLS(prior を logits に適用した fused prob を表示)
    im1 = show(axes[1], prob_i[0], 'IRLS: Prior-applied (fused prob)')
    axes[1].plot(t_sec_ms_irls, y_off, 'w.', ms=6, label='t_sec_ms_irls')
    axes[1].legend(loc='upper left')

    # (3) RANSAC(同上)
    im2 = show(axes[2], prob_r[0], 'RANSAC: Prior-applied (fused prob)')
    axes[2].plot(t_sec_ms_ransac, y_off, 'w.', ms=6, label='t_sec_ms_ransac')
    axes[2].legend(loc='upper left')

    fig.colorbar(im0, ax=axes, fraction=0.046, pad=0.012)
    plt.show()


if __name__ == '__main__':
    demo()
