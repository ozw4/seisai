# %%
# ============================================================
# File: packages/seisai-engine/example/example_trend_prior_op_strategy.py
# 内容: 同じ実験設定（B=1,C=1,H=256,W=512, v_true=2500m/s, 2ms, ノイズあり）で
#       TrendPriorOp（Strategy差し替え: IRLS / RANSAC）を使った可視化
# 可視化: 3枚
#   (1) No Trend（Raw prob + t_sec[parabolic argmax]）
#   (2) IRLS: prior適用後の fused prob + trend_t(irls)
#   (3) RANSAC: prior適用後の fused prob + trend_t(ransac)
# 依存:
#   - seisai_engine.postprocess.trend_prior_op.TrendPriorOp, TrendPriorConfig
#   - seisai_engine.postprocess.trend_fit_strategy.{IRLSStrategy,RANSACStrategy}
#   - seisai_pick.trend._time_pick._argmax_time_parabolic
#   - seisai_pick.trend.gaussian_prior_from_trend（Op内で使用）
# ============================================================
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from seisai_engine.postprocess.trend_prior_op import TrendPriorConfig, TrendPriorOp
from seisai_pick.trend._time_pick import _argmax_time_parabolic
from seisai_pick.trend.trend_fit_strategy import IRLSStrategy, RANSACStrategy
from torch import Tensor

torch.manual_seed(0)


def demo() -> None:
	# ---- 実験設定 ---------------------------------------------------
	B, C, H, W = 1, 1, 256, 512
	offsets = torch.linspace(0.0, 1500.0, H).view(1, -1)  # (B,H) [m]
	fb_idx = torch.zeros(B, H, dtype=torch.long)  # 全トレース有効
	dt_sec = torch.tensor([0.002], dtype=torch.float32)  # (B,) 2ms

	# 疑似logits（真のトレンド v_true=2500m/s に沿って単峰 + ノイズ）※clampしない
	v_true = torch.tensor([2500.0]).view(B, 1, 1)  # [m/s]
	t = torch.arange(W, dtype=torch.float32).view(1, 1, W) * dt_sec.view(B, 1, 1)
	x = offsets.view(B, H, 1)
	t_center = x / v_true
	peak = torch.exp(-0.5 * ((t - t_center) / 0.010) ** 2)  # σ=10ms
	logits = (10.0 * peak).unsqueeze(1) + 4.7 * torch.randn(B, 1, H, W)  # (B,1,H,W)

	# ---- (1) No Trend: Raw prob + t_sec(parabolic argmax) ----------
	prob_raw = F.softmax(logits, dim=-1)[:, 0]  # (B,H,W)
	t_sec = _argmax_time_parabolic(prob_raw, dt_sec)  # (B,H) [s]

	# ---- TrendPriorOp（IRLS / RANSAC） -----------------------------
	batch = {'offsets': offsets, 'fb_idx': fb_idx, 'dt_sec': dt_sec}

	# Strategyのパラメータ（robust_linear_trend と同等のデフォルト）
	section_len, stride = 128, 32
	vmin, vmax = 300.0, 8000.0
	huber_c, iters = 1.345, 3

	# IRLS Strategy
	irls = IRLSStrategy(
		section_len=section_len,
		stride=stride,
		huber_c=huber_c,
		iters=iters,
		vmin=vmin,
		vmax=vmax,
		sort_offsets=True,
		use_taper=True,
	)

	# RANSAC Strategy（セクションRANSAC + 1回IRLSリファイン）
	ransac = RANSACStrategy(
		section_len=section_len,
		stride=stride,
		vmin=vmin,
		vmax=vmax,
		ransac_trials=64,
		ransac_tau=2.0,
		ransac_abs_ms=12.0,
		ransac_pack=16,
		sample_weighted=True,
		refine_irls_iters=1,
		use_inlier_blend=True,
		sort_offsets=True,
	)

	# Op設定（prior_mode='logit'相当、ゲートは適用されやすいように0に）
	cfg_irls = TrendPriorConfig(
		fit=irls,
		prior_alpha=1.0,
		prior_sigma_ms=20.0,
		prior_conf_gate=0.0,  # デモ用：常に適用
		channels=0,
	)
	cfg_ransac = TrendPriorConfig(
		fit=ransac,
		prior_alpha=1.0,
		prior_sigma_ms=20.0,
		prior_conf_gate=0.0,
		channels=0,
	)

	op_irls = TrendPriorOp(cfg_irls)
	op_ransac = TrendPriorOp(cfg_ransac)

	# 適用（logits -> fused logits, aux）
	fused_logits_i, aux_i = op_irls(logits, batch)  # (B,1,H,W), dict
	fused_logits_r, aux_r = op_ransac(logits, batch)  # (B,1,H,W), dict

	# 確率化
	prob_i = F.softmax(fused_logits_i, dim=-1)[:, 0]  # (B,H,W)
	prob_r = F.softmax(fused_logits_r, dim=-1)[:, 0]

	# トレンド時刻（ms）を取得（auxキー: trend_prior_trend_t_ch0）
	t_key = 'trend_prior_trend_t_ch0'
	assert t_key in aux_i and t_key in aux_r, 'auxにtrend_tが見つかりません'
	t_sec_ms = (t_sec[0] * 1000.0).cpu().numpy()
	t_sec_ms_irls = (aux_i[t_key][0] * 1000.0).detach().cpu().numpy()
	t_sec_ms_ransac = (aux_r[t_key][0] * 1000.0).detach().cpu().numpy()

	# ---- 可視化（3枚） ---------------------------------------------
	# カラースケールは Raw/IRLS/RANSAC を合わせて決定
	all_vals = torch.cat(
		[prob_raw[0].flatten(), prob_i[0].flatten(), prob_r[0].flatten()]
	)
	vmax_img = float(np.percentile(all_vals.cpu().numpy(), 96))
	vmin_img = 0.0

	t_ms = (torch.arange(W) * dt_sec[0] * 1000.0).cpu().numpy()
	y_off = offsets[0].cpu().numpy()

	def show(ax, img: Tensor, title: str):
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

	# (1) No Trend（Raw prob + t_sec[parabolic argmax]）
	im0 = show(axes[0], prob_raw[0], 'No Trend (Raw prob + t_sec[parabolic argmax])')
	axes[0].plot(t_sec_ms, y_off, 'w.', ms=6, label='t_sec (parabolic argmax)')
	axes[0].legend(loc='upper left')

	# (2) IRLS（prior適用後の fused prob + trend_t）
	im1 = show(axes[1], prob_i[0], 'IRLS: Prior-applied (fused prob)')
	axes[1].plot(t_sec_ms_irls, y_off, 'w.', ms=6, label='trend_t (IRLS)')
	axes[1].legend(loc='upper left')

	# (3) RANSAC（prior適用後の fused prob + trend_t）
	im2 = show(axes[2], prob_r[0], 'RANSAC: Prior-applied (fused prob)')
	axes[2].plot(t_sec_ms_ransac, y_off, 'w.', ms=6, label='trend_t (RANSAC)')
	axes[2].legend(loc='upper left')

	fig.colorbar(im0, ax=axes, fraction=0.046, pad=0.012)
	plt.show()


if __name__ == '__main__':
	demo()
