# %%
# (B,C,H,W) 専用 TrendPriorOp の可視化付きサンプル
# RawProb / PriorProb は共通カラースケール、Confidence は (H,W) に拡張して2D表示
from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from seisai_engine.postprocess.trend_prior_op import TrendPriorConfig, TrendPriorOp


def demo() -> None:
	# ---- ダミー入力 -------------------------------------------------
	B, C, H, W = 1, 1, 64, 512
	offsets_m = torch.linspace(0.0, 1500.0, H).view(1, -1)  # (B,H)
	fb_idx = torch.zeros(B, H, dtype=torch.long)  # 有効フラグ (>=0)
	dt_sec = torch.tensor([0.002], dtype=torch.float32)  # (B,) = 2ms

	# ---- 疑似logits（v_true に沿ったピーク + ノイズ）--------------
	v_true = torch.tensor([2500.0]).view(B, 1, 1)  # (B,1,1)
	t = torch.arange(W, dtype=torch.float32).view(1, 1, W) * dt_sec.view(B, 1, 1)
	x = offsets_m.abs().view(B, H, 1)  # (B,H,1)
	t_center = x / v_true  # (B,H,1) [s]
	peak = torch.exp(-0.5 * ((t - t_center) / 0.010) ** 2)  # σ=10ms
	logits = (10.0 * peak).unsqueeze(1).clone()  # (B,1,H,W)
	logits = (logits + 5 * torch.randn_like(logits)).clamp_(-12.0, 12.0)

	# ---- TrendPrior（ch=0 のみ適用）--------------------------------
	cfg = TrendPriorConfig(
		channels=0,
		prior_mode='logit',  # 可視化には logit 推奨
		prior_alpha=1.0,
		prior_sigma_ms=20.0,
		prior_conf_gate=0.0,  # デモ用：常に適用
		offsets_key='offsets',
		fb_idx_key='fb_idx',
		dt_key='dt_sec',
	)
	op = TrendPriorOp(cfg)
	batch = {'offsets': offsets_m, 'fb_idx': fb_idx, 'dt_sec': dt_sec}
	logits_prior, aux = op(logits, batch)  # (B,1,H,W), dict

	# ---- softmax ---------------------------------------------------
	prob_raw = F.softmax(logits, dim=-1)[:, 0]  # (B,H,W)
	prob_prior = F.softmax(logits_prior, dim=-1)[:, 0]  # (B,H,W)

	# ---- Confidence (B,H) -> (H,W) へ拡張して2D表示 ----------------
	# aux からキーを探す（例: "trend_prior_w_conf_ch0" を想定）
	conf_key = next((k for k in aux if k.endswith('w_conf_ch0')), None)
	if conf_key is not None:
		w_conf = aux[conf_key]  # (B,H)
		conf_img = w_conf[0].unsqueeze(-1).repeat(1, W)  # (H,W)  ★ここが2D
	else:
		conf_img = torch.ones(H, W, dtype=logits.dtype, device=logits.device)

	# ---- 共通カラースケール（Raw / Prior）-------------------------
	vmin_img = prob_raw.min()
	vmax_img = prob_raw.max()
	vmin_img = max(0.0, vmin_img)
	vmax_img = min(1.0, vmax_img)

	# ---- 可視化 ----------------------------------------------------
	t_ms = (torch.arange(W, dtype=torch.float32) * dt_sec[0] * 1000.0).cpu().numpy()
	y_off = offsets_m[0].cpu().numpy()

	def show(
		ax, img, title: str, *, vmin: float | None = None, vmax: float | None = None
	):
		im = ax.imshow(
			img.detach().cpu().numpy(),
			origin='lower',
			aspect='auto',
			extent=[float(t_ms[0]), float(t_ms[-1]), float(y_off[0]), float(y_off[-1])],
			vmin=vmin,
			vmax=vmax,
		)
		ax.set_title(title)
		ax.set_xlabel('Time [ms]')
		ax.set_ylabel('Offset [m]')
		plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

	fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
	show(axes[0], prob_raw[0], 'Raw probability (noisy)', vmin=vmin_img, vmax=vmax_img)
	show(axes[1], conf_img, 'Confidence (per trace)', vmin=0.0, vmax=1.0)
	show(
		axes[2], prob_prior[0], 'Prior-fused probability', vmin=vmin_img, vmax=vmax_img
	)

	# トレンド線（aux: "trend_t_ch0" を想定）
	trend_key = next((k for k in aux if k.endswith('trend_t_ch0')), None)
	if trend_key is not None:
		trend_t_ms = (aux[trend_key][0] * 1000.0).detach().cpu().numpy()  # (H,)
		for ax in (axes[0], axes[2]):
			ax.plot(trend_t_ms, y_off, lw=1.2, ls='--', color='white')

	diff = float((prob_prior - prob_raw).abs().mean())
	fig.suptitle(f'Mean |PriorProb - RawProb|: {diff:.3e}', y=1.02)
	plt.show()


if __name__ == '__main__':
	demo()
# %%
