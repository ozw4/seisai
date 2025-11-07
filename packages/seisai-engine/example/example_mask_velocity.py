# %%
# (B,C,H,W) 専用版 velocity_filt の可視化付き使用例（ノイズ注入つき）
from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from seisai_engine.postprocess.velocity_filter_op import (
	apply_velocity_filt_logits,
	make_velocity_feasible_filt,
)


def demo() -> None:
	# ---- ダミー入力 -------------------------------------------------
	B, C, H, W = 1, 1, 64, 512
	offsets_m = torch.linspace(0.0, 1500.0, H).view(1, -1)  # (B,H)
	dt_sec = torch.tensor([0.002], dtype=torch.float32)  # (B,) = 2ms

	# 速度制約 [m/s] と t0 スラック/タパー [ms]
	vmin, vmax = 1500.0, 4500.0
	t0_lo_ms, t0_hi_ms, taper_ms = -10.0, 60.0, 8.0

	# ---- 速度円錐フィルタ ------------------------------------------
	filt = make_velocity_feasible_filt(
		offsets_m=offsets_m,
		dt_sec=dt_sec,
		W=W,
		vmin=vmin,
		vmax=vmax,
		t0_lo_ms=t0_lo_ms,
		t0_hi_ms=t0_hi_ms,
		taper_ms=taper_ms,
	)  # (B,H,W)

	# ---- 疑似ログits（真の速度 v_true に沿ってピーク）------------
	v_true = torch.tensor([2500.0]).view(B, 1, 1)  # (B,1,1)
	t = torch.arange(W, dtype=torch.float32).view(1, 1, W) * dt_sec.view(B, 1, 1)
	x = offsets_m.abs().view(B, H, 1)  # (B,H,1)
	t_center = x / v_true  # (B,H,1) [s]
	peak = torch.exp(-0.5 * ((t - t_center) / 0.010) ** 2)  # σ=10ms
	logits = (10.0 * peak).unsqueeze(1).clone()  # (B,1,H,W)

	# ---- ノイズ付加（例：標準偏差指定） ----------------------------
	noise_std = 8.0
	logits = logits + noise_std * torch.randn_like(logits)
	logits = logits.clamp_(-12.0, 12.0)

	# ---- フィルタを log 空間で適用 --------------------------------
	filted_logits = apply_velocity_filt_logits(logits.clone(), filt)  # (B,1,H,W)

	# ---- softmax して確率化 ----------------------------------------
	prob_raw = F.softmax(logits, dim=-1)[:, 0]  # (B,H,W)
	prob_filted = F.softmax(filted_logits, dim=-1)[:, 0]  # (B,H,W)

	# ---- 可視化（RawProb と FiltedProb で vmin/vmax を統一） ------
	t_ms = (torch.arange(W, dtype=torch.float32) * dt_sec[0] * 1000.0).numpy()
	y_off = offsets_m[0].numpy()

	# 共有カラースケールを計算（RawとFiltedの両方から）
	vmin_img = float(torch.minimum(prob_raw.min(), prob_raw.min()))
	vmax_img = float(torch.maximum(prob_raw.max(), prob_raw.max()))
	# ほぼ[0,1]のはずだが、数値誤差を考慮してクリップ
	vmin_img = max(0.0, vmin_img)
	vmax_img = min(1.0, vmax_img)

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

	fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
	show(axes[0], prob_raw[0], 'Raw probability (noisy)', vmin=vmin_img, vmax=vmax_img)
	show(axes[1], filt[0], 'Velocity-feasible filter')  # マスクは独立のスケールでOK
	show(axes[2], prob_filted[0], 'Filted probability', vmin=vmin_img, vmax=vmax_img)

	# 参考境界
	t_lo_ms_curve = (offsets_m[0] / vmax * 1000.0 + t0_lo_ms).numpy()
	t_hi_ms_curve = (offsets_m[0] / vmin * 1000.0 + t0_hi_ms).numpy()
	for ax in axes:
		ax.plot(t_lo_ms_curve, y_off, lw=1.0, ls='--', color='white')
		ax.plot(t_hi_ms_curve, y_off, lw=1.0, ls='--', color='white')

	outside_mass = float(((1.0 - filt[0]) * prob_filted[0]).sum() / (H * W))
	fig.suptitle(f'Avg prob mass outside cone: {outside_mass:.3e}', y=1.02)
	plt.show()


if __name__ == '__main__':
	demo()
