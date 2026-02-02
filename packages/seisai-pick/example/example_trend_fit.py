# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from seisai_pick.trend.confidence_from_prob import trace_confidence_from_prob
from seisai_pick.trend.trend_fit import (
	robust_linear_trend,
	robust_linear_trend_sections_ransac,
)


# ---------- デモ(合成データ生成 → 両手法で推定 → 可視化) ----------
@torch.no_grad()
def _make_synthetic(
	B: int = 1,
	H: int = 400,
	W: int = 256,
	*,
	# 外れ値の制御
	outlier_frac: float = 0.30,  # ランダム外れの割合(全トレースに対して)
	outlier_mode: str = 'shift',  # "shift" | "random"
	outlier_shift_ms: float = 520.0,  # shiftモードの平均量[ms]
	outlier_std_ms: float = 15.0,  # shiftモードの標準偏差[ms]
	outlier_blocks: int = 0,  # 連続ブロック外れの個数(0で無効)
	block_len: int = 24,  # ブロック長(トレース数)
	seed: int = 0,
):
	"""合成の first-break っぽい時刻列と確率マップ(prob)を生成。
	明らかな外れ値をランダム／連続ブロックで注入可能。

	Returns
	-------
	offsets, t_true, t_pred, valid, prob, dt_sec

	"""
	assert B >= 1 and H >= 4 and W >= 4, 'B,H,W must be positive and reasonably large'
	assert 0.0 <= outlier_frac <= 1.0, 'outlier_frac must be in [0,1]'
	assert outlier_mode in ('shift', 'random'), (
		"outlier_mode must be 'shift' or 'random'"
	)
	assert outlier_blocks >= 0 and block_len >= 1, 'outlier_blocks>=0, block_len>=1'

	device = torch.device('cpu')
	g = torch.Generator(device=device).manual_seed(seed)

	offsets = torch.linspace(0.0, 3000.0, H, dtype=torch.float32, device=device).repeat(
		B, 1
	)  # (B,H) [m]
	v_true = 2000.0  # m/s
	a_true = 0.20  # s
	t_true = a_true + (1.0 / v_true) * offsets  # (B,H)
	t_true = t_true + 0.005 * torch.sin(offsets / 400.0)

	valid = torch.ones_like(t_true, dtype=torch.bool)

	# 観測(モデル出力相当): ノイズは強め
	noise = 0.1 * torch.randn_like(t_true)
	t_pred = t_true + noise

	# --- 外れ値インデックスの作成(ランダム + 連続ブロック) ---
	n_rand = int(round(outlier_frac * H))
	rand_idx = (
		torch.randperm(H, generator=g, device=device)[:n_rand]
		if n_rand > 0
		else torch.empty(0, dtype=torch.long, device=device)
	)

	block_idxs = []
	if outlier_blocks > 0:
		starts = torch.randint(
			low=0,
			high=max(1, H - block_len + 1),
			size=(outlier_blocks,),
			generator=g,
			device=device,
		)
		for s in starts.tolist():
			e = min(H, s + block_len)
			block_idxs.append(torch.arange(s, e, device=device))
	if block_idxs:
		block_idx = torch.cat(block_idxs, dim=0)
	else:
		block_idx = torch.empty(0, dtype=torch.long, device=device)

	all_idx = torch.unique(torch.cat([rand_idx, block_idx], dim=0))  # (N_out,)
	N_out = all_idx.numel()

	# --- 外れ値の注入 ---
	if N_out > 0:
		if outlier_mode == 'shift':
			mu = max(outlier_shift_ms * 1e-3, 1e-6)
			sd = max(outlier_std_ms * 1e-3, 0.0)
			sign = torch.where(
				torch.rand((B, N_out), generator=g, device=device) > 0.5, 1.0, -1.0
			)
			delta = torch.normal(
				mean=mu, std=sd, size=(B, N_out), generator=g, device=device
			)
			t_pred[:, all_idx] = t_pred[:, all_idx] + sign * delta
		else:  # "random"
			t_min = t_true.min(dim=1, keepdim=True).values - 0.20
			t_max = t_true.max(dim=1, keepdim=True).values + 0.20
			u = torch.rand((B, N_out), generator=g, device=device)
			t_rand = t_min + u * (t_max - t_min)
			t_pred[:, all_idx] = t_rand

	# サンプル間隔
	dt = 0.002  # 2ms
	dt_sec = torch.full((B, 1), dt, dtype=torch.float32, device=device)

	# prob をガウスで作る(softmax正規化)
	t_idx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
	mu_idx = (t_true / dt).clamp(0, W - 1)  # (B,H)
	sigma_ms = 12.0
	sigma = max(sigma_ms * 1e-3, 1e-6)
	logp = -0.5 * ((t_idx - mu_idx.unsqueeze(-1)) * dt / sigma) ** 2  # (B,H,W)
	logp[:, :, (W // 3) : (W // 3 + 5)] += -2.0  # マルチモード風の邪魔を少し
	prob = F.softmax(logp, dim=-1)

	return offsets, t_true, t_pred, valid, prob, dt_sec


def main():
	B = 1
	offsets, t_true, t_pred, valid, prob, dt_sec = _make_synthetic(B=B)

	conf_floor = 0.2
	conf_power = 0.5
	w_conf_in = trace_confidence_from_prob(
		prob=prob,
		floor=conf_floor,
		power=conf_power,
	).to(t_pred)

	# IRLS(w_conf を直接渡す設計に変更)
	trend_t_i, trend_s_i, v_i, w_conf_used, covered = robust_linear_trend(
		offsets,
		t_pred,
		valid,
		w_conf=w_conf_in,
		section_len=128,
		stride=64,
		iters=3,
		vmin=300.0,
		vmax=6000.0,
		sort_offsets=True,
		use_taper=True,
	)

	# RANSAC(同様に w_conf を渡す)
	trend_t_r, trend_s_r, v_r, _, _ = robust_linear_trend_sections_ransac(
		offsets,
		t_pred,
		valid,
		w_conf=w_conf_in,
		section_len=128,
		stride=64,
		vmin=300.0,
		vmax=6000.0,
		ransac_trials=32,
		ransac_pack=16,
		refine_irls_iters=1,
		ransac_tau=2.0,
		ransac_abs_ms=15.0,
		sample_weighted=True,
		use_inlier_blend=False,
		sort_offsets=True,
	)

	b = 0  # 可視化はバッチ0を表示
	x = offsets[b].cpu().numpy()
	y_true = t_true[b].cpu().numpy()
	y_pred = t_pred[b].cpu().numpy()
	y_i = trend_t_i[b].cpu().numpy()
	y_r = trend_t_r[b].cpu().numpy()
	vtrue = np.full_like(y_true, 2000.0)
	vi = v_i[b].cpu().numpy()
	vr = v_r[b].cpu().numpy()
	wc = w_conf_used[b].cpu().numpy()
	cov = covered[b].cpu().numpy()

	# 1) 到達時刻 t vs オフセット x
	plt.figure()
	plt.title('Arrival time vs offset (points: predicted, lines: trends)')
	plt.xlabel('offset [m]')
	plt.ylabel('time [s]')
	plt.scatter(x, y_pred, s=8, label='predicted t')
	plt.plot(x, y_true, label='true trend')
	plt.plot(x, y_i, label='IRLS trend')
	plt.plot(x, y_r, label='RANSAC trend')
	plt.legend()
	plt.grid(True)

	# 2) 速度(v_trend) vs trace index
	plt.figure()
	plt.title('Trend velocity vs trace index')
	plt.xlabel('trace index')
	plt.ylabel('velocity [m/s]')
	plt.plot(vtrue, label='true v')
	plt.plot(vi, label='IRLS v')
	plt.plot(vr, label='RANSAC v')
	plt.legend()
	plt.grid(True)

	# 3) 信頼度 w_conf とカバレッジ
	plt.figure()
	plt.title('Per-trace confidence and coverage')
	plt.xlabel('trace index')
	plt.ylabel('w_conf / covered')
	plt.plot(wc, label='w_conf')
	plt.plot(cov.astype(float), label='covered (0/1)')
	plt.legend()
	plt.grid(True)

	plt.show()


if __name__ == '__main__':
	main()
