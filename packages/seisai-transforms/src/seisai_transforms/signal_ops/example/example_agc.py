# %%
# example_agc_vs_robust_agc_numpy.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# あなたの実装に合わせてimportパスを調整してください
from seisai_transforms.signal_ops.scaling.agc import (
	agc_np,  # 一般的なRMSベースAGC
)  # ロバスト統計ベースAGC(MAD/IQR)
from seisai_transforms.signal_ops.scaling.robust_agc import (
	robust_agc_np,
)


def make_composite_wave(
	W: int, fs: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
	"""合成波形(1D)を生成：低周波+高周波+トレンド+振幅変化+スパイク+ノイズ
	返り値: t(s), x
	"""
	t = np.arange(W) / float(fs)
	s_low = 0.8 * np.sin(2 * np.pi * 3.0 * t)
	s_high = 0.3 * np.sin(2 * np.pi * 40.0 * t)
	trend = 0.05 * (t - t.mean())
	amp_env = np.ones_like(t)
	amp_env[(t >= 1.5) & (t < 2.5)] = 0.3  # 減衰区間
	amp_env[(t >= 3.5) & (t < 4.5)] = 4.0  # 増幅区間
	spikes = np.zeros_like(t)
	idx = rng.integers(0, W, size=14)
	spikes[idx] = rng.uniform(2.0, 4.0, size=idx.size) * rng.choice(
		[1.0, -1.0], size=idx.size
	)
	noise = 0.05 * rng.standard_normal(W)

	x = amp_env * (s_low + s_high) + trend + spikes + noise
	return t, x.astype(np.float32, copy=False)


def main():
	fs = 1000  # Hz
	dur = 6.0  # sec
	W = int(fs * dur)
	rng = np.random.default_rng(42)

	t, x = make_composite_wave(W, fs, rng)

	# --- AGC(RMSベース) ---
	y_agc, g_agc = agc_np(
		x,
		win=256,
		target_rms=0.3,
		clamp_db=(-20.0, 20.0),
		causal=True,
		eps=1e-8,
		return_gain=True,
	)

	# --- Robust AGC(MAD/IQRベース, 推奨: MAD, gamma=0.75)---
	y_rob, meta = robust_agc_np(
		x,
		win=600,
		hop=None,  # win//4
		method='mad',  # or "iqr"
		gamma=0.75,
		eps=1e-8,
		clamp_pct=(5.0, 95.0),
		causal=False,
		chunk=20000,
		return_meta=True,
	)

	# --- 可視化 ---
	fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
	axs[0].plot(t, x)
	axs[0].set_title('Original')
	axs[0].set_ylabel('Amplitude')

	axs[1].plot(t, y_agc)
	axs[1].set_title('AGC (RMS-based)')
	axs[1].set_ylabel('Amplitude')

	axs[2].plot(t, y_rob)
	axs[2].set_title('Robust AGC (MAD-based, gamma=0.75)')
	axs[2].set_ylabel('Amplitude')
	axs[2].set_xlabel('Time [s]')

	plt.tight_layout()
	plt.show()

	# --- 参考: AGCのゲイン可視化(dB) ---
	# Robust AGCは μ 引き算+σ^γ で正規化するため、純粋な“ゲイン”は定義しない。
	fig2, ax2 = plt.subplots(1, 1, figsize=(12, 3), sharex=True)
	gain_db = 20.0 * np.log10(np.maximum(g_agc, 1e-12))
	ax2.plot(t, gain_db)
	ax2.set_title('AGC Gain (dB)')
	ax2.set_ylabel('dB')
	ax2.set_xlabel('Time [s]')
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	main()
