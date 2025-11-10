# %%
import matplotlib.pyplot as plt
import numpy as np
from seisai_pick.detection.compute_time_support import (
	compute_time_support_from_probabilities,
)
from seisai_pick.detection.detection_pipeline import run_event_detection_pipeline
from seisai_pick.detection.peak_detection import detect_event_peaks

# ==============================
# 合成データ生成と実行例
# ==============================


def _make_synthetic_probabilities(
	H: int,
	T: int,
	*,
	dt_sec: float,
	event_secs: list[float],
	sigma_sec: float,
	slopes_samp_per_ch: list[float],
	amps: list[float],
	noise_level: float = 0.01,
	seed: int = 0,
) -> np.ndarray:
	"""合成 p_ht を作る:
	- イベント k ごとに、中心 t0_k（秒）・振幅 amp_k・時間幅 sigma_sec
	- チャネルごとに線形 moveout（samples/ch）= slopes[k] を適用
	- 各イベントを和して [0,1] にクリップ
	"""
	if not (len(event_secs) == len(slopes_samp_per_ch) == len(amps)):
		raise ValueError('event_secs, slopes_samp_per_ch, amps must have same length')
	rng = np.random.default_rng(seed)
	t_idx = np.arange(T, dtype=np.float64)
	p = np.zeros((H, T), dtype=np.float64)
	sigma = sigma_sec / dt_sec  # samples

	for k, (t0_sec, slope, amp) in enumerate(
		zip(event_secs, slopes_samp_per_ch, amps, strict=False)
	):
		mu = t0_sec / dt_sec
		for h in range(H):
			shift = slope * h  # samples
			p[h] += amp * np.exp(-0.5 * ((t_idx - (mu + shift)) / sigma) ** 2)

	# 0-1 へ正規化（最大1に抑えつつ軽いノイズ）
	p += noise_level * rng.random((H, T))
	p = np.clip(p, 0.0, 1.0)
	return p


if __name__ == '__main__':
	# ---- 合成データ条件 ----
	dt_sec = 0.01  # 100 Hz
	H = 24
	T = 6000  # 60 秒間
	# イベント: 中心時刻（秒）, moveout の傾き（samples/チャネル）, 振幅
	event_secs = [10.0, 30.0, 45.0]
	slopes = [20, -0.2, 0.0]  # 遅い右上がり／やや速い左上がり／ほぼ水平
	amps = [0.9, 0.7, 0.6]
	sigma_sec = 0.50

	p_ht = _make_synthetic_probabilities(
		H,
		T,
		dt_sec=dt_sec,
		event_secs=event_secs,
		sigma_sec=sigma_sec,
		slopes_samp_per_ch=slopes,
		amps=amps,
		noise_level=1,
		seed=42,
	)

	# ---- Step1 の窓（Δ）を秒 → サンプルに変換 ----
	half_window_sec = 0.05
	half_window = int(round(half_window_sec / dt_sec))

	# ---- Step2 のパラメータ（秒 → サンプル） ----
	min_score = 0.6
	min_distance_sec = 5.0
	min_distance = int(round(min_distance_sec / dt_sec))
	smooth_window_sec = 1
	smooth_window = max(1, int(round(smooth_window_sec / dt_sec)))

	# ---- パイプライン実行 ----
	S_fn = lambda x: compute_time_support_from_probabilities(x, half_window=half_window)
	P_fn = lambda S: detect_event_peaks(
		S,
		min_score=min_score,
		min_distance=min_distance,
		smooth_window=smooth_window,
	)

	peak_indices, S_t, event_scores = run_event_detection_pipeline(
		p_ht,
		time_support_fn=S_fn,
		peak_detector_fn=P_fn,
	)

	# ---- 結果表示（秒に変換）----
	t_peaks_sec = peak_indices * dt_sec
	print('ground truth event times (s):', event_secs)
	print(f'Detected {peak_indices.size} events')
	for k, (t_sec, score) in enumerate(zip(t_peaks_sec, event_scores, strict=False)):
		print(f'  #{k:02d}  t={t_sec:6.2f}s  score={score:0.3f}')

	# ---- 可視化 ----
	time_sec = np.arange(T, dtype=np.float64) * dt_sec

	fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

	# 1段目: p_ht のヒートマップ
	im = axes[0].imshow(
		p_ht,
		aspect='auto',
		origin='lower',
		extent=(time_sec[0], time_sec[-1], 0, H),
		interpolation='None',
	)
	fig.colorbar(im, ax=axes[0], label='probability')
	axes[0].set_ylabel('channel index')
	axes[0].set_title('Synthetic onset probabilities p_ht')

	# ground truth イベント時刻の縦線
	for t0 in event_secs:
		axes[0].axvline(t0, linestyle='--', linewidth=1)

	# 2段目: S_t と検出ピーク
	axes[1].plot(time_sec, S_t, label='S(t)')
	if peak_indices.size > 0:
		axes[1].scatter(
			t_peaks_sec,
			S_t[peak_indices],
			marker='o',
			label='detected peaks',
			color='red',
		)
	for t0 in event_secs:
		axes[1].axvline(t0, linestyle='--', linewidth=1)

	axes[1].set_xlabel('time (s)')
	axes[1].set_ylabel('time support S(t)')
	axes[1].set_title('Time support and detected events')
	axes[1].legend()

	plt.tight_layout()
	plt.show()
