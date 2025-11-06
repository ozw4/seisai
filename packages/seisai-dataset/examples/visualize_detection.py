# packages/seisai-dataset/examples/visalize_detection.py
# %%
from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segyio

# seisai-dataset（I/Oとサンプリング）
from seisai_dataset.config import LoaderConfig, TraceSubsetSamplerConfig

# 判定ロジックを共通化
from seisai_dataset.noise_decider import EventDetectConfig, decide_noise
from seisai_dataset.trace_subset_preproc import TraceSubsetLoader
from seisai_dataset.trace_subset_sampler import TraceSubsetSampler

# transforms（必要に応じて有効化）
from seisai_transforms.augment import (
	DeterministicCropOrPad,
	PerTraceStandardize,
	ViewCompose,
)
from seisai_transforms.signal_ops import standardize_per_trace


# ---------------------------- 可視化ヘルパ ----------------------------
def _plot_gather_with_series(
	x: np.ndarray,
	*,
	dt_sec: float,
	series: dict[str, np.ndarray] | None = None,
	title: str = '',
	eps: float = 1e-10,
) -> None:
	"""- 表示前に per-trace z-score 標準化（standardize_per_trace）を適用
	- x軸は [ms]。imshow の extent も ms に合わせる
	- 表示クリップは vmin=-3, vmax=+3
	"""
	if x.ndim != 2:
		raise ValueError(f'x must be (H,T), got {x.shape}')
	H, T = x.shape
	if dt_sec <= 0.0:
		raise ValueError('dt_sec must be > 0')

	# 1) トレース毎 z-score（seisai-transforms 実装を利用）
	xz = standardize_per_trace(x.astype(np.float32, copy=False), eps=eps)

	# 2) 時間軸 [ms]
	scale = 1_000.0
	t_ms = np.arange(T, dtype=np.float64) * float(dt_sec) * scale

	# 3) 描画（画像も series も ms で統一）
	fig = plt.figure(figsize=(10, 6))
	ax0 = fig.add_axes([0.10, 0.35, 0.85, 0.60])
	im = ax0.imshow(
		xz,
		aspect='auto',
		origin='lower',
		cmap='seismic',
		extent=[0.0, float(dt_sec) * scale * T, 0.0, float(H)],
		vmin=-3.0,
		vmax=+3.0,
	)
	ax0.set_ylabel('Trace')
	ax0.set_title(title)
	fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)

	ax1 = fig.add_axes([0.10, 0.10, 0.85, 0.20], sharex=ax0)
	if series:
		for name, y in series.items():
			if y is None:
				continue
			ax1.plot(t_ms, y, label=name)
		ax1.legend(loc='upper right', fontsize=8)
	ax1.set_xlabel('Time [ms]')
	ax1.set_ylabel('value')
	plt.show()


# ---------------------------- SEG-Y file-info 構築 ----------------------------
def _build_file_info(
	segy_path: str,
	ffid_byte,
	chno_byte,
	cmp_byte,
	cache_dir: str | None,
) -> dict:
	meta = load_headers_with_cache(
		segy_path,
		ffid_byte,
		chno_byte,
		cmp_byte,
		cache_dir=cache_dir,
		rebuild=False,
	)
	ffid_values = meta['ffid_values']
	chno_values = meta['chno_values']
	cmp_values = meta['cmp_values']
	dt_us = int(meta['dt_us'])
	dt_sec = dt_us * 1e-6
	n_traces = int(meta['n_traces'])
	n_samples = int(meta['n_samples'])

	f = segyio.open(segy_path, 'r', ignore_geometry=True)  # keep open for mmap
	mmap = f.trace.raw[:]

	def build_index_map(arr: np.ndarray | None) -> dict[int, np.ndarray] | None:
		if arr is None:
			return None
		a = np.asarray(arr)
		uniq, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
		order = np.argsort(inv, kind='mergesort')
		splits = np.cumsum(counts)[:-1]
		groups = np.split(order, splits)
		return {int(k): g.astype(np.int32) for k, g in zip(uniq, groups, strict=False)}

	ffid_key_to_indices = build_index_map(ffid_values)
	chno_key_to_indices = build_index_map(chno_values)
	cmp_key_to_indices = (
		build_index_map(cmp_values) if (cmp_values is not None) else None
	)

	# ★ TraceSubsetSampler.draw が参照する *_unique_keys を必ず入れる（KeyError対策）
	ffid_unique_keys = (
		list(ffid_key_to_indices.keys()) if ffid_key_to_indices is not None else None
	)
	chno_unique_keys = (
		list(chno_key_to_indices.keys()) if chno_key_to_indices is not None else None
	)
	cmp_unique_keys = (
		list(cmp_key_to_indices.keys()) if cmp_key_to_indices is not None else None
	)

	return {
		'path': segy_path,
		'mmap': mmap,
		'segy_obj': f,
		'dt_sec': dt_sec,
		'n_traces': n_traces,
		'n_samples': n_samples,
		'ffid_values': ffid_values,
		'chno_values': chno_values,
		'cmp_values': cmp_values
		if isinstance(cmp_values, np.ndarray) and cmp_values.size > 0
		else None,
		'ffid_key_to_indices': ffid_key_to_indices,
		'chno_key_to_indices': chno_key_to_indices,
		'cmp_key_to_indices': cmp_key_to_indices,
		'ffid_unique_keys': ffid_unique_keys,
		'chno_unique_keys': chno_unique_keys,
		'cmp_unique_keys': cmp_unique_keys,
	}


# ---------------------------- 収集ループ ----------------------------
def collect_examples(
	*,
	segy_files: list[str],
	loader_cfg: LoaderConfig,
	sampler_cfg: TraceSubsetSamplerConfig,
	detect_cfg: EventDetectConfig,
	transform,  # 例: ViewCompose([DeterministicCropOrPad(...), PerTraceStandardize()])
	cache_dir: str | None,
	ffid_byte=segyio.TraceField.FieldRecord,
	chno_byte=segyio.TraceField.TraceNumber,
	cmp_byte=segyio.TraceField.CDP,
	want_per_class: int = 2,
	max_trials: int = 5000,
	seed: int = 0,
):
	"""A棄却（reject_A）/B棄却（reject_B）/ノイズ（noise）の各クラスで want_per_class 件収集。
	未達の場合は RuntimeError に不足カテゴリと観測内訳を含めて失敗させる。
	"""
	if want_per_class < 1:
		raise ValueError('want_per_class must be >= 1')

	rng = np.random.default_rng(seed)
	sampler = TraceSubsetSampler(sampler_cfg)
	loader = TraceSubsetLoader(loader_cfg)
	file_infos = [
		_build_file_info(p, ffid_byte, chno_byte, cmp_byte, cache_dir)
		for p in segy_files
	]

	got_A, got_B, got_N = [], [], []
	# 観測内訳（試行全体で何件ずつ出会ったか）
	observed = {'reject_A': 0, 'reject_B': 0, 'noise': 0, 'other': 0}
	trials = 0

	while (
		len(got_A) < want_per_class
		or len(got_B) < want_per_class
		or len(got_N) < want_per_class
	) and trials < max_trials:
		trials += 1
		info = file_infos[int(rng.integers(0, len(file_infos)))]
		mmap = info['mmap']
		dt_sec = float(info['dt_sec'])

		d = sampler.draw(info, py_random=random.Random(int(rng.integers(0, 2**31 - 1))))
		indices = d['indices'] if 'indices' in d else d['subset_indices']

		x = loader.load(mmap, indices).astype(np.float32, copy=False)
		if transform is not None:
			out = transform(x)
			if isinstance(out, tuple) and len(out) == 2:
				x, _ = out
			else:
				x = out
			if not isinstance(x, np.ndarray) or x.ndim != 2:
				raise ValueError(
					'transform must return 2D numpy array or (array, meta)'
				)

		dec = decide_noise(x, dt_sec, detect_cfg)

		reason = dec.reason if isinstance(dec.reason, str) else 'other'
		if reason not in observed:
			reason = 'other'
		observed[reason] += 1

		if reason == 'reject_B' and len(got_B) < want_per_class:
			got_B.append(
				{
					'x': x.copy(),
					'dt_sec': dt_sec,
					'pick_hist': dec.pick_hist,
					'cluster': dec.cluster,
				}
			)
			continue

		if reason == 'reject_A' and len(got_A) < want_per_class:
			got_A.append({'x': x.copy(), 'dt_sec': dt_sec, 'counts': dec.counts})
			continue

		if reason == 'noise' and len(got_N) < want_per_class:
			got_N.append(
				{
					'x': x.copy(),
					'dt_sec': dt_sec,
					'counts': dec.counts,
					'pick_hist': dec.pick_hist,
					'cluster': dec.cluster,
				}
			)

	# 未達の可読メッセージを作る
	need_A = want_per_class - len(got_A)
	need_B = want_per_class - len(got_B)
	need_N = want_per_class - len(got_N)
	if need_A > 0 or need_B > 0 or need_N > 0:
		parts = []
		if need_A > 0:
			parts.append(f'A(reject_A): {len(got_A)}/{want_per_class} (-{need_A})')
		if need_B > 0:
			parts.append(f'B(reject_B): {len(got_B)}/{want_per_class} (-{need_B})')
		if need_N > 0:
			parts.append(f'Noise: {len(got_N)}/{want_per_class} (-{need_N})')

		msg = (
			'収集件数が不足しました: '
			+ ', '.join(parts)
			+ f' | 試行回数={trials}/{max_trials} | 観測内訳='
			+ f'A:{observed["reject_A"]}, B:{observed["reject_B"]}, noise:{observed["noise"]}, other:{observed["other"]}'
			+ ' | 対処: max_trials↑ / しきい値や本数・連続長の緩和/強化を検討してください。'
		)
		raise RuntimeError(msg)

	for info in file_infos:
		info['segy_obj'].close()

	return got_A, got_B, got_N


# ---------------------------- Main（例） ----------------------------
if __name__ == '__main__':
	# ====== 編集パラメータ ======
	segy_files = [
		'/workspace/example_data/merged_F1.sgy',
	]
	for p in segy_files:
		if not Path(p).exists():
			raise FileNotFoundError(f'SEGY not found: {p}')

	target_len = 2048
	subset_traces = 128
	cache_dir = '../.segy_hdr_cache'
	want_per_class = 2
	# ============================

	transform = ViewCompose([DeterministicCropOrPad(target_len), PerTraceStandardize()])

	loader_cfg = LoaderConfig(pad_traces_to=int(subset_traces))
	sampler_cfg = TraceSubsetSamplerConfig(
		primary_keys=('ffid',),
		use_superwindow=False,
		sw_halfspan=0,
		sw_prob=0.0,
		valid=True,
		subset_traces=int(subset_traces),
	)
	detect_cfg = EventDetectConfig()

	got_A, got_B, got_N = collect_examples(
		segy_files=[str(p) for p in segy_files],
		loader_cfg=loader_cfg,
		sampler_cfg=sampler_cfg,
		detect_cfg=detect_cfg,
		transform=transform,
		cache_dir=cache_dir,
		want_per_class=want_per_class,
		max_trials=1000,
		seed=0,
	)

	for i, s in enumerate(got_A):
		_plot_gather_with_series(
			s['x'],
			dt_sec=s['dt_sec'],
			series={'A_counts': s['counts']},
			title=f'A reject (majority+duration) #{i}',
		)

	for i, s in enumerate(got_B):
		_plot_gather_with_series(
			s['x'],
			dt_sec=s['dt_sec'],
			series={'B_pick_hist': s['pick_hist'], 'B_cluster': s['cluster']},
			title=f'B reject (pick-cluster) #{i}',
		)

	for i, s in enumerate(got_N):
		series = {}
		if s.get('counts') is not None:
			series['A_counts'] = s['counts']
		if s.get('cluster') is not None:
			series['B_cluster'] = s['cluster']
		_plot_gather_with_series(
			s['x'],
			dt_sec=s['dt_sec'],
			series=series,
			title=f'Noise accepted #{i}',
		)
