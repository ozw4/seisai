# packages/seisai-dataset/src/seisai_dataset/noise_decider.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from seisai_pick.detectors import (
	detect_event_pick_cluster,
	detect_event_stalta_majority,
)
from seisai_transforms.signal_ops import compute_envelope


# ---- 設定 ----
@dataclass(frozen=True)
class EventDetectConfig:
	"""包絡→STALTA を用いたイベント検出設定。
	- A方式: 多数決 + 連続長
	- B方式: 初動候補クラスタ（ヒステリシス + リフラクトリ）
	"""

	# Preprocess
	use_envelope: bool = True

	# STA/LTA 共通
	sta_ms: float = 10.0
	lta_ms: float = 100.0
	eps: float = 1e-12

	# A) 多数決 + 連続長
	threshold: float = 4
	min_traces: int = 8
	min_duration_ms: float = 20.0

	# B) 初動候補クラスタ（ヒステリシス＋リフラクトリ）
	thr_on: float = 4.5
	thr_off: float = 1.5
	min_on_ms: float = 8.0
	refr_ms: float = 80.0
	win_ms: float = 30.0


# ---- 結果オブジェクト ----
DecisionReason = Literal['noise', 'reject_A', 'reject_B']


@dataclass(frozen=True)
class NoiseDecision:
	"""判定結果 + デバッグ用の時系列（可視化に使える）"""

	is_noise: bool
	reason: DecisionReason
	counts: np.ndarray | None = None  # A: 各時刻の「閾値以上トレース本数」(T,)
	pick_hist: np.ndarray | None = None  # B: 初動run開始ヒスト (T,)
	cluster: np.ndarray | None = None  # B: pick_hist の移動和 (T,)


# ---- 判定本体（Dataset / Examples で共通利用）----
def decide_noise(
	x: np.ndarray,  # (H, T) 実数
	dt_sec: float,  # サンプリング間隔 [s]
	cfg: EventDetectConfig,
) -> NoiseDecision:
	"""(H,T) 窓に対し、B→A の順でイベントを検出し、ノイズのみ True を返す。

	返却:
	NoiseDecision(
	is_noise=True/False,
	reason="noise" | "reject_A" | "reject_B",
	counts=..., pick_hist=..., cluster=...
	)
	"""
	if x.ndim != 2:
		raise ValueError(f'x must be 2D (H,T), got {x.shape}')
	if not (dt_sec > 0.0):
		raise ValueError('dt_sec must be > 0')
	if not (cfg.sta_ms > 0.0 and cfg.lta_ms > cfg.sta_ms):
		raise ValueError('require 0 < sta_ms < lta_ms')
	if cfg.threshold <= 0.0 or cfg.min_traces < 1:
		raise ValueError('threshold must be >0 and min_traces >=1')
	if cfg.thr_on < cfg.thr_off:
		raise ValueError('thr_on must be >= thr_off')

	# 1) 前処理（包絡は任意）
	sig = compute_envelope(x, axis=-1) if cfg.use_envelope else x
	M = int(cfg.min_traces) if cfg.min_traces >= 1 else 1

	# 2) 方式B：初動候補クラスタ（安全側のフィルタを先に）
	is_event_B, pick_hist, cluster = detect_event_pick_cluster(
		sig,
		dt_sec,
		sta_ms=cfg.sta_ms,
		lta_ms=cfg.lta_ms,
		thr_on=cfg.thr_on,
		thr_off=cfg.thr_off,
		min_on_ms=cfg.min_on_ms,
		refr_ms=cfg.refr_ms,
		win_ms=cfg.win_ms,
		min_traces=M,
		eps=cfg.eps,
	)
	if is_event_B:
		return NoiseDecision(False, 'reject_B', None, pick_hist, cluster)

	# 3) 方式A：多数決 + 連続長
	is_event_A, counts = detect_event_stalta_majority(
		sig,
		dt_sec,
		sta_ms=cfg.sta_ms,
		lta_ms=cfg.lta_ms,
		thr=cfg.threshold,
		min_traces=M,
		min_duration_ms=cfg.min_duration_ms,
		eps=cfg.eps,
	)
	if is_event_A:
		return NoiseDecision(False, 'reject_A', counts, None, None)

	# 4) ノイズ（どちらの検出器にも引っかからない）
	return NoiseDecision(True, 'noise', counts, pick_hist, cluster)


__all__ = [
	'DecisionReason',
	'EventDetectConfig',
	'NoiseDecision',
	'decide_noise',
]
