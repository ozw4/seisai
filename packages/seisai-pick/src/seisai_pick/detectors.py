import numpy as np
from numba import njit

from .stalta import stalta_1d


def _ms_to_samples(dt_sec: float, ms: float) -> int:
	"""Ms をサンプル数へ変換（最低1）。"""
	if dt_sec <= 0.0:
		raise ValueError('dt_sec must be > 0')
	n = int(round(ms / (dt_sec * 1e3)))
	return max(n, 1)


@njit(cache=True, fastmath=True)
def _picks_hist_from_R(
	R: np.ndarray,
	thr_on: float,
	thr_off: float,
	min_on_len: int,
	refr_len: int,
	hist: np.ndarray,
) -> None:
	"""STALTA時系列 R から「初動run開始」を複数抽出し、hist[t] をインクリメントして貯める。
	- 連続 on 長が min_on_len に達した時点の run開始インデックスを pick とする
	- pick 直後は refr_len サンプル分スキップ（リフラクトリ）
	- ヒステリシス: on条件は R>=thr_on、offリセットは R<=thr_off（thr_on>=thr_off を仮定）
	"""
	T = R.size
	if T == 0:
		return
	if thr_on < thr_off:
		raise ValueError('thr_on must be >= thr_off')
	if min_on_len < 1 or refr_len < 1:
		raise ValueError('min_on_len and refr_len must be >= 1')

	t = 0
	run = 0
	while t < T:
		v = R[t]
		if v >= thr_on:
			run += 1
			if run >= min_on_len:
				t0 = t - (run - 1)  # run開始
				if 0 <= t0 < T:
					hist[t0] += 1
				# リフラクトリ突入
				t = t0 + refr_len
				run = 0
				continue
		elif v <= thr_off:
			run = 0
		# thr_off < v < thr_on のときは run を保持（ヒステリシス）
		t += 1


@njit(cache=True, fastmath=True)
def _stalta_pick_hist(
	x_ht: np.ndarray,
	ns: int,
	nl: int,
	eps: float,
	thr_on: float,
	thr_off: float,
	min_on_len: int,
	refr_len: int,
) -> np.ndarray:
	"""(H,T) の各トレースに対して STALTA→初動run抽出→ヒスト加算を行い、hist[T] を返す。"""
	H, T = x_ht.shape
	hist = np.zeros(T, dtype=np.int32)
	for h in range(H):
		R = stalta_1d(x_ht[h], ns, nl, eps)
		_picks_hist_from_R(R, thr_on, thr_off, min_on_len, refr_len, hist)
	return hist


@njit(cache=True, fastmath=True)
def _sliding_sum_same(hist: np.ndarray, win: int) -> np.ndarray:
	"""非巡回・端部寄せの移動和（長さwin）。中心寄せではなく「可能な限り左寄せ」する。
	例: t を中心にせず、区間 [t, t+win) を優先的に取る（端で範囲補正）。
	"""
	T = hist.size
	win = max(win, 1)
	cs = np.zeros(T + 1, dtype=np.int32)
	for i in range(T):
		cs[i + 1] = cs[i] + hist[i]
	out = np.zeros(T, dtype=np.int32)
	half = win // 2
	for t in range(T):
		a = t - half
		a = max(a, 0)
		b = a + win
		if b > T:
			b = T
			a = b - win
			a = max(a, 0)
		out[t] = cs[b] - cs[a]
	return out


def detect_event_pick_cluster(
	x_ht: np.ndarray,
	dt_sec: float,
	*,
	sta_ms: float = 10.0,
	lta_ms: float = 200.0,
	thr_on: float = 3.0,
	thr_off: float = 1.5,
	min_on_ms: float = 6.0,
	refr_ms: float = 80.0,
	win_ms: float = 30.0,
	min_traces: int = 8,
	eps: float = 1e-12,
) -> tuple[bool, np.ndarray, np.ndarray]:
	"""リフラクトリ付き「初動候補クラスタ」方式。
	戻り値: (is_event, pick_hist[T], cluster_counts[T])

	- pick_hist[t]: 各トレースで抽出された初動run開始のヒスト（同一tに複数あれば合算）
	- cluster_counts[t]: 窓long=win_msの移動和（時刻t近傍のクラスタ本数）
	- is_event: max(cluster_counts) >= min_traces
	"""
	if x_ht.ndim != 2:
		raise ValueError(f'x_ht must be (H,T), got {x_ht.shape}')
	if dt_sec <= 0.0:
		raise ValueError('dt_sec must be > 0')
	H, T = x_ht.shape
	if H == 0 or T == 0:
		raise ValueError('empty input')

	ns = _ms_to_samples(dt_sec, sta_ms)
	nl = max(ns + 1, _ms_to_samples(dt_sec, lta_ms))
	min_on_len = _ms_to_samples(dt_sec, min_on_ms)
	refr_len = _ms_to_samples(dt_sec, refr_ms)
	win = _ms_to_samples(dt_sec, win_ms)

	pick_hist = _stalta_pick_hist(
		x_ht.astype(np.float64, copy=False),
		ns,
		nl,
		float(eps),
		float(thr_on),
		float(thr_off),
		int(min_on_len),
		int(refr_len),
	)
	cluster = _sliding_sum_same(pick_hist, int(win))
	is_event = int(cluster.max()) >= int(min_traces)
	return bool(is_event), pick_hist, cluster


@njit(cache=True, fastmath=True)
def _stalta_majority_counts(
	x_ht: np.ndarray, ns: int, nl: int, eps: float, thr: float
) -> np.ndarray:
	"""各トレースで STALTA→thr 判定。時刻 t ごとの合格本数 (T,) を返す。x_ht: (H,T)"""
	H, T = x_ht.shape
	counts = np.zeros(T, dtype=np.int32)
	for h in range(H):
		R = stalta_1d(x_ht[h], ns, nl, eps)
		for t in range(T):
			if R[t] >= thr:
				counts[t] += 1
	return counts


@njit(cache=True, fastmath=True)
def _has_run_geq(counts: np.ndarray, min_count: int, min_len: int) -> bool:
	"""counts[t] >= min_count が min_len サンプル以上 連続 なら True"""
	run = 0
	T = counts.size
	for t in range(T):
		if counts[t] >= min_count:
			run += 1
			if run >= min_len:
				return True
		else:
			run = 0
	return False


def detect_event_stalta_majority(
	x_ht: np.ndarray,
	dt_sec: float,
	*,
	sta_ms: float = 10.0,
	lta_ms: float = 200.0,
	thr: float = 2.5,
	min_traces: int = 8,
	min_duration_ms: float = 20.0,
	eps: float = 1e-12,
) -> tuple[bool, np.ndarray]:
	"""STALTA + 多数決 + 連続長 でイベント有無を返す。
	戻り値: (is_event, counts[T])  ※counts[t]はその時刻の「閾値以上トレース本数」

	前提:
	- x_htは (H,T) の実数配列（float32/float64）
	- dt_sec > 0
	- 0 < sta_ms < lta_ms
	- thr > 0, min_traces >= 1
	"""
	if x_ht.ndim != 2:
		raise ValueError(f'x_ht must be (H,T), got {x_ht.shape}')
	if dt_sec <= 0.0:
		raise ValueError('dt_sec must be > 0')
	if sta_ms <= 0.0 or lta_ms <= sta_ms:
		raise ValueError('require 0 < sta_ms < lta_ms')
	if thr <= 0.0:
		raise ValueError('thr must be > 0')
	if min_traces < 1:
		raise ValueError('min_traces must be >= 1')

	ns = _ms_to_samples(dt_sec, sta_ms)
	nl = max(ns + 1, _ms_to_samples(dt_sec, lta_ms))
	min_len = _ms_to_samples(dt_sec, min_duration_ms)

	counts = _stalta_majority_counts(
		x_ht.astype(np.float64, copy=False), ns, nl, float(eps), float(thr)
	)
	is_event = _has_run_geq(counts, int(min_traces), int(min_len))
	return bool(is_event), counts
