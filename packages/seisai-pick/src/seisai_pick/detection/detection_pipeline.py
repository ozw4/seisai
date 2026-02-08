from collections.abc import Callable

import numpy as np
from seisai_utils.validator import validate_numpy

# Step1: feature_ht(H,T) -> S_t(T,)
TimeSupportFn = Callable[[np.ndarray], np.ndarray]
# Step2: S_t(T,) -> peak_indices(K,)
PeakDetectorFn = Callable[[np.ndarray], np.ndarray]


def run_event_detection_pipeline(
    feature_ht: np.ndarray,
    *,
    time_support_fn: TimeSupportFn,
    peak_detector_fn: PeakDetectorFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step1+Step2 だけで完結するシンプルなイベント検出パイプライン。.

    入力:
    ------
    feature_ht : np.ndarray
            形状 (H, T) の特徴量行列。
            - 深層学習モデル: 初動確率 p_ht (0〜1) を想定
            - STALTA / AIC 等: 「大きいほどイベントらしい」スコアでも可
    time_support_fn : callable
            Step1 用関数。
            シグネチャ: time_support_fn(feature_ht: np.ndarray) -> np.ndarray(形状 (T,))
            例:
                    - compute_time_support_from_probabilities(p_ht, half_window=...)
                    - compute_time_support_from_pick_cluster(feat_ht, ...)
                    - compute_time_support_from_majority(feat_ht, ...)
    peak_detector_fn : callable
            Step2 用関数。
            シグネチャ: peak_detector_fn(S_t: np.ndarray) -> np.ndarray(形状 (K,), int)
            例:
                    - detect_event_peaks(S_t, min_score=..., min_distance=..., smooth_window=...)

    処理内容:
    ---------
    Step1:
            S_t = time_support_fn(feature_ht)
            - 時刻 t ごとの「イベント支持度」1D スコア列 S_t を計算する。

    Step2:
            peak_indices = peak_detector_fn(S_t)
            - S_t からイベント候補の代表時刻インデックスを抽出する。

    さらに:
            event_scores = S_t[peak_indices]
            - 各イベント候補の「確信度」として S_t のピーク値を用いる。

    戻り値:
    -------
    peak_indices : np.ndarray
            形状 (K,) の int64 配列。イベント候補の代表時刻インデックス。
    S_t : np.ndarray
            形状 (T,) の float64 配列。Step1 で得られた time support。
    event_scores : np.ndarray
            形状 (K,) の float64 配列。各イベント候補のスコア(S_t[peak_indices])。

    Raises
    ------
    TypeError, ValueError
            入力・各ステップの戻り値の型や形状が不正な場合。

    """
    # --- 入力検証 ---
    validate_numpy(feature_ht, allowed_ndims=(2,), name='feature_ht')
    H, T = feature_ht.shape
    if H == 0 or T == 0:
        msg = 'feature_ht must be non-empty'
        raise ValueError(msg)

    feature_ht = np.asarray(feature_ht, dtype=np.float64)

    # --- Step1: feature_ht -> S_t (T,) ---
    S_t = time_support_fn(feature_ht)
    validate_numpy(S_t, allowed_ndims=(1,), name='S_t')
    if S_t.size != T:
        msg = 'time_support_fn must return array of shape (T,)'
        raise ValueError(msg)

    S_t = np.asarray(S_t, dtype=np.float64)

    # --- Step2: S_t -> peak_indices (K,) ---
    peak_indices = peak_detector_fn(S_t)
    validate_numpy(peak_indices, allowed_ndims=(1,), name='peak_indices')

    if peak_indices.dtype.kind not in ('i', 'u'):
        msg = 'peak_detector_fn must return an integer array for peak_indices'
        raise TypeError(
            msg
        )

    peak_indices = np.asarray(peak_indices, dtype=np.int64)

    # --- イベント候補スコア: S_t のピーク値をそのまま用いる ---
    if peak_indices.size == 0:
        event_scores = np.zeros(0, dtype=np.float64)
    else:
        # 範囲チェック(バグ検知用)
        if peak_indices.min() < 0 or peak_indices.max() >= T:
            msg = 'peak_indices must be in [0, T-1]'
            raise ValueError(msg)
        event_scores = S_t[peak_indices].astype(np.float64)

    return peak_indices, S_t, event_scores
