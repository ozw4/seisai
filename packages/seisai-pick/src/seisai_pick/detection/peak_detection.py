import numpy as np
from seisai_transforms.signal_ops.smoothing.smooth import smooth_1d_np
from seisai_utils.validator import validate_numpy


def detect_event_peaks(
    S_t: np.ndarray,
    min_score: float,
    min_distance: int,
    smooth_window: int = 1,
) -> np.ndarray:
    """Step2:
    1D スコア列 S_t からイベント候補ピークの時刻インデックスを抽出する。

    パラメータ
    ----------
    S_t : np.ndarray
            形状 (T,) のスコア列。
    min_score : float
            ピークとして採用するための最小スコア。
    min_distance : int
            採用するピーク間の最小距離(サンプル数)。
            この距離未満で競合する場合はスコアが高い方を優先する。
    smooth_window : int
            平滑化窓長(サンプル数)。1 なら平滑なし。

    戻り値
    -------
    peak_indices : np.ndarray
            形状 (K,) の int 配列。イベント候補ピークの時刻インデックス(昇順)。
    """
    validate_numpy(S_t, allowed_ndims=(1,), name='S_t')
    if min_distance < 0:
        msg = 'min_distance must be >= 0'
        raise ValueError(msg)
    if min_score < 0.0:
        msg = 'min_score must be >= 0.0'
        raise ValueError(msg)

    T = S_t.size
    S = smooth_1d_np(S_t, smooth_window)

    is_peak = np.zeros(T, dtype=bool)
    for t in range(T):
        v = S[t]
        if v < min_score:
            continue
        left = S[t - 1] if t > 0 else v
        right = S[t + 1] if t < T - 1 else v
        if v >= left and v >= right:
            is_peak[t] = True

    candidate_indices = np.nonzero(is_peak)[0]
    if candidate_indices.size == 0:
        return candidate_indices.astype(np.int64)

    scores = S[candidate_indices]
    order = np.argsort(scores)[::-1]  # スコア降順
    selected = []

    for idx in candidate_indices[order]:
        if not selected:
            selected.append(int(idx))
            continue
        too_close = False
        for s in selected:
            if abs(idx - s) <= min_distance:
                too_close = True
                break
        if not too_close:
            selected.append(int(idx))

    selected = np.array(sorted(selected), dtype=np.int64)
    return selected
