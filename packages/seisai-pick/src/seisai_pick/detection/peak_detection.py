import numpy as np
import torch
import torch.nn.functional as F
from seisai_utils.validator import validate_numpy


def detect_event_peaks(
    S_t: np.ndarray,
    min_score: float,
    min_distance: int,
    smooth_window: int = 1,
) -> np.ndarray:
    """Step2:
    Torch の avg_pool1d と max_pool1d を使って、1D スコア列 S_t から
    イベント候補ピークの時刻インデックスを抽出する。.

    平滑化は smooth_window > 1 のとき replicate pad + avg_pool1d の移動平均で
    長さ T を維持し、ピーク検出は抑制窓 (2*min_distance+1) の max_pool1d による
    1D NMS を使う。plateau(同値最大の連続)は開始点のみ採用する。

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
    if smooth_window < 1:
        msg = 'smooth_window must be >= 1'
        raise ValueError(msg)
    if __debug__ and not np.isfinite(S_t).all():
        msg = 'S_t must contain only finite values'
        raise ValueError(msg)

    T = S_t.size
    if T == 0:
        return np.empty(0, dtype=np.int64)

    t = torch.as_tensor(S_t, dtype=torch.float32).view(1, 1, T)

    if smooth_window > 1:
        pad_left = smooth_window // 2
        pad_right = smooth_window - 1 - pad_left
        t = F.pad(t, (pad_left, pad_right), mode='replicate')
        t = F.avg_pool1d(t, kernel_size=smooth_window, stride=1)

    S = t.view(T)

    k = 2 * min_distance + 1
    t_nms = F.pad(S.view(1, 1, T), (min_distance, min_distance), mode='replicate')
    Smax = F.max_pool1d(t_nms, kernel_size=k, stride=1).view(T)

    peak_mask = (S >= float(min_score)) & (S == Smax)

    S_left = torch.empty_like(S)
    S_left[0] = -torch.inf
    if T > 1:
        S_left[1:] = S[:-1]
    peak_mask = peak_mask & (S > S_left)

    if min_distance == 0:
        S_right = torch.empty_like(S)
        S_right[-1] = -torch.inf
        if T > 1:
            S_right[:-1] = S[1:]
        peak_mask = peak_mask & (S >= S_right)

    candidates = torch.nonzero(peak_mask, as_tuple=False).flatten()
    if candidates.numel() == 0:
        return np.empty(0, dtype=np.int64)

    candidate_indices = candidates.cpu().numpy().astype(np.int64, copy=False)
    candidate_scores = S[candidates].cpu().numpy()

    order = np.lexsort((candidate_indices, -candidate_scores))
    sorted_candidates = candidate_indices[order]

    if sorted_candidates.size == 1:
        return sorted_candidates.astype(np.int64, copy=False)

    selected: list[int] = []
    for idx in sorted_candidates:
        idx_i = int(idx)
        too_close = False
        for s in selected:
            if abs(idx_i - s) <= min_distance:
                too_close = True
                break
        if not too_close:
            selected.append(idx_i)

    if not selected:
        return np.empty(0, dtype=np.int64)

    return np.sort(np.asarray(selected, dtype=np.int64))
