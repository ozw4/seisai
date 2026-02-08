import numpy as np
from seisai_utils.validator import validate_numpy


def compute_local_event_probabilities(
    p_ht: np.ndarray,
    half_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Step1:
    H×T の初動確率 p_ht から、時間窓 ±half_window 内で少なくとも1回イベントが起こる
    局所イベント確率 q_ht と、チャンネル平均スコア S_t を計算する。.

    パラメータ
    ----------
    p_ht : np.ndarray
        形状 (H, T) の初動確率。0〜1 を想定。
    half_window : int
        時刻 t を中心にした左右の窓半幅(サンプル数)。Δ に相当。

    戻り値
    -------
    q_ht : np.ndarray
        形状 (H, T)。q_ht[h, t] はトレース h の時刻 t 近傍(±half_window)で
        少なくとも1回イベントが起こる確率(Poisson 近似)。
    S_t : np.ndarray
        形状 (T,)。各時刻 t 近傍でイベントが見えているチャンネル割合の期待値。
        S_t[t] = mean_h q_ht[h, t]
    """
    validate_numpy(p_ht, allowed_ndims=(2,), name='p_ht')
    if half_window < 0:
        msg = 'half_window must be >= 0'
        raise ValueError(msg)

    H, T = p_ht.shape
    p_ht = np.asarray(p_ht, dtype=np.float64)

    if half_window == 0:
        m_ht = p_ht
    else:
        pad = half_window
        p_pad = np.pad(
            p_ht,
            pad_width=((0, 0), (pad, pad)),
            mode='constant',
            constant_values=0.0,
        )  # (H, T+2*pad)

        cs = np.cumsum(p_pad, axis=1)
        cs = np.concatenate(
            [np.zeros((H, 1), dtype=np.float64), cs],
            axis=1,
        )  # (H, T+2*pad+1)

        win = 2 * half_window + 1
        # 各 t について実質的に [t-half_window, t+half_window](端部はクリップ)の和
        m_ht = cs[:, win : win + T] - cs[:, :T]  # (H, T)

    q_ht = 1.0 - np.exp(-m_ht)  # Poisson 近似
    q_ht = np.clip(q_ht, 0.0, 1.0)
    S_t = q_ht.mean(axis=0)
    return q_ht, S_t


def smooth_1d(
    x: np.ndarray,
    window: int,
) -> np.ndarray:
    """1次元配列の移動平均による平滑化。.

    パラメータ
    ----------
    x_t : np.ndarray
        形状 (T,) の 1D 配列。
    window : int
        平滑化窓長(サンプル数)。1 以下ならコピーして返す。

    戻り値
    -------
    y_t : np.ndarray
        平滑化後の 1D 配列(形状 (T,))。
    """
    validate_numpy(x, allowed_ndims=(1,), name='x')
    if window <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    if window > x.size:
        msg = 'window must be <= length of x'
        raise ValueError(msg)

    x = np.asarray(x, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode='same')


def detect_event_peaks(
    S_t: np.ndarray,
    min_score: float,
    min_distance: int,
    smooth_window: int = 1,
) -> np.ndarray:
    """Step2:
    1D スコア列 S_t からイベント候補ピークの時刻インデックスを抽出する。.

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
    S = smooth_1d(S_t, smooth_window)

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

    return np.array(sorted(selected), dtype=np.int64)


def compute_window_event_probabilities(
    p_ht: np.ndarray,
    peak_indices: np.ndarray,
    win_left: int,
    win_right: int,
) -> np.ndarray:
    """Step3 (前半):
    各イベント候補ピーク k(時刻 peak_indices[k])に対して、
    [t - win_left, t + win_right] の窓内で少なくとも 1 回イベントが
    起こるトレースごとの確率 q_hk を計算する。.

    パラメータ
    ----------
    p_ht : np.ndarray
        形状 (H, T) の初動確率。
    peak_indices : np.ndarray
        形状 (K,) のイベント候補ピークの時刻インデックス。
    win_left : int
        ピークから左側の窓長(サンプル数)。
    win_right : int
        ピークから右側の窓長(サンプル数)。

    戻り値
    -------
    q_hk : np.ndarray
        形状 (H, K)。q_hk[h, k] はピーク k の窓内でトレース h に
        少なくとも 1 回イベントが起きる確率(Poisson 近似)。
    """
    validate_numpy(p_ht, allowed_ndims=(2,), name='p_ht')
    validate_numpy(peak_indices, allowed_ndims=(1,), name='peak_indices')
    if win_left < 0 or win_right < 0:
        msg = 'win_left and win_right must be >= 0'
        raise ValueError(msg)

    H, T = p_ht.shape
    K = int(peak_indices.size)

    p_ht = np.asarray(p_ht, dtype=np.float64)
    peak_indices = np.asarray(peak_indices, dtype=np.int64)

    cs = np.zeros((H, T + 1), dtype=np.float64)
    cs[:, 1:] = np.cumsum(p_ht, axis=1)

    q_hk = np.zeros((H, K), dtype=np.float64)
    for k in range(K):
        t0 = int(peak_indices[k])
        if t0 < 0 or t0 >= T:
            msg = 'peak index out of range'
            raise ValueError(msg)
        start = t0 - win_left
        end = t0 + win_right + 1  # [start, end)

        start = max(start, 0)
        end = min(end, T)

        if start >= end:
            q_hk[:, k] = 0.0
            continue

        m_h = cs[:, end] - cs[:, start]  # (H,)
        q_hk[:, k] = 1.0 - np.exp(-m_h)

    return np.clip(q_hk, 0.0, 1.0)


def compute_event_scores(
    q_hk: np.ndarray,
    weights_h: np.ndarray | None = None,
) -> np.ndarray:
    """Step3 (後半):
    トレースごとの局所イベント確率 q_hk からイベント全体の確信度 Score_k を計算する。.

    パラメータ
    ----------
    q_hk : np.ndarray
        形状 (H, K)。トレース h × ピーク k の局所イベント確率。
    weights_h : np.ndarray or None
        形状 (H,) のチャネル重み(SNR や距離など)。
        None の場合は一様重み(1/H)が使用される。
        重みは内部で正規化され、合計 1 になる。

    戻り値
    -------
    score_k : np.ndarray
        形状 (K,)。イベント k に対する全体の確信度。
        一様重みの場合は「イベントが見えているトレース割合の期待値」に相当。
    """
    validate_numpy(q_hk, allowed_ndims=(2,), name='q_hk')
    H, _K = q_hk.shape

    q_hk = np.asarray(q_hk, dtype=np.float64)
    if weights_h is None:
        weights = np.full(H, 1.0 / float(H), dtype=np.float64)
    else:
        validate_numpy(weights_h, allowed_ndims=(1,), name='weights_h')
        weights = np.asarray(weights_h, dtype=np.float64)
        if weights.size != H:
            msg = 'weights_h length must match H'
            raise ValueError(msg)
        s = float(weights.sum())
        if s <= 0.0:
            msg = 'sum of weights_h must be > 0'
            raise ValueError(msg)
        weights = weights / s

    score_k = weights @ q_hk
    return np.clip(score_k, 0.0, 1.0)


def detect_events_from_probabilities(
    p_ht: np.ndarray,
    *,
    half_window: int,
    smooth_window: int,
    min_score: float,
    min_peak_distance: int,
    peak_win_left: int,
    peak_win_right: int,
    weights_h: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """パイプライン全体:
    Step1〜3 の関数を組み合わせて、H×T の初動確率からイベント候補とスコアを検出する。.

    パラメータ
    ----------
    p_ht : np.ndarray
        形状 (H, T) の初動確率(Step0 の出力)。
    half_window : int
        Step1 で q_h(t) を計算する際の局所窓半幅(サンプル数)。
    smooth_window : int
        Step2 で S(t) を平滑化する移動平均窓長(サンプル数)。
    min_score : float
        Step2 のピーク検出で採用する最小スコア。
    min_peak_distance : int
        Step2 のピーク間最小距離(サンプル数)。
    peak_win_left : int
        Step3 で各ピークのイベント窓を取る際の左側の窓長(サンプル数)。
    peak_win_right : int
        Step3 で各ピークのイベント窓を取る際の右側の窓長(サンプル数)。
    weights_h : np.ndarray or None
        Step3 でイベントスコアを計算する際のチャネル重み(形状 (H,))。
        None の場合は一様重み。

    戻り値
    -------
    peak_indices : np.ndarray
        形状 (K,) のイベント候補ピークの時刻インデックス(昇順)。
    score_k : np.ndarray
        形状 (K,) のイベント確信度 Score_k。
    S_t : np.ndarray
        形状 (T,) の集約スコア。
    q_ht : np.ndarray
        形状 (H, T) の Step1 局所イベント確率。
    q_hk : np.ndarray
        形状 (H, K) の Step3 局所イベント確率(ピーク窓ごと)。
    """
    validate_numpy(p_ht, allowed_ndims=(2,), name='p_ht')
    if weights_h is not None:
        validate_numpy(weights_h, allowed_ndims=(1,), name='weights_h')

    q_ht, S_t = compute_local_event_probabilities(
        p_ht=p_ht,
        half_window=half_window,
    )

    peak_indices = detect_event_peaks(
        S_t=S_t,
        min_score=min_score,
        min_distance=min_peak_distance,
        smooth_window=smooth_window,
    )

    if peak_indices.size == 0:
        H, _ = p_ht.shape
        q_hk_empty = np.zeros((H, 0), dtype=np.float64)
        score_empty = np.zeros(0, dtype=np.float64)
        return peak_indices, score_empty, S_t, q_ht, q_hk_empty

    q_hk = compute_window_event_probabilities(
        p_ht=p_ht,
        peak_indices=peak_indices,
        win_left=peak_win_left,
        win_right=peak_win_right,
    )

    score_k = compute_event_scores(
        q_hk=q_hk,
        weights_h=weights_h,
    )

    return peak_indices, score_k, S_t, q_ht, q_hk
