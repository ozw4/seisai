import numpy as np
from numba import njit
from seisai_utils.validator import validate_numpy


@njit(cache=True, fastmath=True)
def _sliding_sum_same(hist: np.ndarray, win: int) -> np.ndarray:
    """1 次元ヒストグラム `hist` に対して，長さ `win` のスライディング窓で
    各時刻を中心とした窓内和を計算する。

    端点付近では窓が配列の範囲外にはみ出さないように開始位置を調整する。
    `win` が 1 未満の場合は 1 とみなして計算する。

    Parameters
    ----------
    hist : np.ndarray
            形状 (T,) の 1 次元配列。dtype は int32 を想定する。
    win : int
            スライディング窓の長さ。1 未満の場合は 1 に丸められる。

    Returns
    -------
    np.ndarray
            形状 (T,) の int32 配列。各時刻に対応する窓内和。

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


@njit(cache=True, fastmath=True)
def _picks_hist(
    R: np.ndarray,
    thr_on: float,
    thr_off: float,
    min_on_len: int,
    refr_len: int,
    hist: np.ndarray,
) -> None:
    """閾値ヒステリシスとリフラクトリ期間付きのピーク検出を行い，
    検出開始位置ごとに `hist` をインクリメントする。

    `R[t]` が `thr_on` 以上の状態が `min_on_len` サンプル以上続いた場合，
    その区間の先頭インデックス `t0` に対して `hist[t0]` を 1 増やす。
    検出後は `refr_len` サンプル分インデックスを進めることでリフラクトリ期間に入る。
    `thr_off` はヒステリシスの下側閾値として用いられ，`thr_off < v < thr_on`
    の範囲では連続長 `run` を維持する。

    Parameters
    ----------
    R : np.ndarray
            形状 (T,) の 1 次元スコア列。
    thr_on : float
            ピーク開始判定の上側閾値。
    thr_off : float
            ランのリセットに用いる下側閾値。`thr_on` 以上でなければならない。
    min_on_len : int
            ピーク開始とみなす最小連続長。1 以上でなければならない。
    refr_len : int
            検出後にスキップするリフラクトリ期間の長さ。1 以上でなければならない。
    hist : np.ndarray
            形状 (T,) の int32 配列。検出されたピーク開始位置に対してインクリメントされる。

    Raises
    ------
    ValueError
            `thr_on < thr_off` の場合，または `min_on_len` または `refr_len` が 1 未満の場合。

    """
    T = R.size
    if T == 0:
        return
    if thr_on < thr_off:
        msg = 'thr_on must be >= thr_off'
        raise ValueError(msg)
    if min_on_len < 1 or refr_len < 1:
        msg = 'min_on_len and refr_len must be >= 1'
        raise ValueError(msg)

    t = 0
    run = 0
    while t < T:
        v = R[t]
        if v >= thr_on:
            run += 1
            if run >= min_on_len:
                t0 = t - (run - 1)
                if 0 <= t0 < T:
                    hist[t0] += 1
                t = t0 + refr_len
                run = 0
                continue
        elif v <= thr_off:
            run = 0
        t += 1


@njit(cache=True, fastmath=True)
def _majority_counts_from_feature(
    feat_ht: np.ndarray,
    thr: float,
) -> np.ndarray:
    """特徴量行列 `feat_ht` に対して，閾値 `thr` 以上のチャネル数を
    各時刻ごとにカウントする。

    Parameters
    ----------
    feat_ht : np.ndarray
            形状 (H, T) の 2 次元配列。第 1 軸がチャネル，第 2 軸が時刻。
    thr : float
            閾値。`feat_ht[h, t] >= thr` を満たすチャネルをカウントする。

    Returns
    -------
    np.ndarray
            形状 (T,) の int32 配列。各時刻 t における閾値以上のチャネル数。

    """
    H, T = feat_ht.shape
    counts = np.zeros(T, dtype=np.int32)
    for h in range(H):
        for t in range(T):
            if feat_ht[h, t] >= thr:
                counts[t] += 1
    return counts


def compute_time_support_from_pick_cluster(
    feat_ht: np.ndarray,
    *,
    thr_on: float,
    thr_off: float,
    min_on_len: int,
    refr_len: int,
    win_len: int,
) -> np.ndarray:
    """ピックのクラスタリングに基づいて，時刻 t 近傍でイベントを支持しているチャネルの“割合” S_t (0〜1 スケール) を計算する。

    入力特徴量行列 `feat_ht` の各チャネルに対して `_picks_hist` によるピーク検出を行い，
    ピーク開始位置のヒストグラムを作成する。その後，長さ `win_len` の窓で
    `_sliding_sum_same` により移動和を計算し，クラスタ本数系列を得る。
    最後にチャネル数 H で割ることで，S_t を「1 チャネルあたりの平均ピック数」として
    0 付近〜おおよそ 1 程度のスケールに正規化する(複数イベントが重なると 1 を超えることもある)。

    Parameters
    ----------
    feat_ht : np.ndarray
            形状 (H, T) の 2 次元配列。各行がチャネル，各列が時刻の特徴量。
    thr_on : float
            ピーク開始判定の上側閾値。
    thr_off : float
            ヒステリシスの下側閾値。`thr_on` 以上でなければならない。
    min_on_len : int
            ピーク開始とみなす最小連続長。1 以上でなければならない。
    refr_len : int
            検出後にスキップするリフラクトリ期間の長さ。1 以上でなければならない。
    win_len : int
            ヒストグラムに対して適用するスライディング窓の長さ。1 以上でなければならない。

    Returns
    -------
    np.ndarray
            形状 (T,) の float64 配列。各時刻 t における正規化済みクラスタ指標 S_t。
            典型的には 0〜1 程度の値になるが，窓内に複数イベントが重なると 1 を超える場合もある。

    Raises
    ------
    ValueError
            `feat_ht` が空の場合，`min_on_len` または `refr_len` が 1 未満の場合，
            あるいは `win_len` が 1 未満の場合。

    """
    validate_numpy(feat_ht, allowed_ndims=(2,), name='feat_ht')
    if min_on_len < 1 or refr_len < 1:
        msg = 'min_on_len and refr_len must be >= 1'
        raise ValueError(msg)
    if win_len < 1:
        msg = 'win_len must be >= 1'
        raise ValueError(msg)

    H, T = feat_ht.shape
    if H == 0 or T == 0:
        msg = 'feat_ht must be non-empty'
        raise ValueError(msg)

    feat_ht = np.asarray(feat_ht, dtype=np.float64)

    hist = np.zeros(T, dtype=np.int32)
    for h in range(H):
        R = feat_ht[h]
        _picks_hist(
            R,
            float(thr_on),
            float(thr_off),
            int(min_on_len),
            int(refr_len),
            hist,
        )

    cluster = _sliding_sum_same(hist, int(win_len)).astype(np.float64)
    # チャネル数で割ってスケールを「1 チャネルあたりの平均ピック数」に揃える
    S_t = cluster / float(H)
    return S_t


def compute_time_support_from_majority(
    feat_ht: np.ndarray,
    *,
    thr: float,
) -> np.ndarray:
    """多数決に基づいて時刻 t 近傍でイベントを支持しているチャネルの“割合” S_t (0〜1 スケール) を計算する。

    特徴量行列 `feat_ht` の各時刻 t について，値が `thr` 以上のチャネル数を
    `_majority_counts_from_feature` でカウントし，チャネル数 H で割ることで
    「閾値以上チャネル割合」としての S_t を返す。

    Parameters
    ----------
    feat_ht : np.ndarray
            形状 (H, T) の 2 次元配列。各行がチャネル，各列が時刻の特徴量。
    thr : float
            閾値。`feat_ht[h, t] >= thr` を満たすチャネルをカウントする。

    Returns
    -------
    np.ndarray
            形状 (T,) の float64 配列。各時刻 t における閾値以上チャネル割合 S_t(0〜1)。

    Raises
    ------
    ValueError
            `feat_ht` が空の場合。

    """
    validate_numpy(feat_ht, allowed_ndims=(2,), name='feat_ht')

    H, T = feat_ht.shape
    if H == 0 or T == 0:
        msg = 'feat_ht must be non-empty'
        raise ValueError(msg)

    feat_ht = np.asarray(feat_ht, dtype=np.float64)

    counts = _majority_counts_from_feature(
        feat_ht,
        float(thr),
    ).astype(np.float64)

    S_t = counts / float(H)
    return S_t


def compute_time_support_from_probabilities(
    p_ht: np.ndarray,
    *,
    half_window: int,
) -> np.ndarray:
    """Step1:
    H×T の初動確率 p_ht から，時間方向の窓平均スコアに基づく
    time support S_t を計算する。

    - 各トレース h・各時刻 t について，
      窓 [t-half_window, t+half_window] の平均値を m_ht[h,t] とする。
    - S_t[t] = mean_h m_ht[h,t] を time support として返す。

    Parameters
    ----------
    p_ht : np.ndarray
        形状 (H, T) の初動“確率風”スコア。確率出力を想定するので [0,1] 範囲を要求。
    half_window : int
        時刻 t を中心にした左右の窓半幅(サンプル数)。Δ に相当。0 以上。

    Returns
    -------
    S_t : np.ndarray
        形状 (T,) の float64 配列。各時刻 t における time support スコア。
        0〜1 の範囲に収まる(p_ht が [0,1] の場合)。

    Raises
    ------
    TypeError, ValueError
        入力の型・次元・範囲，half_window の値が不正な場合。

    """
    validate_numpy(p_ht, allowed_ndims=(2,), name='p_ht')
    if half_window < 0:
        msg = 'half_window must be >= 0'
        raise ValueError(msg)

    H, T = p_ht.shape
    if H == 0 or T == 0:
        msg = 'p_ht must be non-empty'
        raise ValueError(msg)

    x = np.asarray(p_ht, dtype=np.float64)

    # p が [0,1] に入っていることを確認(確率前提の Step1 なのでここは厳しめに見る)
    if np.any(x < 0.0) or np.any(x > 1.0):
        msg = 'p_ht must be in [0, 1]'
        raise ValueError(msg)

    if half_window == 0:
        # 窓長1 → そのままチャネル平均
        return x.mean(axis=0)

    # half_window が極端に大きい設定は明示的に弾いておく
    # (全域より十分大きい窓は意味を持ちにくいため)
    if 2 * half_window + 1 > T:
        msg = '2 * half_window + 1 must be <= T'
        raise ValueError(msg)

    pad = half_window
    # 時間方向にゼロパディング
    p_pad = np.pad(
        x,
        pad_width=((0, 0), (pad, pad)),
        mode='constant',
        constant_values=0.0,
    )  # (H, T + 2*pad)

    # 累積和で全時刻の窓和を一括計算
    cs = np.cumsum(p_pad, axis=1)
    cs = np.concatenate(
        [np.zeros((H, 1), dtype=np.float64), cs],
        axis=1,
    )  # (H, T + 2*pad + 1)

    win = 2 * half_window + 1
    # 各 t について [t-half_window, t+half_window] の和 → 窓平均 m_ht
    window_sum = cs[:, win : win + T] - cs[:, :T]  # (H, T)
    m_ht = window_sum / float(win)  # (H, T)

    # チャネル平均で time support S_t を作る
    S_t = m_ht.mean(axis=0)
    return S_t.astype(np.float64)
