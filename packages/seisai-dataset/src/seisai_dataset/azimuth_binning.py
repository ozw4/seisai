from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class AzimuthBinConfig:
    # 入力角度の単位
    degrees: bool = True

    # ヒストグラム
    bins: int = 720  # 0.5°刻み相当(degrees=Trueの場合)
    # 平滑カーネル(円周)
    smooth: Literal['moving', 'gaussian'] = 'moving'
    win_bins: int = 15  # 移動平均の窓(bin数)。~7–21が無難
    gauss_sigma_bins: float = 6.0  # gaussianの場合のσ[bin]

    # 谷検出のしきい値
    min_prominence: float = 0.02  # 谷のプロミネンス閾値(全カウントに対する比率)
    min_sep_bins: int = 20  # 隣接する谷の最小距離(bin)

    # フォールバック(CDF等分)
    min_traces_per_bin: int = (
        200  # これを満たすKに自動調整(K=floor(N/min_traces_per_bin))
    )
    max_bins_when_fallback: int = 12


def _to_rad(a: np.ndarray, degrees: bool) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64)
    if degrees:
        x = np.deg2rad(x)
    # [0, 2π) 正規化
    x = np.mod(x, 2.0 * np.pi)
    return x


def _max_gap_rotation(a_rad: np.ndarray) -> tuple[np.ndarray, float]:
    # 角度を昇順→最大ギャップを探し、その直後を原点に回転
    s = np.sort(a_rad)
    dif = np.diff(s, append=s[0] + 2.0 * np.pi)
    k = int(np.argmax(dif))  # 最大ギャップの開始インデックス
    origin = float(s[(k + 1) % s.size])  # ギャップ直後
    r = np.mod(a_rad - origin, 2.0 * np.pi)
    return r, origin


def _circ_hist(a_rad: np.ndarray, bins: int) -> np.ndarray:
    h, _ = np.histogram(a_rad, bins=bins, range=(0.0, 2.0 * np.pi))
    return h.astype(np.float64)


def _circ_smooth(h: np.ndarray, cfg: AzimuthBinConfig) -> np.ndarray:
    n = int(h.size)
    if cfg.smooth == 'moving':
        w = int(max(1, cfg.win_bins))
        # FFT畳み込み(円周)
        ker = np.zeros(n)
        half = w // 2
        ker[: half + 1] = 1.0
        ker[-(w - (half + 1)) :] = 1.0 if w > (half + 1) else ker[-(w - (half + 1)) :]
        ker = ker / ker.sum()
    else:  # gaussian
        # 円環ガウシアン核をサンプル
        x = np.arange(n)
        c = 0
        d = np.minimum(np.abs(x - c), n - np.abs(x - c))
        ker = np.exp(-0.5 * (d / float(cfg.gauss_sigma_bins)) ** 2)
        ker = ker / ker.sum()
    H = np.fft.ifft(np.fft.fft(h) * np.fft.fft(ker)).real
    return H


def _find_valleys(
    H: np.ndarray, min_prominence_abs: float, min_sep_bins: int
) -> np.ndarray:
    # 単純な極小：H[i-1] > H[i] < H[i+1] を円周で判定 → プロミネンス閾で間引き → 最小距離でNMS
    n = H.size
    # 隣接比較用のロール
    Hp = np.roll(H, 1)
    Hn = np.roll(H, -1)
    cand = (Hp > H) & (Hn > H)

    idx = np.where(cand)[0]
    if idx.size == 0:
        return idx

    # プロミネンス近似：左右の局所最大までの差の小さい方
    # 左右ピークは単純に単調増加が止まるまで辿る近似でOK
    def left_peak(i):
        j = (i - 1) % n
        while (H[(j - 1) % n] <= H[j]) and ((j - 1) % n != i):
            j = (j - 1) % n
        return H[j]

    def right_peak(i):
        j = (i + 1) % n
        while (H[(j + 1) % n] <= H[j]) and ((j + 1) % n != i):
            j = (j + 1) % n
        return H[j]

    keep = []
    for i in idx:
        lp = left_peak(i)
        rp = right_peak(i)
        prom = min(lp - H[i], rp - H[i])
        if prom >= min_prominence_abs:
            keep.append(i)
    if not keep:
        return np.asarray([], dtype=np.int64)

    # NMS 的な最小距離間引き(谷が近すぎる場合は深い方を残す)
    keep = np.asarray(keep, dtype=np.int64)
    order = np.argsort(H[keep])  # 深い順に処理(小さい方が先)
    sel = []
    used = np.zeros(n, dtype=bool)
    for j in order:
        i = int(keep[j])
        if used[i]:
            continue
        sel.append(i)
        # 半径 min_sep_bins を使用不可に
        for d in range(-min_sep_bins, min_sep_bins + 1):
            used[(i + d) % n] = True
    sel = np.sort(np.asarray(sel, dtype=np.int64))
    return sel


def _labels_from_cuts(a_bin: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    # cuts は「谷」のbinインデックス(境界)。谷[i]〜谷[i+1]の間が1ビン。
    if cuts.size == 0:
        return np.zeros_like(a_bin, dtype=np.int64)
    cuts_sorted = np.sort(cuts)
    # a_bin ∈ [0, bins-1] を、循環的に区間割り当て
    # まず各サンプルがどの cut より右にあるかを数えてラベル化
    # 例: cuts=[10, 50, 300] → ラベルは {0,1,2}(最後の境界〜先頭も1区間)
    labels = np.zeros_like(a_bin, dtype=np.int64)
    for k, c in enumerate(cuts_sorted):
        labels += (a_bin >= c).astype(np.int64)
    labels %= cuts_sorted.size  # 循環
    return labels


def bin_azimuth_dynamic(
    azimuths: np.ndarray,
    cfg: AzimuthBinConfig = AzimuthBinConfig(),
) -> dict:
    """ヒストグラム+円周平滑で“谷”を境界に方位ビンを自動生成。
    失敗時(谷が無い・制約未満)は CDF 等分でフォールバック(要警告)。

    Returns:
      {
        'labels': ndarray[int] shape (N,),  # 各サンプルのビンラベル
        'cuts_bins': ndarray[int],          # 境界のbinインデックス(回転後座標)
        'origin_rad': float,                # 最大ギャップ回転の原点(rad)
        'used_fallback': bool,
        'hist': ndarray[float],             # 回転後ヒスト
        'hist_smooth': ndarray[float],      # 平滑後ヒスト
      }

    """
    a = _to_rad(azimuths, cfg.degrees)
    if a.size == 0:
        msg = 'azimuths is empty'
        raise ValueError(msg)

    a_rot, origin = _max_gap_rotation(a)
    h = _circ_hist(a_rot, cfg.bins)
    H = _circ_smooth(h, cfg)

    min_prom_abs = max(float(cfg.min_prominence) * float(a_rot.size), 1.0)
    valleys = _find_valleys(
        H,
        min_prominence_abs=min_prom_abs,
        min_sep_bins=int(cfg.min_sep_bins),
    )

    used_fallback = False
    if valleys.size == 0:
        # フォールバック：CDF等分(Kは件数から決める)
        K = max(
            1,
            min(cfg.max_bins_when_fallback, a_rot.size // int(cfg.min_traces_per_bin)),
        )
        if K == 1:
            labels = np.zeros(a_rot.size, dtype=np.int64)
            return {
                'labels': labels,
                'cuts_bins': np.asarray([], dtype=np.int64),
                'origin_rad': float(origin),
                'used_fallback': True,
                'hist': h,
                'hist_smooth': H,
            }
        # 分位点(回転後座標で)
        q = np.linspace(0.0, 1.0, K + 1, endpoint=False)[1:]
        edges = np.quantile(a_rot, q, method='linear')
        # 分位点をbinインデックスに変換
        cuts = np.floor(edges / (2.0 * np.pi) * cfg.bins).astype(np.int64) % cfg.bins
        # ラベル
        a_bin = np.floor(a_rot / (2.0 * np.pi) * cfg.bins).astype(np.int64) % cfg.bins
        labels = _labels_from_cuts(a_bin, cuts)
        used_fallback = True
        return {
            'labels': labels,
            'cuts_bins': np.sort(cuts),
            'origin_rad': float(origin),
            'used_fallback': used_fallback,
            'hist': h,
            'hist_smooth': H,
        }

    # 谷で区切る
    a_bin = np.floor(a_rot / (2.0 * np.pi) * cfg.bins).astype(np.int64) % cfg.bins
    labels = _labels_from_cuts(a_bin, valleys)
    return {
        'labels': labels,
        'cuts_bins': np.sort(valleys),
        'origin_rad': float(origin),
        'used_fallback': used_fallback,
        'hist': h,
        'hist_smooth': H,
    }
