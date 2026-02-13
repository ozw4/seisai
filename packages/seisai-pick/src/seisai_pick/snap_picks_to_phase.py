from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import signal

if TYPE_CHECKING:
    from numpy.typing import NDArray

PickMode = Literal['peak', 'trough', 'rising', 'trailing']


def _choose_offset(candidates: NDArray[np.int_]) -> int:
    """Choose an offset with smallest |offset|. If tie, prefer negative (earlier)."""
    if candidates.size == 0:
        return 0

    abs_vals = np.abs(candidates)
    best_abs = abs_vals.min()
    best = candidates[abs_vals == best_abs]

    neg = best[best < 0]
    if neg.size:
        # closest-to-zero negative (same abs tie => unique anyway)
        return int(neg.max())
    return int(best.min())


def _clip_window(n_samples: int, center: int, half_width: int) -> tuple[int, int, int]:
    """Return (left, right, center_in_window) for inclusive window [left, right]."""
    if half_width <= 0:
        return center, center, 0
    left = max(0, center - half_width)
    right = min(n_samples - 1, center + half_width)
    return left, right, center - left


def snap_picks_to_phase(
    max_index: NDArray[np.integer],
    seis: NDArray[np.floating],
    mode: PickMode = 'peak',
    ltcor: int = 5,
) -> NDArray[np.int32]:
    """予測されたピックを近傍の特定phaseへシフトさせる。.

    Args:
        max_index: 予測ピック位置(サンプル番号)の1次元配列 (n_traces,)。
            0 はピック無しとして扱う。
        seis: 地震波形セクションの2次元配列 (n_traces, n_samples)。
        mode: 移動させるphaseの指定。
            - "peak": 近傍の相対最大(ピーク)
            - "trough": 近傍の相対最小(トラフ)
            - "rising": 立ち上がり(符号が負→正になる境界側へ寄せる発想)
            - "trailing": 立下がり(符号が正→負になる境界側へ寄せる発想)
        ltcor: 最大補正量(サンプル数)。
            "peak"/"trough" の探索窓にのみ使用。
            "rising"/"trailing" では元仕様通り考慮しない。

    Returns:
        補正後のピック位置(サンプル番号)の配列 (n_traces,) を int32 で返す。
        入力が 0(ピック無し)のトレースは 0 のまま。

    Raises:
        ValueError: seis が2次元でない、max_index の長さが一致しない、
            ltcor が負、または max_index が範囲外の場合。
        ValueError: mode が不正な場合。

    """
    seis_arr = np.asarray(seis)
    if seis_arr.ndim != 2:
        msg = f'seis must be 2D (n_traces, n_samples), got shape={seis_arr.shape}'
        raise ValueError(msg)

    n_traces, n_samples = seis_arr.shape
    picks = np.asarray(max_index)
    if picks.ndim != 1 or picks.shape[0] != n_traces:
        msg = f'max_index must be 1D with length n_traces={n_traces}, got shape={picks.shape}'
        raise ValueError(msg)

    if ltcor < 0:
        msg = 'ltcor must be >= 0.'
        raise ValueError(msg)

    if mode not in ('peak', 'trough', 'rising', 'trailing'):
        msg = "mode must be one of: 'peak', 'trough', 'rising', 'trailing'."
        raise ValueError(msg)

    out = picks.astype(np.int32, copy=True)

    for tr in range(n_traces):
        p0 = int(picks[tr])
        if p0 == 0:
            continue

        if not (0 <= p0 < n_samples):
            msg = f'pick out of bounds: trace={tr}, pick={p0}, n_samples={n_samples}'
            raise ValueError(msg)

        if mode in ('peak', 'trough'):
            left, right, center_in_win = _clip_window(n_samples, p0, ltcor)
            win = seis_arr[tr, left : right + 1]

            if win.size == 1:
                continue

            if mode == 'peak':
                idx = signal.argrelmax(win)[0]  # relative maxima indices in window
                if idx.size:
                    offsets = idx.astype(np.int32) - center_in_win
                    off = _choose_offset(offsets)
                else:
                    # fallback: no local maxima found -> choose max(|x|) in window?
                    # 元仕様は「ピーク検出」だが、空になるケースがあるため
                    # 「最大値(そのまま)」でfallback(同距離なら時間マイナス優先)
                    score = win
                    best = (
                        np.flatnonzero(score == score.max()).astype(np.int32)
                        - center_in_win
                    )
                    off = _choose_offset(best)
            else:  # trough
                idx = signal.argrelmin(win)[0]
                if idx.size:
                    offsets = idx.astype(np.int32) - center_in_win
                    off = _choose_offset(offsets)
                else:
                    score = win
                    best = (
                        np.flatnonzero(score == score.min()).astype(np.int32)
                        - center_in_win
                    )
                    off = _choose_offset(best)

            out[tr] = np.int32(p0 + off)
            continue

        amp = float(seis_arr[tr, p0])
        if amp == 0.0:
            continue

        if mode == 'trailing':
            if amp < 0.0:
                # backward: last >=0 before point
                cand = np.flatnonzero(seis_arr[tr, : p0 + 1] >= 0).astype(np.int32) - p0
            else:
                # forward: first <=0 after point
                cand = np.flatnonzero(seis_arr[tr, p0:] <= 0).astype(np.int32)
        elif amp > 0.0:
            # backward: last <=0 before point
            cand = np.flatnonzero(seis_arr[tr, : p0 + 1] <= 0).astype(np.int32) - p0
        else:
            # forward: first >=0 after point
            cand = np.flatnonzero(seis_arr[tr, p0:] >= 0).astype(np.int32)

        off = _choose_offset(cand)
        out[tr] = np.int32(p0 + off)

    return out
