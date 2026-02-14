from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np

NormalizeMode = Optional[Literal['trace', 'global']]
PickUnit = Literal['sample', 'time']


@dataclass(frozen=True)
class PickOverlay:
    y: np.ndarray  # (nt,) per-trace picks
    unit: PickUnit = 'sample'  # "sample" or "time"
    label: str = ''
    marker: str = 'o'
    size: float = 14.0
    color: str | None = None  # None -> matplotlib cycle
    alpha: float = 0.9
    zorder: int = 6


@dataclass(frozen=True)
class WiggleConfig:
    dt: float | None = None
    t0: float = 0.0
    time_axis: Literal[0, 1] = 0  # 0: (ns, nt), 1: (nt, ns)

    x: np.ndarray | None = None
    trace_spacing: float = 1.0

    normalize: NormalizeMode = 'trace'  # "trace" | "global" | None
    gain: float = 1.0
    scale: float | None = None

    polarity: float = 1.0
    clip: float | None = None

    line_color: str = 'k'
    line_width: float = 0.5

    # ★ここが塗りつぶし設定
    fill_positive: bool = True
    fill_color: str = 'k'
    fill_alpha: float = 0.85  # ★見えるように少し濃く
    fill_zorder: int = 1  # ★線より下

    invert_y: bool = True
    show_baseline: bool = False
    baseline_color: str = '0.7'
    baseline_width: float = 0.5

    picks: tuple[PickOverlay, ...] = ()
    show_legend: bool = True


def _picks_to_y(
    pick: PickOverlay,
    *,
    ns: int,
    dt: float | None,
    t0: float,
    axis_is_time: bool,
) -> np.ndarray:
    y = np.asarray(pick.y, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError('pick.y must be 1D (nt,)')

    if pick.unit == 'sample':
        m = np.isfinite(y) & (y >= 0) & (y < ns)
        if axis_is_time:
            if dt is None:
                raise ValueError(
                    'cfg.dt is required to convert sample picks to seconds'
                )
            yv = t0 + y * float(dt)
        else:
            yv = t0 + y
        return np.where(m, yv, np.nan)

    if pick.unit == 'time':
        if not axis_is_time:
            raise ValueError('time picks require cfg.dt (time axis)')
        if dt is None:
            raise ValueError('cfg.dt is required for time axis')
        t_min = t0
        t_max = t0 + (ns - 1) * float(dt)
        m = np.isfinite(y) & (y >= t_min) & (y <= t_max)
        return np.where(m, y, np.nan)

    raise ValueError("pick.unit must be 'sample' or 'time'")


def plot_wiggle(
    data: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    cfg: WiggleConfig = WiggleConfig(),
) -> plt.Axes:
    x = np.asarray(data)
    if x.ndim != 2:
        raise ValueError('data must be 2D')

    if cfg.time_axis == 1:
        x = x.T  # -> (ns, nt)

    ns, nt = x.shape
    x = x.astype(np.float32, copy=False) * float(cfg.polarity)

    if cfg.clip is not None:
        if cfg.clip <= 0:
            raise ValueError('clip must be positive')
        x = np.clip(x, -cfg.clip, cfg.clip)

    axis_is_time = cfg.dt is not None
    if cfg.dt is None:
        t = cfg.t0 + np.arange(ns, dtype=np.float32)
        t_label = 'sample'
    else:
        t = cfg.t0 + np.arange(ns, dtype=np.float32) * float(cfg.dt)
        t_label = 'time (s)'

    if cfg.x is None:
        xpos = np.arange(nt, dtype=np.float32) * float(cfg.trace_spacing)
    else:
        xpos = np.asarray(cfg.x, dtype=np.float32)
        if xpos.ndim != 1 or xpos.shape[0] != nt:
            raise ValueError('x must be 1D with length == n_traces')

    # normalize
    if cfg.normalize == 'trace':
        denom = np.max(np.abs(x), axis=0)
        denom = np.where(denom == 0.0, 1.0, denom)
        xn = x / denom
        ref = 1.0
    elif cfg.normalize == 'global':
        denom = float(np.max(np.abs(x)))
        denom = 1.0 if denom == 0.0 else denom
        xn = x / denom
        ref = 1.0
    elif cfg.normalize is None:
        xn = x
        ref = float(np.max(np.abs(x)))
        ref = 1.0 if ref == 0.0 else ref
    else:
        raise ValueError("normalize must be 'trace', 'global', or None")

    # scale
    if cfg.scale is None:
        if nt >= 2:
            dx = np.median(np.diff(np.sort(xpos)))
            dx = 1.0 if dx == 0.0 else float(dx)
        else:
            dx = 1.0
        s = 0.45 * dx
    else:
        s = float(cfg.scale)

    amp = xn * (
        float(cfg.gain) * s / ref if cfg.normalize is None else float(cfg.gain) * s
    )

    if ax is None:
        _, ax = plt.subplots()

    if cfg.show_baseline:
        for i in range(nt):
            ax.plot(
                [xpos[i], xpos[i]],
                [t[0], t[-1]],
                color=cfg.baseline_color,
                lw=cfg.baseline_width,
                zorder=0,
            )

    # traces + fill
    for i in range(nt):
        xt = xpos[i] + amp[:, i]

        # ★正側(>=0)を塗りつぶし
        if cfg.fill_positive:
            m = amp[:, i] >= 0.0
            if np.any(m):
                ax.fill_betweenx(
                    t,
                    xpos[i],
                    xt,
                    where=m,
                    interpolate=True,  # ★0交差部を綺麗に閉じる
                    facecolor=cfg.fill_color,
                    alpha=float(cfg.fill_alpha),
                    linewidth=0.0,
                    zorder=int(cfg.fill_zorder),
                )

        # 線は塗りの上に
        ax.plot(xt, t, color=cfg.line_color, lw=cfg.line_width, zorder=2)

    # pick overlays
    if cfg.picks:
        for pk in cfg.picks:
            yv = _picks_to_y(pk, ns=ns, dt=cfg.dt, t0=cfg.t0, axis_is_time=axis_is_time)
            if yv.shape[0] != nt:
                raise ValueError('pick.y length must match n_traces')
            m = np.isfinite(yv)
            if np.any(m):
                ax.scatter(
                    xpos[m],
                    yv[m],
                    s=float(pk.size),
                    marker=pk.marker,
                    alpha=float(pk.alpha),
                    label=pk.label if pk.label else None,
                    color=pk.color,
                    zorder=int(pk.zorder),
                )

        if cfg.show_legend and any(pk.label for pk in cfg.picks):
            ax.legend(loc='best')

    ax.set_xlabel('trace')
    ax.set_ylabel(t_label)
    ax.set_xlim(float(np.min(xpos) - s), float(np.max(xpos) + s))
    ax.set_ylim(float(t[0]), float(t[-1]))
    if cfg.invert_y:
        ax.invert_yaxis()

    return ax
