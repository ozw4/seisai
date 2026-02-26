from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jogsarar_shared import valid_pick_mask


def save_conf_scatter(
    *,
    out_png: Path,
    x_abs: np.ndarray,
    picks_i: np.ndarray,
    dt_ms: float,
    conf_prob_viz01: np.ndarray,
    conf_rs: np.ndarray,
    conf_trend: np.ndarray,
    title: str,
    trend_hat_ms: np.ndarray | None = None,
) -> None:
    x = np.asarray(x_abs, dtype=np.float32)
    pk = np.asarray(picks_i)
    if x.ndim != 1 or pk.ndim != 1 or x.shape[0] != pk.shape[0]:
        msg = f'x_abs/picks_i must be (H,), got {x.shape}, {pk.shape}'
        raise ValueError(msg)

    y_ms = pk.astype(np.float32, copy=False) * float(dt_ms)
    valid = valid_pick_mask(pk)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    panels = [
        ('conf_prob(viz01)', np.asarray(conf_prob_viz01, dtype=np.float32)),
        ('conf_trend', np.asarray(conf_trend, dtype=np.float32)),
        ('conf_rs', np.asarray(conf_rs, dtype=np.float32)),
    ]

    for ax, (name, cval) in zip(axes.ravel(), panels, strict=True):
        if cval.shape != x.shape:
            msg = f'{name} shape mismatch: {cval.shape}, expected {x.shape}'
            raise ValueError(msg)

        if np.any(valid):
            sc = ax.scatter(
                x[valid],
                y_ms[valid],
                c=cval[valid],
                s=12.0,
                cmap='viridis',
                vmin=0.0,
                vmax=1.0,
            )
            if trend_hat_ms is not None:
                th = np.asarray(trend_hat_ms, dtype=np.float32)
                if th.shape != x.shape:
                    msg = f'trend_hat_ms shape mismatch: {th.shape}, expected {x.shape}'
                    raise ValueError(msg)
                tmask = valid & np.isfinite(th)
                if np.any(tmask):
                    ord_i = np.argsort(x[tmask], kind='mergesort')
                    ax.plot(
                        x[tmask][ord_i],
                        th[tmask][ord_i],
                        color='white',
                        lw=1.0,
                        alpha=0.8,
                    )
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(name)
        ax.grid(alpha=0.2)

    axes[0].set_xlabel('|offset| [m]')
    axes[1].set_xlabel('|offset| [m]')
    axes[2].set_xlabel('|offset| [m]')
    axes[0].set_ylabel('pick [ms]')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


__all__ = ['save_conf_scatter']
