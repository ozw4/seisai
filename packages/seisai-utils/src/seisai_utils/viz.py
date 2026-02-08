from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class ImshowPanel:
    title: str
    data_hw: np.ndarray
    cmap: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    alpha: float = 1.0


def _as_hw(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        msg = f'data must be 2D (H,W), got shape={a.shape}'
        raise ValueError(msg)
    return a


def to_numpy_bchw(x: torch.Tensor | np.ndarray, *, name: str) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        if int(x.ndim) != 4:
            msg = f'{name} must be (B,C,H,W) tensor, got shape={tuple(x.shape)}'
            raise ValueError(
                msg
            )
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if int(x.ndim) != 4:
            msg = f'{name} must be (B,C,H,W) array, got shape={x.shape}'
            raise ValueError(msg)
        return x
    msg = f'{name} must be torch.Tensor or numpy.ndarray'
    raise TypeError(msg)


def select_hw(x_bchw: np.ndarray, *, b: int, c: int, name: str) -> np.ndarray:
    if not (0 <= int(b) < int(x_bchw.shape[0])):
        msg = f'{name}: batch index out of range: b={b} B={int(x_bchw.shape[0])}'
        raise ValueError(
            msg
        )
    if not (0 <= int(c) < int(x_bchw.shape[1])):
        msg = f'{name}: channel index out of range: c={c} C={int(x_bchw.shape[1])}'
        raise ValueError(
            msg
        )
    hw = x_bchw[int(b), int(c)]
    if hw.ndim != 2:
        msg = f'{name}: expected (H,W), got {hw.shape}'
        raise ValueError(msg)
    return np.asarray(hw)


def imshow_hw(
    ax,
    data_hw: np.ndarray,
    *,
    transpose_for_trace_time: bool = True,
    title: str | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 1.0,
) -> None:
    """(H,W) を表示。transpose_for_trace_time=True で x=Trace, y=Time になるように表示する。."""
    hw = _as_hw(data_hw)
    img = hw.T if transpose_for_trace_time else hw
    ax.imshow(img, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)

    if title is not None:
        ax.set_title(title)

    if transpose_for_trace_time:
        ax.set_xlabel('Trace (H)')
        ax.set_ylabel('Time (samples)')
    else:
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Trace (H)')


def imshow_overlay_hw(
    ax,
    base_hw: np.ndarray,
    overlay_hw: np.ndarray,
    *,
    transpose_for_trace_time: bool = True,
    base_title: str | None = None,
    base_cmap: str | None = None,
    base_vmin: float | None = None,
    base_vmax: float | None = None,
    overlay_cmap: str | None = None,
    overlay_vmin: float | None = None,
    overlay_vmax: float | None = None,
    overlay_alpha: float = 0.5,
) -> None:
    """Base の上に overlay を重ねる(example_segy_gather_pipline_ds.py のやつ)。."""
    base = _as_hw(base_hw)
    ov = _as_hw(overlay_hw)
    if base.shape != ov.shape:
        msg = f'shape mismatch: base={base.shape} overlay={ov.shape}'
        raise ValueError(msg)

    imshow_hw(
        ax,
        base,
        transpose_for_trace_time=transpose_for_trace_time,
        title=base_title,
        cmap=base_cmap,
        vmin=base_vmin,
        vmax=base_vmax,
        alpha=1.0,
    )
    imshow_hw(
        ax,
        ov,
        transpose_for_trace_time=transpose_for_trace_time,
        title=None,
        cmap=overlay_cmap,
        vmin=overlay_vmin,
        vmax=overlay_vmax,
        alpha=overlay_alpha,
    )


def save_imshow_row(
    png_path: str | Path,
    panels: Iterable[ImshowPanel],
    *,
    suptitle: str | None = None,
    transpose_for_trace_time: bool = True,
    figsize: tuple[float, float] = (21.0, 5.0),
    dpi: int = 150,
) -> None:
    """横一列に並べて保存(入力/GT/Pred みたいな triptych 用)。."""
    panels = list(panels)
    if len(panels) == 0:
        msg = 'panels must be non-empty'
        raise ValueError(msg)

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1, len(panels), figsize=figsize, sharey=True, constrained_layout=True
    )
    if len(panels) == 1:
        axes = [axes]

    for ax, p in zip(axes, panels, strict=False):
        imshow_hw(
            ax,
            p.data_hw,
            transpose_for_trace_time=transpose_for_trace_time,
            title=p.title,
            cmap=p.cmap,
            vmin=p.vmin,
            vmax=p.vmax,
            alpha=p.alpha,
        )

    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)


def per_trace_zscore_hw(a_hw: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(H,W) をトレース毎(行ごと)に z-score。.

    主に可視化用途を想定。
    """
    hw = _as_hw(a_hw).astype(np.float32, copy=False)
    m = hw.mean(axis=1, keepdims=True)
    s = hw.std(axis=1, keepdims=True)
    return (hw - m) / (s + float(eps))


def save_triptych_bchw(
    *,
    x_in_bchw: np.ndarray | torch.Tensor,
    x_tg_bchw: np.ndarray | torch.Tensor,
    x_pr_bchw: np.ndarray | torch.Tensor,
    out_path: str | Path,
    titles: tuple[str, str, str] = ('Input', 'Target', 'Pred'),
    suptitle: str | None = None,
    batch_index: int = 0,
    channel_index: int = 0,
    per_trace_norm: bool = False,
    per_trace_eps: float = 1e-8,
    transpose_for_trace_time: bool = True,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (20.0, 15.0),
    dpi: int = 300,
) -> None:
    """入力/正解/予測を横3枚(triptych)で保存する。.

    Args:
            x_in_bchw/x_tg_bchw/x_pr_bchw: (B,C,H,W)
            out_path: 保存先 PNG
            batch_index/channel_index: 表示する B/C の index
            per_trace_norm: True の場合、各 (H,W) を行方向に z-score

    """
    in_bchw = to_numpy_bchw(x_in_bchw, name='x_in_bchw')
    tg_bchw = to_numpy_bchw(x_tg_bchw, name='x_tg_bchw')
    pr_bchw = to_numpy_bchw(x_pr_bchw, name='x_pr_bchw')
    if tg_bchw.shape != in_bchw.shape or pr_bchw.shape != in_bchw.shape:
        msg = 'shape mismatch among input/target/pred'
        raise ValueError(msg)
    in_hw = select_hw(in_bchw, b=batch_index, c=channel_index, name='x_in_bchw').astype(
        np.float32, copy=False
    )
    tg_hw = select_hw(tg_bchw, b=batch_index, c=channel_index, name='x_tg_bchw').astype(
        np.float32, copy=False
    )
    pr_hw = select_hw(pr_bchw, b=batch_index, c=channel_index, name='x_pr_bchw').astype(
        np.float32, copy=False
    )

    if per_trace_norm:
        in_hw = per_trace_zscore_hw(in_hw, eps=per_trace_eps)
        tg_hw = per_trace_zscore_hw(tg_hw, eps=per_trace_eps)
        pr_hw = per_trace_zscore_hw(pr_hw, eps=per_trace_eps)

    save_imshow_row(
        out_path,
        [
            ImshowPanel(
                title=titles[0],
                data_hw=in_hw,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            ),
            ImshowPanel(
                title=titles[1],
                data_hw=tg_hw,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            ),
            ImshowPanel(
                title=titles[2],
                data_hw=pr_hw,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            ),
        ],
        suptitle=suptitle,
        transpose_for_trace_time=transpose_for_trace_time,
        figsize=figsize,
        dpi=dpi,
    )
