from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from seisai_utils.viz import imshow_hw, select_hw, to_numpy_bchw


def save_psn_debug_png(
    out_path: str | Path,
    *,
    x_bchw: torch.Tensor | np.ndarray,
    target_b3hw: torch.Tensor | np.ndarray,
    logits_b3hw: torch.Tensor,
    b: int = 0,
    title: str | None = None,
    transpose_for_trace_time: bool = True,
    dpi: int = 150,
) -> None:
    """Save a minimal PSN debug visualization PNG.

    Panels (2x3):
    - waveform (x[b,0])
    - target P/S (target[b,0], target[b,1])
    - pred P/S (softmax(logits)[b,0], [b,1])
    - pred Noise (softmax(logits)[b,2]) in the last panel

    Args:
            out_path: output PNG path
            x_bchw: (B,1,H,W) (or (B,C,H,W), uses channel 0)
            target_b3hw: (B,3,H,W) probability targets
            logits_b3hw: (B,3,H,W) logits (no softmax)
            b: batch index to visualize
            title: optional figure title
            transpose_for_trace_time: display orientation (True: x=Trace, y=Time)
            dpi: PNG dpi

    """
    x_np = to_numpy_bchw(x_bchw, name='x_bchw')
    tg_np = to_numpy_bchw(target_b3hw, name='target_b3hw')

    if not isinstance(logits_b3hw, torch.Tensor) or int(logits_b3hw.ndim) != 4:
        msg = f'logits_b3hw must be torch.Tensor (B,3,H,W), got {type(logits_b3hw).__name__} {getattr(logits_b3hw, "shape", None)}'
        raise ValueError(
            msg
        )
    if int(logits_b3hw.shape[1]) != 3:
        msg = f'logits_b3hw must have C==3, got shape={tuple(logits_b3hw.shape)}'
        raise ValueError(
            msg
        )

    B, Cx, H, W = x_np.shape
    if B <= 0 or H <= 0 or W <= 0:
        msg = f'invalid x_bchw shape: {tuple(x_np.shape)}'
        raise ValueError(msg)
    if Cx <= 0:
        msg = f'x_bchw must have C>=1, got shape={tuple(x_np.shape)}'
        raise ValueError(msg)
    if tuple(tg_np.shape) != (B, 3, H, W):
        msg = f'target_b3hw shape mismatch: got {tuple(tg_np.shape)} expected {(B, 3, H, W)}'
        raise ValueError(
            msg
        )
    if tuple(logits_b3hw.shape) != (B, 3, H, W):
        msg = f'logits_b3hw shape mismatch: got {tuple(logits_b3hw.shape)} expected {(B, 3, H, W)}'
        raise ValueError(
            msg
        )
    if not (0 <= int(b) < int(B)):
        msg = f'b out of range: b={b} for B={B}'
        raise ValueError(msg)

    pred = torch.softmax(logits_b3hw.detach().float(), dim=1)
    pr_np = to_numpy_bchw(pred, name='pred_b3hw')

    x_hw = select_hw(x_np, b=int(b), c=0, name='x_bchw').astype(np.float32, copy=False)
    tp_hw = select_hw(tg_np, b=int(b), c=0, name='target_b3hw').astype(
        np.float32, copy=False
    )
    ts_hw = select_hw(tg_np, b=int(b), c=1, name='target_b3hw').astype(
        np.float32, copy=False
    )
    pp_hw = select_hw(pr_np, b=int(b), c=0, name='pred_b3hw').astype(
        np.float32, copy=False
    )
    ps_hw = select_hw(pr_np, b=int(b), c=1, name='pred_b3hw').astype(
        np.float32, copy=False
    )
    pn_hw = select_hw(pr_np, b=int(b), c=2, name='pred_b3hw').astype(
        np.float32, copy=False
    )

    # Robust waveform scaling (avoid over/under-saturation when not standardized).
    abs_q = float(np.nanpercentile(np.abs(x_hw), 99.0)) if x_hw.size > 0 else 1.0
    wave_v = abs_q if abs_q > 0.0 else 1.0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.0), sharey=True)

    imshow_hw(
        axes[0, 0],
        x_hw,
        transpose_for_trace_time=bool(transpose_for_trace_time),
        title='waveform (ch0)',
        cmap='gray',
        vmin=-wave_v,
        vmax=wave_v,
    )
    imshow_hw(
        axes[0, 1],
        tp_hw,
        transpose_for_trace_time=bool(transpose_for_trace_time),
        title='target P',
        cmap=None,
        vmin=0.0,
        vmax=1.0,
    )
    imshow_hw(
        axes[0, 2],
        pp_hw,
        transpose_for_trace_time=bool(transpose_for_trace_time),
        title='pred P (softmax)',
        cmap=None,
        vmin=0.0,
        vmax=1.0,
    )

    imshow_hw(
        axes[1, 0],
        ts_hw,
        transpose_for_trace_time=bool(transpose_for_trace_time),
        title='target S',
        cmap=None,
        vmin=0.0,
        vmax=1.0,
    )
    imshow_hw(
        axes[1, 1],
        ps_hw,
        transpose_for_trace_time=bool(transpose_for_trace_time),
        title='pred S (softmax)',
        cmap=None,
        vmin=0.0,
        vmax=1.0,
    )
    imshow_hw(
        axes[1, 2],
        pn_hw,
        transpose_for_trace_time=bool(transpose_for_trace_time),
        title='pred Noise (softmax)',
        cmap=None,
        vmin=0.0,
        vmax=1.0,
    )

    if title is not None:
        fig.suptitle(str(title))

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _pick_str(v: Any, *, b: int) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)) and 0 <= int(b) < len(v):
        return _pick_str(v[int(b)], b=b)
    return None


def make_title_from_batch_meta(
    batch: dict[str, Any] | None,
    *,
    b: int = 0,
) -> str | None:
    """Build a lightweight title string from a collated batch/meta.

    This is intended for debug visualization (best-effort).
    """
    if batch is None:
        return None

    meta = batch.get('meta', None)
    if not isinstance(meta, dict):
        meta = {}

    # Prefer top-level keys, but allow meta fallbacks.
    file_path = _pick_str(batch.get('file_path'), b=b) or _pick_str(
        meta.get('file_path'), b=b
    )
    group_id = _pick_str(batch.get('group_id'), b=b) or _pick_str(
        meta.get('group_id'), b=b
    )
    primary_unique = _pick_str(batch.get('primary_unique'), b=b) or _pick_str(
        meta.get('primary_unique'), b=b
    )
    key_name = _pick_str(batch.get('key_name'), b=b) or _pick_str(
        meta.get('key_name'), b=b
    )
    secondary_key = _pick_str(batch.get('secondary_key'), b=b) or _pick_str(
        meta.get('secondary_key'), b=b
    )

    parts: list[str] = []
    if key_name is not None and secondary_key is not None:
        parts.append(f'key={key_name}/{secondary_key}')
    elif key_name is not None:
        parts.append(f'key={key_name}')

    if primary_unique is not None:
        parts.append(f'primary={primary_unique}')
    if group_id is not None:
        parts.append(f'group={group_id}')
    if file_path is not None:
        parts.append(f'file={Path(file_path).name}')

    if len(parts) == 0:
        return None
    return ' | '.join(parts)


__all__ = [
    'make_title_from_batch_meta',
    'save_psn_debug_png',
]
