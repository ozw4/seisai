from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from seisai_dataset.ffid_gather_iter import FFIDGatherIterator, SortWithinGather
from seisai_utils.segy_write import write_segy_like_input

from seisai_engine.predict import _run_tiled

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


@dataclass(frozen=True)
class Tiled2DConfig:
    tile_h: int = 128
    overlap_h: int = 64
    tile_w: int = 6016
    overlap_w: int = 1024
    tiles_per_batch: int = 16
    amp: bool = True
    use_tqdm: bool = False


def _validate_tiled2d_cfg(cfg: Tiled2DConfig) -> None:
    if cfg.tile_h <= 0 or cfg.tile_w <= 0:
        msg = 'tile_h/tile_w must be positive'
        raise ValueError(msg)
    if cfg.overlap_h < 0 or cfg.overlap_w < 0:
        msg = 'overlap_h/overlap_w must be non-negative'
        raise ValueError(msg)
    if cfg.overlap_h >= cfg.tile_h or cfg.overlap_w >= cfg.tile_w:
        msg = 'overlap must be < tile'
        raise ValueError(msg)
    if cfg.tiles_per_batch <= 0:
        msg = 'tiles_per_batch must be positive'
        raise ValueError(msg)


def _infer_hw_denorm_like_input(
    model: torch.nn.Module,
    x_hw: np.ndarray,
    *,
    device: torch.device,
    cfg: Tiled2DConfig,
    eps_std: float,
) -> np.ndarray:
    """x_hw(=元スケール) -> per-trace標準化 -> tiled推論 -> denormして元スケールで返す."""
    if x_hw.ndim != 2:
        msg = f'x_hw must be (H,W), got {x_hw.shape}'
        raise ValueError(msg)

    x = np.asarray(x_hw, dtype=np.float32)
    h = int(x.shape[0])
    w = int(x.shape[1])
    if h <= 0 or w <= 0:
        msg = f'invalid x_hw shape: {x.shape}'
        raise ValueError(msg)

    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True) + float(eps_std)
    xn = (x - m) / s

    x_bchw = torch.from_numpy(xn[None, None, :, :]).to(
        device=device, dtype=torch.float32
    )

    tile_h = min(int(cfg.tile_h), h)
    tile_w = min(int(cfg.tile_w), w)

    ov_h = int(cfg.overlap_h)
    ov_w = int(cfg.overlap_w)
    ov_h = 0 if tile_h == 1 else min(ov_h, tile_h - 1)
    ov_w = 0 if tile_w == 1 else min(ov_w, tile_w - 1)

    y = _run_tiled(
        model,
        x_bchw,
        tile=(tile_h, tile_w),
        overlap=(ov_h, ov_w),
        amp=bool(cfg.amp),
        use_tqdm=bool(cfg.use_tqdm),
        tiles_per_batch=int(cfg.tiles_per_batch),
    )

    y_hw = y[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    return y_hw * s + m


@torch.no_grad()
def run_ffid_gather_infer_and_write_segy(
    model: torch.nn.Module,
    *,
    segy_files: list[str],
    out_dir: str | Path,
    out_suffix: str = '_pred.sgy',
    overwrite: bool = False,
    ffids: Iterable[int] | None = None,
    sort_within: SortWithinGather = 'chno',
    tiled_cfg: Tiled2DConfig | None = None,
    eps_std: float = 1e-8,
    device: torch.device | None = None,
) -> list[Path]:
    """FFID gather 単位で推論して、入力SEGYと同じサイズ/ヘッダ維持でSEGY出力する。.

    - FFIDごとに gather(H,W) を作る(H=trace数, W=全サンプル)
    - per-trace mean/std で標準化して推論
    - 出力を mean/std で denorm して元スケールに戻す(A案)
    - trace_indices を使って file 内の元のtrace順へ戻し、fileごとに full (n_traces,n_samples) を作る
    - write_segy_like_input で保存(headerは触らない、sample format維持)

    Returns:
            各入力ファイルに対応する出力SEGYパス

    """
    if len(segy_files) == 0:
        msg = 'segy_files must be non-empty'
        raise ValueError(msg)
    if not isinstance(out_suffix, str) or len(out_suffix) == 0:
        msg = 'out_suffix must be non-empty str'
        raise ValueError(msg)

    cfg = tiled_cfg or Tiled2DConfig()
    _validate_tiled2d_cfg(cfg)

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    ffid_list = None
    if ffids is not None:
        ffid_list = [int(x) for x in ffids]
        if len(ffid_list) == 0:
            msg = 'ffids must be non-empty if provided'
            raise ValueError(msg)

    out_paths: list[Path] = []

    with FFIDGatherIterator(segy_files, sort_within=sort_within) as it:
        for fi, src_path in enumerate(it.segy_files):
            info = it.file_infos[fi]
            n_traces = int(info['n_traces'])
            n_samples = int(info['n_samples'])
            if n_traces <= 0 or n_samples <= 0:
                msg = f'invalid segy shape: n_traces={n_traces}, n_samples={n_samples} ({src_path})'
                raise ValueError(
                    msg
                )

            out_hw = np.zeros((n_traces, n_samples), dtype=np.float32)
            seen = np.zeros((n_traces,), dtype=np.bool_)

            for g in it.iter_gathers(file_indices=[fi], ffids=ffid_list):
                idx = np.asarray(g.trace_indices, dtype=np.int64)
                if idx.size == 0:
                    continue
                if idx.min() < 0 or idx.max() >= n_traces:
                    msg = 'trace_indices out of range'
                    raise ValueError(msg)
                if int(g.x_hw.shape[1]) != n_samples:
                    msg = f'gather W mismatch: got {int(g.x_hw.shape[1])}, want {n_samples} ({src_path})'
                    raise ValueError(
                        msg
                    )

                y_hw = _infer_hw_denorm_like_input(
                    model,
                    g.x_hw,
                    device=device,
                    cfg=cfg,
                    eps_std=float(eps_std),
                )

                out_hw[idx] = y_hw
                seen[idx] = True

            if not bool(seen.all()):
                miss = int((~seen).sum())
                msg = f'some traces were not filled (miss={miss}) in {src_path}'
                raise ValueError(
                    msg
                )

            dst = write_segy_like_input(
                src_path=src_path,
                out_dir=out_dir,
                out_suffix=out_suffix,
                data_hw=out_hw,
                overwrite=bool(overwrite),
            )
            out_paths.append(dst)

    return out_paths


__all__ = [
    'Tiled2DConfig',
    'run_ffid_gather_infer_and_write_segy',
]
