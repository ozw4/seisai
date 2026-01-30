from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from seisai_utils.viz import ImshowPanel, save_imshow_row


@dataclass(frozen=True)
class PairTriptychVisConfig:
	cmap: str | None = 'seismic'
	vmin: float | None = -3.0
	vmax: float | None = 3.0
	transpose_for_trace_time: bool = True
	per_trace_norm: bool = True
	per_trace_eps: float = 1e-8
	figsize: tuple[float, float] = (20.0, 15.0)
	dpi: int = 300


def _to_numpy_bchw(x: torch.Tensor | np.ndarray, *, name: str) -> np.ndarray:
	if isinstance(x, torch.Tensor):
		if int(x.ndim) != 4:
			raise ValueError(
				f'{name} must be (B,C,H,W) tensor, got shape={tuple(x.shape)}'
			)
		return x.detach().cpu().numpy()
	if isinstance(x, np.ndarray):
		if int(x.ndim) != 4:
			raise ValueError(f'{name} must be (B,C,H,W) array, got shape={x.shape}')
		return x
	raise TypeError(f'{name} must be torch.Tensor or numpy.ndarray')


def _select_hw(x_bchw: np.ndarray, *, b: int, c: int, name: str) -> np.ndarray:
	if not (0 <= int(b) < int(x_bchw.shape[0])):
		raise ValueError(
			f'{name}: batch index out of range: b={b} B={int(x_bchw.shape[0])}'
		)
	if not (0 <= int(c) < int(x_bchw.shape[1])):
		raise ValueError(
			f'{name}: channel index out of range: c={c} C={int(x_bchw.shape[1])}'
		)
	hw = x_bchw[int(b), int(c)]
	if hw.ndim != 2:
		raise ValueError(f'{name}: expected (H,W), got {hw.shape}')
	return np.asarray(hw)


def _per_trace_zscore_hw(x_hw: np.ndarray, *, eps: float) -> np.ndarray:
	x = np.asarray(x_hw, dtype=np.float32)
	if x.ndim != 2:
		raise ValueError(f'x_hw must be (H,W), got {x.shape}')
	m = x.mean(axis=1, keepdims=True)
	s = x.std(axis=1, keepdims=True) + float(eps)
	return (x - m) / s


def _pick_str_from_batch(batch: dict[str, Any], key: str, b: int) -> str | None:
	if key not in batch:
		return None
	v = batch[key]
	if isinstance(v, str):
		return v
	if isinstance(v, (Path,)):
		return str(v)
	if isinstance(v, list):
		if 0 <= int(b) < len(v):
			vi = v[int(b)]
			if isinstance(vi, str):
				return vi
			if isinstance(vi, Path):
				return str(vi)
	return None


def make_pair_suptitle(batch: dict[str, Any] | None, *, b: int = 0) -> str | None:
	if batch is None:
		return None

	key_name = _pick_str_from_batch(batch, 'key_name', b)
	secondary_key = _pick_str_from_batch(batch, 'secondary_key', b)

	fp_in = _pick_str_from_batch(batch, 'file_path_input', b)
	fp_tg = _pick_str_from_batch(batch, 'file_path_target', b)

	parts: list[str] = []

	if key_name is not None and secondary_key is not None:
		parts.append(f'key={key_name}/{secondary_key}')
	elif key_name is not None:
		parts.append(f'key={key_name}')

	if fp_in is not None:
		parts.append(f'in={Path(fp_in).name}')
	if fp_tg is not None:
		parts.append(f'tg={Path(fp_tg).name}')

	if len(parts) == 0:
		return None
	return ' | '.join(parts)


def save_pair_triptych_png(
	png_path: str | Path,
	*,
	x_in_bchw: torch.Tensor | np.ndarray,
	x_tg_bchw: torch.Tensor | np.ndarray,
	x_pr_bchw: torch.Tensor | np.ndarray,
	cfg: PairTriptychVisConfig,
	b: int = 0,
	c: int = 0,
	suptitle: str | None = None,
	batch: dict[str, Any] | None = None,
) -> None:
	x_in = _to_numpy_bchw(x_in_bchw, name='x_in_bchw')
	x_tg = _to_numpy_bchw(x_tg_bchw, name='x_tg_bchw')
	x_pr = _to_numpy_bchw(x_pr_bchw, name='x_pr_bchw')

	if x_in.shape != x_tg.shape or x_in.shape != x_pr.shape:
		raise ValueError(
			f'shape mismatch: in={x_in.shape} tg={x_tg.shape} pr={x_pr.shape}'
		)

	in_hw = _select_hw(x_in, b=b, c=c, name='x_in_bchw')
	tg_hw = _select_hw(x_tg, b=b, c=c, name='x_tg_bchw')
	pr_hw = _select_hw(x_pr, b=b, c=c, name='x_pr_bchw')

	if cfg.per_trace_norm:
		in_hw = _per_trace_zscore_hw(in_hw, eps=cfg.per_trace_eps)
		tg_hw = _per_trace_zscore_hw(tg_hw, eps=cfg.per_trace_eps)
		pr_hw = _per_trace_zscore_hw(pr_hw, eps=cfg.per_trace_eps)

	if suptitle is None:
		suptitle = make_pair_suptitle(batch, b=b)

	panels = [
		ImshowPanel(
			title='Input', data_hw=in_hw, cmap=cfg.cmap, vmin=cfg.vmin, vmax=cfg.vmax
		),
		ImshowPanel(
			title='Target', data_hw=tg_hw, cmap=cfg.cmap, vmin=cfg.vmin, vmax=cfg.vmax
		),
		ImshowPanel(
			title='Pred', data_hw=pr_hw, cmap=cfg.cmap, vmin=cfg.vmin, vmax=cfg.vmax
		),
	]

	save_imshow_row(
		png_path,
		panels,
		suptitle=suptitle,
		transpose_for_trace_time=cfg.transpose_for_trace_time,
		figsize=cfg.figsize,
		dpi=int(cfg.dpi),
	)


def save_pair_triptych_step_png(
	out_dir: str | Path,
	*,
	step: int,
	x_in_bchw: torch.Tensor | np.ndarray,
	x_tg_bchw: torch.Tensor | np.ndarray,
	x_pr_bchw: torch.Tensor | np.ndarray,
	cfg: PairTriptychVisConfig,
	b: int = 0,
	c: int = 0,
	batch: dict[str, Any] | None = None,
	prefix: str = 'pair_triptych_step',
) -> Path:
	out_dir = Path(out_dir)
	png_path = out_dir / f'{prefix}{int(step):04d}.png'

	save_pair_triptych_png(
		png_path,
		x_in_bchw=x_in_bchw,
		x_tg_bchw=x_tg_bchw,
		x_pr_bchw=x_pr_bchw,
		cfg=cfg,
		b=b,
		c=c,
		batch=batch,
	)

	return png_path


__all__ = [
	'PairTriptychVisConfig',
	'make_pair_suptitle',
	'save_pair_triptych_png',
	'save_pair_triptych_step_png',
]
