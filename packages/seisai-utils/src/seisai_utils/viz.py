from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
		raise ValueError(f'data must be 2D (H,W), got shape={a.shape}')
	return a


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
	"""(H,W) を表示。transpose_for_trace_time=True で x=Trace, y=Time になるように表示する。"""
	hw = _as_hw(data_hw)
	img = hw.T if transpose_for_trace_time else hw
	ax.imshow(
		img, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha
	)

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
	"""Base の上に overlay を重ねる（example_segy_gather_pipline_ds.py のやつ）。"""
	base = _as_hw(base_hw)
	ov = _as_hw(overlay_hw)
	if base.shape != ov.shape:
		raise ValueError(f'shape mismatch: base={base.shape} overlay={ov.shape}')

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
	"""横一列に並べて保存（入力/GT/Pred みたいな triptych 用）。"""
	panels = list(panels)
	if len(panels) == 0:
		raise ValueError('panels must be non-empty')

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
