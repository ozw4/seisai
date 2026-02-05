from __future__ import annotations

from seisai_utils.viz_pair import PairTriptychVisConfig, save_pair_triptych_step_png

__all__ = ['build_triptych_cfg', 'save_triptych_step']


def build_triptych_cfg(
	*,
	cmap: str,
	vmin: float,
	vmax: float,
	transpose_for_trace_time: bool,
	per_trace_norm: bool,
	per_trace_eps: float,
	figsize: tuple[float, float],
	dpi: int,
) -> PairTriptychVisConfig:
	return PairTriptychVisConfig(
		cmap=str(cmap),
		vmin=float(vmin),
		vmax=float(vmax),
		transpose_for_trace_time=bool(transpose_for_trace_time),
		per_trace_norm=bool(per_trace_norm),
		per_trace_eps=float(per_trace_eps),
		figsize=figsize,
		dpi=int(dpi),
	)


def save_triptych_step(
	*,
	out_dir: str,
	step: int,
	x_in_bchw,
	x_tg_bchw,
	x_pr_bchw,
	cfg: PairTriptychVisConfig,
	batch: dict,
	c: int = 0,
) -> None:
	save_pair_triptych_step_png(
		out_dir,
		step=step,
		x_in_bchw=x_in_bchw,
		x_tg_bchw=x_tg_bchw,
		x_pr_bchw=x_pr_bchw,
		cfg=cfg,
		batch=batch,
		c=c,
	)
