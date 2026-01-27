from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from seisai_engine.predict import PostTileTransform, TileTransform, _run_tiled


@dataclass(frozen=True)
class TiledWConfig:
	"""W 方向タイル推論の設定。

	想定:
	- 入力は (B,C,H,W)
	- H は分割しない（tile_h=H, overlap_h=0）
	- W は tile_w を基準に分割する

	注意:
	- W < tile_w は設定ミスとして即失敗（フォールバックしない）
	- model は out_chans 属性を持ち、out_chans==1 を前提
	"""

	tile_w: int = 6016
	overlap_w: int = 1024
	tiles_per_batch: int = 16
	amp: bool = True
	use_tqdm: bool = False


def _validate_cfg(cfg: TiledWConfig) -> None:
	if cfg.tile_w <= 0:
		raise ValueError('tile_w must be positive')
	if cfg.overlap_w < 0:
		raise ValueError('overlap_w must be non-negative')
	if cfg.overlap_w >= cfg.tile_w:
		raise ValueError('overlap_w must be < tile_w')
	if cfg.tiles_per_batch <= 0:
		raise ValueError('tiles_per_batch must be positive')


def _validate_model_for_infer(model: torch.nn.Module) -> None:
	if not hasattr(model, 'out_chans'):
		raise ValueError('model must have attribute out_chans')
	c_out = int(getattr(model, 'out_chans'))
	if c_out != 1:
		raise ValueError(f'model.out_chans must be 1 for this runner, got {c_out}')


@torch.no_grad()
def infer_batch_tiled_w(
	model: torch.nn.Module,
	x_bchw: torch.Tensor,
	*,
	cfg: TiledWConfig,
	tile_transform: TileTransform | None = None,
	post_tile_transform: PostTileTransform | None = None,
) -> torch.Tensor:
	"""1バッチ分の W 方向タイル推論。

	Args:
		model: out_chans==1 を前提
		x_bchw: (B,C,H,W) tensor
		cfg: TiledWConfig

	Returns:
		logits: (B,1,H,W)
	"""
	_validate_cfg(cfg)
	_validate_model_for_infer(model)

	if not isinstance(x_bchw, torch.Tensor) or x_bchw.ndim != 4:
		raise ValueError(
			f'x_bchw must be torch.Tensor with shape (B,C,H,W), got {type(x_bchw)} {getattr(x_bchw, "shape", None)}'
		)

	b, _c, h, w = x_bchw.shape
	if b <= 0 or h <= 0 or w <= 0:
		raise ValueError(f'invalid input shape: {tuple(x_bchw.shape)}')
	if w < cfg.tile_w:
		raise ValueError(f'W ({w}) must be >= tile_w ({cfg.tile_w})')

	return _run_tiled(
		model,
		x_bchw,
		tile=(int(h), int(cfg.tile_w)),
		overlap=(0, int(cfg.overlap_w)),
		amp=bool(cfg.amp),
		use_tqdm=bool(cfg.use_tqdm),
		tiles_per_batch=int(cfg.tiles_per_batch),
		tile_transform=tile_transform,
		post_tile_transform=post_tile_transform,
	)


@torch.no_grad()
def iter_infer_loader_tiled_w(
	model: torch.nn.Module,
	loader: Any,
	*,
	device: torch.device | None = None,
	cfg: TiledWConfig | None = None,
	output_to_cpu: bool = False,
	max_batches: int | None = None,
	tile_transform: TileTransform | None = None,
	post_tile_transform: PostTileTransform | None = None,
) -> Iterator[tuple[torch.Tensor, list[dict[str, Any]]]]:
	"""DataLoader を回して推論し、(logits, metas) を逐次返す。

	loader は (x_bchw, metas) を返す想定:
	- x_bchw: torch.Tensor (B,C,H,W)
	- metas: list[dict] （長さ B）

	例: InferenceGatherWindowsDataset + collate_pad_w_right
	"""
	cfg = cfg or TiledWConfig()
	_validate_cfg(cfg)
	_validate_model_for_infer(model)

	if device is None:
		device = next(model.parameters()).device

	model.eval()

	for step, batch in enumerate(loader):
		if max_batches is not None and step >= int(max_batches):
			break

		if not isinstance(batch, (tuple, list)) or len(batch) != 2:
			raise ValueError('loader must yield (x_bchw, metas)')
		x_bchw, metas = batch

		if not isinstance(x_bchw, torch.Tensor) or x_bchw.ndim != 4:
			raise ValueError(
				f'x_bchw must be torch.Tensor (B,C,H,W), got {type(x_bchw)} {getattr(x_bchw, "shape", None)}'
			)
		if not isinstance(metas, list):
			raise ValueError('metas must be a list[dict]')
		if len(metas) != int(x_bchw.shape[0]):
			raise ValueError(
				f'len(metas) must equal batch size B={int(x_bchw.shape[0])}, got {len(metas)}'
			)

		non_blocking = bool(device.type == 'cuda')
		x_bchw = x_bchw.to(device=device, non_blocking=non_blocking)

		logits = infer_batch_tiled_w(
			model,
			x_bchw,
			cfg=cfg,
			tile_transform=tile_transform,
			post_tile_transform=post_tile_transform,
		)

		if output_to_cpu:
			logits = logits.detach().cpu()

		yield logits, metas


@torch.no_grad()
def run_infer_loader_tiled_w(
	model: torch.nn.Module,
	loader: Any,
	*,
	device: torch.device | None = None,
	cfg: TiledWConfig | None = None,
	output_to_cpu: bool = False,
	max_batches: int | None = None,
	tile_transform: TileTransform | None = None,
	post_tile_transform: PostTileTransform | None = None,
) -> list[tuple[torch.Tensor, list[dict[str, Any]]]]:
	"""iter_infer_loader_tiled_w の収集版（小規模デモ用）。"""
	return list(
		iter_infer_loader_tiled_w(
			model,
			loader,
			device=device,
			cfg=cfg,
			output_to_cpu=output_to_cpu,
			max_batches=max_batches,
			tile_transform=tile_transform,
			post_tile_transform=post_tile_transform,
		)
	)


__all__ = [
	'TiledWConfig',
	'infer_batch_tiled_w',
	'iter_infer_loader_tiled_w',
	'run_infer_loader_tiled_w',
]
