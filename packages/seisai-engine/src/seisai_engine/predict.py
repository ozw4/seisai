from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def _run_tiled(
	model: torch.nn.Module,
	x: torch.Tensor,
	*,
	tile: tuple[int, int] = (128, 6016),
	overlap: int = 32,
	amp: bool = True,
) -> torch.Tensor:
	"""Sliding-window 推論ユーティリティ（多出力チャネル対応）。
	前提: model に int 属性 out_chans が存在し、forward は (B,C,H,W)->(B,out_chans,h,w)。

	Args:
	  model: nn.Module（model.out_chans を必須とする）
	  x: (B,C,H,W)
	  tile: (tile_h, tile_w)
	  overlap: 0 <= overlap < tile_h かつ < tile_w
	  amp: CUDA 環境で autocast を有効化

	Returns:
	  (B, out_chans, H, W) float32

	"""
	assert x.ndim == 4, 'x must be (B,C,H,W)'
	assert hasattr(model, 'out_chans'), 'model.out_chans is required'
	c_out = int(model.out_chans)
	assert c_out > 0

	b, _, h, w = x.shape
	tile_h, tile_w = tile
	assert tile_h > 0 and tile_w > 0
	assert 0 <= overlap < tile_h and 0 <= overlap < tile_w

	stride_h = tile_h - overlap
	stride_w = tile_w - overlap

	out = torch.zeros((b, c_out, h, w), device=x.device, dtype=torch.float32)
	weight = torch.zeros((b, 1, h, w), device=x.device, dtype=torch.float32)

	use_amp = bool(amp and torch.cuda.is_available())
	model.eval()

	for top in range(0, h, stride_h):
		for left in range(0, w, stride_w):
			bottom = min(top + tile_h, h)
			right = min(left + tile_w, w)

			h0 = max(0, bottom - tile_h)
			w0 = max(0, right - tile_w)

			patch = x[:, :, h0:bottom, w0:right]
			ph, pw = patch.shape[-2], patch.shape[-1]

			pad_h = max(0, tile_h - ph)
			pad_w = max(0, tile_w - pw)
			if pad_h or pad_w:
				patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='constant', value=0.0)

			with torch.cuda.amp.autocast(enabled=use_amp):
				yp = model(patch)  # (B, c_out, tile_h, tile_w) を想定
			assert yp.ndim == 4 and yp.shape[0] == b and yp.shape[1] == c_out

			yp = yp[..., :ph, :pw].to(torch.float32)

			out[:, :, h0:bottom, w0:right] += yp
			weight[:, :, h0:bottom, w0:right] += 1.0

	out /= weight.clamp_min(1.0)
	return out


@torch.no_grad()
def infer_tiled_chw(
	model: torch.nn.Module,
	x_chw: torch.Tensor,
	*,
	tile: tuple[int, int] = (128, 6016),
	overlap: int = 32,
	amp: bool = True,
) -> torch.Tensor:
	"""(C,H,W) -> (out_chans,H,W) の推論ラッパ。バッチ=1で _run_tiled を呼ぶ。
	事前の正規化等は呼び出し側で行うこと。
	"""
	assert x_chw.ndim == 3, 'x_chw must be (C,H,W)'
	x = x_chw.unsqueeze(0)  # (1,C,H,W)
	y = _run_tiled(model, x, tile=tile, overlap=overlap, amp=amp)  # (1,out_chans,H,W)
	return y.squeeze(0)  # (out_chans,H,W)
