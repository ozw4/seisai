# predict.py 内：tqdm だけで途中経過を表示する ＋ タイルの“バッチ化”対応版
import importlib.util
from contextlib import nullcontext

import torch


def _tile_starts(full: int, tile: int, overlap: int) -> list[int]:
	assert tile > 0 and 0 <= overlap < tile
	stride = tile - overlap
	starts = [0]
	while starts[-1] + tile < full:
		nxt = starts[-1] + stride
		if nxt + tile >= full:
			starts.append(max(full - tile, 0))
			break
		starts.append(nxt)
	return sorted(set(starts))


@torch.no_grad()
def _run_tiled(
	model: torch.nn.Module,
	x: torch.Tensor,
	*,
	tile: tuple[int, int] = (128, 128),
	overlap: tuple[int, int] = (32, 32),
	amp: bool = True,
	use_tqdm: bool = False,
	tiles_per_batch: int = 8,
) -> torch.Tensor:
	"""Sliding-window 推論（多出力チャネル）。overlap を軸別指定し、tqdm で進捗表示。
	タイルをまとめて1回の forward に投入（バッチ化）して起動オーバーヘッドを削減する。
	前提: model に int 属性 out_chans が存在し、forward は (N,C,h,w)->(N,out_chans,h,w)。
	Returns: (B, out_chans, H, W) float32
	"""
	assert x.ndim == 4, 'x must be (B,C,H,W)'
	assert hasattr(model, 'out_chans'), 'model.out_chans is required'
	c_out = int(model.out_chans)
	assert c_out > 0
	assert tiles_per_batch > 0

	b, c, h, w = x.shape
	tile_h, tile_w = tile
	ov_h, ov_w = overlap
	assert tile_h > 0 and tile_w > 0
	assert 0 <= ov_h < tile_h and 0 <= ov_w < tile_w

	stride_h = tile_h - ov_h
	stride_w = tile_w - ov_w
	assert stride_h > 0 and stride_w > 0

	ys = _tile_starts(h, tile_h, ov_h)
	xs = _tile_starts(w, tile_w, ov_w)

	# すべてのタイル矩形を列挙（h0,w0 と実サイズ ph,pw）
	rects: list[tuple[int, int, int, int]] = []
	for top in ys:
		bottom = min(top + tile_h, h)
		h0 = max(0, bottom - tile_h)
		ph = bottom - h0
		for left in xs:
			right = min(left + tile_w, w)
			w0 = max(0, right - tile_w)
			pw = right - w0
			rects.append((h0, w0, ph, pw))

	total_tiles = len(rects)

	pbar = None
	if use_tqdm:
		assert importlib.util.find_spec('tqdm') is not None, (
			'tqdm is required when use_tqdm=True'
		)
		from tqdm import tqdm

		pbar = tqdm(total=total_tiles, desc='infer_tiled', unit='tile', leave=False)

	out = torch.zeros((b, c_out, h, w), device=x.device, dtype=torch.float32)
	weight = torch.zeros((b, 1, h, w), device=x.device, dtype=torch.float32)

	use_amp = bool(amp and (x.device.type == 'cuda'))
	model.eval()

	# ステージング・バッファ（タイル束×バッチ）を1度だけ確保して使い回す
	max_slots = tiles_per_batch * b
	stage_x = torch.empty(
		(max_slots, c, tile_h, tile_w), device=x.device, dtype=x.dtype
	)

	with torch.cuda.device(x.device) if x.device.type == 'cuda' else nullcontext():
		for start in range(0, total_tiles, tiles_per_batch):
			batch_rects = rects[start : start + tiles_per_batch]
			slots = len(batch_rects) * b

			# ゼロ埋めしてからスライスコピー（右下パディング）
			stage_x.zero_()
			for t, (h0, w0, ph, pw) in enumerate(batch_rects):
				patch = x[:, :, h0 : h0 + ph, w0 : w0 + pw]  # (B,C,ph,pw)
				stage_x[t * b : (t + 1) * b, :, :ph, :pw].copy_(patch)

			with torch.amp.autocast('cuda', enabled=use_amp):
				yb = model(stage_x[:slots])  # (slots, c_out, tile_h, tile_w)
			assert yb.ndim == 4 and yb.shape[1] == c_out

			# 元のテンソルへ合成
			for t, (h0, w0, ph, pw) in enumerate(batch_rects):
				sl = slice(t * b, (t + 1) * b)
				yp = yb[sl, :, :ph, :pw].to(torch.float32)  # (B,c_out,ph,pw)
				out[:, :, h0 : h0 + ph, w0 : w0 + pw] += yp
				weight[:, :, h0 : h0 + ph, w0 : w0 + pw] += 1.0

			if pbar is not None:
				pbar.update(len(batch_rects))

	if pbar is not None:
		pbar.close()

	# 未カバーがあれば即時失敗
	assert torch.all(weight > 0), (
		'uncovered pixels detected; check tile/overlap settings'
	)

	out /= weight
	return out


@torch.no_grad()
def infer_tiled_chw(
	model: torch.nn.Module,
	x_chw: torch.Tensor,
	*,
	tile: tuple[int, int] = (128, 128),
	overlap: tuple[int, int] = (32, 32),
	amp: bool = True,
	use_tqdm: bool = False,
	tiles_per_batch: int = 8,
) -> torch.Tensor:
	"""(C,H,W) 入力のタイル推論ラッパ。"""
	assert x_chw.ndim == 3, 'x_chw must be (C,H,W)'
	x = x_chw.unsqueeze(0)  # (1,C,H,W)
	y = _run_tiled(
		model,
		x,
		tile=tile,
		overlap=overlap,
		amp=amp,
		use_tqdm=use_tqdm,
		tiles_per_batch=tiles_per_batch,
	)  # (1,out_chans,H,W)
	return y.squeeze(0)
