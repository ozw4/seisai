from collections.abc import Callable

import numpy as np
import torch
from seisai_transforms.augment import ViewCompose
from seisai_utils.validator import validate_array
from tqdm.auto import tqdm

# (H,W) -> (H,W) を想定（ViewCompose 相当）
PreView = Callable[[np.ndarray], np.ndarray]
# (B,C,ph,pw) -> (B,C,ph,pw) を想定（タイル単位の正規化など）
TileTransform = Callable[[torch.Tensor], torch.Tensor]
# (B,C,ph,pw) -> (B,C,ph,pw) を想定（タイル単位の予測後処理など）
PostTileTransform = Callable[[torch.Tensor], torch.Tensor]


class _NoRandRNG:
	# 乱数使用を禁止するダミー RNG（PreView 内で rng を使ったら即失敗）
	def random(self, *a, **k):
		raise RuntimeError('random() forbidden in deterministic inference')

	def uniform(self, *a, **k):
		raise RuntimeError('uniform() forbidden in deterministic inference')

	def integers(self, *a, **k):
		raise RuntimeError('integers() forbidden in deterministic inference')

	bit_generator = object()


@torch.no_grad()
def _run_tiled(
	model: torch.nn.Module,
	x: torch.Tensor,  # (B,C,H,W)
	*,
	tile: tuple[int, int] = (128, 128),
	overlap: tuple[int, int] = (32, 32),
	amp: bool = True,
	use_tqdm: bool = False,
	tiles_per_batch: int = 8,
	tile_transform: ViewCompose | None = None,
	post_tile_transform: ViewCompose | None = None,
) -> torch.Tensor:
	assert x.ndim == 4 and hasattr(model, 'out_chans')
	c_out = int(model.out_chans)
	b, c, h, w = x.shape
	tile_h, tile_w = tile
	ov_h, ov_w = overlap
	assert 0 < tile_h <= h and 0 < tile_w <= w
	assert 0 <= ov_h < tile_h and 0 <= ov_w < tile_w
	stride_h = tile_h - ov_h
	stride_w = tile_w - ov_w
	assert tiles_per_batch > 0 and stride_h > 0 and stride_w > 0

	def _tile_starts(full: int, tile_size: int, overlap_size: int) -> list[int]:
		stride = tile_size - overlap_size
		starts = [0]
		while starts[-1] + tile_size < full:
			nxt = starts[-1] + stride
			if nxt + tile_size >= full:
				starts.append(max(full - tile_size, 0))
				break
			starts.append(nxt)
		return sorted(set(starts))

	# Hann 窓ベースの 2D 重み（中心が大きく、端が小さい）
	def _make_tile_weight(
		tile_h: int, tile_w: int, device: torch.device
	) -> torch.Tensor:
		wy = torch.hann_window(
			tile_h, periodic=False, device=device, dtype=torch.float32
		)
		wx = torch.hann_window(
			tile_w, periodic=False, device=device, dtype=torch.float32
		)
		w2d = wy.view(tile_h, 1) * wx.view(1, tile_w)  # outer product

		# 端を完全な 0 にすると画像端ピクセルの weight が 0 になりうるので、
		# ごく小さい値で下限を切り上げておく
		eps = 1e-3
		w2d = torch.clamp(w2d, min=eps)

		# shape: (1,1,H,W) にしておく（B,C に broadcast させる）
		return w2d.view(1, 1, tile_h, tile_w)

	ys = _tile_starts(h, tile_h, ov_h)
	xs = _tile_starts(w, tile_w, ov_w)

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
	out = torch.zeros((b, c_out, h, w), device=x.device, dtype=torch.float32)
	weight = torch.zeros((b, 1, h, w), device=x.device, dtype=torch.float32)
	use_amp = bool(amp and (x.device.type == 'cuda'))
	model.eval()

	max_slots = tiles_per_batch * b
	stage_x = torch.empty(
		(max_slots, c, tile_h, tile_w), device=x.device, dtype=x.dtype
	)

	# 全タイル共通の Hann 窓重み
	tile_weight_full = _make_tile_weight(tile_h, tile_w, x.device)

	if use_tqdm:
		num_batches = (total_tiles + tiles_per_batch - 1) // tiles_per_batch
		batch_iter = tqdm(
			range(0, total_tiles, tiles_per_batch),
			total=num_batches,
			desc='tiled inference',
		)
	else:
		batch_iter = range(0, total_tiles, tiles_per_batch)

	for start_idx in batch_iter:
		batch_rects = rects[start_idx : start_idx + tiles_per_batch]
		slots = len(batch_rects) * b
		stage_x.zero_()

		for t, (h0, w0, ph, pw) in enumerate(batch_rects):
			patch = x[:, :, h0 : h0 + ph, w0 : w0 + pw]

			if tile_transform is not None:
				patch = tile_transform(patch, return_meta=False)
				assert patch.shape == (b, c, ph, pw)
				assert patch.device == x.device

			stage_x[t * b : (t + 1) * b, :, :ph, :pw].copy_(patch)

		with torch.amp.autocast('cuda', enabled=use_amp):
			yb = model(stage_x[:slots])

		for t, (h0, w0, ph, pw) in enumerate(batch_rects):
			sl = slice(t * b, (t + 1) * b)
			yp = yb[sl, :, :ph, :pw].to(torch.float32)

			if post_tile_transform is not None:
				yp = post_tile_transform(yp, return_meta=False)
				assert yp.shape == (b, c_out, ph, pw)
				assert yp.device == x.device
				yp = yp.to(torch.float32)

			# タイル内の有効部分だけ切り出し
			w_patch = tile_weight_full[:, :, :ph, :pw]  # (1,1,ph,pw)

			out[:, :, h0 : h0 + ph, w0 : w0 + pw] += yp * w_patch
			weight[:, :, h0 : h0 + ph, w0 : w0 + pw] += w_patch

	assert torch.all(weight > 0), (
		'未カバー領域があります。tile/overlap を見直してください。'
	)
	out /= weight
	return out


@torch.no_grad()
def infer_tiled_chw(
	model: torch.nn.Module,
	x_chw: np.ndarray,  # ndarray(CHW) 受け取り
	*,
	tile: tuple[int, int] = (128, 128),
	overlap: tuple[int, int] = (32, 32),
	amp: bool = True,
	use_tqdm: bool = False,
	tiles_per_batch: int = 4,
	tile_transform: ViewCompose | None = None,
	post_tile_transform: ViewCompose | None = None,
) -> torch.Tensor:
	# 1) 入力検証
	validate_array(x_chw, allowed_ndims=(2, 3), name='x', backend='numpy')
	if x_chw.ndim == 2:
		x_chw = x_chw[None, :]

	x_np = np.ascontiguousarray(x_chw, dtype=np.float32)

	# 3) Torch へ 1 回だけ変換（以降はTorchでタイル推論）
	device = next(model.parameters()).device
	x_t = torch.from_numpy(x_np).unsqueeze(0).to(device=device, dtype=torch.float32)

	y = _run_tiled(
		model,
		x_t,
		tile=tile,
		overlap=overlap,
		amp=amp,
		use_tqdm=use_tqdm,
		tiles_per_batch=tiles_per_batch,
		tile_transform=tile_transform,
		post_tile_transform=post_tile_transform,
	)
	return y.squeeze(0)
