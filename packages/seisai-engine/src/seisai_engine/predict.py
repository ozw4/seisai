# predict.py — 入力を ndarray(CHW) で受け取り、分割“前”に ViewCompose を1回だけ適用
from collections.abc import Callable

import numpy as np
import torch
from seisai_utils.validator import validate_array

# (H,W) -> (H,W) を想定（ViewCompose 相当）
TileView = Callable[[np.ndarray], np.ndarray]


# 乱数使用を禁止するダミー RNG（ViewCompose 内で rng.* が呼ばれたら即失敗）
class _NoRandRNG:
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
	x: torch.Tensor,  # (B,C,H,W) torch.Tensor（すでにTorch）
	*,
	tile: tuple[int, int] = (128, 128),
	overlap: tuple[int, int] = (32, 32),
	amp: bool = True,
	use_tqdm: bool = False,
	tiles_per_batch: int = 8,
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

	def _tile_starts(full: int, tile: int, overlap: int) -> list[int]:
		stride = tile - overlap
		starts = [0]
		while starts[-1] + tile < full:
			nxt = starts[-1] + stride
			if nxt + tile >= full:
				starts.append(max(full - tile, 0))
				break
			starts.append(nxt)
		return sorted(set(starts))

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

	# tqdm の準備（use_tqdm=True のときのみ import、無ければ ImportError で即失敗）
	if use_tqdm:
		from tqdm.auto import tqdm

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
			patch = x[:, :, h0 : h0 + ph, w0 : w0 + pw]  # (B,C,ph,pw)
			stage_x[t * b : (t + 1) * b, :, :ph, :pw].copy_(patch)

		with torch.amp.autocast('cuda', enabled=use_amp):
			yb = model(stage_x[:slots])  # (slots, c_out, tile_h, tile_w)

		for t, (h0, w0, ph, pw) in enumerate(batch_rects):
			sl = slice(t * b, (t + 1) * b)
			yp = yb[sl, :, :ph, :pw].to(torch.float32)
			out[:, :, h0 : h0 + ph, w0 : w0 + pw] += yp
			weight[:, :, h0 : h0 + ph, w0 : w0 + pw] += 1.0

	assert torch.all(weight > 0), (
		'未カバー領域があります。tile/overlap を見直してください。'
	)
	out /= weight
	return out


@torch.no_grad()
def infer_tiled_chw(
	model: torch.nn.Module,
	x_chw: np.ndarray,  # ← ndarray(CHW) 受け取りに変更
	*,
	tile: tuple[int, int] = (128, 128),
	overlap: tuple[int, int] = (32, 32),
	amp: bool = True,
	use_tqdm: bool = False,
	tiles_per_batch: int = 8,
	view_compose: Callable[..., np.ndarray]
	| None = None,  # 分割“前”に1回だけ適用（決定論のみ）
) -> torch.Tensor:
	# 1) 入力検証
	validate_array(x_chw, allowed_ndims=(2, 3), name='x', backend='numpy')
	if x_chw.ndim == 2:
		x_chw = x_chw[None, ...]

	c, h, w = x_chw.shape
	x_np = np.ascontiguousarray(x_chw, dtype=np.float32)

	# 2) 分割前に ViewCompose を 1 回だけ（乱数は全面禁止）
	if view_compose is not None:
		_rng = _NoRandRNG()
		y0 = view_compose(x_np[0], rng=_rng, return_meta=False) if c == 1 else None
		if c == 1:
			assert isinstance(y0, np.ndarray) and y0.shape == (h, w)
			x_np = y0[None, ...]
		else:
			ys = []
			for ci in range(c):
				y_ci = view_compose(x_np[ci], rng=_rng, return_meta=False)
				assert isinstance(y_ci, np.ndarray) and y_ci.shape == (h, w)
				ys.append(y_ci)
			x_np = np.stack(ys, axis=0).astype(np.float32, copy=False)

	# 3) Torch へ 1 回だけ変換（以降はTorchでタイル推論）
	device = next(model.parameters()).device
	x_t = (
		torch.from_numpy(x_np).unsqueeze(0).to(device=device, dtype=torch.float32)
	)  # (1,C,H,W)

	y = _run_tiled(
		model,
		x_t,
		tile=tile,
		overlap=overlap,
		amp=amp,
		use_tqdm=use_tqdm,
		tiles_per_batch=tiles_per_batch,
	)
	return y.squeeze(0)  # (C_out,H,W) torch.Tensor
