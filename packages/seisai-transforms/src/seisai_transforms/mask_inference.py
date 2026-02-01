import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch.amp import autocast


@torch.no_grad()
def cover_all_traces_predict_striped(
	model: torch.nn.Module,
	x: torch.Tensor,  # (B,1,H,W)
	*,
	mask_ratio: float = 0.5,  # 1パスで隠すトレース割合
	band_width: int = 1,  # 連続バンド幅（>=1）
	noise_std: float = 1.0,
	mask_noise_mode: Literal['replace', 'add'] = 'replace',
	use_amp: bool = True,
	device=None,
	# 等間隔ストライプの「開始オフセット」を複数与えると TTA（平均化）になる
	offsets: Sequence[int] = (0,),
	passes_batch: int = 4,
) -> torch.Tensor:
	"""等間隔（striped）マスクで全トレースを一度は隠し、その位置の予測を合成。
	offsets を複数与えると、開始位置をずらした複数ラウンドの平均（TTA）を行う。

	- band_width=1 で従来の「1トレース幅」
	- mask_ratio * H / band_width ≒ 1パスに含めるバンド本数
	- num_passes は自動決定（完全被覆になる）
	"""
	if x.dim() != 4 or x.size(1) != 1:
		raise ValueError('x must be (B,1,H,W)')
	if not (0.0 < mask_ratio <= 1.0):
		raise ValueError('mask_ratio must be in (0,1]')
	if band_width < 1:
		raise ValueError('band_width must be >= 1')
	if noise_std < 0:
		raise ValueError('noise_std must be >= 0')
	if mask_noise_mode not in ('replace', 'add'):
		raise ValueError(f'Invalid mask_noise_mode: {mask_noise_mode}')
	if len(offsets) == 0:
		raise ValueError('offsets must be non-empty')

	device = device or x.device
	B, _, H, W = x.shape
	w = int(band_width)

	# ---- 連続バンド（ブロック）を構成 ----
	# 例: H=10, w=4 -> [0..3],[4..7],[8..9]
	blocks = []
	i = 0
	while i < H:
		j = min(i + w, H)
		blocks.append(torch.arange(i, j, dtype=torch.long))
		i = j
	num_blocks = len(blocks)

	# 1パスあたりの目標トレース本数と、必要ブロック数
	traces_per_pass = max(1, min(int(round(mask_ratio * H)), H))
	blocks_per_pass = max(1, min(math.ceil(traces_per_pass / w), num_blocks))

	# 等間隔化のためのパス数（ブロックを K 分割して mod K ごとに採る）
	K = max(1, math.ceil(num_blocks / blocks_per_pass))

	# 出力は TTA（offsets の数）で平均
	y_sum = torch.zeros_like(x)
	hits = torch.zeros((B, 1, H, 1), dtype=torch.int32, device=device)

	for off in offsets:
		off = int(off) % K
		# パス p ごとに、(block_index + off) % K == p を採択 → 等間隔ストライプ
		for p in range(K):
			# ブロックインデックス集合
			idx_blocks = [bi for bi in range(num_blocks) if ((bi + off) % K) == p]
			if not idx_blocks:
				continue
			# まとめ推論を passes_batch ごとに回す
			for s in range(0, len(idx_blocks), passes_batch):
				batch_bi = idx_blocks[s : s + passes_batch]
				xmb = []
				row_index_list = []

				for bi in batch_bi:
					rows = blocks[bi]  # (Nh,)
					row_index_list.append(rows)

					xm = x.clone()  # (B,1,H,W)
					if noise_std > 0:
						n = (
							torch.randn((B, 1, rows.numel(), W), device=device)
							* noise_std
						)
					else:
						n = torch.zeros((B, 1, rows.numel(), W), device=device)

					rows_dev = rows.to(device=device)
					if mask_noise_mode == 'replace':
						xm[:, :, rows_dev, :] = n
					else:  # "add"
						xm[:, :, rows_dev, :] += n

					xmb.append(xm)

				xmb = torch.cat(xmb, dim=0)  # (len(batch_bi)*B, 1, H, W)
				with autocast('cuda', enabled=use_amp):
					yb = model(xmb)  # (len(batch_bi)*B, C, H, W)
				# 元の B に戻して対応行だけ加算
				yb = yb.view(len(batch_bi), B, -1, H, W)  # (Nb,B,C,H,W)
				for k, rows in enumerate(row_index_list):
					rows_dev = rows.to(device=device)
					y_sum[:, :, rows_dev, :] += yb[k, :, :, rows_dev, :]
					hits[:, :, rows_dev, :] += 1

	# 平均化（全行 hit>=1 のはず。もし0があれば設計ミス）
	hits_clamped = hits.clamp_min(1)
	y_full = y_sum / hits_clamped
	return y_full
