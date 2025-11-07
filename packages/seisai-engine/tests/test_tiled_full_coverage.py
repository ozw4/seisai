# test_tiled_full_coverage_ones.py
# 目的: EncDec2Dは使わず、OnesModelで infer_tiled_chw のフルカバレッジ(未処理画素ゼロ)のみ検証する。
# 使い方: そのまま実行。変数は下の「設定」を書き換える。


import torch

# ===== 設定（必要なら変更） =====
IN_CHANS: int = 1
OUT_CHANS: int = 1
DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# 複数ケースを一括検証（端が割り切れないサイズも含めてストレステスト）
SIZES: list[tuple[int, int]] = [
	(64, 64),
	(127, 111),
	(128, 128),
	(129, 257),
	(256, 256),
	(313, 427),
]
TILES: list[tuple[int, int]] = [
	(64, 64),
	(96, 128),
	(128, 128),
]


# overlapは各tileに対して自動で {0, ~1/4, ~1/2 - 1} を使う（tileより小さいことを保証）
def overlaps_for(tile: tuple[int, int]) -> list[int]:
	th, tw = tile
	m = min(th, tw)
	cand = [0, max(m // 4, 1), max(m // 2 - 1, 1)]
	uniq = sorted({o for o in cand if 0 <= o < th and o < tw})
	return uniq


from seisai_engine.predict import infer_tiled_chw  # /mnt/data/predict.py


# ===== モデル（常に1を出力） =====
class OnesModel(torch.nn.Module):
	def __init__(self, out_chans: int):
		super().__init__()
		self.out_chans = int(out_chans)
		self.use_tta = False

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		n, c, h, w = x.shape
		return torch.ones((n, self.out_chans, h, w), device=x.device, dtype=x.dtype)


# ===== テスト本体 =====
def run_case(h: int, w: int, tile: tuple[int, int], overlap: int, amp: bool) -> None:
	th, tw = tile
	assert th > 0 and tw > 0
	assert 0 <= overlap < th and overlap < tw
	assert h > 0 and w > 0
	assert IN_CHANS > 0 and OUT_CHANS > 0

	if DEVICE == 'cuda':
		assert torch.cuda.is_available(), 'CUDA requested but not available'

	model = OnesModel(OUT_CHANS).to(DEVICE).eval()
	x = torch.zeros(IN_CHANS, h, w, device=DEVICE, dtype=torch.float32)

	y = infer_tiled_chw(
		model, x, tile=tile, overlap=overlap, amp=amp
	)  # 期待形状: (C,H,W)
	assert y.shape == (OUT_CHANS, h, w), (
		f'shape mismatch: got {tuple(y.shape)} expected {(OUT_CHANS, h, w)}'
	)

	# “常に1を出す”モデルなので、フルカバレッジなら全要素が1（平均合成でも1）のはず。
	# どこか未処理なら0が残る→0.5未満が検出される。
	holes = int((y < 0.5).sum().item())
	assert holes == 0, (
		f'uncovered elements={holes} (H={h},W={w},tile={tile},overlap={overlap},amp={amp})'
	)

	# 参考: 数値的健全性と統計
	mn = float(y.min().item())
	mx = float(y.max().item())
	mean = float(y.mean().item())
	print(
		f'[OK] HxW={h}x{w} tile={tile} overlap={overlap} amp={amp}  min={mn:.3f} max={mx:.3f} mean={mean:.3f}'
	)


def main() -> None:
	amp_options = [False] + ([True] if (DEVICE == 'cuda') else [])
	for h, w in SIZES:
		for tile in TILES:
			for overlap in overlaps_for(tile):
				for amp in amp_options:
					run_case(h, w, tile, overlap, amp)
	print('[ALL PASSED] infer_tiled_chw provides full coverage with OnesModel.')


if __name__ == '__main__':
	main()
