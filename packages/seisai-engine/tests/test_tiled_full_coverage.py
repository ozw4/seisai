# test_tiled_full_coverage_ones.py
# pytest で収集・実行される形に修正（parametrize で全ケース展開）
import pytest
import torch

# ===== 設定（必要なら変更） =====
IN_CHANS: int = 1
OUT_CHANS: int = 1
DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# ストレステスト用の入力サイズ（割り切れない端も含む）
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


# overlap_h / overlap_w は各tileに対して {0, ~1/4, ~1/2 - 1} の直積
def overlaps_for_hw(tile: tuple[int, int]) -> list[tuple[int, int]]:
	th, tw = tile
	cand_h = [0, max(th // 4, 1), max(th // 2 - 1, 1)]
	cand_w = [0, max(tw // 4, 1), max(tw // 2 - 1, 1)]
	res: set[tuple[int, int]] = set()
	for oh in cand_h:
		if not (0 <= oh < th):
			continue
		for ow in cand_w:
			if 0 <= ow < tw:
				res.add((oh, ow))
	return sorted(res)


# あなたの環境の実装に合わせる（overlap_hw=(oh, ow) を受け取る前提）
from seisai_engine.predict import infer_tiled_chw  # noqa: E402


# ===== モデル（常に1を出力） =====
class OnesModel(torch.nn.Module):
	def __init__(self, out_chans: int):
		super().__init__()
		self.out_chans = int(out_chans)
		self.use_tta = False

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		n, c, h, w = x.shape
		return torch.ones((n, self.out_chans, h, w), device=x.device, dtype=x.dtype)


# ===== テストケースの全展開 =====
AMP_OPTIONS = [False] + ([True] if torch.cuda.is_available() else [])
CASES: list[tuple[int, int, tuple[int, int], tuple[int, int], bool]] = []
for h, w in SIZES:
	for tile in TILES:
		for overlap_hw in overlaps_for_hw(tile):
			for amp in AMP_OPTIONS:
				CASES.append((h, w, tile, overlap_hw, amp))


@pytest.mark.parametrize('h,w,tile,overlap_hw,amp', CASES)
def test_tiled_full_coverage_ones(
	h: int, w: int, tile: tuple[int, int], overlap_hw: tuple[int, int], amp: bool
) -> None:
	th, tw = tile
	oh, ow = overlap_hw
	assert th > 0 and tw > 0
	assert 0 <= oh < th and 0 <= ow < tw
	assert h > 0 and w > 0
	assert IN_CHANS > 0 and OUT_CHANS > 0

	# CUDAが無いのにamp=Trueにならないようにケース生成時に制御済み
	if DEVICE == 'cuda':
		assert torch.cuda.is_available(), 'CUDA requested but not available'

	model = OnesModel(OUT_CHANS).to(DEVICE).eval()
	x = torch.zeros(IN_CHANS, h, w, device=DEVICE, dtype=torch.float32)

	# 期待形状: (C,H,W) かつ overlap を軸別指定（overlap_hw）
	y = infer_tiled_chw(model, x, tile=tile, overlap=overlap_hw, amp=amp)
	assert y.shape == (OUT_CHANS, h, w), (
		f'shape mismatch: got {tuple(y.shape)} expected {(OUT_CHANS, h, w)}'
	)

	# “常に1を出す”モデル → フルカバレッジなら平均合成後も全要素が1のはず。
	holes = int((y < 0.5).sum().item())
	assert holes == 0, (
		f'uncovered elements={holes} (H={h},W={w},tile={tile},overlap_hw={overlap_hw},amp={amp})'
	)
