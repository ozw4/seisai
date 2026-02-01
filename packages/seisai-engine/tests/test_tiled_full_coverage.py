# test_tiled_full_coverage_ones.py
import numpy as np
import pytest
import torch

IN_CHANS: int = 1
OUT_CHANS: int = 1
DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

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


from seisai_engine.predict import infer_tiled_chw  # noqa: E402


class OnesModel(torch.nn.Module):
	def __init__(self, out_chans: int):
		super().__init__()
		self.out_chans = int(out_chans)
		self.use_tta = False
		# infer_tiled_chw 内の next(model.parameters()) 用のダミーパラメータ
		self._dummy = torch.nn.Parameter(torch.zeros(1))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		n, c, h, w = x.shape
		return torch.ones((n, self.out_chans, h, w), device=x.device, dtype=x.dtype)


AMP_OPTIONS = [False] + ([True] if torch.cuda.is_available() else [])

CASES: list[tuple[int, int, tuple[int, int], tuple[int, int], bool]] = []
for h, w in SIZES:
	for tile in TILES:
		th, tw = tile
		# _run_tiled の前提: 0 < tile_h <= h, 0 < tile_w <= w
		if th > h or tw > w:
			continue
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
	# _run_tiled と同じ前提をテスト側でも明示しておく
	assert th <= h and tw <= w

	if DEVICE == 'cuda':
		assert torch.cuda.is_available(), 'CUDA requested but not available'

	model = OnesModel(OUT_CHANS).to(DEVICE).eval()

	x_np = np.zeros((IN_CHANS, h, w), dtype=np.float32)

	y = infer_tiled_chw(
		model,
		x_np,
		tile=tile,
		overlap=overlap_hw,
		amp=amp,
		tiles_per_batch=8,
		tile_transform=None,
	)

	assert isinstance(y, torch.Tensor)
	assert tuple(y.shape) == (OUT_CHANS, h, w), (
		f'shape mismatch: got {tuple(y.shape)} expected {(OUT_CHANS, h, w)}'
	)

	holes = int((y < 0.5).sum().item())
	assert holes == 0, (
		f'uncovered elements={holes} (H={h},W={w},tile={tile},'
		f'overlap_hw={overlap_hw},amp={amp})'
	)
