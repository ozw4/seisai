import numpy as np
import pytest
import torch
from seisai_dataset.infer_window_dataset import collate_pad_w_right


def test_collate_pad_w_right_pads_varlen_w_and_keeps_meta_order() -> None:
	C, H = 2, 3

	x0 = torch.ones((C, H, 4), dtype=torch.float32)
	x1 = torch.full((C, H, 6), 2.0, dtype=torch.float32)
	x2 = torch.arange(C * H * 5, dtype=torch.float32).reshape(C, H, 5)

	meta0 = {'id': 0}
	meta1 = {'id': 1}
	meta2 = {'id': 2}

	batch = [
		{'input': x0, 'meta': meta0},
		{'input': x1, 'meta': meta1},
		{'input': x2, 'meta': meta2},
	]

	out, metas = collate_pad_w_right(batch)

	assert isinstance(out, torch.Tensor)
	assert out.shape == (3, C, H, 6)
	assert out.dtype == torch.float32

	# W=4 -> Wmax=6 で右0pad
	assert torch.equal(out[0, :, :, :4], x0)
	assert out[0, :, :, 4:].eq(0).all().item() is True

	# W=6 -> padなし
	assert torch.equal(out[1, :, :, :6], x1)

	# W=5 -> 1列だけ右0pad
	assert torch.equal(out[2, :, :, :5], x2)
	assert out[2, :, :, 5:].eq(0).all().item() is True

	# meta順番維持（コピー/新規dictでもOK、内容の順序のみ保証）
	assert isinstance(metas, list)
	assert len(metas) == 3
	assert [m['id'] for m in metas] == [0, 1, 2]


def test_collate_pad_w_right_raises_on_empty_batch() -> None:
	with pytest.raises(ValueError, match=r'empty'):
		collate_pad_w_right([])


def test_collate_pad_w_right_raises_when_input_is_not_tensor() -> None:
	C, H = 1, 2
	x0 = torch.zeros((C, H, 3), dtype=torch.float32)

	batch = [
		{'input': x0, 'meta': {'id': 0}},
		{'input': np.zeros((C, H, 2), dtype=np.float32), 'meta': {'id': 1}},
	]

	with pytest.raises(TypeError, match=r'torch\.Tensor'):
		collate_pad_w_right(batch)


def test_collate_pad_w_right_raises_when_ch_mismatch() -> None:
	x0 = torch.zeros((2, 3, 4), dtype=torch.float32)
	x_bad_h = torch.zeros((2, 4, 4), dtype=torch.float32)

	batch = [
		{'input': x0, 'meta': {'id': 0}},
		{'input': x_bad_h, 'meta': {'id': 1}},
	]

	with pytest.raises(ValueError, match=r'\(C,H\)'):
		collate_pad_w_right(batch)


def test_collate_pad_w_right_raises_when_input_ndim_is_not_3() -> None:
	x0 = torch.zeros((2, 3, 4), dtype=torch.float32)
	x_bad_4d = torch.zeros((2, 3, 4, 1), dtype=torch.float32)

	batch = [
		{'input': x0, 'meta': {'id': 0}},
		{'input': x_bad_4d, 'meta': {'id': 1}},
	]

	with pytest.raises(ValueError, match=r'\(C,H,W\)'):
		collate_pad_w_right(batch)
