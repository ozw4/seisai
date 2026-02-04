import matplotlib

matplotlib.use('Agg')

import torch  # noqa: E402

from seisai_utils.viz_phase import make_title_from_batch_meta, save_psn_debug_png  # noqa: E402


def test_save_psn_debug_png_smoke(tmp_path) -> None:
	B, H, W = 2, 4, 6
	x = torch.randn((B, 1, H, W), dtype=torch.float32)
	logits = torch.randn((B, 3, H, W), dtype=torch.float32)
	target = torch.softmax(torch.randn((B, 3, H, W), dtype=torch.float32), dim=1)

	out = tmp_path / 'psn.png'
	save_psn_debug_png(out, x_bchw=x, target_b3hw=target, logits_b3hw=logits, b=0)
	assert out.exists()
	assert out.stat().st_size > 0


def test_make_title_from_batch_meta_best_effort() -> None:
	batch = {
		'file_path': ['a/b/c.sgy'],
		'key_name': ['ffid'],
		'secondary_key': ['chno'],
		'primary_unique': ['123'],
		'meta': {},
	}
	t = make_title_from_batch_meta(batch, b=0)
	assert isinstance(t, str)
	assert 'c.sgy' in t

