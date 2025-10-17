import numpy as np
from seisai_builders import FBGaussMap


def _make_sample(H=3, W=11, dt=0.001):
	x = np.zeros((H, W), dtype=np.float32)  # 中身は不要、形状だけ使う
	fb = np.array([5, -1, 8], dtype=np.int64)
	meta = {'hflip': False, 'factor': 1.0, 'start': 0}
	return {'x_view': x, 'fb_idx': fb, 'meta': meta, 'dt_sec': dt}


def test_fb_gauss_basic_maxima_positions():
	s = _make_sample(H=3, W=11)
	FBGaussMap(dst='g', sigma=1.0)(s)
	g = s['g']  # numpy (H,W)
	assert g.shape == (3, 11)
	assert int(np.argmax(g[0])) == 5  # fb=5
	assert np.allclose(g[1], 0.0)  # fb=-1 → 全ゼロ
	assert int(np.argmax(g[2])) == 8  # fb=8


def test_fb_gauss_meta_factor_start_hflip():
	H, W = 2, 15
	s = {
		'x_view': np.zeros((H, W), dtype=np.float32),
		'fb_idx': np.array([4, 6], dtype=np.int64),
		'meta': {'hflip': True, 'factor': 2.0, 'start': 3},  # 旧設計の補正に相当
		'dt_sec': 0.001,
	}
	FBGaussMap(dst='g', sigma=1.0)(s)
	g = s['g']
	# hflipで [4,6] → [6,4]、factor=2 & start=3 の補正
	# H=0 は元の2本目(6): 2*6-3=9
	# H=1 は元の1本目(4): 2*4-3=5
	assert int(np.argmax(g[0])) == 9
	assert int(np.argmax(g[1])) == 5
