import numpy as np
from seisai_builders import FBGaussMap


def _sample_center(H=1, W=31, idx=15):
	s = {
		'x_view': np.zeros((H, W), dtype=np.float32),
		'fb_idx': np.array([idx], dtype=np.int64),
		'meta': {'hflip': False, 'factor': 1.0, 'start': 0},
		'dt_sec': 0.001,
	}
	return s


def test_sigma_controls_peak_spread():
	# 同じ fb に対して、sigma 小さい方が尖る／大きい方が広がる
	s1 = _sample_center()
	s2 = _sample_center()
	FBGaussMap(dst='g1', sigma=1.0)(s1)
	FBGaussMap(dst='g2', sigma=3.0)(s2)
	g1, g2 = s1['g1'][0], s2['g2'][0]
	center = int(np.argmax(g1))
	# 距離 d=3 の点で比較：大きい sigma の方が値が大きい（広い）
	assert g2[center + 3] > g1[center + 3]
	# 中心値は小さい sigma の方が高い（尖る）
	assert g1[center] > g2[center]
