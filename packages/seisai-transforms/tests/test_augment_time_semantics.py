import numpy as np
from seisai_transforms.augment import RandomTimeStretch
from seisai_transforms.config import TimeAugConfig


def test_time_aug_noop_when_prob_zero():
	H, W = 4, 64
	x = np.random.randn(H, W).astype(np.float32)
	op = RandomTimeStretch(TimeAugConfig(prob=0.0))
	y, meta = op(x, rng=np.random.default_rng(), return_meta=True)
	assert meta.get('factor', 1.0) == 1.0
	np.testing.assert_allclose(y, x, atol=0, rtol=0)
