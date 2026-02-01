import numpy as np
from seisai_transforms.augment import RandomFreqFilter
from seisai_transforms.config import FreqAugConfig


def test_freq_aug_noop_when_prob_zero():
	H, W = 8, 64
	x = np.random.randn(H, W).astype(np.float32)
	aug = RandomFreqFilter(FreqAugConfig(prob=0.0))
	y = aug(x, rng=np.random.default_rng(0))
	np.testing.assert_allclose(y, x, atol=0, rtol=0)
