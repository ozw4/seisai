import numpy as np
from seisai_transforms.config import SpaceAugConfig
from seisai_transforms.augment import RandomSpatialStretchSameH

def test_space_aug_noop_when_prob_zero():
    H, W = 8, 32
    x = np.random.randn(H, W).astype(np.float32)
    op = RandomSpatialStretchSameH(SpaceAugConfig(prob=0.0))
    y, meta = op(x, rng=np.random.default_rng(0), return_meta=True)
    assert meta["did_space"] is False and meta["factor_h"] == 1.0
    np.testing.assert_allclose(y, x, atol=0, rtol=0)