from seisai_transforms.augment import ViewCompose, RandomHFlip, RandomCropOrPad
import numpy as np

def test_viewcompose_meta():
    x = np.zeros((4, 10), dtype=np.float32)
    v = ViewCompose([RandomHFlip(1.0), RandomCropOrPad(8)])
    y, meta = v(x, return_meta=True)
    assert y.shape[1] == 8
    assert meta["hflip"] is True
    assert "start" in meta