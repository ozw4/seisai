import numpy as np

from seisai_transforms import RandomPolarityFlip


def test_polarity_flip_prob_one() -> None:
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    op = RandomPolarityFlip(prob=1.0)
    y = op(x)
    assert np.allclose(y, -x)


def test_polarity_flip_prob_zero() -> None:
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    op = RandomPolarityFlip(prob=0.0)
    y = op(x)
    assert np.array_equal(y, x)


def test_polarity_flip_meta() -> None:
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    op_on = RandomPolarityFlip(prob=1.0)
    y_on, meta_on = op_on(x, return_meta=True)
    assert np.allclose(y_on, -x)
    assert meta_on['polarity_flip'] is True

    op_off = RandomPolarityFlip(prob=0.0)
    y_off, meta_off = op_off(x, return_meta=True)
    assert np.array_equal(y_off, x)
    assert meta_off['polarity_flip'] is False
