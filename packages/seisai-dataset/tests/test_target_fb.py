import numpy as np


def test_target_fb_module_importable() -> None:
    from seisai_dataset.target_fb import FBTargetBuilder, FBTargetConfig

    assert FBTargetBuilder is not None
    assert FBTargetConfig is not None


def test_fb_target_builder_spatial_stretch_preserves_shape_and_dtype() -> None:
    from seisai_dataset.target_fb import FBTargetBuilder, FBTargetConfig

    fb_idx_win = np.array([1, 2, -1, 3], dtype=np.int64)
    builder = FBTargetBuilder(FBTargetConfig(sigma=1.0))

    y = builder.build(fb_idx_win, W=8, did_space=True, f_h=1.2)

    assert y.shape == (1, 4, 8)
    assert y.dtype == np.float32


def test_fb_target_builder_spatial_stretch_identity_matches_no_space() -> None:
    from seisai_dataset.target_fb import FBTargetBuilder, FBTargetConfig

    fb_idx_win = np.array([1, 2, -1, 3], dtype=np.int64)
    builder = FBTargetBuilder(FBTargetConfig(sigma=1.0))

    without_space = builder.build(fb_idx_win, W=8, did_space=False)
    identity_space = builder.build(fb_idx_win, W=8, did_space=True, f_h=1.0)

    np.testing.assert_array_equal(identity_space, without_space)
