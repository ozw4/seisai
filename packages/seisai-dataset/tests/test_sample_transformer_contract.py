import numpy as np
import pytest
from seisai_dataset import LoaderConfig, TraceSubsetLoader
from seisai_dataset.file_info import FileInfo
from seisai_dataset.segy_gather_pipeline_dataset import SampleTransformer


def _make_file_info(
    *, mmap: np.ndarray, offsets: np.ndarray, dt_sec: float
) -> FileInfo:
    if mmap.ndim != 2:
        msg = 'mmap must be 2D (n_traces, n_samples)'
        raise ValueError(msg)
    n_traces, n_samples = mmap.shape
    if offsets.shape != (n_traces,):
        msg = 'offsets must have shape (n_traces,)'
        raise ValueError(msg)

    ffid_values = np.zeros(n_traces, dtype=np.int32)
    chno_values = np.arange(n_traces, dtype=np.int32)

    return FileInfo(
        path='synthetic.sgy',
        mmap=mmap,
        segy_obj=None,
        dt_sec=float(dt_sec),
        n_traces=int(n_traces),
        n_samples=int(n_samples),
        ffid_values=ffid_values,
        chno_values=chno_values,
        cmp_values=None,
        ffid_key_to_indices=None,
        chno_key_to_indices=None,
        cmp_key_to_indices=None,
        ffid_unique_keys=None,
        chno_unique_keys=None,
        cmp_unique_keys=None,
        offsets=np.asarray(offsets, dtype=np.float32),
        ffid_centroids=None,
        chno_centroids=None,
        fb=None,
    )


def _make_transformer(*, H: int, W0: int, transform):
    n_traces = max(8, H + 2)
    mmap = np.arange(n_traces * W0, dtype=np.float32).reshape(n_traces, W0)
    offsets = (np.arange(n_traces, dtype=np.float32) + 1.0) * 10.0
    info = _make_file_info(mmap=mmap, offsets=offsets, dt_sec=0.002)

    indices = np.arange(H, dtype=np.int64)
    fb_subset = np.full(H, 3, dtype=np.int64)

    loader = TraceSubsetLoader(LoaderConfig(pad_traces_to=H))
    st = SampleTransformer(loader, transform)
    return st, info, indices, fb_subset


def test_sample_transformer_raises_when_meta_is_not_dict() -> None:
    H, W0 = 4, 16

    def bad_meta_transform(
        x: np.ndarray, *, rng: np.random.Generator, return_meta: bool = True
    ):
        return x, ['not', 'a', 'dict']

    st, info, indices, fb_subset = _make_transformer(
        H=H, W0=W0, transform=bad_meta_transform
    )
    with pytest.raises(TypeError, match=r'transform meta must be dict'):
        st.load_transform(info, indices, fb_subset, rng=np.random.default_rng(0))


def test_sample_transformer_raises_when_output_is_not_2d_numpy() -> None:
    H, W0 = 4, 16

    def bad_ndim_transform(
        x: np.ndarray, *, rng: np.random.Generator, return_meta: bool = True
    ):
        return x[None, :, :]

    st, info, indices, fb_subset = _make_transformer(
        H=H, W0=W0, transform=bad_ndim_transform
    )
    with pytest.raises(ValueError, match=r'transform は 2D numpy'):
        st.load_transform(info, indices, fb_subset, rng=np.random.default_rng(0))


def test_sample_transformer_raises_when_transform_changes_H() -> None:
    H, W0 = 5, 16

    def change_h_transform(
        x: np.ndarray, *, rng: np.random.Generator, return_meta: bool = True
    ):
        return x[:-1, :]

    st, info, indices, fb_subset = _make_transformer(
        H=H, W0=W0, transform=change_h_transform
    )
    with pytest.raises(ValueError, match=r'transform must keep H'):
        st.load_transform(info, indices, fb_subset, rng=np.random.default_rng(0))


def test_sample_transformer_adds_required_meta_fields() -> None:
    H, W0 = 6, 20
    W = 10

    def ok_transform(
        x: np.ndarray, *, rng: np.random.Generator, return_meta: bool = True
    ):
        x_view = x[:, :W]
        meta = {'factor': 1.0, 'start': 0, 'hflip': False, 'factor_h': 1.0}
        return x_view, meta

    st, info, indices, fb_subset = _make_transformer(H=H, W0=W0, transform=ok_transform)
    x_view, meta, offsets, fb_out, idx_out, trace_valid = st.load_transform(
        info, indices, fb_subset, rng=np.random.default_rng(0)
    )

    assert isinstance(x_view, np.ndarray)
    assert x_view.dtype == np.float32
    assert x_view.shape == (H, W)

    assert isinstance(meta, dict)
    for k in ('trace_valid', 'fb_idx_view', 'offsets_view', 'time_view'):
        assert k in meta

    assert isinstance(meta['trace_valid'], np.ndarray)
    assert meta['trace_valid'].dtype == np.bool_
    assert meta['trace_valid'].shape == (H,)
    np.testing.assert_array_equal(meta['trace_valid'], trace_valid)

    assert isinstance(meta['fb_idx_view'], np.ndarray)
    assert meta['fb_idx_view'].dtype == np.int64
    assert meta['fb_idx_view'].shape == (H,)

    assert isinstance(meta['offsets_view'], np.ndarray)
    assert meta['offsets_view'].dtype == np.float32
    assert meta['offsets_view'].shape == (H,)
    np.testing.assert_allclose(meta['offsets_view'], offsets, atol=0.0, rtol=0.0)

    assert isinstance(meta['time_view'], np.ndarray)
    assert meta['time_view'].dtype == np.float32
    assert meta['time_view'].shape == (W,)

    assert isinstance(offsets, np.ndarray)
    assert offsets.dtype == np.float32
    assert offsets.shape == (H,)

    assert isinstance(fb_out, np.ndarray)
    assert fb_out.dtype == np.int64
    assert fb_out.shape == (H,)

    assert isinstance(idx_out, np.ndarray)
    assert idx_out.dtype == np.int64
    assert idx_out.shape == (H,)

    assert isinstance(trace_valid, np.ndarray)
    assert trace_valid.dtype == np.bool_
    assert trace_valid.shape == (H,)
