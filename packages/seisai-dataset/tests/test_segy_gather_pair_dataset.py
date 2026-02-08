from pathlib import Path

import numpy as np
import pytest
import segyio
import torch
from seisai_dataset import BuildPlan, SegyGatherPairDataset
from seisai_dataset.builder.builder import IdentitySignal, SelectStack


class Crop1DTransform:
    """rng-driven crop along time axis. Keeps H unchanged."""

    def __init__(self, out_len: int) -> None:
        self.out_len = int(out_len)

    def __call__(
        self,
        x: np.ndarray,
        *,
        rng: np.random.Generator,
        return_meta: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            msg = 'x must be a 2D numpy array'
            raise ValueError(msg)
        H, W0 = x.shape
        if H <= 0 or W0 <= 0:
            msg = 'x must have positive shape'
            raise ValueError(msg)
        out_len = int(self.out_len)
        if out_len <= 0:
            msg = 'out_len must be > 0'
            raise ValueError(msg)
        if out_len > W0:
            msg = f'W0({W0}) < out_len({out_len}); test expects crop-only'
            raise ValueError(msg)

        start = int(rng.integers(0, W0 - out_len + 1))
        x_view = x[:, start : start + out_len]
        meta = {'start': start, 'factor': 1.0}
        return (x_view, meta) if return_meta else x_view


def write_unstructured_segy(path: str, traces: np.ndarray, dt_us: int) -> None:
    arr = np.asarray(traces, dtype=np.float32)
    if arr.ndim != 2:
        msg = 'traces must be 2D (n_traces, n_samples)'
        raise ValueError(msg)
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        msg = 'traces must be non-empty'
        raise ValueError(msg)

    n_traces, n_samples = arr.shape

    spec = segyio.spec()
    # Mandatory fields for segyio.create
    spec.iline = 189
    spec.xline = 193
    spec.format = 5  # IEEE float32
    spec.sorting = 2
    spec.samples = np.arange(n_samples, dtype=np.int32)
    spec.tracecount = int(n_traces)

    with segyio.create(path, spec) as f:
        f.bin[segyio.BinField.Interval] = int(dt_us)
        for i in range(n_traces):
            sx = 100
            sy = 2000
            gx = 1000 + i * 10
            gy = 2000
            f.header[i] = {
                segyio.TraceField.FieldRecord: 1,
                segyio.TraceField.TraceNumber: int(i + 1),
                segyio.TraceField.CDP: 1,
                segyio.TraceField.offset: int((i + 1) * 10),
                segyio.TraceField.SourceX: int(sx),
                segyio.TraceField.SourceY: int(sy),
                segyio.TraceField.GroupX: int(gx),
                segyio.TraceField.GroupY: int(gy),
                segyio.TraceField.SourceGroupScalar: 1,
            }
            f.trace[i] = arr[i]


def make_pair_plan() -> BuildPlan:
    return BuildPlan(
        wave_ops=[
            IdentitySignal(source_key='x_view_input', dst='x_in', copy=False),
            IdentitySignal(source_key='x_view_target', dst='x_tg', copy=False),
        ],
        label_ops=[],
        input_stack=SelectStack(keys='x_in', dst='input'),
        target_stack=SelectStack(keys='x_tg', dst='target'),
    )


def as_chw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    msg = f'expected 2D or 3D tensor, got shape={tuple(x.shape)}'
    raise ValueError(msg)


def test_segy_gather_pair_dataset_sync_transform_and_plan(tmp_path: Path) -> None:
    n_traces = 12
    n_samples = 64
    dt_us = 2000

    t = np.arange(n_samples, dtype=np.float32)
    target = np.stack([t + (10.0 * i) for i in range(n_traces)], axis=0)
    inp = 2.0 * target

    input_path = str(tmp_path / 'noisy.sgy')
    target_path = str(tmp_path / 'clean.sgy')
    write_unstructured_segy(input_path, inp, dt_us)
    write_unstructured_segy(target_path, target, dt_us)

    transform = Crop1DTransform(out_len=32)
    plan = make_pair_plan()

    ds = SegyGatherPairDataset(
        input_segy_files=[input_path],
        target_segy_files=[target_path],
        transform=transform,
        plan=plan,
        primary_keys=('ffid',),
        subset_traces=8,
        use_header_cache=False,
        secondary_key_fixed=False,
        verbose=True,
        max_trials=64,
    )
    # Make sampling deterministic for the test
    ds._rng = np.random.default_rng(0)

    out = ds[0]
    assert 'input' in out
    assert 'target' in out
    assert isinstance(out['input'], torch.Tensor)
    assert isinstance(out['target'], torch.Tensor)

    x_in = as_chw(out['input']).to(dtype=torch.float32)
    x_tg = as_chw(out['target']).to(dtype=torch.float32)
    assert x_in.shape == x_tg.shape
    assert torch.allclose(x_in, 2.0 * x_tg, atol=0.0, rtol=0.0)

    assert isinstance(out['dt_sec'], torch.Tensor)
    assert float(out['dt_sec'].item()) == pytest.approx(dt_us * 1e-6)
    assert out['file_path_input'] == input_path
    assert out['file_path_target'] == target_path
    assert isinstance(out['indices'], np.ndarray)

    assert out.get('did_superwindow') is True or out.get('did_superwindow') is False

    ds.close()


def test_segy_gather_pair_dataset_mismatch_raises(tmp_path: Path) -> None:
    n_traces = 6
    dt_us = 2000

    t1 = np.arange(32, dtype=np.float32)
    t2 = np.arange(40, dtype=np.float32)
    target = np.stack([t1 for _ in range(n_traces)], axis=0)
    inp = 2.0 * target
    bad_target = np.stack([t2 for _ in range(n_traces)], axis=0)

    input_path = str(tmp_path / 'noisy.sgy')
    good_target_path = str(tmp_path / 'clean_good.sgy')
    bad_target_path = str(tmp_path / 'clean_bad.sgy')
    write_unstructured_segy(input_path, inp, dt_us)
    write_unstructured_segy(good_target_path, target, dt_us)
    write_unstructured_segy(bad_target_path, bad_target, dt_us)

    transform = Crop1DTransform(out_len=16)
    plan = make_pair_plan()

    SegyGatherPairDataset(
        input_segy_files=[input_path],
        target_segy_files=[good_target_path],
        transform=transform,
        plan=plan,
        primary_keys=('ffid',),
        subset_traces=4,
        use_header_cache=False,
        max_trials=8,
    ).close()

    with pytest.raises(ValueError):
        SegyGatherPairDataset(
            input_segy_files=[input_path],
            target_segy_files=[bad_target_path],
            transform=transform,
            plan=plan,
            primary_keys=('ffid',),
            subset_traces=4,
            use_header_cache=False,
            max_trials=8,
        )


def test_segy_gather_pair_dataset_pads_when_gather_is_short(tmp_path: Path) -> None:
    # gather(FFID=1)が subset_traces より短い場合、loader が H 方向に pad する
    n_traces = 6
    n_samples = 64
    dt_us = 2000

    t = np.arange(n_samples, dtype=np.float32)
    target = np.stack([t + (10.0 * i) for i in range(n_traces)], axis=0)
    inp = 2.0 * target

    input_path = str(tmp_path / 'noisy_pad.sgy')
    target_path = str(tmp_path / 'clean_pad.sgy')
    write_unstructured_segy(input_path, inp, dt_us)
    write_unstructured_segy(target_path, target, dt_us)

    transform = Crop1DTransform(out_len=32)
    plan = make_pair_plan()

    ds = SegyGatherPairDataset(
        input_segy_files=[input_path],
        target_segy_files=[target_path],
        transform=transform,
        plan=plan,
        primary_keys=('ffid',),
        subset_traces=8,  # pad を発生させる
        use_header_cache=False,
        secondary_key_fixed=True,
        max_trials=64,
    )

    out = ds[0]
    x_in = as_chw(out['input'])[0]
    x_tg = as_chw(out['target'])[0]

    # H が pad 後のサイズになっていること
    assert x_in.shape[0] == 8
    assert x_tg.shape[0] == 8

    # pad 行はゼロ(関係式も維持される)
    assert torch.allclose(x_in[6:], torch.zeros_like(x_in[6:]))
    assert torch.allclose(x_tg[6:], torch.zeros_like(x_tg[6:]))
    assert torch.allclose(x_in, 2.0 * x_tg)

    # indices / offsets も pad されている(仕様にしている場合)
    idx = out['indices']
    assert idx.shape == (8,)
    assert int(idx[6]) == -1
    assert int(idx[7]) == -1
    off = out['offsets'].cpu().numpy()
    assert off.shape == (8,)
    assert float(off[6]) == 0.0
    assert float(off[7]) == 0.0

    ds.close()


def _read_trace_headers(path: str, indices: np.ndarray) -> dict[str, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1:
        msg = 'indices must be 1D'
        raise ValueError(msg)
    if idx.size == 0:
        msg = 'indices must be non-empty'
        raise ValueError(msg)
    if np.any(idx < 0):
        msg = 'indices must be non-negative for header read'
        raise ValueError(msg)

    with segyio.open(path, 'r', ignore_geometry=True) as f:
        ffid = np.array(
            [int(f.header[int(i)][segyio.TraceField.FieldRecord]) for i in idx],
            dtype=np.int64,
        )
        chno = np.array(
            [int(f.header[int(i)][segyio.TraceField.TraceNumber]) for i in idx],
            dtype=np.int64,
        )
        off = np.array(
            [int(f.header[int(i)][segyio.TraceField.offset]) for i in idx],
            dtype=np.int64,
        )
    return {'ffid': ffid, 'chno': chno, 'offset': off}


def test_segy_gather_pair_dataset_pair_consistency_headers_shape_dtype_and_sync_crop(
    tmp_path: Path,
) -> None:
    from seisai_transforms.augment import RandomCropOrPad

    n_traces = 20
    n_samples = 64
    dt_us = 2000

    subset_traces = 8
    time_len = 32

    t = np.arange(n_samples, dtype=np.float32)
    target = np.stack([t + (1000.0 * i) for i in range(n_traces)], axis=0)
    inp = 2.0 * target

    input_path = str(tmp_path / 'noisy.sgy')
    target_path = str(tmp_path / 'clean.sgy')
    write_unstructured_segy(input_path, inp, dt_us)
    write_unstructured_segy(target_path, target, dt_us)

    transform = RandomCropOrPad(target_len=time_len)
    plan = make_pair_plan()

    ds = SegyGatherPairDataset(
        input_segy_files=[input_path],
        target_segy_files=[target_path],
        transform=transform,
        plan=plan,
        primary_keys=('ffid',),
        subset_traces=subset_traces,
        use_header_cache=False,
        secondary_key_fixed=True,
        verbose=True,
        max_trials=64,
    )
    ds._rng = np.random.default_rng(0)

    out = ds[0]

    # ---- shape/dtype contract
    assert 'input' in out
    assert 'target' in out
    assert isinstance(out['input'], torch.Tensor)
    assert isinstance(out['target'], torch.Tensor)
    assert out['input'].dtype == torch.float32
    assert out['target'].dtype == torch.float32

    x_in = as_chw(out['input'])
    x_tg = as_chw(out['target'])

    assert x_in.ndim == 3
    assert x_tg.ndim == 3
    assert x_in.shape == x_tg.shape
    assert x_in.shape[0] == 1
    assert x_in.shape[1] == subset_traces
    assert x_in.shape[2] == time_len

    # ---- indices/offsets contract
    assert isinstance(out['indices'], np.ndarray)
    indices = np.asarray(out['indices'], dtype=np.int64)
    assert indices.shape == (subset_traces,)
    assert np.all(indices >= 0)
    assert np.all(indices < n_traces)

    assert isinstance(out['offsets'], torch.Tensor)
    offsets = out['offsets'].detach().cpu().numpy()
    assert offsets.shape == (subset_traces,)

    # ---- transform sync (same crop window) + exact content check
    assert isinstance(out.get('meta', {}), dict)
    meta = out.get('meta', {})
    assert 'start' in meta
    start = int(meta['start'])
    assert 0 <= start <= (n_samples - time_len)

    exp_tg_hw = target[indices, start : start + time_len]
    exp_in_hw = inp[indices, start : start + time_len]

    got_tg_hw = x_tg[0].detach().cpu().numpy()
    got_in_hw = x_in[0].detach().cpu().numpy()

    assert np.array_equal(got_tg_hw, exp_tg_hw)
    assert np.array_equal(got_in_hw, exp_in_hw)

    # ---- pair trace correspondence (header-derived identifiers)
    h_in = _read_trace_headers(input_path, indices)
    h_tg = _read_trace_headers(target_path, indices)

    assert np.array_equal(h_in['ffid'], h_tg['ffid'])
    assert np.array_equal(h_in['chno'], h_tg['chno'])
    assert np.array_equal(h_in['offset'], h_tg['offset'])

    # dataset-provided offsets must match header offsets (float32)
    assert np.array_equal(offsets.astype(np.float32), h_in['offset'].astype(np.float32))

    # ---- trivial consistency: input is exactly 2x target in this synthetic setup
    assert torch.allclose(x_in, 2.0 * x_tg, atol=0.0, rtol=0.0)

    ds.close()
