from pathlib import Path

import numpy as np
import pytest
import segyio
import torch

from seisai_dataset import (
    InferenceGatherWindowsConfig,
    InferenceGatherWindowsDataset,
    InputOnlyPlan,
)
from seisai_dataset.builder import (
    IdentitySignal,
    MakeOffsetChannel,
    MakeTimeChannel,
    NormalizeOffsetByConst,
    NormalizeTimeByConst,
    SelectStack,
)


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
    spec.iline = 189
    spec.xline = 193
    spec.format = 5
    spec.sorting = 2
    spec.samples = np.arange(n_samples, dtype=np.int32)
    spec.tracecount = int(n_traces)

    with segyio.create(path, spec) as f:
        f.bin[segyio.BinField.Interval] = int(dt_us)
        for i in range(n_traces):
            f.header[i] = {
                segyio.TraceField.FieldRecord: 1,
                segyio.TraceField.TraceNumber: int(i + 1),
                segyio.TraceField.CDP: 1,
                segyio.TraceField.offset: int((i + 1) * 10),
                segyio.TraceField.SourceX: 100,
                segyio.TraceField.SourceY: 2000,
                segyio.TraceField.GroupX: 1000 + i * 10,
                segyio.TraceField.GroupY: 2000,
                segyio.TraceField.SourceGroupScalar: 1,
            }
            f.trace[i] = arr[i]


def _make_plan() -> InputOnlyPlan:
    return InputOnlyPlan(
        wave_ops=[
            IdentitySignal(src='x_view', dst='wave_ch', copy=False),
            MakeOffsetChannel(dst='offset_ch_raw', normalize=False),
            NormalizeOffsetByConst(
                src='offset_ch_raw',
                dst='offset_ch',
                ref_m=40.0,
                use_abs=True,
                clip_lo=0.0,
                clip_hi=1.5,
            ),
            MakeTimeChannel(dst='time_ch_raw'),
            NormalizeTimeByConst(
                src='time_ch_raw',
                dst='time_ch',
                ref_sec=0.02,
                clip_lo=0.0,
                clip_hi=1.5,
            ),
        ],
        label_ops=[],
        input_stack=SelectStack(keys=('wave_ch', 'offset_ch', 'time_ch'), dst='input'),
    )


def _make_cfg() -> InferenceGatherWindowsConfig:
    return InferenceGatherWindowsConfig(
        domains=('shot',),
        win_size_traces=8,
        stride_traces=4,
        pad_last=True,
        target_len=16,
    )


def _make_segy(tmp_path: Path, *, n_traces: int = 5, n_samples: int = 12) -> str:
    t = np.arange(n_samples, dtype=np.float32)
    traces = np.stack([t + (100.0 * i) for i in range(n_traces)], axis=0)
    segy_path = str(tmp_path / 'infer_windows.sgy')
    write_unstructured_segy(segy_path, traces, dt_us=2000)
    return segy_path


def test_inference_gather_windows_dataset_supports_raw_only_inference(
    tmp_path: Path,
) -> None:
    segy_path = _make_segy(tmp_path)
    ds = InferenceGatherWindowsDataset(
        segy_files=[segy_path],
        fb_files=None,
        plan=_make_plan(),
        cfg=_make_cfg(),
        use_header_cache=False,
    )

    try:
        assert len(ds) == 1
        out = ds[0]
    finally:
        ds.close()

    x = out['input']
    meta = out['meta']

    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float32
    assert x.shape == (3, 8, 16)

    assert np.array_equal(
        meta['trace_valid'],
        np.array([True, True, True, True, True, False, False, False], dtype=np.bool_),
    )
    assert np.array_equal(
        meta['fb_idx_view'],
        -np.ones((8,), dtype=np.int64),
    )
    assert np.array_equal(
        meta['raw_idx_global'],
        np.array([0, 1, 2, 3, 4, -1, -1, -1], dtype=np.int64),
    )
    assert np.array_equal(
        meta['abs_h'],
        np.array([0, 1, 2, 3, 4, -1, -1, -1], dtype=np.int64),
    )
    np.testing.assert_allclose(
        meta['offsets_view'],
        np.array([10, 20, 30, 40, 50, 0, 0, 0], dtype=np.float32),
        atol=1e-6,
    )
    assert float(meta['dt_sec']) == pytest.approx(0.002)

    np.testing.assert_allclose(
        x[1, :, 0].detach().cpu().numpy(),
        np.array([0.25, 0.5, 0.75, 1.0, 1.25, 0.0, 0.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        x[2, 0, :].detach().cpu().numpy(),
        np.clip(np.arange(16, dtype=np.float32) * 0.1, 0.0, 1.5),
        atol=1e-6,
    )
    assert x[0, :, 12:].eq(0).all().item() is True


def test_inference_gather_windows_dataset_keeps_labeled_path(tmp_path: Path) -> None:
    segy_path = _make_segy(tmp_path)
    fb_path = str(tmp_path / 'infer_windows_fb.npy')
    np.save(fb_path, np.full((5,), 7, dtype=np.int64))

    ds = InferenceGatherWindowsDataset(
        segy_files=[segy_path],
        fb_files=[fb_path],
        plan=_make_plan(),
        cfg=_make_cfg(),
        use_header_cache=False,
    )

    try:
        out = ds[0]
    finally:
        ds.close()

    x = out['input']
    meta = out['meta']

    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 8, 16)
    assert np.array_equal(
        meta['fb_idx_view'],
        np.array([7, 7, 7, 7, 7, -1, -1, -1], dtype=np.int64),
    )
    assert np.array_equal(
        meta['trace_valid'],
        np.array([True, True, True, True, True, False, False, False], dtype=np.bool_),
    )
