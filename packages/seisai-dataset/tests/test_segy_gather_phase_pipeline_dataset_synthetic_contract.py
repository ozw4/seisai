from pathlib import Path

import numpy as np
import pytest
import segyio
import torch
from seisai_dataset import (
    BuildPlan,
    FirstBreakGate,
    FirstBreakGateConfig,
    SegyGatherPhasePipelineDataset,
)
from seisai_dataset.builder.builder import IdentitySignal, PhasePSNMap, SelectStack
from seisai_dataset.config import LoaderConfig
from seisai_dataset.segy_gather_pipeline_dataset import SampleTransformer
from seisai_dataset.trace_subset_preproc import TraceSubsetLoader
from seisai_transforms.augment import DeterministicCropOrPad, ViewCompose


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


def _csr_from_rows(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    lengths = np.array([len(r) for r in rows], dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    data = np.array([v for r in rows for v in r], dtype=np.int64)
    return indptr, data


def make_plan(*, sigma: float) -> BuildPlan:
    return BuildPlan(
        wave_ops=[IdentitySignal(src='x_view', dst='x', copy=False)],
        label_ops=[PhasePSNMap(dst='psn_map', sigma=float(sigma))],
        input_stack=SelectStack(keys='x', dst='input'),
        target_stack=SelectStack(keys='psn_map', dst='target'),
    )


def make_transform(target_len: int):
    return ViewCompose([DeterministicCropOrPad(target_len)])


def test_phase_pipeline_dataset_synthetic_smoke_contract(tmp_path: Path) -> None:
    n_traces = 12
    n_samples = 64
    dt_us = 2000

    subset_traces = 8
    target_len = n_samples  # keep factor=1,start=0 for a clean mapping

    t = np.arange(n_samples, dtype=np.float32)
    traces = np.stack([t + (100.0 * i) for i in range(n_traces)], axis=0)

    segy_path = str(tmp_path / 'synthetic_phase.sgy')
    pick_path = str(tmp_path / 'synthetic_phase_picks.npz')

    write_unstructured_segy(segy_path, traces, dt_us)

    # Only trace 4 has a P pick, ensuring any contiguous window of length 8 contains at
    # least one pick (for this synthetic gather layout).
    p_rows = [[] for _ in range(n_traces)]
    s_rows = [[] for _ in range(n_traces)]
    p_rows[4] = [10]
    # S picks exist but should be invalidated because s_first(=5) < p_first(=10).
    s_rows[4] = [5, 15]

    p_indptr, p_data = _csr_from_rows(p_rows)
    s_indptr, s_data = _csr_from_rows(s_rows)
    np.savez_compressed(
        pick_path,
        p_indptr=p_indptr,
        p_data=p_data,
        s_indptr=s_indptr,
        s_data=s_data,
    )

    transform = make_transform(target_len)
    fbgate = FirstBreakGate(
        FirstBreakGateConfig(
            apply_on='off',
            min_pick_ratio=0.0,
        )
    )
    plan = make_plan(sigma=2.0)

    ds = SegyGatherPhasePipelineDataset(
        segy_files=[segy_path],
        phase_pick_files=[pick_path],
        transform=transform,
        fbgate=fbgate,
        plan=plan,
        primary_keys=('ffid',),
        subset_traces=int(subset_traces),
        use_superwindow=False,
        use_header_cache=False,
        secondary_key_fixed=True,
        verbose=False,
        max_trials=64,
        include_empty_gathers=False,
    )
    # Make sampling deterministic for the test
    ds._rng = np.random.default_rng(0)

    try:
        out = ds[0]
    finally:
        ds.close()

    required = {
        'input',
        'target',
        'trace_valid',
        'fb_idx',
        'p_idx',
        's_idx',
        'label_valid',
        'offsets',
        'dt_sec',
        'indices',
        'meta',
        'file_path',
        'key_name',
        'secondary_key',
        'primary_unique',
        'did_superwindow',
    }
    missing = required.difference(out.keys())
    assert not missing, f'missing keys: {sorted(missing)}'

    x = out['input']
    y = out['target']
    trace_valid = out['trace_valid']
    fb_idx = out['fb_idx']
    p_idx = out['p_idx']
    s_idx = out['s_idx']
    label_valid = out['label_valid']
    indices = out['indices']
    meta = out['meta']

    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float32
    assert x.shape == (1, subset_traces, target_len)

    assert isinstance(y, torch.Tensor)
    assert y.dtype == torch.float32
    assert y.shape == (3, subset_traces, target_len)

    assert isinstance(trace_valid, torch.Tensor)
    assert trace_valid.dtype == torch.bool
    assert trace_valid.shape == (subset_traces,)
    assert bool(trace_valid.all().item()) is True

    assert isinstance(label_valid, torch.Tensor)
    assert label_valid.dtype == torch.bool
    assert label_valid.shape == (subset_traces,)

    assert isinstance(indices, np.ndarray)
    assert indices.dtype == np.int64
    assert indices.shape == (subset_traces,)
    assert 4 in indices.tolist()

    pos = int(np.where(indices == 4)[0][0])
    assert bool(label_valid[pos].item()) is True
    assert int(label_valid.sum().item()) == 1

    assert isinstance(fb_idx, torch.Tensor)
    assert fb_idx.dtype == torch.int64
    assert fb_idx.shape == (subset_traces,)

    assert isinstance(p_idx, torch.Tensor)
    assert p_idx.dtype == torch.int64
    assert p_idx.shape == (subset_traces,)

    assert isinstance(s_idx, torch.Tensor)
    assert s_idx.dtype == torch.int64
    assert s_idx.shape == (subset_traces,)

    fb_np = fb_idx.detach().cpu().numpy()
    assert int(fb_np[pos]) == 10
    assert int((fb_np > 0).sum()) == 1

    s_np = s_idx.detach().cpu().numpy()
    assert int(s_np[pos]) == 0

    assert isinstance(meta, dict)
    assert meta['p_idx_view'].shape == (subset_traces,)
    assert meta['s_idx_view'].shape == (subset_traces,)

    y_np = y.detach().cpu().numpy()
    np.testing.assert_allclose(
        y_np.sum(axis=0),
        np.ones((subset_traces, target_len), dtype=np.float32),
        atol=1e-6,
        rtol=0,
    )
    # S picks were invalidated for the picked trace, so at the P peak bin:
    # P=1, S=0, Noise=0.
    assert float(y_np[0, pos, 10]) == pytest.approx(1.0, abs=1e-6)
    assert float(y_np[1, pos, 10]) == pytest.approx(0.0, abs=1e-6)
    assert float(y_np[2, pos, 10]) == pytest.approx(0.0, abs=1e-6)


def test_phase_pipeline_dataset_include_empty_gathers_skips_fb_gates(
    tmp_path: Path,
) -> None:
    n_traces = 12
    n_samples = 64
    dt_us = 2000

    subset_traces = 8
    target_len = n_samples

    t = np.arange(n_samples, dtype=np.float32)
    traces = np.stack([t + (100.0 * i) for i in range(n_traces)], axis=0)

    segy_path = str(tmp_path / 'synthetic_phase_empty.sgy')
    pick_path = str(tmp_path / 'synthetic_phase_empty_picks.npz')

    write_unstructured_segy(segy_path, traces, dt_us)

    # Completely empty picks.
    p_rows = [[] for _ in range(n_traces)]
    s_rows = [[] for _ in range(n_traces)]
    p_indptr, p_data = _csr_from_rows(p_rows)
    s_indptr, s_data = _csr_from_rows(s_rows)
    np.savez_compressed(
        pick_path,
        p_indptr=p_indptr,
        p_data=p_data,
        s_indptr=s_indptr,
        s_data=s_data,
    )

    transform = make_transform(target_len)
    # These gates would reject empty samples if applied.
    fbgate = FirstBreakGate(
        FirstBreakGateConfig(
            apply_on='any',
            min_pick_ratio=0.5,
            min_pairs=16,
        )
    )
    plan = make_plan(sigma=2.0)

    ds = SegyGatherPhasePipelineDataset(
        segy_files=[segy_path],
        phase_pick_files=[pick_path],
        transform=transform,
        fbgate=fbgate,
        plan=plan,
        primary_keys=('ffid',),
        subset_traces=int(subset_traces),
        use_superwindow=False,
        use_header_cache=False,
        secondary_key_fixed=True,
        verbose=False,
        max_trials=64,
        include_empty_gathers=True,
    )
    ds._rng = np.random.default_rng(0)

    try:
        out = ds[0]
    finally:
        ds.close()

    assert 'dt_eff_sec' in out['meta']
    assert isinstance(out['meta']['dt_eff_sec'], float)

    y = out['target']
    assert isinstance(y, torch.Tensor)
    assert y.shape == (3, subset_traces, target_len)
    assert torch.allclose(y[2], torch.ones_like(y[2]))

    label_valid = out['label_valid']
    assert isinstance(label_valid, torch.Tensor)
    assert label_valid.dtype == torch.bool
    assert bool(label_valid.any().item()) is False


class _MetaInjectSampleTransformer:
    """Inject meta keys to force label projection onto padded traces.

    This wrapper makes the test sensitive to whether label_valid correctly respects
    trace_valid after view projection (hflip/factor_h).
    """

    def __init__(
        self, base: SampleTransformer, *, hflip: bool, factor_h: float
    ) -> None:
        self.base = base
        self.hflip = bool(hflip)
        self.factor_h = float(factor_h)

    def load_transform(self, info, indices, fb, rng):
        x_view, meta, offsets, fb_subset, indices_pad, trace_valid = (
            self.base.load_transform(info, indices, fb, rng)
        )
        meta2 = dict(meta)
        meta2['hflip'] = self.hflip
        meta2['factor_h'] = self.factor_h
        meta2['trace_valid'] = np.asarray(trace_valid, dtype=np.bool_)
        return x_view, meta2, offsets, fb_subset, indices_pad, trace_valid


def test_phase_pipeline_dataset_padding_forces_label_valid_false_on_padded_traces(
    tmp_path: Path,
) -> None:
    # Make n_traces < subset_traces so padding happens.
    n_traces = 4
    n_samples = 64
    dt_us = 2000

    subset_traces = 8
    target_len = n_samples

    t = np.arange(n_samples, dtype=np.float32)
    traces = np.stack([t + (100.0 * i) for i in range(n_traces)], axis=0)

    segy_path = str(tmp_path / 'synthetic_phase_pad.sgy')
    pick_path = str(tmp_path / 'synthetic_phase_pad_picks.npz')
    write_unstructured_segy(segy_path, traces, dt_us)

    # Put a single P pick on trace 2.
    # With meta injection (hflip=True, factor_h=2.0), projection can move this pick
    # onto padded destination traces; label_valid must still remain False there.
    p_rows = [[] for _ in range(n_traces)]
    s_rows = [[] for _ in range(n_traces)]
    p_rows[2] = [10]
    p_indptr, p_data = _csr_from_rows(p_rows)
    s_indptr, s_data = _csr_from_rows(s_rows)
    np.savez_compressed(
        pick_path,
        p_indptr=p_indptr,
        p_data=p_data,
        s_indptr=s_indptr,
        s_data=s_data,
    )

    transform = make_transform(target_len)
    fbgate = FirstBreakGate(
        FirstBreakGateConfig(
            apply_on='off',
            min_pick_ratio=0.0,
        )
    )
    plan = make_plan(sigma=2.0)

    # Build a base SampleTransformer then wrap it to inject meta.
    subsetloader = TraceSubsetLoader(LoaderConfig(pad_traces_to=int(subset_traces)))
    base_st = SampleTransformer(subsetloader, transform)
    sample_transformer = _MetaInjectSampleTransformer(base_st, hflip=True, factor_h=2.0)

    ds = SegyGatherPhasePipelineDataset(
        segy_files=[segy_path],
        phase_pick_files=[pick_path],
        transform=transform,
        fbgate=fbgate,
        plan=plan,
        primary_keys=('ffid',),
        subset_traces=int(subset_traces),
        use_superwindow=False,
        use_header_cache=False,
        secondary_key_fixed=True,
        verbose=False,
        max_trials=64,
        include_empty_gathers=False,
        sample_transformer=sample_transformer,
    )
    ds._rng = np.random.default_rng(0)

    try:
        out = ds[0]
    finally:
        ds.close()

    tv = out['trace_valid']
    lv = out['label_valid']
    y = out['target']

    assert isinstance(tv, torch.Tensor)
    assert tv.dtype == torch.bool
    assert isinstance(lv, torch.Tensor)
    assert lv.dtype == torch.bool
    assert isinstance(y, torch.Tensor)
    assert y.dtype == torch.float32

    assert tv.shape == (subset_traces,)
    assert lv.shape == (subset_traces,)
    assert y.shape == (3, subset_traces, target_len)

    # Padding must exist.
    pad_mask = ~tv
    assert bool(pad_mask.any().item()) is True
    assert int(pad_mask.sum().item()) == subset_traces - n_traces

    # Core contract: padded traces must never be label-valid.
    assert bool(lv[pad_mask].any().item()) is False

    # Ensure this test is non-trivial: projection should create a strong P peak on at
    # least one padded trace (otherwise the above could pass vacuously).
    p_peaks = y[0, pad_mask, 10]
    assert bool((p_peaks > 0.9).any().item()) is True
