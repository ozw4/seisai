from pathlib import Path

import numpy as np
import pytest
import segyio
import torch
from seisai_dataset import (
	BuildPlan,
	FirstBreakGate,
	FirstBreakGateConfig,
	SegyGatherPipelineDataset,
)
from seisai_dataset.builder.builder import IdentitySignal, SelectStack
from seisai_transforms.augment import DeterministicCropOrPad, ViewCompose


def write_unstructured_segy(path: str, traces: np.ndarray, dt_us: int) -> None:
	arr = np.asarray(traces, dtype=np.float32)
	if arr.ndim != 2:
		raise ValueError('traces must be 2D (n_traces, n_samples)')
	if arr.shape[0] <= 0 or arr.shape[1] <= 0:
		raise ValueError('traces must be non-empty')

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


def make_plan() -> BuildPlan:
	return BuildPlan(
		wave_ops=[IdentitySignal(src='x_view', dst='x', copy=False)],
		label_ops=[],
		input_stack=SelectStack(keys='x', dst='input'),
		target_stack=SelectStack(keys='x', dst='target'),
	)


def make_transform(target_len: int):
	return ViewCompose([DeterministicCropOrPad(target_len)])


def make_synthetic_segy_and_fb(
	tmp_path: Path,
	*,
	n_traces: int,
	n_samples: int,
	dt_us: int,
	fb_value: int,
) -> tuple[str, str]:
	t = np.arange(n_samples, dtype=np.float32)
	traces = np.stack([t + (100.0 * i) for i in range(n_traces)], axis=0)

	segy_path = str(tmp_path / 'synthetic.sgy')
	fb_path = str(tmp_path / 'synthetic_fb.npy')

	write_unstructured_segy(segy_path, traces, dt_us)
	fb = np.full(n_traces, int(fb_value), dtype=np.int64)
	np.save(fb_path, fb)

	return segy_path, fb_path


def build_dataset(
	*,
	segy_path: str,
	fb_path: str,
	subset_traces: int,
	target_len: int,
	max_trials: int = 64,
) -> SegyGatherPipelineDataset:
	transform = make_transform(target_len)
	fbgate = FirstBreakGate(
		FirstBreakGateConfig(
			apply_on='off',
			min_pick_ratio=0.0,
		)
	)
	plan = make_plan()

	ds = SegyGatherPipelineDataset(
		segy_files=[segy_path],
		fb_files=[fb_path],
		transform=transform,
		fbgate=fbgate,
		plan=plan,
		primary_keys=('ffid',),
		subset_traces=int(subset_traces),
		use_superwindow=False,
		use_header_cache=False,
		valid=True,
		verbose=False,
		max_trials=int(max_trials),
	)
	# Make sampling deterministic for the test
	ds._rng = np.random.default_rng(0)
	return ds


def test_pipeline_dataset_synthetic_smoke_contract_no_padding(tmp_path: Path) -> None:
	n_traces = 12
	n_samples = 64
	dt_us = 2000

	subset_traces = 8
	target_len = 32

	crop_start = (n_samples - target_len) // 2
	fb_raw = crop_start + 8  # -> fb_idx_view == 8

	segy_path, fb_path = make_synthetic_segy_and_fb(
		tmp_path,
		n_traces=n_traces,
		n_samples=n_samples,
		dt_us=dt_us,
		fb_value=fb_raw,
	)

	ds = build_dataset(
		segy_path=segy_path,
		fb_path=fb_path,
		subset_traces=subset_traces,
		target_len=target_len,
	)
	try:
		out = ds[0]
	finally:
		ds.close()

	required = {
		'input',
		'target',
		'trace_valid',
		'fb_idx',
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
	offsets = out['offsets']
	dt_sec = out['dt_sec']
	indices = out['indices']
	meta = out['meta']

	assert isinstance(x, torch.Tensor)
	assert x.dtype == torch.float32
	assert x.ndim == 3
	C, H, W = x.shape
	assert C >= 1
	assert subset_traces == H
	assert target_len == W

	assert isinstance(y, torch.Tensor)
	assert y.dtype == torch.float32
	assert y.shape == x.shape

	assert isinstance(trace_valid, torch.Tensor)
	assert trace_valid.dtype == torch.bool
	assert trace_valid.shape == (H,)
	assert bool(trace_valid.all().item()) is True

	assert isinstance(indices, np.ndarray)
	assert indices.dtype == np.int64
	assert indices.shape == (H,)
	assert np.all(indices >= 0)
	assert np.all(indices < n_traces)

	assert isinstance(offsets, torch.Tensor)
	assert offsets.dtype == torch.float32
	assert offsets.shape == (H,)
	off_np = offsets.detach().cpu().numpy()
	exp_off = ((indices + 1) * 10).astype(np.float32)
	assert np.array_equal(off_np, exp_off)

	assert isinstance(fb_idx, torch.Tensor)
	assert fb_idx.dtype == torch.int64
	assert fb_idx.shape == (H,)
	assert np.all(fb_idx.detach().cpu().numpy() == fb_raw)

	assert isinstance(dt_sec, torch.Tensor)
	assert dt_sec.dtype == torch.float32
	assert float(dt_sec.item()) == pytest.approx(dt_us * 1e-6)

	assert isinstance(meta, dict)
	meta_required = {
		'time_view',
		'offsets_view',
		'fb_idx_view',
		'dt_eff_sec',
		'trace_valid',
	}
	missing_meta = meta_required.difference(meta.keys())
	assert not missing_meta, f'missing meta keys: {sorted(missing_meta)}'

	assert isinstance(meta['time_view'], np.ndarray)
	assert meta['time_view'].dtype == np.float32
	assert meta['time_view'].shape == (W,)

	assert isinstance(meta['offsets_view'], np.ndarray)
	assert meta['offsets_view'].dtype == np.float32
	assert meta['offsets_view'].shape == (H,)
	assert np.array_equal(meta['offsets_view'], off_np)

	assert isinstance(meta['fb_idx_view'], np.ndarray)
	assert meta['fb_idx_view'].dtype == np.int64
	assert meta['fb_idx_view'].shape == (H,)
	assert np.all(meta['fb_idx_view'] == 8)

	assert isinstance(meta['dt_eff_sec'], float)

	assert isinstance(meta['trace_valid'], np.ndarray)
	assert meta['trace_valid'].dtype == np.bool_
	assert meta['trace_valid'].shape == (H,)
	assert np.array_equal(meta['trace_valid'], trace_valid.detach().cpu().numpy())

	assert out['file_path'] == segy_path
	assert out['key_name'] == 'ffid'
	assert out['secondary_key'] == 'chno'
	assert out['primary_unique'] == '1'
	assert isinstance(out['did_superwindow'], bool)
	assert out['did_superwindow'] is False


def test_pipeline_dataset_synthetic_pads_short_gather(tmp_path: Path) -> None:
	n_traces = 6
	n_samples = 64
	dt_us = 2000

	subset_traces = 8
	target_len = 32

	crop_start = (n_samples - target_len) // 2
	fb_raw = crop_start + 8

	segy_path, fb_path = make_synthetic_segy_and_fb(
		tmp_path,
		n_traces=n_traces,
		n_samples=n_samples,
		dt_us=dt_us,
		fb_value=fb_raw,
	)

	ds = build_dataset(
		segy_path=segy_path,
		fb_path=fb_path,
		subset_traces=subset_traces,
		target_len=target_len,
	)
	try:
		out = ds[0]
	finally:
		ds.close()

	x = out['input']
	y = out['target']
	trace_valid = out['trace_valid']
	fb_idx = out['fb_idx']
	offsets = out['offsets']
	indices = out['indices']
	meta = out['meta']

	assert x.shape[1] == subset_traces
	assert x.shape[2] == target_len
	assert y.shape == x.shape

	assert trace_valid.shape == (subset_traces,)
	tv = trace_valid.detach().cpu().numpy()
	assert np.array_equal(
		tv, np.array([True] * n_traces + [False] * (subset_traces - n_traces))
	)

	assert indices.shape == (subset_traces,)
	assert np.array_equal(indices[:n_traces], np.arange(n_traces, dtype=np.int64))
	assert np.array_equal(
		indices[n_traces:], -np.ones(subset_traces - n_traces, dtype=np.int64)
	)

	off_np = offsets.detach().cpu().numpy()
	exp_off = (np.arange(n_traces, dtype=np.int64) + 1) * 10
	exp_off = exp_off.astype(np.float32)
	assert np.array_equal(off_np[:n_traces], exp_off)
	assert np.array_equal(
		off_np[n_traces:], np.zeros(subset_traces - n_traces, dtype=np.float32)
	)

	fb_np = fb_idx.detach().cpu().numpy()
	assert np.all(fb_np[:n_traces] == fb_raw)
	assert np.array_equal(
		fb_np[n_traces:], -np.ones(subset_traces - n_traces, dtype=np.int64)
	)

	x0 = x[0]
	y0 = y[0]
	assert torch.allclose(
		x0[n_traces:], torch.zeros_like(x0[n_traces:]), atol=0.0, rtol=0.0
	)
	assert torch.allclose(
		y0[n_traces:], torch.zeros_like(y0[n_traces:]), atol=0.0, rtol=0.0
	)

	assert isinstance(meta, dict)
	assert np.array_equal(meta['trace_valid'], tv)
	assert meta['fb_idx_view'].shape == (subset_traces,)
	assert np.all(meta['fb_idx_view'][:n_traces] == 8)
	assert np.array_equal(
		meta['fb_idx_view'][n_traces:],
		-np.ones(subset_traces - n_traces, dtype=np.int64),
	)
	assert np.array_equal(meta['offsets_view'], off_np)
