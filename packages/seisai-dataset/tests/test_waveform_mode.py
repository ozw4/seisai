from __future__ import annotations

from pathlib import Path

import numpy as np
import segyio
import torch
from seisai_dataset import (
	BuildPlan,
	LoaderConfig,
	SegyGatherPairDataset,
	TraceSubsetLoader,
)
from seisai_dataset.builder.builder import IdentitySignal, SelectStack
from seisai_dataset.file_info import build_file_info_dataclass


class IdentityTransform:
	def __call__(
		self,
		x: np.ndarray,
		*,
		rng: np.random.Generator,
		return_meta: bool = False,
	) -> np.ndarray | tuple[np.ndarray, dict]:
		if not isinstance(x, np.ndarray) or x.ndim != 2:
			msg = 'x must be 2D numpy array'
			raise ValueError(msg)
		meta = {'factor': 1.0, 'start': 0, 'hflip': False, 'factor_h': 1.0}
		return (x, meta) if return_meta else x


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


def make_synthetic_traces(*, n_traces: int, n_samples: int) -> np.ndarray:
	t = np.arange(n_samples, dtype=np.float32)
	return np.stack([t + (100.0 * i) for i in range(n_traces)], axis=0)


def make_synthetic_pair(tmp_path: Path) -> tuple[str, str, np.ndarray, np.ndarray]:
	n_traces = 12
	n_samples = 64
	dt_us = 2000

	target = make_synthetic_traces(n_traces=n_traces, n_samples=n_samples)
	inp = 2.0 * target

	input_path = str(tmp_path / 'waveform_mode_input.sgy')
	target_path = str(tmp_path / 'waveform_mode_target.sgy')
	write_unstructured_segy(input_path, inp, dt_us)
	write_unstructured_segy(target_path, target, dt_us)
	return input_path, target_path, inp, target


def as_chw(x: torch.Tensor) -> torch.Tensor:
	if x.ndim == 2:
		return x.unsqueeze(0)
	if x.ndim == 3:
		return x
	msg = f'expected 2D or 3D tensor, got shape={tuple(x.shape)}'
	raise ValueError(msg)


def _make_pair_plan() -> BuildPlan:
	return BuildPlan(
		wave_ops=[
			IdentitySignal(source_key='x_view_input', dst='x_in', copy=False),
			IdentitySignal(source_key='x_view_target', dst='x_tg', copy=False),
		],
		label_ops=[],
		input_stack=SelectStack(keys='x_in', dst='input'),
		target_stack=SelectStack(keys='x_tg', dst='target'),
	)


def test_file_info_waveform_modes_and_trace_loader(tmp_path: Path) -> None:
	segy_path, _target_path, traces, _target = make_synthetic_pair(tmp_path)

	info = build_file_info_dataclass(
		segy_path,
		ffid_byte=segyio.TraceField.FieldRecord,
		chno_byte=segyio.TraceField.TraceNumber,
		cmp_byte=segyio.TraceField.CDP,
		use_header_cache=True,
		waveform_mode='eager',
	)
	try:
		assert isinstance(info.mmap, np.ndarray)
		assert info.mmap.shape == traces.shape
		assert np.array_equal(info.mmap, traces)
	finally:
		if info.segy_obj is not None:
			info.segy_obj.close()

	info_mmap = build_file_info_dataclass(
		segy_path,
		ffid_byte=segyio.TraceField.FieldRecord,
		chno_byte=segyio.TraceField.TraceNumber,
		cmp_byte=segyio.TraceField.CDP,
		use_header_cache=True,
		waveform_mode='mmap',
	)
	try:
		assert not isinstance(info_mmap.mmap, np.ndarray)

		indices = np.array([0, 2, 5], dtype=np.int64)
		loader = TraceSubsetLoader(LoaderConfig(pad_traces_to=int(indices.size)))
		x = loader.load(info_mmap.mmap, indices)
		assert x.shape == (indices.size, info_mmap.n_samples)
		assert x.dtype == np.float32
		assert np.array_equal(x, traces[indices])
	finally:
		if info_mmap.segy_obj is not None:
			info_mmap.segy_obj.close()


def test_segy_gather_pair_dataset_mmap_roundtrip(tmp_path: Path) -> None:
	input_path, target_path, inp, target = make_synthetic_pair(tmp_path)
	transform = IdentityTransform()
	plan = _make_pair_plan()

	ds = SegyGatherPairDataset(
		input_segy_files=[input_path],
		target_segy_files=[target_path],
		input_transform=transform,
		target_transform=transform,
		plan=plan,
		primary_keys=('ffid',),
		subset_traces=8,
		use_header_cache=True,
		secondary_key_fixed=True,
		verbose=False,
		max_trials=64,
		waveform_mode='mmap',
	)
	try:
		ds._rng = np.random.default_rng(0)

		assert len(ds.file_infos) == 1
		info = ds.file_infos[0]
		assert not isinstance(info.input_info.mmap, np.ndarray)
		assert not isinstance(info.target_mmap, np.ndarray)

		out = ds[0]
		assert 'input' in out
		assert 'target' in out
		assert isinstance(out['input'], torch.Tensor)
		assert isinstance(out['target'], torch.Tensor)
		assert out['input'].dtype == torch.float32
		assert out['target'].dtype == torch.float32

		x_in = as_chw(out['input'])[0].detach().cpu().numpy()
		x_tg = as_chw(out['target'])[0].detach().cpu().numpy()
		indices = np.asarray(out['indices'], dtype=np.int64)

		assert x_in.shape == x_tg.shape
		assert x_in.shape == (8, target.shape[1])
		assert np.array_equal(x_in, inp[indices])
		assert np.array_equal(x_tg, target[indices])
	finally:
		ds.close()
