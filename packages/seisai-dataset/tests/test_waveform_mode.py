from __future__ import annotations

from pathlib import Path

import numpy as np
import segyio
import torch

from seisai_dataset import BuildPlan, LoaderConfig, SegyGatherPairDataset, TraceSubsetLoader
from seisai_dataset.builder.builder import IdentitySignal, SelectStack
from seisai_dataset.file_info import build_file_info_dataclass


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / 'tests' / 'data'


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


def test_file_info_waveform_modes_and_trace_loader() -> None:
    segy_path = str(_data_dir() / '20200623002546.sgy')

    info = build_file_info_dataclass(
        segy_path,
        ffid_byte=segyio.TraceField.FieldRecord,
        chno_byte=segyio.TraceField.TraceNumber,
        cmp_byte=segyio.TraceField.CDP,
        use_header_cache=True,
        waveform_mode='eager',
    )
    assert isinstance(info.mmap, np.ndarray)
    assert info.mmap.shape == (info.n_traces, info.n_samples)
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
    assert not isinstance(info_mmap.mmap, np.ndarray)

    indices = np.array([0, 2, 5], dtype=np.int64)
    loader = TraceSubsetLoader(LoaderConfig(pad_traces_to=int(indices.size)))
    x = loader.load(info_mmap.mmap, indices)
    assert x.shape == (indices.size, info_mmap.n_samples)
    assert x.dtype == np.float32

    if info_mmap.segy_obj is not None:
        info_mmap.segy_obj.close()


def test_segy_gather_pair_dataset_mmap_roundtrip() -> None:
    segy_path = str(_data_dir() / '20200623002546.sgy')
    transform = IdentityTransform()
    plan = _make_pair_plan()

    ds = SegyGatherPairDataset(
        input_segy_files=[segy_path],
        target_segy_files=[segy_path],
        transform=transform,
        plan=plan,
        primary_keys=('ffid',),
        subset_traces=32,
        use_header_cache=True,
        secondary_key_fixed=False,
        verbose=False,
        max_trials=64,
        waveform_mode='mmap',
    )
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
    assert out['input'].shape == out['target'].shape
    assert out['input'].dtype == torch.float32
    assert out['target'].dtype == torch.float32

    ds.close()
