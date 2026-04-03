from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import segyio
import torch
from torch.utils.data import DataLoader, Subset

from seisai_engine.pipelines.common import load_checkpoint
from seisai_engine.pipelines.fbpick.common import COARSE_REQUIRED_KEYS, load_coarse_npz
from seisai_engine.pipelines.fbpick.coarse import (
    build_fbgate,
    build_plan,
    build_train_dataset,
    load_coarse_train_config,
    load_train_bundle,
    run_coarse_infer,
    run_train,
)
from seisai_engine.train_loop import train_one_epoch


def write_unstructured_segy(path: str, traces: np.ndarray, dt_us: int) -> None:
    arr = np.asarray(traces, dtype=np.float32)
    if arr.ndim != 2:
        msg = 'traces must be 2D (n_traces, n_samples)'
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


def _make_plan() -> object:
    return build_plan(
        sigma_ms=10.0,
        time_ref_sec=20.0,
        offset_ref_m=2000.0,
    )


def _make_training_config(tmp_path: Path, *, segy_path: str, fb_path: str) -> dict:
    return {
        'paths': {
            'segy_files': [segy_path],
            'fb_files': [fb_path],
            'out_dir': str(tmp_path / 'train_out'),
        },
        'dataset': {
            'use_header_cache': False,
            'verbose': False,
            'progress': False,
            'primary_keys': ['ffid'],
            'waveform_mode': 'eager',
            'train_endian': 'big',
            'infer_endian': 'big',
        },
        'transform': {
            'time_len': 6016,
            'standardize_eps': 1.0e-8,
        },
        'norm_refs': {
            'time_ref_sec': 20.0,
            'offset_ref_m': 2000.0,
        },
        'train': {
            'seed': 0,
            'epochs': 1,
            'samples_per_epoch': 1,
            'batch_size': 1,
            'num_workers': 0,
            'max_norm': 1.0,
            'use_amp': False,
            'lr': 1.0e-3,
            'weight_decay': 0.0,
            'subset_traces': 128,
            'fb_sigma_ms': 10.0,
        },
        'infer': {
            'seed': 0,
            'batch_size': 1,
            'num_workers': 0,
            'max_batches': 1,
            'subset_traces': 128,
        },
        'vis': {
            'n': 0,
            'out_subdir': 'vis',
        },
        'ckpt': {
            'save_best_only': True,
            'metric': 'infer_loss',
            'mode': 'min',
        },
        'model': {
            'backbone': 'resnet18',
            'pretrained': False,
            'in_chans': 3,
            'out_chans': 1,
        },
    }


def test_load_coarse_train_config_returns_fixed_contract_values(
    tmp_path: Path,
) -> None:
    cfg = _make_training_config(tmp_path, segy_path='dummy.sgy', fb_path='dummy.npy')

    typed = load_coarse_train_config(cfg)

    assert typed.transform.time_len == 6016
    assert typed.train.fb_sigma_ms == pytest.approx(10.0)
    assert typed.model_sig['in_chans'] == 3
    assert typed.model_sig['out_chans'] == 1
    assert typed.ckpt.pipeline == 'fbpick'
    assert typed.ckpt.output_ids == ('P',)
    assert typed.ckpt.softmax_axis == 'time'


@pytest.mark.parametrize(
    ('mutate', 'message'),
    [
        (
            lambda cfg: cfg['model'].__setitem__('in_chans', 1),
            'model.in_chans must be 3',
        ),
        (
            lambda cfg: cfg['model'].__setitem__('out_chans', 2),
            'model.out_chans must be 1',
        ),
        (
            lambda cfg: cfg['transform'].__setitem__('time_len', 6000),
            'transform.time_len must be 6016',
        ),
        (
            lambda cfg: cfg['train'].__setitem__('fb_sigma_ms', 0.0),
            'train.fb_sigma_ms must be > 0',
        ),
        (
            lambda cfg: cfg['norm_refs'].__setitem__('time_ref_sec', 0.0),
            'norm_refs.time_ref_sec must be > 0',
        ),
        (
            lambda cfg: cfg['norm_refs'].__setitem__('offset_ref_m', 0.0),
            'norm_refs.offset_ref_m must be > 0',
        ),
    ],
)
def test_load_coarse_train_config_rejects_invalid_fixed_contract(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    cfg = _make_training_config(tmp_path, segy_path='dummy.sgy', fb_path='dummy.npy')
    mutate(cfg)

    with pytest.raises(ValueError) as exc:
        load_coarse_train_config(cfg)

    assert message in str(exc.value)


def test_coarse_build_plan_produces_expected_shapes() -> None:
    plan = _make_plan()
    sample = {
        'x_view': np.ones((4, 8), dtype=np.float32),
        'meta': {
            'fb_idx_view': np.array([2, 3, 4, -1], dtype=np.int64),
            'trace_valid': np.array([True, True, True, False], dtype=np.bool_),
            'offsets_view': np.array([10, 20, 30, 0], dtype=np.float32),
            'time_view': np.arange(8, dtype=np.float32) * 0.004,
            'dt_sec': np.float32(0.004),
            'dt_eff_sec': np.float32(0.004),
        },
    }

    plan.run(sample, rng=np.random.default_rng(0))

    assert tuple(sample['input'].shape) == (3, 4, 8)
    assert tuple(sample['target'].shape) == (1, 4, 8)


def test_pick_aware_crop_keeps_all_valid_picks_in_view(tmp_path: Path) -> None:
    n_traces = 128
    n_samples = 7000
    traces = np.stack(
        [np.linspace(-1.0, 1.0, n_samples, dtype=np.float32) + i for i in range(n_traces)],
        axis=0,
    )
    segy_path = str(tmp_path / 'pick_crop.sgy')
    write_unstructured_segy(segy_path, traces, dt_us=2000)
    fb = np.linspace(6400, 6500, n_traces, dtype=np.int64)
    fb_path = str(tmp_path / 'pick_crop_fb.npy')
    np.save(fb_path, fb)

    ds = build_train_dataset(
        segy_files=[segy_path],
        fb_files=[fb_path],
        sampling_overrides=None,
        plan=_make_plan(),
        fbgate=build_fbgate(apply_on='off', min_pick_ratio=0.0, verbose=False),
        subset_traces=128,
        time_len=6016,
        standardize_eps=1.0e-8,
        trace_decimate_prob=0.0,
        trace_decimate_stride_range=(1, 1),
        primary_keys=('ffid',),
        secondary_key_fixed=False,
        verbose=False,
        progress=False,
        max_trials=8,
        use_header_cache=False,
        waveform_mode='eager',
        segy_endian='big',
    )

    try:
        ds._rng = np.random.default_rng(0)
        sample = ds[0]
    finally:
        ds.close()

    fb_idx_view = np.asarray(sample['meta']['fb_idx_view'], dtype=np.int64)
    trace_valid = np.asarray(sample['meta']['trace_valid'], dtype=np.bool_)
    assert int(sample['meta']['start']) > 0
    assert np.all(fb_idx_view[trace_valid] > 0)
    assert np.all(fb_idx_view[trace_valid] < 6016)
    assert tuple(sample['input'].shape) == (3, 128, 6016)
    assert tuple(sample['target'].shape) == (1, 128, 6016)


def test_coarse_train_smoke_one_epoch(tmp_path: Path) -> None:
    n_traces = 128
    n_samples = 6200
    t = np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)
    traces = np.stack([t + (0.01 * i) for i in range(n_traces)], axis=0)
    segy_path = str(tmp_path / 'train_smoke.sgy')
    write_unstructured_segy(segy_path, traces, dt_us=2000)
    fb = np.full((n_traces,), 3000, dtype=np.int64)
    fb_path = str(tmp_path / 'train_smoke_fb.npy')
    np.save(fb_path, fb)

    cfg = _make_training_config(tmp_path, segy_path=segy_path, fb_path=fb_path)
    cfg_path = tmp_path / 'config_train_fbpick_coarse.yaml'
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')

    bundle = load_train_bundle(cfg_path, device=torch.device('cpu'))
    try:
        bundle.ds_train_full._rng = np.random.default_rng(0)
        loader = DataLoader(
            Subset(bundle.ds_train_full, range(1)),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        stats = train_one_epoch(
            bundle.model,
            loader,
            bundle.optimizer,
            bundle.criterion,
            device=torch.device('cpu'),
            gradient_accumulation_steps=1,
            max_norm=1.0,
            use_amp=False,
            print_freq=1,
        )
    finally:
        bundle.ds_train_full.close()
        bundle.ds_infer_full.close()

    assert np.isfinite(stats['loss'])
    assert stats['loss'] >= 0.0
    assert stats['steps'] == 1.0
    assert stats['samples'] == 1.0


def test_coarse_run_train_writes_fbpick_ckpt_metadata(tmp_path: Path) -> None:
    n_traces = 128
    n_samples = 6016
    t = np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)
    traces = np.stack([t + (0.01 * i) for i in range(n_traces)], axis=0)
    segy_path = str(tmp_path / 'train_skeleton.sgy')
    write_unstructured_segy(segy_path, traces, dt_us=2000)
    fb = np.full((n_traces,), 3000, dtype=np.int64)
    fb_path = str(tmp_path / 'train_skeleton_fb.npy')
    np.save(fb_path, fb)

    cfg = _make_training_config(tmp_path, segy_path=segy_path, fb_path=fb_path)
    cfg_path = tmp_path / 'config_train_fbpick_coarse_skeleton.yaml'
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')

    run_train(cfg_path, device=torch.device('cpu'))

    ckpt = load_checkpoint(tmp_path / 'train_out' / 'ckpt' / 'best.pt')
    assert ckpt['pipeline'] == 'fbpick'
    assert ckpt['output_ids'] == ['P']
    assert ckpt['softmax_axis'] == 'time'
    assert ckpt['model_sig']['in_chans'] == 3
    assert ckpt['model_sig']['out_chans'] == 1


class _TimeChannelModel(torch.nn.Module):
    out_chans = 1

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 2:3, :, :] * self.scale.view(1, 1, 1, 1)


def test_coarse_raw_only_infer_writes_npz(tmp_path: Path) -> None:
    n_traces = 160
    n_samples = 7000
    t = np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)
    traces = np.stack([t + float(i) for i in range(n_traces)], axis=0)
    segy_path = str(tmp_path / 'coarse_infer.sgy')
    write_unstructured_segy(segy_path, traces, dt_us=2000)

    cfg = {
        'paths': {
            'segy_files': [segy_path],
            'out_dir': str(tmp_path / 'infer_out'),
        },
        'dataset': {
            'use_header_cache': False,
            'verbose': False,
            'progress': False,
            'primary_keys': ['ffid'],
            'waveform_mode': 'eager',
            'train_endian': 'big',
            'infer_endian': 'big',
        },
        'transform': {
            'time_len': 6016,
            'standardize_eps': 1.0e-8,
        },
        'norm_refs': {
            'time_ref_sec': 20.0,
            'offset_ref_m': 2000.0,
        },
        'infer': {
            'subset_traces': 128,
            'batch_size': 2,
            'num_workers': 0,
            'overlap_h': 96,
            'tile_w': 6016,
            'overlap_w': 1024,
            'tiles_per_batch': 4,
            'amp': False,
            'use_tqdm': False,
        },
        'model': {
            'backbone': 'resnet18',
            'pretrained': False,
            'in_chans': 3,
            'out_chans': 1,
        },
    }

    out_path = run_coarse_infer(
        model=_TimeChannelModel(),
        cfg=cfg,
        device=torch.device('cpu'),
        source_model_id='coarse-smoke',
        iter_id=7,
    )
    data = load_coarse_npz(out_path)

    assert out_path.is_file()
    assert set(COARSE_REQUIRED_KEYS).issubset(data.keys())
    assert int(np.asarray(data['n_traces']).item()) == n_traces
    assert int(np.asarray(data['n_samples_orig']).item()) == n_samples
    np.testing.assert_array_equal(
        data['trace_indices'],
        np.arange(n_traces, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        data['coarse_pick_i'],
        np.full((n_traces,), n_samples - 1, dtype=np.int32),
    )
    np.testing.assert_allclose(
        data['coarse_pick_t_sec'],
        np.full((n_traces,), (n_samples - 1) * 0.002, dtype=np.float32),
        atol=1.0e-6,
    )
    assert np.asarray(data['lineage']).ndim == 0
