from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import segyio
import torch
from seisai_engine.pipelines.common import load_checkpoint
from seisai_engine.pipelines.fbpick.coarse import (
    build_fbgate,
    build_labeled_infer_dataset,
    build_plan,
    build_train_dataset,
    load_coarse_infer_config,
    load_coarse_train_config,
    load_train_bundle,
    run_coarse_infer,
    run_train,
)
from seisai_engine.pipelines.fbpick.common import COARSE_REQUIRED_KEYS, load_coarse_npz
from seisai_engine.train_loop import train_one_epoch
from torch.utils.data import DataLoader, Subset


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
        'coarse': {
            'input_mode': 'global_anchor_resize',
        },
        'transform': {
            'trace_len': 256,
            'time_len': 2048,
            'standardize_eps': 1.0e-8,
        },
        'trace_anchor': {
            'gap_ratio': 5.0,
            'min_gap_m': None,
            'train_mode': 'random',
            'infer_mode': 'center',
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
            'subset_traces': 256,
            'fb_sigma_ms': 10.0,
        },
        'infer': {
            'seed': 0,
            'batch_size': 1,
            'num_workers': 0,
            'max_batches': 1,
            'subset_traces': 256,
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

    assert typed.coarse.input_mode == 'global_anchor_resize'
    assert typed.transform.trace_len == 256
    assert typed.transform.time_len == 2048
    assert typed.trace_anchor.gap_ratio == pytest.approx(5.0)
    assert typed.trace_anchor.min_gap_m is None
    assert typed.trace_anchor.train_mode == 'random'
    assert typed.trace_anchor.infer_mode == 'center'
    assert typed.train.fb_sigma_ms == pytest.approx(10.0)
    assert typed.model_sig['in_chans'] == 3
    assert typed.model_sig['out_chans'] == 1
    assert typed.ckpt.pipeline == 'fbpick'
    assert typed.ckpt.output_ids == ('P',)
    assert typed.ckpt.softmax_axis == 'time'


@pytest.mark.parametrize(
    'mutate_paths',
    [
        lambda paths: paths.__setitem__('infer_segy_files', ['valid.sgy']),
        lambda paths: paths.__setitem__('infer_fb_files', ['valid.npy']),
    ],
)
def test_load_coarse_train_config_rejects_partial_infer_pairs(
    tmp_path: Path,
    mutate_paths,
) -> None:
    cfg = _make_training_config(tmp_path, segy_path='train.sgy', fb_path='train.npy')
    mutate_paths(cfg['paths'])

    with pytest.raises(ValueError) as exc:
        load_coarse_train_config(cfg)

    assert (
        'paths.infer_segy_files and paths.infer_fb_files must be provided together'
        in str(exc.value)
    )


def test_load_coarse_train_config_defaults_missing_infer_pairs_to_training_pairs(
    tmp_path: Path,
) -> None:
    cfg = _make_training_config(tmp_path, segy_path='train.sgy', fb_path='train.npy')

    typed = load_coarse_train_config(cfg)

    assert typed.paths.infer_segy_files == typed.paths.segy_files
    assert typed.paths.infer_fb_files == typed.paths.fb_files


def test_load_coarse_train_config_uses_explicit_infer_pairs(
    tmp_path: Path,
) -> None:
    cfg = _make_training_config(tmp_path, segy_path='train.sgy', fb_path='train.npy')
    cfg['paths']['infer_segy_files'] = ['valid.sgy']
    cfg['paths']['infer_fb_files'] = ['valid.npy']

    typed = load_coarse_train_config(cfg)

    assert typed.paths.infer_segy_files == ('valid.sgy',)
    assert typed.paths.infer_fb_files == ('valid.npy',)


def test_load_coarse_infer_config_returns_global_anchor_contract(
    tmp_path: Path,
) -> None:
    cfg = _make_training_config(tmp_path, segy_path='dummy.sgy', fb_path='dummy.npy')
    cfg['paths'].pop('fb_files')

    typed = load_coarse_infer_config(cfg)

    assert typed.coarse.input_mode == 'global_anchor_resize'
    assert typed.transform.trace_len == 256
    assert typed.transform.time_len == 2048
    assert typed.trace_anchor.gap_ratio == pytest.approx(5.0)
    assert typed.trace_anchor.infer_mode == 'center'
    assert typed.model_sig['in_chans'] == 3


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
            lambda cfg: cfg['coarse'].__setitem__('input_mode', 'legacy_tiled'),
            "coarse.input_mode must be 'global_anchor_resize'",
        ),
        (
            lambda cfg: cfg['transform'].__setitem__('trace_len', 128),
            'transform.trace_len must be 256',
        ),
        (
            lambda cfg: cfg['transform'].__setitem__('time_len', 6000),
            'transform.time_len must be 2048',
        ),
        (
            lambda cfg: cfg['trace_anchor'].__setitem__('gap_ratio', 1.0),
            'trace_anchor.gap_ratio must be > 1.0',
        ),
        (
            lambda cfg: cfg['trace_anchor'].__setitem__('min_gap_m', 0.0),
            'trace_anchor.min_gap_m must be null or > 0',
        ),
        (
            lambda cfg: cfg['trace_anchor'].__setitem__('train_mode', 'center'),
            'trace_anchor.train_mode must be "random"',
        ),
        (
            lambda cfg: cfg['trace_anchor'].__setitem__('infer_mode', 'random'),
            'trace_anchor.infer_mode must be "center"',
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


def test_global_anchor_train_dataset_returns_fixed_shape_and_masks_pad_rows(
    tmp_path: Path,
) -> None:
    n_traces = 128
    n_samples = 4097
    traces = np.stack(
        [
            np.linspace(-1.0, 1.0, n_samples, dtype=np.float32) + i
            for i in range(n_traces)
        ],
        axis=0,
    )
    segy_path = str(tmp_path / 'global_train_pad.sgy')
    write_unstructured_segy(segy_path, traces, dt_us=2000)
    fb = np.linspace(0, n_samples - 1, n_traces, dtype=np.int64)
    fb_path = str(tmp_path / 'global_train_pad_fb.npy')
    np.save(fb_path, fb)

    ds = build_train_dataset(
        segy_files=[segy_path],
        fb_files=[fb_path],
        sampling_overrides=None,
        plan=_make_plan(),
        fbgate=build_fbgate(apply_on='off', min_pick_ratio=0.0, verbose=False),
        subset_traces=256,
        trace_len=256,
        time_len=2048,
        standardize_eps=1.0e-8,
        anchor_mode='random',
        gap_ratio=5.0,
        min_gap_m=None,
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

    trace_valid = np.asarray(sample['meta']['trace_valid'], dtype=np.bool_)
    assert tuple(sample['input'].shape) == (3, 256, 2048)
    assert tuple(sample['target'].shape) == (1, 256, 2048)
    assert 'anchor_raw_indices' in sample
    assert 'anchor_source_pos' in sample
    assert 'anchor_offsets_m' in sample
    assert 'anchor_raw_indices' not in sample['meta']
    assert 'anchor_source_pos' not in sample['meta']
    assert 'anchor_offsets_m' not in sample['meta']
    for key in (
        'trace_valid',
        'fb_idx_view',
        'offsets_view',
        'time_view',
        'dt_eff_sec',
    ):
        assert key in sample['meta']
    np.testing.assert_array_equal(
        np.asarray(sample['indices'], dtype=np.int64),
        sample['anchor_raw_indices'].numpy(),
    )
    np.testing.assert_array_equal(trace_valid[:n_traces], np.ones(n_traces, dtype=bool))
    assert not np.any(trace_valid[n_traces:])
    torch.testing.assert_close(
        sample['input'][0, n_traces:],
        torch.zeros_like(sample['input'][0, n_traces:]),
    )
    torch.testing.assert_close(
        sample['input'][1, n_traces:],
        torch.zeros_like(sample['input'][1, n_traces:]),
    )
    torch.testing.assert_close(
        sample['target'][:, n_traces:],
        torch.zeros_like(sample['target'][:, n_traces:]),
    )
    np.testing.assert_array_equal(
        sample['meta']['fb_idx_coarse_for_anchors'][n_traces:],
        np.full((256 - n_traces,), -1, dtype=np.int64),
    )
    assert int(sample['meta']['raw_time_len']) == n_samples
    assert int(sample['meta']['coarse_time_len']) == 2048
    assert sample['meta']['time_view'][0] == pytest.approx(0.0)
    assert sample['meta']['time_view'][-1] == pytest.approx((n_samples - 1) * 0.002)
    assert sample['meta']['fb_idx_coarse_for_anchors'][0] == 0
    assert sample['meta']['fb_idx_coarse_for_anchors'][n_traces - 1] == 2047


def test_build_train_bundle_uses_train_and_infer_endian_for_dataset_builders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import seisai_engine.pipelines.fbpick.coarse.train as coarse_train

    cfg = _make_training_config(tmp_path, segy_path='train.sgy', fb_path='train.npy')
    cfg['paths']['infer_segy_files'] = ['valid.sgy']
    cfg['paths']['infer_fb_files'] = ['valid.npy']
    cfg['dataset']['train_endian'] = 'little'
    cfg['dataset']['infer_endian'] = 'big'

    seen_endian: dict[str, str] = {}

    class StubDataset:
        def close(self) -> None:
            return None

    def fake_build_train_dataset(**kwargs):
        seen_endian['train'] = kwargs['segy_endian']
        return StubDataset()

    def fake_build_labeled_infer_dataset(**kwargs):
        seen_endian['infer'] = kwargs['segy_endian']
        return StubDataset()

    monkeypatch.setattr(coarse_train, 'build_train_dataset', fake_build_train_dataset)
    monkeypatch.setattr(
        coarse_train,
        'build_labeled_infer_dataset',
        fake_build_labeled_infer_dataset,
    )
    monkeypatch.setattr(
        coarse_train,
        'build_model',
        lambda _model_sig: torch.nn.Conv2d(3, 1, kernel_size=1),
    )

    bundle = coarse_train.build_train_bundle(
        cfg,
        base_dir=tmp_path,
        device=torch.device('cpu'),
    )

    assert seen_endian == {'train': 'little', 'infer': 'big'}
    assert bundle.ds_train_full is not bundle.ds_infer_full


def test_global_anchor_train_dataset_uses_random_anchors(tmp_path: Path) -> None:
    n_traces = 512
    n_samples = 2049
    traces = np.stack(
        [
            np.linspace(-1.0, 1.0, n_samples, dtype=np.float32) + i
            for i in range(n_traces)
        ],
        axis=0,
    )
    segy_path = str(tmp_path / 'global_train_random.sgy')
    write_unstructured_segy(segy_path, traces, dt_us=2000)
    fb = np.full((n_traces,), 1000, dtype=np.int64)
    fb_path = str(tmp_path / 'global_train_random_fb.npy')
    np.save(fb_path, fb)

    ds = build_train_dataset(
        segy_files=[segy_path],
        fb_files=[fb_path],
        sampling_overrides=None,
        plan=_make_plan(),
        fbgate=build_fbgate(apply_on='off', min_pick_ratio=0.0, verbose=False),
        subset_traces=256,
        trace_len=256,
        time_len=2048,
        standardize_eps=1.0e-8,
        anchor_mode='random',
        gap_ratio=5.0,
        min_gap_m=None,
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
        ds._rng = np.random.default_rng(123)
        first = ds[0]
        ds._rng = np.random.default_rng(124)
        second = ds[0]
    finally:
        ds.close()

    assert tuple(first['input'].shape) == (3, 256, 2048)
    assert tuple(second['target'].shape) == (1, 256, 2048)
    assert not torch.equal(first['anchor_source_pos'], second['anchor_source_pos'])


def test_global_anchor_validation_dataset_is_deterministic(tmp_path: Path) -> None:
    n_traces = 512
    n_samples = 2049
    traces = np.stack(
        [
            np.linspace(-1.0, 1.0, n_samples, dtype=np.float32) + i
            for i in range(n_traces)
        ],
        axis=0,
    )
    segy_path = str(tmp_path / 'global_validation.sgy')
    write_unstructured_segy(segy_path, traces, dt_us=2000)
    fb = np.full((n_traces,), 1000, dtype=np.int64)
    fb_path = str(tmp_path / 'global_validation_fb.npy')
    np.save(fb_path, fb)

    ds = build_labeled_infer_dataset(
        segy_files=[segy_path],
        fb_files=[fb_path],
        sampling_overrides=None,
        plan=_make_plan(),
        fbgate=build_fbgate(apply_on='off', min_pick_ratio=0.0, verbose=False),
        subset_traces=256,
        trace_len=256,
        time_len=2048,
        standardize_eps=1.0e-8,
        anchor_mode='center',
        gap_ratio=5.0,
        min_gap_m=None,
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
        first = ds[0]
        second = ds[0]
    finally:
        ds.close()

    assert tuple(first['input'].shape) == (3, 256, 2048)
    assert tuple(first['target'].shape) == (1, 256, 2048)
    torch.testing.assert_close(first['input'], second['input'])
    torch.testing.assert_close(first['target'], second['target'])
    assert torch.equal(first['anchor_source_pos'], second['anchor_source_pos'])


def test_coarse_train_smoke_one_epoch(tmp_path: Path) -> None:
    n_traces = 256
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
    n_traces = 256
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
    assert ckpt['coarse_input_mode'] == 'global_anchor_resize'
    assert ckpt['coarse_trace_len'] == 256
    assert ckpt['coarse_time_len'] == 2048
    assert ckpt['coarse_in_chans'] == 3
    assert ckpt['coarse_input_channels'] == ['waveform', 'offset_ch', 'time_ch']
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
    n_traces = 256
    n_samples = 2048
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
        'coarse': {
            'input_mode': 'global_anchor_resize',
        },
        'transform': {
            'trace_len': 256,
            'time_len': 2048,
            'standardize_eps': 1.0e-8,
        },
        'trace_anchor': {
            'gap_ratio': 5.0,
            'min_gap_m': None,
            'train_mode': 'random',
            'infer_mode': 'center',
        },
        'norm_refs': {
            'time_ref_sec': 20.0,
            'offset_ref_m': 2000.0,
        },
        'infer': {
            'subset_traces': 256,
            'batch_size': 2,
            'num_workers': 0,
            'overlap_h': 192,
            'tile_w': 2048,
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
        repo_root=tmp_path,
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
    lineage = json.loads(np.asarray(data['lineage']).item())
    assert lineage['iter_id'] == 7
    assert lineage['source_model_id'] == 'coarse-smoke'
    assert lineage['cfg_hash']
    assert lineage['git_sha'] is None
