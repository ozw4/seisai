from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cli.run_fbpick_fine_infer as fine_infer_cli
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Subset

import seisai_dataset.infer_window_dataset as infer_window_dataset
import seisai_dataset.segy_gather_pipeline_dataset as segy_gather_pipeline_dataset
import seisai_engine.pipelines.fbpick.fine.infer as fine_infer_module
from seisai_dataset.file_info import FileInfo
from seisai_engine.pipelines.common import load_checkpoint
from seisai_engine.pipelines.fbpick.common import (
    FINE_RESULT_REQUIRED_KEYS,
    ROBUST_SOURCE_COARSE_OBSERVED,
    build_final_npz_name,
    load_fbpick_final_npz,
    save_coarse_npz,
    save_robust_npz,
    validate_fbpick_final_payload,
    validate_fine_result_payload,
)
from seisai_engine.pipelines.fbpick.coarse import build_fbgate, build_model as build_coarse_model
from seisai_engine.pipelines.fbpick.fine import (
    FineCenterAugmentCfg,
    FineUniformJitterCfg,
    build_labeled_infer_dataset,
    build_model as build_fine_model,
    build_plan,
    build_raw_infer_dataset,
    build_train_dataset,
    extract_local_windowed_view,
    load_fine_infer_config,
    load_fine_train_config,
    load_train_bundle,
    run_fine_infer,
    run_fine_local_infer,
    run_train,
    restore_local_pick_to_raw,
    sample_center_jitter,
)
from seisai_engine.pipelines.fbpick.fine.infer import (
    _derive_final_npz_path,
    _prepare_fine_infer_cfg,
    _save_fine_gather_qc_pngs,
    main as run_fine_infer_main,
    require_existing_coarse_npz_path,
    resolve_fine_coarse_npz_path,
)
from seisai_engine.pipelines.fbpick.fine.init_from_coarse import (
    build_fine_init_state_dict,
    load_fine_init_from_coarse_checkpoint,
)
from seisai_engine.train_loop import train_one_epoch


class _DummySegy:
    def close(self) -> None:
        return None


class _IdentityFineModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _coarse_model_sig() -> dict[str, object]:
    return {
        'backbone': 'resnet18',
        'pretrained': False,
        'in_chans': 3,
        'out_chans': 1,
    }


def _fine_model_sig() -> dict[str, object]:
    return {
        'backbone': 'resnet18',
        'pretrained': False,
        'in_chans': 1,
        'out_chans': 1,
    }


def _make_file_info(path: str, traces: np.ndarray, *, dt_sec: float) -> FileInfo:
    arr = np.asarray(traces, dtype=np.float32)
    n_traces, n_samples = arr.shape
    trace_idx = np.arange(n_traces, dtype=np.int32)
    return FileInfo(
        path=str(Path(path).expanduser().resolve()),
        mmap=arr,
        segy_obj=_DummySegy(),
        dt_sec=float(dt_sec),
        n_traces=int(n_traces),
        n_samples=int(n_samples),
        ffid_values=np.ones(n_traces, dtype=np.int32),
        chno_values=np.arange(1, n_traces + 1, dtype=np.int32),
        cmp_values=None,
        ffid_key_to_indices={1: trace_idx},
        chno_key_to_indices=None,
        cmp_key_to_indices=None,
        ffid_unique_keys=[1],
        chno_unique_keys=None,
        cmp_unique_keys=None,
        offsets=np.arange(n_traces, dtype=np.float32) * 10.0,
        ffid_centroids=None,
        chno_centroids=None,
    )


def _make_qc_info(
    *,
    traces: np.ndarray | object,
    key_to_indices: dict[int, list[int] | np.ndarray],
    n_traces: int,
    n_samples: int,
    dt_sec: float = 0.002,
) -> dict[str, object]:
    return {
        'path': 'qc.sgy',
        'mmap': traces,
        'segy_obj': _DummySegy(),
        'dt_sec': float(dt_sec),
        'n_traces': int(n_traces),
        'n_samples': int(n_samples),
        'ffid_values': np.ones(n_traces, dtype=np.int32),
        'chno_values': np.arange(1, n_traces + 1, dtype=np.int32),
        'cmp_values': None,
        'ffid_key_to_indices': {
            key: np.asarray(indices, dtype=np.int64)
            for key, indices in key_to_indices.items()
        },
        'chno_key_to_indices': None,
        'cmp_key_to_indices': None,
        'offsets': np.arange(n_traces, dtype=np.float32),
    }


def _make_final_payload_for_qc(
    *,
    n_traces: int,
    n_samples: int,
    dt_sec: float = 0.002,
) -> dict[str, np.ndarray]:
    base = np.arange(n_traces, dtype=np.int32)
    coarse = np.clip(10 + base, 0, n_samples - 1).astype(np.int32)
    robust = np.clip(coarse + 1, 0, n_samples - 1).astype(np.int32)
    final = np.clip(robust + 1, 0, n_samples - 1).astype(np.int32)
    return {
        'dt_sec': np.asarray(dt_sec, dtype=np.float32),
        'n_samples_orig': np.asarray(n_samples, dtype=np.int32),
        'n_traces': np.asarray(n_traces, dtype=np.int32),
        'trace_indices': np.arange(n_traces, dtype=np.int64),
        'coarse_pick_i': coarse,
        'robust_pick_i': robust,
        'window_start_i': np.maximum(final - 128, 0).astype(np.int32),
        'window_end_i': np.minimum(final + 127, n_samples - 1).astype(np.int32),
        'final_pick_i': final,
        'high_conf_mask': np.ones(n_traces, dtype=np.bool_),
        'reject_mask': np.zeros(n_traces, dtype=np.bool_),
        'final_conf': np.ones(n_traces, dtype=np.float32),
    }


def _patch_synthetic_file_infos(
    monkeypatch: pytest.MonkeyPatch,
    *,
    traces_by_path: dict[str, np.ndarray],
    dt_sec: float,
) -> None:
    normalized = {
        str(Path(path).expanduser().resolve()): np.asarray(traces, dtype=np.float32)
        for path, traces in traces_by_path.items()
    }

    def _build(path: str, *args, **kwargs):
        _ = (args, kwargs)
        key = str(Path(path).expanduser().resolve())
        return _make_file_info(key, normalized[key], dt_sec=dt_sec)

    monkeypatch.setattr(segy_gather_pipeline_dataset, 'build_file_info_dataclass', _build)
    monkeypatch.setattr(infer_window_dataset, 'build_file_info', _build)


def _register_synthetic_segy(tmp_path: Path, name: str, traces: np.ndarray) -> str:
    path = tmp_path / name
    path.touch()
    return str(path.resolve())


def _write_fb(path: Path, fb: np.ndarray) -> str:
    np.save(path, np.asarray(fb, dtype=np.int64))
    return str(path.resolve())


def _write_robust(
    path: Path,
    *,
    robust_pick_i: np.ndarray,
    dt_sec: float,
) -> str:
    robust_pick_i_arr = np.asarray(robust_pick_i, dtype=np.int32)
    n_traces = int(robust_pick_i_arr.shape[0])
    save_robust_npz(
        path,
        dt_sec=float(dt_sec),
        n_samples_orig=int(np.max(robust_pick_i_arr)) + 256,
        n_traces=n_traces,
        ffid_values=np.ones(n_traces, dtype=np.int32),
        chno_values=np.arange(1, n_traces + 1, dtype=np.int32),
        offsets_m=np.arange(n_traces, dtype=np.float32) * 10.0,
        trace_indices=np.arange(n_traces, dtype=np.int64),
        robust_pick_i=robust_pick_i_arr,
        robust_pick_t_sec=robust_pick_i_arr.astype(np.float32) * np.float32(dt_sec),
        robust_conf=np.ones(n_traces, dtype=np.float32),
        robust_source=np.full(
            n_traces,
            ROBUST_SOURCE_COARSE_OBSERVED,
            dtype=np.uint8,
        ),
        used_theoretical_mask=np.zeros(n_traces, dtype=np.bool_),
        reason_mask=np.zeros(n_traces, dtype=np.uint8),
        conf_prob1=np.ones(n_traces, dtype=np.float32),
        conf_trend1=np.ones(n_traces, dtype=np.float32),
        conf_rs1=np.ones(n_traces, dtype=np.float32),
        lineage=np.asarray('synthetic-lineage'),
    )
    return str(path.resolve())


def _write_robust_with_n_samples(
    path: Path,
    *,
    robust_pick_i: np.ndarray,
    n_samples_orig: int,
    dt_sec: float,
    fine_center_i: np.ndarray | None = None,
) -> str:
    robust_pick_i_arr = np.asarray(robust_pick_i, dtype=np.int32)
    n_traces = int(robust_pick_i_arr.shape[0])
    payload = {
        'dt_sec': float(dt_sec),
        'n_samples_orig': int(n_samples_orig),
        'n_traces': n_traces,
        'ffid_values': np.ones(n_traces, dtype=np.int32),
        'chno_values': np.arange(1, n_traces + 1, dtype=np.int32),
        'offsets_m': np.arange(n_traces, dtype=np.float32) * 10.0,
        'trace_indices': np.arange(n_traces, dtype=np.int64),
        'robust_pick_i': robust_pick_i_arr,
        'robust_pick_t_sec': robust_pick_i_arr.astype(np.float32) * np.float32(dt_sec),
        'robust_conf': np.ones(n_traces, dtype=np.float32),
        'robust_source': np.full(
            n_traces,
            ROBUST_SOURCE_COARSE_OBSERVED,
            dtype=np.uint8,
        ),
        'used_theoretical_mask': np.zeros(n_traces, dtype=np.bool_),
        'reason_mask': np.zeros(n_traces, dtype=np.uint8),
        'conf_prob1': np.ones(n_traces, dtype=np.float32),
        'conf_trend1': np.ones(n_traces, dtype=np.float32),
        'conf_rs1': np.ones(n_traces, dtype=np.float32),
        'lineage': np.asarray('synthetic-lineage'),
    }
    if fine_center_i is not None:
        fine_center_i_arr = np.asarray(fine_center_i, dtype=np.int32)
        payload['fine_center_i'] = fine_center_i_arr
        payload['fine_center_t_sec'] = (
            fine_center_i_arr.astype(np.float32) * np.float32(dt_sec)
        )
    save_robust_npz(path, **payload)
    return str(path.resolve())


def _write_coarse_with_n_samples(
    path: Path,
    *,
    coarse_pick_i: np.ndarray,
    n_samples_orig: int,
    dt_sec: float,
) -> str:
    coarse_pick_i_arr = np.asarray(coarse_pick_i, dtype=np.int32)
    n_traces = int(coarse_pick_i_arr.shape[0])
    save_coarse_npz(
        path,
        dt_sec=float(dt_sec),
        n_samples_orig=int(n_samples_orig),
        n_traces=n_traces,
        ffid_values=np.ones(n_traces, dtype=np.int32),
        chno_values=np.arange(1, n_traces + 1, dtype=np.int32),
        offsets_m=np.arange(n_traces, dtype=np.float32) * 10.0,
        trace_indices=np.arange(n_traces, dtype=np.int64),
        coarse_pick_i=coarse_pick_i_arr,
        coarse_pick_t_sec=coarse_pick_i_arr.astype(np.float32) * np.float32(dt_sec),
        coarse_pmax=np.ones(n_traces, dtype=np.float32),
        coarse_prob_summary=np.ones(n_traces, dtype=np.float32),
        lineage=np.asarray('synthetic-lineage'),
    )
    return str(path.resolve())


def _make_fine_train_config(
    tmp_path: Path,
    *,
    segy_path: str,
    fb_path: str,
    robust_path: str,
    coarse_ckpt_path: str | None = None,
    use_coarse_init: bool = False,
) -> dict:
    return {
        'paths': {
            'segy_files': [segy_path],
            'fb_files': [fb_path],
            'robust_npz_files': [robust_path],
            'out_dir': str(tmp_path / 'fine_train_out'),
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
            'trace_len': 128,
            'time_len': 256,
            'center_index': 128,
            'standardize_eps': 1.0e-8,
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
            'fb_sigma_ms': 3.0,
            'sigma_samples_min': 1.5,
            'sigma_samples_max': 12.0,
        },
        'infer': {
            'seed': 0,
            'batch_size': 1,
            'num_workers': 0,
            'max_batches': 1,
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
        'model': _fine_model_sig(),
        'init': {
            'coarse_ckpt_path': coarse_ckpt_path,
            'use_coarse_init': use_coarse_init,
            'reset_seg_head': True,
            'reset_first_bn_stats': True,
        },
    }


def _make_fine_infer_config(
    tmp_path: Path,
    *,
    segy_path: str,
    robust_path: str,
    coarse_path: str | None = None,
) -> dict:
    paths = {
        'segy_files': [segy_path],
        'robust_npz_files': [robust_path],
        'out_dir': str(tmp_path / 'fine_infer_out'),
    }
    if coarse_path is not None:
        paths['coarse_npz_files'] = [coarse_path]

    return {
        'paths': paths,
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
            'trace_len': 128,
            'time_len': 256,
            'center_index': 128,
            'standardize_eps': 1.0e-8,
        },
        'infer': {
            'batch_size': 2,
            'num_workers': 0,
            'overlap_h': 96,
            'amp': False,
            'use_tqdm': False,
        },
        'model': _fine_model_sig(),
    }


def _build_test_fine_train_dataset(
    *,
    segy_path: str,
    fb_path: str,
    robust_path: str,
    window_center_npz_key: str = 'robust_pick_i',
    window_center_fallback_npz_key: str | None = None,
    center_augment: FineCenterAugmentCfg | None = None,
    max_trials: int = 8,
):
    return build_train_dataset(
        segy_files=[segy_path],
        fb_files=[fb_path],
        robust_npz_files=[robust_path],
        sampling_overrides=None,
        plan=build_plan(sigma_ms=3.0, sigma_samples_min=1.5, sigma_samples_max=12.0),
        fbgate=build_fbgate(apply_on='off', min_pick_ratio=0.0, verbose=False),
        trace_len=128,
        time_len=256,
        center_index=128,
        standardize_eps=1.0e-8,
        trace_decimate_prob=0.0,
        trace_decimate_stride_range=(1, 1),
        primary_keys=('ffid',),
        secondary_key_fixed=False,
        verbose=False,
        progress=False,
        max_trials=max_trials,
        use_header_cache=False,
        waveform_mode='eager',
        segy_endian='big',
        window_center_npz_key=window_center_npz_key,
        window_center_fallback_npz_key=window_center_fallback_npz_key,
        center_augment=center_augment,
    )


def _build_test_fine_labeled_infer_dataset(
    *,
    segy_path: str,
    fb_path: str,
    robust_path: str,
    window_center_npz_key: str = 'robust_pick_i',
    window_center_fallback_npz_key: str | None = None,
    max_trials: int = 8,
):
    return build_labeled_infer_dataset(
        segy_files=[segy_path],
        fb_files=[fb_path],
        robust_npz_files=[robust_path],
        sampling_overrides=None,
        plan=build_plan(sigma_ms=3.0, sigma_samples_min=1.5, sigma_samples_max=12.0),
        fbgate=build_fbgate(apply_on='off', min_pick_ratio=0.0, verbose=False),
        trace_len=128,
        time_len=256,
        center_index=128,
        standardize_eps=1.0e-8,
        primary_keys=('ffid',),
        secondary_key_fixed=False,
        verbose=False,
        progress=False,
        max_trials=max_trials,
        use_header_cache=False,
        waveform_mode='eager',
        segy_endian='big',
        window_center_npz_key=window_center_npz_key,
        window_center_fallback_npz_key=window_center_fallback_npz_key,
    )


def _write_coarse_ckpt(tmp_path: Path) -> str:
    path = tmp_path / 'coarse_init.pt'
    model = build_coarse_model(_coarse_model_sig())
    ckpt = {
        'version': 1,
        'pipeline': 'fbpick',
        'epoch': 0,
        'global_step': 0,
        'model_sig': _coarse_model_sig(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},
        'lr_scheduler_sig': None,
        'lr_scheduler_state_dict': None,
        'cfg': {},
        'output_ids': ['P'],
        'softmax_axis': 'time',
    }
    torch.save(ckpt, path)
    return str(path.resolve())


def _write_fine_ckpt_for_infer(tmp_path: Path) -> str:
    path = tmp_path / 'fine_infer.pt'
    model = build_fine_model(_fine_model_sig())
    ckpt = {
        'version': 1,
        'pipeline': 'fbpick',
        'stage': 'fine',
        'epoch': 0,
        'global_step': 0,
        'model_sig': _fine_model_sig(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},
        'lr_scheduler_sig': None,
        'lr_scheduler_state_dict': None,
        'cfg': _make_fine_train_config(
            tmp_path,
            segy_path='train.sgy',
            fb_path='train_fb.npy',
            robust_path='train.robust.npz',
        ),
        'output_ids': ['P'],
        'softmax_axis': 'time',
    }
    torch.save(ckpt, path)
    return str(path.resolve())


def test_derive_final_npz_path_uses_parent_prefixed_name(tmp_path: Path) -> None:
    out_path = _derive_final_npz_path(
        segy_path=tmp_path / 'line' / 'survey.sgy',
        out_dir=tmp_path / 'out',
    )

    assert out_path == tmp_path / 'out' / 'line__survey.fbpick_final.npz'


def test_fine_build_plan_returns_single_channel_shapes() -> None:
    plan = build_plan(sigma_ms=3.0, sigma_samples_min=1.5, sigma_samples_max=12.0)
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

    assert tuple(sample['input'].shape) == (1, 4, 8)
    assert tuple(sample['target'].shape) == (1, 4, 8)


def test_load_fine_train_config_returns_fixed_contract_values(tmp_path: Path) -> None:
    cfg = _make_fine_train_config(
        tmp_path,
        segy_path='dummy.sgy',
        fb_path='dummy.npy',
        robust_path='dummy.robust.npz',
    )

    typed = load_fine_train_config(cfg, base_dir=tmp_path)

    assert typed.transform.trace_len == 128
    assert typed.transform.time_len == 256
    assert typed.transform.center_index == 128
    assert typed.train.fb_sigma_ms == pytest.approx(3.0)
    assert typed.model_sig['in_chans'] == 1
    assert typed.model_sig['out_chans'] == 1
    assert typed.window_center.npz_key == 'robust_pick_i'
    assert typed.window_center.fallback_npz_key is None
    assert typed.center_augment.enabled is False
    assert typed.center_augment.train_only is True
    assert typed.center_augment.p_no_jitter == pytest.approx(1.0)
    assert typed.center_augment.uniform_jitter_samples == ()
    assert typed.center_augment.clip_to_record is True
    assert typed.center_augment.require_fb_inside is True
    assert typed.ckpt.pipeline == 'fbpick'
    assert typed.ckpt.stage == 'fine'
    assert typed.ckpt.output_ids == ('P',)
    assert typed.ckpt.softmax_axis == 'time'


def test_load_fine_train_config_parses_window_center(tmp_path: Path) -> None:
    cfg = _make_fine_train_config(
        tmp_path,
        segy_path='dummy.sgy',
        fb_path='dummy.npy',
        robust_path='dummy.robust.npz',
    )
    cfg['window_center'] = {
        'npz_key': 'fine_center_i',
        'fallback_npz_key': 'robust_pick_i',
    }

    typed = load_fine_train_config(cfg, base_dir=tmp_path)

    assert typed.window_center.npz_key == 'fine_center_i'
    assert typed.window_center.fallback_npz_key == 'robust_pick_i'


def test_load_fine_train_config_parses_center_augment(tmp_path: Path) -> None:
    cfg = _make_fine_train_config(
        tmp_path,
        segy_path='dummy.sgy',
        fb_path='dummy.npy',
        robust_path='dummy.robust.npz',
    )
    cfg['center_augment'] = {
        'enabled': True,
        'train_only': True,
        'p_no_jitter': 0.7,
        'uniform_jitter_samples': [
            {'prob': 0.2, 'lo': -32, 'hi': 32},
            {'prob': 0.1, 'lo': -64, 'hi': 64},
        ],
        'clip_to_record': True,
        'require_fb_inside': True,
    }

    typed = load_fine_train_config(cfg, base_dir=tmp_path)

    assert typed.center_augment.enabled is True
    assert typed.center_augment.train_only is True
    assert typed.center_augment.p_no_jitter == pytest.approx(0.7)
    assert typed.center_augment.clip_to_record is True
    assert typed.center_augment.require_fb_inside is True
    assert typed.center_augment.uniform_jitter_samples == (
        FineUniformJitterCfg(prob=0.2, lo=-32, hi=32),
        FineUniformJitterCfg(prob=0.1, lo=-64, hi=64),
    )


@pytest.mark.parametrize(
    ('center_augment', 'error_type', 'message'),
    [
        (
            {'enabled': True, 'p_no_jitter': -0.1},
            ValueError,
            'center_augment.p_no_jitter must lie in [0, 1]',
        ),
        (
            {'enabled': True, 'p_no_jitter': 0.0, 'uniform_jitter_samples': []},
            ValueError,
            'center_augment probabilities must sum to > 0',
        ),
        (
            {
                'enabled': True,
                'uniform_jitter_samples': [{'prob': 1.0, 'lo': 4, 'hi': 3}],
            },
            ValueError,
            'center_augment.uniform_jitter_samples[0].lo must be <= hi',
        ),
        (
            {'enabled': True, 'require_fb_inside': False},
            ValueError,
            'center_augment.require_fb_inside must be true',
        ),
        (
            {'enabled': True, 'uniform_jitter_samples': [{'prob': True, 'lo': 0, 'hi': 0}]},
            TypeError,
            'center_augment.uniform_jitter_samples[0].prob must be float',
        ),
    ],
)
def test_load_fine_train_config_rejects_invalid_center_augment(
    tmp_path: Path,
    center_augment: dict,
    error_type: type[Exception],
    message: str,
) -> None:
    cfg = _make_fine_train_config(
        tmp_path,
        segy_path='dummy.sgy',
        fb_path='dummy.npy',
        robust_path='dummy.robust.npz',
    )
    cfg['center_augment'] = center_augment

    with pytest.raises(error_type) as exc:
        load_fine_train_config(cfg, base_dir=tmp_path)

    assert message in str(exc.value)


@pytest.mark.parametrize(
    ('mutate', 'message'),
    [
        (
            lambda cfg: cfg['model'].__setitem__('in_chans', 2),
            'model.in_chans must be 1',
        ),
        (
            lambda cfg: cfg['model'].__setitem__('out_chans', 2),
            'model.out_chans must be 1',
        ),
        (
            lambda cfg: cfg['transform'].__setitem__('trace_len', 64),
            'transform.trace_len must be 128',
        ),
        (
            lambda cfg: cfg['transform'].__setitem__('time_len', 128),
            'transform.time_len must be 256',
        ),
        (
            lambda cfg: cfg['transform'].__setitem__('center_index', 64),
            'transform.center_index must be 128',
        ),
        (
            lambda cfg: cfg['train'].__setitem__('fb_sigma_ms', 0.0),
            'train.fb_sigma_ms must be > 0',
        ),
    ],
)
def test_load_fine_train_config_rejects_invalid_fixed_contract(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    cfg = _make_fine_train_config(
        tmp_path,
        segy_path='dummy.sgy',
        fb_path='dummy.npy',
        robust_path='dummy.robust.npz',
    )
    mutate(cfg)

    with pytest.raises(ValueError) as exc:
        load_fine_train_config(cfg, base_dir=tmp_path)

    assert message in str(exc.value)


def test_load_fine_infer_config_returns_default_high_conf_threshold(
    tmp_path: Path,
) -> None:
    typed = load_fine_infer_config(
        _make_fine_infer_config(
            tmp_path,
            segy_path='dummy.sgy',
            robust_path='dummy.robust.npz',
        )
    )

    assert typed.infer.high_conf_threshold == pytest.approx(0.5)
    assert typed.window_center.npz_key == 'robust_pick_i'
    assert typed.window_center.fallback_npz_key is None
    assert typed.viewer.enabled is False
    assert typed.viewer.save_overview_png is False
    assert typed.viewer.save_gather_png is False
    assert typed.viewer.gather_selection == 'first'
    assert typed.viewer.first_panel_only is False


def test_load_fine_infer_config_parses_gather_viewer(tmp_path: Path) -> None:
    cfg = _make_fine_infer_config(
        tmp_path,
        segy_path='dummy.sgy',
        robust_path='dummy.robust.npz',
    )
    cfg['viewer'] = {
        'enabled': True,
        'save_overview_png': False,
        'save_gather_png': True,
        'max_gathers_per_file': 3,
        'gather_selection': 'even',
        'skip_gather_keys': {'ffid': [0], 'cmp': [-1]},
        'max_traces_per_gather': 10000,
        'waveform_norm': 'per_trace',
        'dpi': 120,
        'clip_percentile': 98.5,
        'first_panel_only': True,
        'overlays': {'robust_pick': False, 'gt_pick': True},
    }
    cfg['paths']['viewer_fb_files'] = ['viewer.fb.npy']

    typed = load_fine_infer_config(cfg)

    assert typed.paths.viewer_fb_files == ('viewer.fb.npy',)
    assert typed.viewer.enabled is True
    assert typed.viewer.save_overview_png is False
    assert typed.viewer.save_gather_png is True
    assert typed.viewer.max_gathers_per_file == 3
    assert typed.viewer.gather_selection == 'even'
    assert typed.viewer.skip_gather_keys == {
        'ffid': frozenset({0}),
        'cmp': frozenset({-1}),
    }
    assert typed.viewer.max_traces_per_gather == 10000
    assert typed.viewer.waveform_norm == 'per_trace'
    assert typed.viewer.dpi == 120
    assert typed.viewer.clip_percentile == pytest.approx(98.5)
    assert typed.viewer.first_panel_only is True
    assert typed.viewer.overlays['gt_pick'] is True
    assert typed.viewer.overlays['robust_pick'] is False
    assert typed.viewer.overlays['coarse_pick'] is True


def test_load_fine_infer_config_rejects_invalid_gather_selection(
    tmp_path: Path,
) -> None:
    cfg = _make_fine_infer_config(
        tmp_path,
        segy_path='dummy.sgy',
        robust_path='dummy.robust.npz',
    )
    cfg['viewer'] = {'gather_selection': 'evenly_spaced'}

    with pytest.raises(ValueError) as exc:
        load_fine_infer_config(cfg)

    assert 'viewer.gather_selection must be one of: first, even' in str(exc.value)


def test_load_fine_infer_config_rejects_non_bool_first_panel_only(
    tmp_path: Path,
) -> None:
    cfg = _make_fine_infer_config(
        tmp_path,
        segy_path='dummy.sgy',
        robust_path='dummy.robust.npz',
    )
    cfg['viewer'] = {'first_panel_only': 1}

    with pytest.raises(TypeError) as exc:
        load_fine_infer_config(cfg)

    assert 'first_panel_only' in str(exc.value)


def test_load_fine_infer_config_parses_window_center(tmp_path: Path) -> None:
    cfg = _make_fine_infer_config(
        tmp_path,
        segy_path='dummy.sgy',
        robust_path='dummy.robust.npz',
    )
    cfg['window_center'] = {
        'npz_key': 'fine_center_i',
        'fallback_npz_key': 'robust_pick_i',
    }

    typed = load_fine_infer_config(cfg)

    assert typed.window_center.npz_key == 'fine_center_i'
    assert typed.window_center.fallback_npz_key == 'robust_pick_i'


def test_load_fine_infer_config_parses_explicit_coarse_npz_files(
    tmp_path: Path,
) -> None:
    cfg = _make_fine_infer_config(
        tmp_path,
        segy_path='dummy.sgy',
        robust_path='dummy.robust.npz',
        coarse_path='other_dir/dummy.coarse.npz',
    )

    typed = load_fine_infer_config(cfg)

    assert typed.paths.coarse_npz_files == ('other_dir/dummy.coarse.npz',)


def test_sample_center_jitter_is_reproducible_and_uses_weight_mixture() -> None:
    cfg = FineCenterAugmentCfg(
        enabled=True,
        train_only=True,
        p_no_jitter=0.5,
        uniform_jitter_samples=(FineUniformJitterCfg(prob=0.5, lo=10, hi=10),),
        clip_to_record=True,
        require_fb_inside=True,
    )

    first = sample_center_jitter(size=64, cfg=cfg, rng=np.random.default_rng(123))
    second = sample_center_jitter(size=64, cfg=cfg, rng=np.random.default_rng(123))

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.int64
    assert first.shape == (64,)
    assert set(first.tolist()).issubset({0, 10})
    assert 0 in first
    assert 10 in first


def test_prepare_fine_infer_cfg_expands_coarse_npz_listfile(tmp_path: Path) -> None:
    data_dir = tmp_path / 'data'
    physics_dir = tmp_path / 'physics'
    coarse_dir = tmp_path / 'coarse'
    list_dir = tmp_path / 'lists'
    for directory in (data_dir, physics_dir, coarse_dir, list_dir):
        directory.mkdir()

    segy_path = data_dir / 'sample.sgy'
    robust_path = physics_dir / 'sample.robust.npz'
    coarse_path = coarse_dir / 'sample.coarse.npz'
    fb_path = data_dir / 'sample.fb.npy'
    for path in (segy_path, robust_path, coarse_path, fb_path):
        path.touch()

    segy_list = list_dir / 'segy.txt'
    fb_list = list_dir / 'fb.txt'
    robust_list = list_dir / 'robust.txt'
    coarse_list = list_dir / 'coarse.txt'
    segy_list.write_text('../data/sample.sgy\n', encoding='utf-8')
    fb_list.write_text('../data/sample.fb.npy\n', encoding='utf-8')
    robust_list.write_text('../physics/sample.robust.npz\n', encoding='utf-8')
    coarse_list.write_text('../coarse/sample.coarse.npz\n', encoding='utf-8')

    cfg = _make_fine_infer_config(
        tmp_path,
        segy_path='unused.sgy',
        robust_path='unused.robust.npz',
    )
    cfg['paths'] = {
        'segy_files': 'lists/segy.txt',
        'fb_files': 'lists/fb.txt',
        'robust_npz_files': 'lists/robust.txt',
        'coarse_npz_files': 'lists/coarse.txt',
        'out_dir': 'out',
    }

    prepared = _prepare_fine_infer_cfg(cfg, base_dir=tmp_path)
    typed = load_fine_infer_config(prepared)

    assert typed.paths.segy_files == (str(segy_path.resolve()),)
    assert typed.paths.fb_files == (str(fb_path.resolve()),)
    assert typed.paths.robust_npz_files == (str(robust_path.resolve()),)
    assert typed.paths.coarse_npz_files == (str(coarse_path.resolve()),)


def test_load_fine_infer_config_rejects_explicit_coarse_length_mismatch(
    tmp_path: Path,
) -> None:
    cfg = _make_fine_infer_config(
        tmp_path,
        segy_path='dummy.sgy',
        robust_path='dummy.robust.npz',
    )
    cfg['paths']['coarse_npz_files'] = [
        'one.coarse.npz',
        'two.coarse.npz',
    ]

    with pytest.raises(ValueError) as exc:
        load_fine_infer_config(cfg)

    assert 'must have the same length' in str(exc.value)


def test_load_fine_infer_config_rejects_invalid_overlap(tmp_path: Path) -> None:
    cfg = _make_fine_infer_config(
        tmp_path,
        segy_path='dummy.sgy',
        robust_path='dummy.robust.npz',
    )
    cfg['infer']['overlap_h'] = 128

    with pytest.raises(ValueError) as exc:
        load_fine_infer_config(cfg)

    assert 'infer.overlap_h must satisfy 0 <= overlap_h < transform.trace_len' in str(
        exc.value
    )


def test_load_fine_infer_config_rejects_invalid_high_conf_threshold(
    tmp_path: Path,
) -> None:
    cfg = _make_fine_infer_config(
        tmp_path,
        segy_path='dummy.sgy',
        robust_path='dummy.robust.npz',
    )
    cfg['infer']['high_conf_threshold'] = 1.1

    with pytest.raises(ValueError) as exc:
        load_fine_infer_config(cfg)

    assert 'infer.high_conf_threshold must lie in [0, 1]' in str(exc.value)


def test_resolve_fine_coarse_npz_path_prefers_explicit_path(tmp_path: Path) -> None:
    robust_path = tmp_path / 'physics' / 'sample.robust.npz'
    explicit_path = tmp_path / 'coarse' / 'sample.coarse.npz'

    resolved = resolve_fine_coarse_npz_path(
        robust_npz_path=robust_path,
        explicit_coarse_npz_path=explicit_path,
    )

    assert resolved == explicit_path.resolve()


def test_resolve_fine_coarse_npz_path_infers_from_robust_path(tmp_path: Path) -> None:
    robust_path = tmp_path / 'same_dir' / 'sample.robust.npz'

    resolved = resolve_fine_coarse_npz_path(
        robust_npz_path=robust_path,
        explicit_coarse_npz_path=None,
    )

    assert resolved == (tmp_path / 'same_dir' / 'sample.coarse.npz').resolve()


def test_require_existing_coarse_npz_path_reports_missing_explicit(
    tmp_path: Path,
) -> None:
    coarse_path = tmp_path / 'missing' / 'sample.coarse.npz'

    with pytest.raises(FileNotFoundError) as exc:
        require_existing_coarse_npz_path(
            coarse_npz_path=coarse_path,
            robust_npz_path=tmp_path / 'sample.robust.npz',
            was_explicit=True,
        )

    message = str(exc.value)
    assert 'coarse npz file not found' in message
    assert str(coarse_path.resolve()) in message
    assert 'Set paths.coarse_npz_files explicitly' not in message


def test_require_existing_coarse_npz_path_suggests_explicit_for_missing_inferred(
    tmp_path: Path,
) -> None:
    robust_path = tmp_path / 'physics' / 'sample.robust.npz'
    coarse_path = tmp_path / 'physics' / 'sample.coarse.npz'

    with pytest.raises(FileNotFoundError) as exc:
        require_existing_coarse_npz_path(
            coarse_npz_path=coarse_path,
            robust_npz_path=robust_path,
            was_explicit=False,
        )

    message = str(exc.value)
    assert 'coarse npz file not found' in message
    assert f'inferred from robust npz: {robust_path.resolve()}' in message
    assert 'Set paths.coarse_npz_files explicitly' in message


def test_extract_local_windowed_view_zero_pads_and_restores_indices() -> None:
    x = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [10, 11, 12, 13, 14, 15],
        ],
        dtype=np.float32,
    )
    local = extract_local_windowed_view(
        x,
        center_raw_i=np.array([1, 4], dtype=np.int64),
        trace_valid=np.array([True, True], dtype=np.bool_),
        fb_raw_i=np.array([2, 5], dtype=np.int64),
        time_len=4,
        center_index=2,
        require_fb_inside=True,
    )

    assert local is not None
    np.testing.assert_array_equal(
        local.x_view,
        np.array(
            [
                [0, 0, 1, 2],
                [12, 13, 14, 15],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(local.fb_idx_view, np.array([3, 3], dtype=np.int64))
    np.testing.assert_array_equal(local.window_start_i, np.array([-1, 2], dtype=np.int32))
    np.testing.assert_array_equal(local.window_end_i, np.array([3, 6], dtype=np.int32))

    restored = restore_local_pick_to_raw(
        np.array([3.0, 3.0], dtype=np.float32),
        window_start_i=local.window_start_i,
        n_samples_orig=6,
    )
    np.testing.assert_allclose(restored, np.array([2.0, 5.0], dtype=np.float32))


def test_fine_train_dataset_rejects_when_gt_pick_is_outside_local_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 128
    n_samples = 400
    dt_sec = 0.002
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    fb = np.full((n_traces,), 390, dtype=np.int64)
    robust = np.full((n_traces,), 10, dtype=np.int32)

    segy_path = _register_synthetic_segy(tmp_path, 'reject.sgy', traces)
    fb_path = _write_fb(tmp_path / 'reject_fb.npy', fb)
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'reject.robust.npz',
        robust_pick_i=robust,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )

    ds = build_train_dataset(
        segy_files=[segy_path],
        fb_files=[fb_path],
        robust_npz_files=[robust_path],
        sampling_overrides=None,
        plan=build_plan(sigma_ms=3.0, sigma_samples_min=1.5, sigma_samples_max=12.0),
        fbgate=build_fbgate(apply_on='off', min_pick_ratio=0.0, verbose=False),
        trace_len=128,
        time_len=256,
        center_index=128,
        standardize_eps=1.0e-8,
        trace_decimate_prob=0.0,
        trace_decimate_stride_range=(1, 1),
        primary_keys=('ffid',),
        secondary_key_fixed=False,
        verbose=False,
        progress=False,
        max_trials=1,
        use_header_cache=False,
        waveform_mode='eager',
        segy_endian='big',
    )

    try:
        ds._rng = np.random.default_rng(0)
        with pytest.raises(RuntimeError, match='local_window=1'):
            _ = ds[0]
    finally:
        ds.close()


def test_fine_train_center_augment_constant_jitter_shifts_center(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 128
    n_samples = 512
    dt_sec = 0.002
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    robust = np.full((n_traces,), 100, dtype=np.int32)
    fb = np.full((n_traces,), 110, dtype=np.int64)

    segy_path = _register_synthetic_segy(tmp_path, 'jitter_shift.sgy', traces)
    fb_path = _write_fb(tmp_path / 'jitter_shift_fb.npy', fb)
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'jitter_shift.robust.npz',
        robust_pick_i=robust,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )
    center_augment = FineCenterAugmentCfg(
        enabled=True,
        train_only=True,
        p_no_jitter=0.0,
        uniform_jitter_samples=(FineUniformJitterCfg(prob=1.0, lo=10, hi=10),),
        clip_to_record=True,
        require_fb_inside=True,
    )
    ds = _build_test_fine_train_dataset(
        segy_path=segy_path,
        fb_path=fb_path,
        robust_path=robust_path,
        center_augment=center_augment,
    )

    try:
        ds._rng = np.random.default_rng(0)
        sample = ds[0]
    finally:
        ds.close()

    meta = sample['meta']
    np.testing.assert_array_equal(meta['center_raw_i'], np.full((128,), 110, dtype=np.int32))
    np.testing.assert_array_equal(meta['window_start_i'], np.full((128,), -18, dtype=np.int32))


def test_fine_labeled_infer_dataset_does_not_apply_center_jitter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 128
    n_samples = 512
    dt_sec = 0.002
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    robust = np.full((n_traces,), 100, dtype=np.int32)
    fb = np.full((n_traces,), 110, dtype=np.int64)

    segy_path = _register_synthetic_segy(tmp_path, 'jitter_valid.sgy', traces)
    fb_path = _write_fb(tmp_path / 'jitter_valid_fb.npy', fb)
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'jitter_valid.robust.npz',
        robust_pick_i=robust,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )
    ds = _build_test_fine_labeled_infer_dataset(
        segy_path=segy_path,
        fb_path=fb_path,
        robust_path=robust_path,
    )

    try:
        ds._rng = np.random.default_rng(0)
        sample = ds[0]
    finally:
        ds.close()

    meta = sample['meta']
    np.testing.assert_array_equal(meta['center_raw_i'], robust)
    np.testing.assert_array_equal(meta['window_start_i'], robust.astype(np.int32) - 128)


def test_fine_train_dataset_uses_configured_fine_center(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 128
    n_samples = 512
    dt_sec = 0.002
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    robust = np.full((n_traces,), 80, dtype=np.int32)
    fine_center = np.linspace(180, 220, n_traces, dtype=np.int32)
    fb = fine_center.astype(np.int64)

    segy_path = _register_synthetic_segy(tmp_path, 'train_fine_center.sgy', traces)
    fb_path = _write_fb(tmp_path / 'train_fine_center_fb.npy', fb)
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'train_fine_center.robust.npz',
        robust_pick_i=robust,
        fine_center_i=fine_center,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )
    ds = _build_test_fine_train_dataset(
        segy_path=segy_path,
        fb_path=fb_path,
        robust_path=robust_path,
        window_center_npz_key='fine_center_i',
        window_center_fallback_npz_key='robust_pick_i',
    )

    try:
        ds._rng = np.random.default_rng(0)
        sample = ds[0]
    finally:
        ds.close()

    meta = sample['meta']
    np.testing.assert_array_equal(meta['center_raw_i'], fine_center)
    np.testing.assert_array_equal(
        meta['window_start_i'],
        fine_center.astype(np.int32) - 128,
    )


def test_fine_train_dataset_falls_back_to_robust_pick_when_fine_center_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 128
    n_samples = 512
    dt_sec = 0.002
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    robust = np.full((n_traces,), 200, dtype=np.int32)
    fb = robust.astype(np.int64)

    segy_path = _register_synthetic_segy(tmp_path, 'train_center_fallback.sgy', traces)
    fb_path = _write_fb(tmp_path / 'train_center_fallback_fb.npy', fb)
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'train_center_fallback.robust.npz',
        robust_pick_i=robust,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )
    ds = _build_test_fine_train_dataset(
        segy_path=segy_path,
        fb_path=fb_path,
        robust_path=robust_path,
        window_center_npz_key='fine_center_i',
        window_center_fallback_npz_key='robust_pick_i',
    )

    try:
        ds._rng = np.random.default_rng(0)
        sample = ds[0]
    finally:
        ds.close()

    np.testing.assert_array_equal(sample['meta']['center_raw_i'], robust)


def test_fine_train_dataset_raises_when_center_key_and_fallback_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 128
    n_samples = 512
    dt_sec = 0.002
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    robust = np.full((n_traces,), 200, dtype=np.int32)
    fb = robust.astype(np.int64)

    segy_path = _register_synthetic_segy(tmp_path, 'train_center_missing.sgy', traces)
    fb_path = _write_fb(tmp_path / 'train_center_missing_fb.npy', fb)
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'train_center_missing.robust.npz',
        robust_pick_i=robust,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )

    with pytest.raises(KeyError, match='fine window center key'):
        _build_test_fine_train_dataset(
            segy_path=segy_path,
            fb_path=fb_path,
            robust_path=robust_path,
            window_center_npz_key='fine_center_i',
        )


def test_fine_train_center_augment_clips_to_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 128
    n_samples = 512
    dt_sec = 0.002
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    robust = np.full((n_traces,), 5, dtype=np.int32)
    fb = np.full((n_traces,), 5, dtype=np.int64)

    segy_path = _register_synthetic_segy(tmp_path, 'jitter_clip.sgy', traces)
    fb_path = _write_fb(tmp_path / 'jitter_clip_fb.npy', fb)
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'jitter_clip.robust.npz',
        robust_pick_i=robust,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )
    center_augment = FineCenterAugmentCfg(
        enabled=True,
        train_only=True,
        p_no_jitter=0.0,
        uniform_jitter_samples=(FineUniformJitterCfg(prob=1.0, lo=-100, hi=-100),),
        clip_to_record=True,
        require_fb_inside=True,
    )
    ds = _build_test_fine_train_dataset(
        segy_path=segy_path,
        fb_path=fb_path,
        robust_path=robust_path,
        center_augment=center_augment,
    )

    try:
        ds._rng = np.random.default_rng(0)
        sample = ds[0]
    finally:
        ds.close()

    np.testing.assert_array_equal(
        sample['meta']['center_raw_i'],
        np.zeros((128,), dtype=np.int32),
    )


def test_fine_train_center_augment_rejects_out_of_window_jitter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 128
    n_samples = 1000
    dt_sec = 0.002
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    robust = np.full((n_traces,), 500, dtype=np.int32)
    fb = np.full((n_traces,), 500, dtype=np.int64)

    segy_path = _register_synthetic_segy(tmp_path, 'jitter_reject.sgy', traces)
    fb_path = _write_fb(tmp_path / 'jitter_reject_fb.npy', fb)
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'jitter_reject.robust.npz',
        robust_pick_i=robust,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )
    center_augment = FineCenterAugmentCfg(
        enabled=True,
        train_only=True,
        p_no_jitter=0.0,
        uniform_jitter_samples=(FineUniformJitterCfg(prob=1.0, lo=300, hi=300),),
        clip_to_record=True,
        require_fb_inside=True,
    )
    ds = _build_test_fine_train_dataset(
        segy_path=segy_path,
        fb_path=fb_path,
        robust_path=robust_path,
        center_augment=center_augment,
        max_trials=1,
    )

    try:
        ds._rng = np.random.default_rng(0)
        with pytest.raises(RuntimeError, match='center_jitter=1'):
            _ = ds[0]
    finally:
        ds.close()


def test_fine_infer_dataset_meta_includes_raw_restore_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 160
    n_samples = 512
    dt_sec = 0.002
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    robust = np.linspace(180, 220, n_traces, dtype=np.int32)

    segy_path = _register_synthetic_segy(tmp_path, 'infer_meta.sgy', traces)
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'infer_meta.robust.npz',
        robust_pick_i=robust,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )

    ds = build_raw_infer_dataset(
        segy_files=[segy_path],
        robust_npz_files=[robust_path],
        plan=build_plan(sigma_ms=3.0, sigma_samples_min=1.5, sigma_samples_max=12.0),
        trace_len=128,
        overlap_h=96,
        time_len=256,
        center_index=128,
        standardize_eps=1.0e-8,
        waveform_mode='eager',
        segy_endian='big',
        use_header_cache=False,
    )

    try:
        sample = ds[0]
    finally:
        ds.close()

    meta = sample['meta']
    assert tuple(sample['input'].shape) == (1, 128, 256)
    assert meta['trace_slice_start'] == 0
    assert meta['trace_slice_end'] == 128
    assert meta['window_start_i'].shape == (128,)
    assert meta['window_end_i'].shape == (128,)
    assert meta['center_raw_i'].shape == (128,)
    assert meta['raw_idx_global'].shape == (128,)
    np.testing.assert_array_equal(meta['raw_idx_global'][:5], np.arange(5, dtype=np.int64))
    assert int(meta['window_end_i'][0] - meta['window_start_i'][0]) == 256
    assert int(meta['center_raw_i'][0]) == int(robust[0])


def test_fine_infer_dataset_uses_configured_fine_center(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 160
    n_samples = 512
    dt_sec = 0.002
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    robust = np.full((n_traces,), 80, dtype=np.int32)
    fine_center = np.linspace(180, 220, n_traces, dtype=np.int32)

    segy_path = _register_synthetic_segy(tmp_path, 'infer_fine_center.sgy', traces)
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'infer_fine_center.robust.npz',
        robust_pick_i=robust,
        fine_center_i=fine_center,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )

    ds = build_raw_infer_dataset(
        segy_files=[segy_path],
        robust_npz_files=[robust_path],
        plan=build_plan(sigma_ms=3.0, sigma_samples_min=1.5, sigma_samples_max=12.0),
        trace_len=128,
        overlap_h=96,
        time_len=256,
        center_index=128,
        standardize_eps=1.0e-8,
        waveform_mode='eager',
        segy_endian='big',
        use_header_cache=False,
        window_center_npz_key='fine_center_i',
        window_center_fallback_npz_key='robust_pick_i',
    )

    try:
        sample = ds[0]
    finally:
        ds.close()

    meta = sample['meta']
    np.testing.assert_array_equal(meta['center_raw_i'][:5], fine_center[:5])
    np.testing.assert_array_equal(meta['window_start_i'][:5], fine_center[:5] - 128)


def test_build_fine_init_state_dict_slices_first_conv_waveform_channel() -> None:
    coarse_model = build_coarse_model(_coarse_model_sig())
    fine_model = build_fine_model(_fine_model_sig())
    coarse_sd = coarse_model.state_dict()
    fine_sd = fine_model.state_dict()

    shape = tuple(coarse_sd['backbone.conv1.weight'].shape)
    coarse_sd['backbone.conv1.weight'] = torch.arange(
        int(np.prod(shape)),
        dtype=torch.float32,
    ).reshape(shape)

    load_sd = build_fine_init_state_dict(
        coarse_sd,
        fine_sd,
        reset_seg_head=True,
        reset_first_bn_stats=False,
    )

    expected = coarse_sd['backbone.conv1.weight'][:, 0:1, :, :]
    assert torch.equal(load_sd['backbone.conv1.weight'], expected)


def test_build_fine_init_state_dict_skips_seg_head_keys() -> None:
    coarse_model = build_coarse_model(_coarse_model_sig())
    fine_model = build_fine_model(_fine_model_sig())
    load_sd = build_fine_init_state_dict(
        coarse_model.state_dict(),
        fine_model.state_dict(),
        reset_seg_head=True,
        reset_first_bn_stats=False,
    )

    assert not any(key.startswith('seg_head.') for key in load_sd)


def test_load_fine_init_from_coarse_checkpoint_resets_first_bn_stats_and_keeps_seg_head(
    tmp_path: Path,
) -> None:
    coarse_model = build_coarse_model(_coarse_model_sig())
    fine_model = build_fine_model(_fine_model_sig())
    fine_before = {key: value.clone() for key, value in fine_model.state_dict().items()}

    coarse_sd = coarse_model.state_dict()
    coarse_sd['backbone.bn1.running_mean'].fill_(9.0)
    coarse_sd['backbone.bn1.running_var'].fill_(5.0)
    coarse_sd['backbone.bn1.num_batches_tracked'].fill_(11)

    ckpt_path = tmp_path / 'coarse_for_fine.pt'
    torch.save(
        {
            'version': 1,
            'pipeline': 'fbpick',
            'epoch': 0,
            'global_step': 0,
            'model_sig': _coarse_model_sig(),
            'model_state_dict': coarse_sd,
            'optimizer_state_dict': {},
            'lr_scheduler_sig': None,
            'lr_scheduler_state_dict': None,
            'cfg': {},
            'output_ids': ['P'],
            'softmax_axis': 'time',
        },
        ckpt_path,
    )

    load_fine_init_from_coarse_checkpoint(
        fine_model,
        ckpt_path,
        fine_model_sig=_fine_model_sig(),
        reset_seg_head=True,
        reset_first_bn_stats=True,
    )
    fine_after = fine_model.state_dict()

    assert torch.equal(
        fine_after['backbone.bn1.running_mean'],
        fine_before['backbone.bn1.running_mean'],
    )
    assert torch.equal(
        fine_after['backbone.bn1.running_var'],
        fine_before['backbone.bn1.running_var'],
    )
    assert torch.equal(
        fine_after['backbone.bn1.num_batches_tracked'],
        fine_before['backbone.bn1.num_batches_tracked'],
    )
    for key, value in fine_before.items():
        if key.startswith('seg_head.'):
            assert torch.equal(fine_after[key], value)


def test_fine_train_smoke_one_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 128
    n_samples = 400
    dt_sec = 0.002
    robust = np.full((n_traces,), 200, dtype=np.int32)
    fb = robust.astype(np.int64) + 10
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    for trace_idx, pick_idx in enumerate(fb.tolist()):
        traces[trace_idx, int(pick_idx)] = 5.0 + 0.01 * float(trace_idx)

    segy_path = _register_synthetic_segy(tmp_path, 'train_smoke.sgy', traces)
    fb_path = _write_fb(tmp_path / 'train_smoke_fb.npy', fb)
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'train_smoke.robust.npz',
        robust_pick_i=robust,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    coarse_ckpt_path = _write_coarse_ckpt(tmp_path)
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )

    cfg = _make_fine_train_config(
        tmp_path,
        segy_path=segy_path,
        fb_path=fb_path,
        robust_path=robust_path,
        coarse_ckpt_path=coarse_ckpt_path,
        use_coarse_init=True,
    )
    cfg_path = tmp_path / 'config_train_fbpick_fine.yaml'
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


def test_fine_run_train_writes_fbpick_ckpt_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 128
    n_samples = 400
    dt_sec = 0.002
    robust = np.full((n_traces,), 200, dtype=np.int32)
    fb = robust.astype(np.int64) + 8
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    for trace_idx, pick_idx in enumerate(fb.tolist()):
        traces[trace_idx, int(pick_idx)] = 6.0 + 0.01 * float(trace_idx)

    segy_path = _register_synthetic_segy(tmp_path, 'train_ckpt.sgy', traces)
    fb_path = _write_fb(tmp_path / 'train_ckpt_fb.npy', fb)
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'train_ckpt.robust.npz',
        robust_pick_i=robust,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )

    cfg = _make_fine_train_config(
        tmp_path,
        segy_path=segy_path,
        fb_path=fb_path,
        robust_path=robust_path,
    )
    cfg_path = tmp_path / 'config_train_fbpick_fine_ckpt.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')

    run_train(cfg_path, device=torch.device('cpu'))

    ckpt = load_checkpoint(tmp_path / 'fine_train_out' / 'ckpt' / 'best.pt')
    assert ckpt['pipeline'] == 'fbpick'
    assert ckpt['stage'] == 'fine'
    assert ckpt['output_ids'] == ['P']
    assert ckpt['softmax_axis'] == 'time'
    assert ckpt['model_sig']['in_chans'] == 1
    assert ckpt['model_sig']['out_chans'] == 1


def test_run_fine_local_infer_restores_raw_indices_and_covers_all_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 160
    n_samples = 512
    dt_sec = 0.002
    trace_axis = np.arange(n_traces, dtype=np.int32)
    robust = 220 + ((trace_axis % 7) - 3) * 4
    offset = ((trace_axis % 5) - 2) * 7
    final_pick = robust + offset

    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    for trace_idx, pick_idx in enumerate(final_pick.tolist()):
        traces[trace_idx, int(pick_idx)] = 10.0 + 0.01 * float(trace_idx)

    segy_path = _register_synthetic_segy(tmp_path, 'fine_infer.sgy', traces)
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'fine_infer.robust.npz',
        robust_pick_i=robust,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )

    result = run_fine_local_infer(
        model=_IdentityFineModel(),
        cfg=_make_fine_infer_config(
            tmp_path,
            segy_path=segy_path,
            robust_path=robust_path,
        ),
        device=torch.device('cpu'),
    )

    validate_fine_result_payload(result)
    assert set(FINE_RESULT_REQUIRED_KEYS).issubset(result.keys())
    np.testing.assert_array_equal(
        result['trace_indices'],
        np.arange(n_traces, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        result['window_start_i'],
        (robust - 128).astype(np.int32),
    )
    np.testing.assert_array_equal(
        result['window_end_i'],
        (robust + 128).astype(np.int32),
    )
    np.testing.assert_array_equal(
        result['fine_pick_local_i'],
        (final_pick - (robust - 128)).astype(np.int32),
    )
    np.testing.assert_array_equal(
        result['final_pick_i'],
        final_pick.astype(np.int32),
    )
    np.testing.assert_allclose(
        result['final_pick_t_sec'],
        final_pick.astype(np.float32) * np.float32(dt_sec),
        atol=1.0e-6,
    )


def test_run_fine_local_infer_uses_configured_fine_center(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 160
    n_samples = 512
    dt_sec = 0.002
    trace_axis = np.arange(n_traces, dtype=np.int32)
    robust = np.full((n_traces,), 80, dtype=np.int32)
    fine_center = 220 + ((trace_axis % 7) - 3) * 4
    final_pick = fine_center + ((trace_axis % 5) - 2) * 7

    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    for trace_idx, pick_idx in enumerate(final_pick.tolist()):
        traces[trace_idx, int(pick_idx)] = 10.0 + 0.01 * float(trace_idx)

    segy_path = _register_synthetic_segy(tmp_path, 'fine_center_infer.sgy', traces)
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'fine_center_infer.robust.npz',
        robust_pick_i=robust,
        fine_center_i=fine_center,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )

    cfg = _make_fine_infer_config(
        tmp_path,
        segy_path=segy_path,
        robust_path=robust_path,
    )
    cfg['window_center'] = {
        'npz_key': 'fine_center_i',
        'fallback_npz_key': 'robust_pick_i',
    }
    result = run_fine_local_infer(
        model=_IdentityFineModel(),
        cfg=cfg,
        device=torch.device('cpu'),
    )

    np.testing.assert_array_equal(
        result['window_start_i'],
        (fine_center - 128).astype(np.int32),
    )
    np.testing.assert_array_equal(result['final_pick_i'], final_pick.astype(np.int32))


def test_run_fine_infer_builds_and_saves_final_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 160
    n_samples = 512
    dt_sec = 0.002
    trace_axis = np.arange(n_traces, dtype=np.int32)
    robust = 220 + ((trace_axis % 7) - 3) * 4
    offset = ((trace_axis % 5) - 2) * 7
    final_pick = robust + offset

    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    for trace_idx, pick_idx in enumerate(final_pick.tolist()):
        traces[trace_idx, int(pick_idx)] = 10.0 + 0.01 * float(trace_idx)

    segy_path = _register_synthetic_segy(tmp_path, 'fine_public.sgy', traces)
    coarse_path = _write_coarse_with_n_samples(
        tmp_path / 'fine_public.coarse.npz',
        coarse_pick_i=robust.astype(np.int32),
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'fine_public.robust.npz',
        robust_pick_i=robust,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )

    result = run_fine_infer(
        model=_IdentityFineModel(),
        cfg=_make_fine_infer_config(
            tmp_path,
            segy_path=segy_path,
            robust_path=robust_path,
        ),
        device=torch.device('cpu'),
    )

    _ = coarse_path
    validate_fbpick_final_payload(result, high_conf_threshold=0.5)
    np.testing.assert_array_equal(result['final_pick_i'], final_pick.astype(np.int32))
    np.testing.assert_array_equal(result['reject_mask'], np.zeros(n_traces, dtype=np.bool_))
    np.testing.assert_array_equal(result['high_conf_mask'], np.ones(n_traces, dtype=np.bool_))
    np.testing.assert_array_equal(
        result['window_end_i'],
        (result['window_start_i'] + 255).astype(np.int32),
    )

    saved = load_fbpick_final_npz(
        tmp_path / 'fine_infer_out' / build_final_npz_name(segy_path)
    )
    np.testing.assert_array_equal(saved['final_pick_i'], final_pick.astype(np.int32))


def test_run_fine_infer_uses_explicit_coarse_npz_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 160
    n_samples = 512
    dt_sec = 0.002
    trace_axis = np.arange(n_traces, dtype=np.int32)
    robust = 220 + ((trace_axis % 7) - 3) * 4
    final_pick = robust + ((trace_axis % 5) - 2) * 7

    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    for trace_idx, pick_idx in enumerate(final_pick.tolist()):
        traces[trace_idx, int(pick_idx)] = 10.0 + 0.01 * float(trace_idx)

    segy_path = _register_synthetic_segy(tmp_path, 'fine_explicit.sgy', traces)
    coarse_path = _write_coarse_with_n_samples(
        tmp_path / 'coarse_out' / 'fine_explicit.coarse.npz',
        coarse_pick_i=robust.astype(np.int32),
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'physics_out' / 'fine_explicit.robust.npz',
        robust_pick_i=robust,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )

    result = run_fine_infer(
        model=_IdentityFineModel(),
        cfg=_make_fine_infer_config(
            tmp_path,
            segy_path=segy_path,
            robust_path=robust_path,
            coarse_path=coarse_path,
        ),
        device=torch.device('cpu'),
    )

    validate_fbpick_final_payload(result, high_conf_threshold=0.5)
    np.testing.assert_array_equal(result['final_pick_i'], final_pick.astype(np.int32))


def test_save_fine_gather_qc_pngs_skips_configured_key_and_counts_accepted(
    tmp_path: Path,
) -> None:
    traces = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
    info = _make_qc_info(
        traces=traces,
        key_to_indices={0: [0], 1: [1], 2: [2]},
        n_traces=3,
        n_samples=8,
    )
    cfg = _make_fine_infer_config(
        tmp_path,
        segy_path='dummy.sgy',
        robust_path='dummy.robust.npz',
    )
    cfg['viewer'] = {
        'enabled': True,
        'save_gather_png': True,
        'max_gathers_per_file': 2,
        'skip_gather_keys': {'ffid': [0]},
        'max_traces_per_gather': None,
        'waveform_norm': 'per_trace',
        'first_panel_only': True,
    }
    viewer = load_fine_infer_config(cfg).viewer
    captured: list[np.ndarray] = []

    def _save(out_png: Path, **kwargs: object) -> Path:
        captured.append(np.asarray(kwargs['trace_indices'], dtype=np.int64))
        assert kwargs['waveform_norm'] == 'per_trace'
        assert kwargs['first_panel_only'] is True
        return Path(out_png)

    out_paths = _save_fine_gather_qc_pngs(
        info=info,
        segy_path=tmp_path / 'line' / 'small.sgy',
        out_dir=tmp_path / 'out',
        final_payload=_make_final_payload_for_qc(n_traces=3, n_samples=8),
        viewer=viewer,
        primary_keys=('ffid',),
        save_png_func=_save,
    )

    assert len(out_paths) == 2
    assert [int(indices[0]) for indices in captured] == [1, 2]


def test_save_fine_gather_qc_pngs_passes_viewer_gather_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = np.arange(8, dtype=np.float32).reshape(1, 8)
    info = _make_qc_info(
        traces=traces,
        key_to_indices={10: [0]},
        n_traces=1,
        n_samples=8,
    )
    cfg = _make_fine_infer_config(
        tmp_path,
        segy_path='dummy.sgy',
        robust_path='dummy.robust.npz',
    )
    cfg['viewer'] = {
        'enabled': True,
        'save_gather_png': True,
        'max_gathers_per_file': 1,
        'gather_selection': 'even',
    }
    viewer = load_fine_infer_config(cfg).viewer
    captured: dict[str, object] = {}

    def _iter_qc_gathers(
        info_arg: object,
        *,
        primary_keys: object,
        max_gathers: object,
        skip_gather_keys: object,
        max_traces_per_gather: object,
        segy_path: object,
        gather_selection: object,
    ) -> Iterator[tuple[str, int, np.ndarray]]:
        _ = (
            info_arg,
            primary_keys,
            max_gathers,
            skip_gather_keys,
            max_traces_per_gather,
            segy_path,
        )
        captured['gather_selection'] = gather_selection
        yield 'ffid', 10, np.asarray([0], dtype=np.int64)

    def _save(out_png: Path, **kwargs: object) -> Path:
        _ = kwargs
        return Path(out_png)

    monkeypatch.setattr(fine_infer_module, 'iter_qc_gathers', _iter_qc_gathers)

    out_paths = _save_fine_gather_qc_pngs(
        info=info,
        segy_path=tmp_path / 'line' / 'small.sgy',
        out_dir=tmp_path / 'out',
        final_payload=_make_final_payload_for_qc(n_traces=1, n_samples=8),
        viewer=viewer,
        primary_keys=('ffid',),
        save_png_func=_save,
    )

    assert len(out_paths) == 1
    assert captured == {'gather_selection': 'even'}


def test_save_fine_gather_qc_pngs_skips_oversized_before_mmap_access(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingMmap:
        def __getitem__(self, index: int) -> np.ndarray:
            raise AssertionError(f'mmap should not be read for skipped gather {index}')

    def _unexpected_save(*args, **kwargs) -> Path:
        raise AssertionError('PNG writer should not be called')

    info = _make_qc_info(
        traces=FailingMmap(),
        key_to_indices={1: np.arange(5)},
        n_traces=5,
        n_samples=8,
    )
    cfg = _make_fine_infer_config(
        tmp_path,
        segy_path='dummy.sgy',
        robust_path='dummy.robust.npz',
    )
    cfg['viewer'] = {
        'enabled': True,
        'save_gather_png': True,
        'max_gathers_per_file': 1,
        'skip_gather_keys': {},
        'max_traces_per_gather': 3,
    }
    viewer = load_fine_infer_config(cfg).viewer
    segy_path = tmp_path / 'line' / 'huge.sgy'

    out_paths = _save_fine_gather_qc_pngs(
        info=info,
        segy_path=segy_path,
        out_dir=tmp_path / 'out',
        final_payload=_make_final_payload_for_qc(n_traces=5, n_samples=8),
        viewer=viewer,
        primary_keys=('ffid',),
        save_png_func=_unexpected_save,
    )

    assert out_paths == []
    captured = capsys.readouterr()
    assert 'skip oversized gather:' in captured.out
    assert f'No fine gather PNGs written for {segy_path}: all candidates were skipped' in (
        captured.out
    )


def test_run_fine_infer_gather_qc_does_not_materialize_full_waveform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_traces = 160
    n_samples = 512
    dt_sec = 0.002
    trace_axis = np.arange(n_traces, dtype=np.int32)
    robust = 220 + ((trace_axis % 7) - 3) * 4
    final_pick = robust + ((trace_axis % 5) - 2) * 7

    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    for trace_idx, pick_idx in enumerate(final_pick.tolist()):
        traces[trace_idx, int(pick_idx)] = 10.0 + 0.01 * float(trace_idx)

    segy_path = _register_synthetic_segy(tmp_path, 'fine_gather_qc.sgy', traces)
    _write_coarse_with_n_samples(
        tmp_path / 'fine_gather_qc.coarse.npz',
        coarse_pick_i=robust.astype(np.int32),
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'fine_gather_qc.robust.npz',
        robust_pick_i=robust,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )

    def _fail_extract(*args, **kwargs) -> np.ndarray:
        raise AssertionError('full waveform overview extraction should not run')

    captured: dict[str, object] = {}

    def _fake_build_qc_info(*args, **kwargs) -> dict[str, object]:
        captured['build_qc_info'] = True
        return {'segy_obj': _DummySegy()}

    def _fake_save_qc(**kwargs: object) -> list[Path]:
        captured['save_qc'] = True
        return []

    monkeypatch.setattr(fine_infer_module, '_extract_raw_wave_hw', _fail_extract)
    monkeypatch.setattr(fine_infer_module, '_build_fine_qc_info', _fake_build_qc_info)
    monkeypatch.setattr(fine_infer_module, '_save_fine_gather_qc_pngs', _fake_save_qc)

    cfg = _make_fine_infer_config(
        tmp_path,
        segy_path=segy_path,
        robust_path=robust_path,
    )
    cfg['viewer'] = {
        'enabled': True,
        'save_overview_png': False,
        'save_gather_png': True,
    }

    result = run_fine_infer(
        model=_IdentityFineModel(),
        cfg=cfg,
        device=torch.device('cpu'),
    )

    validate_fbpick_final_payload(result, high_conf_threshold=0.5)
    assert captured == {'build_qc_info': True, 'save_qc': True}


def test_fine_infer_main_merges_ckpt_cfg_and_writes_final_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    n_traces = 160
    n_samples = 512
    dt_sec = 0.002
    trace_axis = np.arange(n_traces, dtype=np.int32)
    robust = 220 + ((trace_axis % 7) - 3) * 4

    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    for trace_idx, pick_idx in enumerate(robust.tolist()):
        traces[trace_idx, int(pick_idx)] = 5.0 + 0.01 * float(trace_idx)

    segy_path = _register_synthetic_segy(tmp_path, 'fine_cli.sgy', traces)
    _write_coarse_with_n_samples(
        tmp_path / 'fine_cli.coarse.npz',
        coarse_pick_i=robust.astype(np.int32),
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    robust_path = _write_robust_with_n_samples(
        tmp_path / 'fine_cli.robust.npz',
        robust_pick_i=robust,
        n_samples_orig=n_samples,
        dt_sec=dt_sec,
    )
    ckpt_path = _write_fine_ckpt_for_infer(tmp_path)
    _patch_synthetic_file_infos(
        monkeypatch,
        traces_by_path={segy_path: traces},
        dt_sec=dt_sec,
    )

    cfg = {
        'paths': {
            'segy_files': [segy_path],
            'robust_npz_files': [robust_path],
            'out_dir': str(tmp_path / 'fine_cli_out'),
        },
        'infer': {
            'ckpt_path': ckpt_path,
            'device': 'cpu',
            'batch_size': 2,
            'num_workers': 0,
            'overlap_h': 96,
            'amp': False,
            'use_tqdm': False,
            'high_conf_threshold': 0.5,
        },
    }
    cfg_path = tmp_path / 'config_infer_fbpick_fine.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')

    run_fine_infer_main(['--config', str(cfg_path)])

    out_path = tmp_path / 'fine_cli_out' / build_final_npz_name(segy_path)
    saved = load_fbpick_final_npz(out_path)
    validate_fbpick_final_payload(saved, high_conf_threshold=0.5)
    assert capsys.readouterr().out.strip() == str(out_path)


def test_run_fbpick_fine_infer_cli_forwards_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[str] | None] = {}

    def _fake_pipeline_main(argv: list[str] | None = None) -> None:
        captured['argv'] = argv

    monkeypatch.setattr(fine_infer_cli, 'pipeline_main', _fake_pipeline_main)

    fine_infer_cli.main(['--config', 'dummy.yaml'])

    assert captured['argv'] == ['--config', 'dummy.yaml']
