from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from seisai_engine.pipelines.common.train_skeleton import (
    TrainSkeletonSpec,
    build_ckpt_payload,
)
from seisai_engine.pipelines.fbpick.coarse import (
    COARSE_IN_CHANS,
    COARSE_INPUT_MODE_GLOBAL_ANCHOR_RESIZE,
    COARSE_TIME_LEN,
    COARSE_TRACE_LEN,
    CoarseTrainBundle,
    build_coarse_ckpt_extra,
    build_train_spec,
    load_coarse_infer_config,
    load_coarse_train_config,
    validate_coarse_checkpoint_metadata,
)


class _DummyDataset:
    def close(self) -> None:
        return


def _dummy_infer_epoch_fn(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    vis_epoch_dir: Path,
    vis_n: int,
    max_batches: int,
) -> float:
    _ = (model, loader, device, vis_epoch_dir, vis_n, max_batches)
    return 0.0


def _make_train_cfg() -> dict[str, Any]:
    return {
        'coarse': {'input_mode': 'global_anchor_resize'},
        'paths': {
            'segy_files': ['dummy.sgy'],
            'fb_files': ['dummy.npy'],
            'out_dir': './out',
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
            'trace_len': 256,
            'time_len': 2048,
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
        'vis': {'n': 0, 'out_subdir': 'vis'},
        'ckpt': {'save_best_only': True, 'metric': 'infer_loss', 'mode': 'min'},
        'model': {
            'backbone': 'resnet18',
            'pretrained': False,
            'in_chans': 3,
            'out_chans': 1,
        },
    }


def _make_spec(*, ckpt_extra: dict[str, Any]) -> TrainSkeletonSpec:
    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return TrainSkeletonSpec(
        pipeline='fbpick',
        cfg=_make_train_cfg(),
        base_dir=Path('/tmp'),
        out_dir=Path('/tmp'),
        vis_subdir='vis',
        model_sig={'backbone': 'resnet18', 'in_chans': 3, 'out_chans': 1},
        model=model,
        optimizer=optimizer,
        criterion=lambda *args, **kwargs: None,
        ds_train_full=_DummyDataset(),
        ds_infer_full=_DummyDataset(),
        device=torch.device('cpu'),
        seed_train=1,
        seed_infer=2,
        epochs=1,
        train_batch_size=1,
        train_num_workers=0,
        samples_per_epoch=1,
        max_norm=1.0,
        use_amp_train=False,
        gradient_accumulation_steps=1,
        infer_batch_size=1,
        infer_num_workers=0,
        infer_max_batches=1,
        vis_n=0,
        infer_epoch_fn=_dummy_infer_epoch_fn,
        ckpt_extra=ckpt_extra,
        print_freq=10,
    )


def _make_bundle(cfg: dict[str, Any]) -> CoarseTrainBundle:
    typed = load_coarse_train_config(cfg)
    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return CoarseTrainBundle(
        cfg=cfg,
        base_dir=Path('/tmp'),
        typed=typed,
        out_dir=Path('/tmp/out'),
        model_sig=typed.model_sig,
        model=model,
        optimizer=optimizer,
        criterion=lambda *args, **kwargs: None,
        ds_train_full=_DummyDataset(),
        ds_infer_full=_DummyDataset(),
        device=torch.device('cpu'),
    )


def test_load_coarse_train_config_global_anchor_contract() -> None:
    typed = load_coarse_train_config(_make_train_cfg())

    assert typed.coarse.input_mode == COARSE_INPUT_MODE_GLOBAL_ANCHOR_RESIZE
    assert typed.transform.trace_len == COARSE_TRACE_LEN
    assert typed.transform.time_len == COARSE_TIME_LEN
    assert typed.model_sig['in_chans'] == COARSE_IN_CHANS


def test_load_coarse_infer_config_global_anchor_contract() -> None:
    cfg = _make_train_cfg()
    cfg['paths'].pop('fb_files')
    cfg['infer']['ckpt_path'] = 'best.pt'

    typed = load_coarse_infer_config(cfg)

    assert typed.coarse.input_mode == 'global_anchor_resize'
    assert typed.transform.trace_len == 256
    assert typed.transform.time_len == 2048


@pytest.mark.parametrize(
    ('mutate', 'message'),
    [
        (
            lambda cfg: cfg['coarse'].__setitem__('input_mode', 'legacy_tiled'),
            'coarse.input_mode must be',
        ),
        (
            lambda cfg: cfg['transform'].__setitem__('trace_len', 128),
            'transform.trace_len must be 256',
        ),
        (
            lambda cfg: cfg['transform'].__setitem__('time_len', 6016),
            'transform.time_len must be 2048',
        ),
        (
            lambda cfg: cfg['model'].__setitem__('in_chans', 1),
            'model.in_chans must be 3',
        ),
    ],
)
def test_load_coarse_train_config_rejects_contract_mismatch(
    mutate,
    message: str,
) -> None:
    cfg = _make_train_cfg()
    mutate(cfg)

    with pytest.raises(ValueError, match=message):
        load_coarse_train_config(cfg)


def test_coarse_ckpt_extra_is_saved_in_train_skeleton_payload() -> None:
    payload = build_ckpt_payload(
        spec=_make_spec(ckpt_extra=build_coarse_ckpt_extra()),
        epoch=3,
        global_step=9,
        scheduler_sig=None,
        scheduler_state_dict=None,
    )

    assert payload['coarse_input_mode'] == 'global_anchor_resize'
    assert payload['coarse_trace_len'] == 256
    assert payload['coarse_time_len'] == 2048
    assert payload['coarse_in_chans'] == 3
    assert payload['coarse_input_channels'] == ['waveform', 'offset_ch', 'time_ch']
    assert payload['output_ids'] == ['P']
    assert payload['softmax_axis'] == 'time'


def test_build_train_spec_attaches_coarse_ckpt_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import seisai_engine.pipelines.fbpick.coarse.train as train_module

    cfg = _make_train_cfg()
    monkeypatch.setattr(
        train_module,
        'build_train_bundle',
        lambda cfg, *, base_dir, device: _make_bundle(cfg),
    )

    spec = build_train_spec(
        cfg,
        base_dir=Path('/tmp'),
        device=torch.device('cpu'),
    )

    assert spec.ckpt_extra is not None
    assert spec.ckpt_extra['coarse_input_mode'] == 'global_anchor_resize'
    assert spec.ckpt_extra['coarse_trace_len'] == 256
    assert spec.ckpt_extra['coarse_time_len'] == 2048
    assert spec.ckpt_extra['coarse_in_chans'] == 3


def test_validate_coarse_checkpoint_metadata_accepts_new_contract() -> None:
    validate_coarse_checkpoint_metadata(
        {
            'pipeline': 'fbpick',
            'coarse_input_mode': 'global_anchor_resize',
            'coarse_trace_len': 256,
            'coarse_time_len': 2048,
            'coarse_in_chans': 3,
        }
    )


def test_validate_coarse_checkpoint_metadata_rejects_legacy_missing_metadata() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "expected coarse_input_mode='global_anchor_resize', got None. "
            'This checkpoint appears to be from the legacy tiled coarse pipeline.'
        ),
    ):
        validate_coarse_checkpoint_metadata({'pipeline': 'fbpick'})


def test_validate_coarse_checkpoint_metadata_includes_expected_and_actual() -> None:
    with pytest.raises(
        ValueError,
        match='expected coarse_time_len=2048, got 6016',
    ):
        validate_coarse_checkpoint_metadata(
            {
                'pipeline': 'fbpick',
                'coarse_input_mode': 'global_anchor_resize',
                'coarse_trace_len': 256,
                'coarse_time_len': 6016,
                'coarse_in_chans': 3,
            }
        )
