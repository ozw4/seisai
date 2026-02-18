from __future__ import annotations

import pytest
import torch

from seisai_engine.loss import composite
from seisai_engine.pipelines.pair.config import load_pair_train_config
from seisai_engine.pipelines.psn.config import load_psn_train_config
from seisai_engine.pipelines.psn.loss import build_psn_criterion


def _make_pair_cfg() -> dict:
    return {
        'paths': {
            'input_segy_files': ['in.sgy'],
            'target_segy_files': ['tg.sgy'],
            'infer_input_segy_files': ['in.sgy'],
            'infer_target_segy_files': ['tg.sgy'],
            'out_dir': './out',
        },
        'dataset': {
            'max_trials': 8,
            'use_header_cache': True,
            'verbose': False,
            'primary_keys': ['ffid'],
            'secondary_key_fixed': False,
        },
        'train': {
            'seed': 42,
            'batch_size': 1,
            'epochs': 1,
            'samples_per_epoch': 1,
            'num_workers': 0,
            'use_amp': False,
            'max_norm': 1.0,
            'lr': 1.0e-4,
            'subset_traces': 8,
            'loss_scope': 'all',
            'losses': [
                {
                    'kind': 'l1',
                    'weight': 1.0,
                    'scope': 'all',
                    'params': {},
                }
            ],
        },
        'transform': {
            'time_len': 32,
        },
        'infer': {
            'seed': 43,
            'batch_size': 1,
            'max_batches': 1,
            'subset_traces': 8,
            'num_workers': 0,
        },
        'tile': {
            'tile_h': 8,
            'overlap_h': 4,
            'tiles_per_batch': 1,
            'amp': False,
            'use_tqdm': False,
        },
        'vis': {
            'out_subdir': 'vis',
            'n': 1,
            'cmap': 'seismic',
            'vmin': -3.0,
            'vmax': 3.0,
            'transpose_for_trace_time': True,
            'per_trace_norm': True,
            'per_trace_eps': 1.0e-8,
            'figsize': [8.0, 6.0],
            'dpi': 120,
        },
        'ckpt': {
            'save_best_only': True,
            'metric': 'infer_loss',
            'mode': 'min',
        },
        'model': {
            'backbone': 'resnet18',
            'in_chans': 1,
            'out_chans': 1,
        },
    }


def _make_psn_cfg() -> dict:
    return {
        'paths': {
            'out_dir': './out',
        },
        'train': {
            'seed': 42,
            'batch_size': 1,
            'epochs': 1,
            'samples_per_epoch': 1,
            'num_workers': 0,
            'use_amp': False,
            'max_norm': 1.0,
            'lr': 1.0e-4,
            'subset_traces': 8,
            'loss_scope': 'all',
            'losses': [
                {
                    'kind': 'soft_label_ce',
                    'weight': 1.0,
                    'scope': 'all',
                    'params': {},
                }
            ],
        },
        'infer': {
            'seed': 43,
            'batch_size': 1,
            'max_batches': 1,
            'subset_traces': 8,
            'num_workers': 0,
        },
        'vis': {
            'out_subdir': 'vis',
            'n': 1,
        },
        'ckpt': {
            'save_best_only': True,
            'metric': 'infer_loss',
            'mode': 'min',
        },
        'model': {
            'backbone': 'resnet18',
            'in_chans': 1,
            'out_chans': 3,
        },
    }


def _make_psn_batch(
    *,
    logits: torch.Tensor,
    target: torch.Tensor,
    with_mask: bool,
    mask_fill: bool = True,
) -> dict:
    batch: dict = {
        'target': target,
        'trace_valid': torch.ones(
            (int(logits.shape[0]), int(logits.shape[-2])), dtype=torch.bool
        ),
        'label_valid': torch.ones(
            (int(logits.shape[0]), int(logits.shape[-2])), dtype=torch.bool
        ),
    }
    if with_mask:
        batch['mask_bool'] = torch.full(
            (int(logits.shape[0]), int(logits.shape[-2]), int(logits.shape[-1])),
            fill_value=bool(mask_fill),
            dtype=torch.bool,
        )
    return batch


def test_pair_rejects_deprecated_train_loss_kind_key() -> None:
    cfg = _make_pair_cfg()
    cfg['train']['loss_kind'] = 'l1'
    with pytest.raises(
        ValueError,
        match='deprecated key: train.loss_kind; use train.losses',
    ):
        load_pair_train_config(cfg)


def test_pair_accepts_train_losses_and_builds_criterion() -> None:
    cfg = _make_pair_cfg()
    typed = load_pair_train_config(cfg)

    assert len(typed.loss_specs_train) == 1
    assert typed.loss_specs_eval == typed.loss_specs_train

    criterion = composite.build_weighted_criterion(list(typed.loss_specs_train))
    pred = torch.randn((1, 1, 2, 4), dtype=torch.float32)
    target = torch.randn_like(pred)
    loss = criterion(pred, target, {})

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_psn_requires_train_losses() -> None:
    cfg = _make_psn_cfg()
    del cfg['train']['losses']
    with pytest.raises(ValueError, match='train.losses is required'):
        load_psn_train_config(cfg)


def test_psn_rejects_empty_train_losses() -> None:
    cfg = _make_psn_cfg()
    cfg['train']['losses'] = []
    with pytest.raises(ValueError, match='train.losses must be non-empty'):
        load_psn_train_config(cfg)


@pytest.mark.parametrize(
    ('kind', 'params'),
    [
        ('soft_label_ce', {}),
        ('prob_l1', {}),
        ('prob_mse', {}),
        ('prob_huber', {'huber_delta': 0.5}),
    ],
)
def test_psn_build_criterion_supported_kinds(kind: str, params: dict) -> None:
    specs = composite.parse_loss_specs(
        [
            {
                'kind': kind,
                'weight': 1.0,
                'scope': 'all',
                'params': params,
            }
        ],
        default_scope='all',
        label='train.losses',
        scope_label='train.loss_scope',
    )
    criterion = build_psn_criterion(specs)

    logits = torch.randn((2, 3, 4, 5), dtype=torch.float32)
    target = torch.softmax(torch.randn_like(logits), dim=1)
    batch = _make_psn_batch(logits=logits, target=target, with_mask=False)
    loss = criterion(logits, target, batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_psn_scope_all_ignores_mask_bool() -> None:
    specs = composite.parse_loss_specs(
        [
            {
                'kind': 'soft_label_ce',
                'weight': 1.0,
                'scope': 'all',
                'params': {},
            }
        ],
        default_scope='all',
        label='train.losses',
        scope_label='train.loss_scope',
    )
    criterion = build_psn_criterion(specs)

    logits = torch.randn((2, 3, 4, 5), dtype=torch.float32)
    target = torch.softmax(torch.randn_like(logits), dim=1)

    batch_no_mask = _make_psn_batch(logits=logits, target=target, with_mask=False)
    batch_zero_mask = _make_psn_batch(
        logits=logits,
        target=target,
        with_mask=True,
        mask_fill=False,
    )

    loss_no_mask = criterion(logits, target, batch_no_mask)
    loss_zero_mask = criterion(logits, target, batch_zero_mask)

    assert torch.allclose(loss_no_mask, loss_zero_mask)


def test_psn_scope_masked_only_applies_optional_mask_bool() -> None:
    specs = composite.parse_loss_specs(
        [
            {
                'kind': 'soft_label_ce',
                'weight': 1.0,
                'scope': 'masked_only',
                'params': {},
            }
        ],
        default_scope='all',
        label='train.losses',
        scope_label='train.loss_scope',
    )
    criterion = build_psn_criterion(specs)

    logits = torch.randn((2, 3, 4, 5), dtype=torch.float32)
    target = torch.softmax(torch.randn_like(logits), dim=1)

    batch_no_mask = _make_psn_batch(logits=logits, target=target, with_mask=False)
    batch_zero_mask = _make_psn_batch(
        logits=logits,
        target=target,
        with_mask=True,
        mask_fill=False,
    )

    loss_no_mask = criterion(logits, target, batch_no_mask)
    loss_zero_mask = criterion(logits, target, batch_zero_mask)

    assert loss_no_mask.ndim == 0
    assert torch.isfinite(loss_no_mask)
    assert torch.allclose(loss_zero_mask, torch.tensor(0.0, dtype=loss_zero_mask.dtype))
