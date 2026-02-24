from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from seisai_engine.pipelines.common.train_skeleton import (
    TrainSkeletonSpec,
    build_ckpt_payload,
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


def _make_spec(*, ckpt_extra: dict[str, Any] | None) -> TrainSkeletonSpec:
    model = torch.nn.Conv2d(1, 3, kernel_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return TrainSkeletonSpec(
        pipeline='psn',
        cfg={'train': {}, 'infer': {}},
        base_dir=Path('/tmp'),
        out_dir=Path('/tmp'),
        vis_subdir='vis',
        model_sig={'backbone': 'resnet18', 'in_chans': 1, 'out_chans': 3},
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
        vis_n=1,
        infer_epoch_fn=_dummy_infer_epoch_fn,
        ckpt_extra=ckpt_extra,
        print_freq=10,
    )


def test_build_ckpt_payload_merges_ckpt_extra() -> None:
    spec = _make_spec(
        ckpt_extra={
            'output_ids': ['P', 'S', 'N'],
            'softmax_axis': 'channel',
        }
    )
    payload = build_ckpt_payload(
        spec=spec,
        epoch=3,
        global_step=9,
        scheduler_sig=None,
        scheduler_state_dict=None,
    )
    assert payload['version'] == 1
    assert payload['pipeline'] == 'psn'
    assert payload['epoch'] == 3
    assert payload['global_step'] == 9
    assert payload['output_ids'] == ['P', 'S', 'N']
    assert payload['softmax_axis'] == 'channel'


def test_build_ckpt_payload_rejects_collision() -> None:
    spec = _make_spec(ckpt_extra={'version': 2})
    with pytest.raises(ValueError):
        build_ckpt_payload(
            spec=spec,
            epoch=0,
            global_step=0,
            scheduler_sig=None,
            scheduler_state_dict=None,
        )


def test_build_ckpt_payload_rejects_non_str_ckpt_extra_key() -> None:
    spec = _make_spec(ckpt_extra={1: 'x'})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_ckpt_payload(
            spec=spec,
            epoch=0,
            global_step=0,
            scheduler_sig=None,
            scheduler_state_dict=None,
        )
