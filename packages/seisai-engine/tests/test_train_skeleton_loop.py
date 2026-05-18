from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from seisai_engine.pipelines.common import train_skeleton_loop
from seisai_engine.pipelines.common.checkpoint_io import load_checkpoint
from seisai_engine.pipelines.common.train_skeleton import TrainSkeletonSpec
from seisai_engine.pipelines.common.train_skeleton_tracking import TrackingRunState
from seisai_engine.tracking.config import TrackingConfig


class _DummyDataset:
    def __init__(self, size: int = 4) -> None:
        self.size = int(size)
        self._rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        _ = idx
        return torch.zeros(1)

    def close(self) -> None:
        return


def _make_spec(
    *,
    tmp_path: Path,
    ckpt_metric: str,
    ckpt_mode: str,
) -> TrainSkeletonSpec:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def infer_epoch_fn(
        model: torch.nn.Module,
        loader: Any,
        device: torch.device,
        vis_epoch_dir: Path,
        vis_n: int,
        max_batches: int,
    ) -> float:
        _ = (model, loader, device, vis_epoch_dir, vis_n, max_batches)
        return 100.0

    return TrainSkeletonSpec(
        pipeline='fbpick',
        cfg={},
        base_dir=tmp_path,
        out_dir=tmp_path / 'out',
        vis_subdir='vis',
        model_sig={'backbone': 'linear', 'in_chans': 1, 'out_chans': 1},
        model=model,
        optimizer=optimizer,
        criterion=lambda *args, **kwargs: None,
        ds_train_full=_DummyDataset(),
        ds_infer_full=_DummyDataset(),
        device=torch.device('cpu'),
        seed_train=1,
        seed_infer=2,
        epochs=2,
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
        infer_epoch_fn=infer_epoch_fn,
        ckpt_metric=ckpt_metric,
        ckpt_mode=ckpt_mode,
        print_freq=10,
    )


def test_run_training_loop_last_metric_saves_last_epoch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train_one_epoch(*args, **kwargs) -> dict[str, float]:
        _ = (args, kwargs)
        return {'loss': 1.0, 'steps': 1.0, 'samples': 1.0}

    monkeypatch.setattr(
        train_skeleton_loop,
        'train_one_epoch',
        fake_train_one_epoch,
    )
    ckpt_dir = tmp_path / 'ckpt'
    vis_root = tmp_path / 'vis'
    ckpt_dir.mkdir()
    vis_root.mkdir()
    tracking_state = TrackingRunState(
        tracking_cfg=TrackingConfig(),
        tracker=object(),  # type: ignore[arg-type]
        tracking_enabled=False,
        run_started=False,
        run_name='disabled',
    )

    stats = train_skeleton_loop.run_training_loop(
        spec=_make_spec(tmp_path=tmp_path, ckpt_metric='last', ckpt_mode='max'),
        ckpt_dir=ckpt_dir,
        vis_root=vis_root,
        ema_cfg_obj=None,
        ema_controller=None,
        tracking_state=tracking_state,
    )

    ckpt = load_checkpoint(ckpt_dir / 'best.pt')
    assert ckpt['epoch'] == 1
    assert stats.best_epoch == 1
    assert stats.best_ckpt_value == 1.0
