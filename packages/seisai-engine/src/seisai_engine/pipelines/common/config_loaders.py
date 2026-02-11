from __future__ import annotations

from seisai_utils.config import (
    optional_value,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_value,
)

from .config_schema import (
    CommonTrainConfig,
    InferLoopConfig,
    OutputConfig,
    SeedsConfig,
    TrainLoopConfig,
)

__all__ = ['load_common_train_config']


def load_common_train_config(cfg: dict) -> CommonTrainConfig:
    paths = require_dict(cfg, 'paths')
    train_cfg = require_dict(cfg, 'train')
    infer_cfg = require_dict(cfg, 'infer')
    vis_cfg = require_dict(cfg, 'vis')

    out_dir = require_value(
        paths,
        'out_dir',
        str,
        type_message='config.paths.out_dir must be str',
    )
    vis_subdir = require_value(
        vis_cfg,
        'out_subdir',
        str,
        type_message='config.vis.out_subdir must be str',
    )

    seed_train = require_int(train_cfg, 'seed')
    seed_infer = require_int(infer_cfg, 'seed')

    print_freq = 10
    if 'print_freq' in train_cfg:
        print_freq = require_int(train_cfg, 'print_freq')

    gradient_accumulation_steps = optional_value(
        train_cfg,
        'gradient_accumulation_steps',
        1,
        int,
        type_message='config.train.gradient_accumulation_steps must be int',
        coerce=int,
        coerce_default=True,
    )

    train = TrainLoopConfig(
        epochs=int(require_int(train_cfg, 'epochs')),
        samples_per_epoch=int(require_int(train_cfg, 'samples_per_epoch')),
        train_batch_size=int(require_int(train_cfg, 'batch_size')),
        train_num_workers=int(require_int(train_cfg, 'num_workers')),
        max_norm=float(require_float(train_cfg, 'max_norm')),
        use_amp_train=bool(require_bool(train_cfg, 'use_amp')),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        print_freq=int(print_freq),
    )
    infer = InferLoopConfig(
        infer_batch_size=int(require_int(infer_cfg, 'batch_size')),
        infer_num_workers=int(require_int(infer_cfg, 'num_workers')),
        infer_max_batches=int(require_int(infer_cfg, 'max_batches')),
        vis_n=int(require_int(vis_cfg, 'n')),
    )
    output = OutputConfig(out_dir=out_dir, vis_subdir=str(vis_subdir))
    seeds = SeedsConfig(seed_train=int(seed_train), seed_infer=int(seed_infer))

    return CommonTrainConfig(output=output, seeds=seeds, train=train, infer=infer)
