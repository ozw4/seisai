from __future__ import annotations

from typing import TYPE_CHECKING

from seisai_dataset import BuildPlan, SegyGatherPairDataset
from seisai_transforms.augment import RandomCropOrPad, ViewCompose

from seisai_engine.pipelines.common.augment import build_train_augment_ops
from seisai_engine.pipelines.common.validate_files import validate_files_exist

if TYPE_CHECKING:
    from .config import PairDatasetCfg, PairPaths

__all__ = [
    'build_infer_transform',
    'build_pair_dataset',
    'build_train_transform',
]


def build_train_transform(
    time_len: int,
    eps: float = 1e-8,
    augment_cfg: dict | None = None,
) -> ViewCompose:
    geom_ops, post_ops = build_train_augment_ops(augment_cfg)
    return ViewCompose(
        [
            *geom_ops,
            RandomCropOrPad(target_len=int(time_len)),
            *post_ops,
        ]
    )


def build_infer_transform(eps: float = 1e-8) -> ViewCompose:
    _ = eps
    return ViewCompose([])


def build_pair_dataset(
    paths: PairPaths,
    ds_cfg: PairDatasetCfg,
    transform: ViewCompose,
    plan: BuildPlan,
    subset_traces: int,
    secondary_key_fixed: bool,
    standardize_eps: float = 1e-8,
) -> SegyGatherPairDataset:
    validate_files_exist(list(paths.input_segy_files) + list(paths.target_segy_files))

    return SegyGatherPairDataset(
        input_segy_files=paths.input_segy_files,
        target_segy_files=paths.target_segy_files,
        transform=transform,
        plan=plan,
        subset_traces=int(subset_traces),
        primary_keys=ds_cfg.primary_keys,
        secondary_key_fixed=bool(secondary_key_fixed),
        waveform_mode=str(ds_cfg.waveform_mode),
        verbose=bool(ds_cfg.verbose),
        progress=bool(ds_cfg.progress),
        max_trials=int(ds_cfg.max_trials),
        use_header_cache=bool(ds_cfg.use_header_cache),
        standardize_from_input=True,
        standardize_eps=float(standardize_eps),
    )
