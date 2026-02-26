from __future__ import annotations

from typing import TYPE_CHECKING

from seisai_dataset import BuildPlan, SegyGatherPairDataset
from seisai_transforms.augment import RandomCropOrPad, ViewCompose
from seisai_utils.fs import validate_files_exist

from seisai_engine.pipelines.common.augment import build_train_augment_ops
from seisai_engine.pipelines.common.noise_add import maybe_build_noise_add_op

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
    noise_provider_ctx: dict[str, object] | None = None,
) -> tuple[ViewCompose, ViewCompose]:
    _ = eps
    geom_ops, post_ops = build_train_augment_ops(augment_cfg)
    shared_ops = [
        *geom_ops,
        RandomCropOrPad(target_len=int(time_len)),
        *post_ops,
    ]

    if (
        noise_provider_ctx is None
        and isinstance(augment_cfg, dict)
        and augment_cfg.get('noise_add') is not None
    ):
        msg = 'noise_provider_ctx is required when augment.noise_add is set'
        raise ValueError(msg)

    noise_op = None
    if noise_provider_ctx is not None:
        noise_ctx = dict(noise_provider_ctx)
        noise_op = maybe_build_noise_add_op(
            augment_cfg=augment_cfg,
            subset_traces=int(noise_ctx['subset_traces']),
            primary_keys=tuple(noise_ctx['primary_keys']),
            secondary_key_fixed=bool(noise_ctx['secondary_key_fixed']),
            waveform_mode=str(noise_ctx['waveform_mode']),
            segy_endian=str(noise_ctx['segy_endian']),
            header_cache_dir=noise_ctx['header_cache_dir'],
            use_header_cache=bool(noise_ctx['use_header_cache']),
        )

    target_transform = ViewCompose(list(shared_ops))
    if noise_op is None:
        input_transform = ViewCompose(list(shared_ops))
    else:
        input_transform = ViewCompose([*shared_ops, noise_op])
    return input_transform, target_transform


def build_infer_transform(eps: float = 1e-8) -> ViewCompose:
    _ = eps
    return ViewCompose([])


def build_pair_dataset(
    paths: PairPaths,
    ds_cfg: PairDatasetCfg,
    input_transform: ViewCompose,
    target_transform: ViewCompose,
    plan: BuildPlan,
    subset_traces: int,
    secondary_key_fixed: bool,
    input_segy_endian: str,
    target_segy_endian: str,
    standardize_eps: float = 1e-8,
) -> SegyGatherPairDataset:
    validate_files_exist(list(paths.input_segy_files) + list(paths.target_segy_files))

    return SegyGatherPairDataset(
        input_segy_files=paths.input_segy_files,
        target_segy_files=paths.target_segy_files,
        input_transform=input_transform,
        target_transform=target_transform,
        plan=plan,
        subset_traces=int(subset_traces),
        primary_keys=ds_cfg.primary_keys,
        secondary_key_fixed=bool(secondary_key_fixed),
        waveform_mode=str(ds_cfg.waveform_mode),
        input_segy_endian=str(input_segy_endian),
        target_segy_endian=str(target_segy_endian),
        verbose=bool(ds_cfg.verbose),
        progress=bool(ds_cfg.progress),
        max_trials=int(ds_cfg.max_trials),
        use_header_cache=bool(ds_cfg.use_header_cache),
        standardize_from_input=True,
        standardize_eps=float(standardize_eps),
    )
