from __future__ import annotations

from seisai_dataset import (
    FirstBreakGate,
    FirstBreakGateConfig,
    SegyGatherPipelineDataset,
)
from seisai_transforms.augment import (
    DeterministicCropOrPad,
    PerTraceStandardize,
    RandomCropOrPad,
    ViewCompose,
)

from seisai_engine.pipelines.common.augment import build_train_augment_ops
from seisai_engine.pipelines.common.validate_files import validate_files_exist

__all__ = [
    'build_dataset',
    'build_fbgate',
    'build_infer_transform',
    'build_train_transform',
]


def build_train_transform(
    *,
    time_len: int,
    per_trace_standardize: bool,
    augment_cfg: dict | None = None,
) -> ViewCompose:
    geom_ops, post_ops = build_train_augment_ops(augment_cfg)
    ops: list = [
        *geom_ops,
        RandomCropOrPad(target_len=int(time_len)),
        *post_ops,
    ]
    if per_trace_standardize:
        ops.append(PerTraceStandardize(eps=1e-8))
    return ViewCompose(ops)


def build_infer_transform(*, time_len: int, per_trace_standardize: bool) -> ViewCompose:
    ops: list = [DeterministicCropOrPad(target_len=int(time_len))]
    if per_trace_standardize:
        ops.append(PerTraceStandardize(eps=1e-8))
    return ViewCompose(ops)


def build_fbgate(
    *, apply_on: str, min_pick_ratio: float, verbose: bool
) -> FirstBreakGate:
    ap = str(apply_on).lower()
    if ap == 'on':
        ap = 'any'
    if ap not in ('any', 'super_only', 'off'):
        msg = 'fbgate.apply_on must be "any", "super_only", or "off"'
        raise ValueError(msg)

    cfg = FirstBreakGateConfig(
        apply_on=ap,
        min_pick_ratio=float(min_pick_ratio),
        verbose=bool(verbose),
    )
    return FirstBreakGate(cfg)


def build_dataset(
    *,
    segy_files: list[str],
    fb_files: list[str] | None,
    transform: ViewCompose,
    fbgate: FirstBreakGate,
    plan,
    subset_traces: int,
    primary_keys: tuple[str, ...],
    secondary_key_fixed: bool,
    verbose: bool,
    progress: bool,
    max_trials: int,
    use_header_cache: bool,
) -> SegyGatherPipelineDataset:
    if fb_files is None:
        validate_files_exist(list(segy_files))
    else:
        validate_files_exist(list(segy_files) + list(fb_files))
    return SegyGatherPipelineDataset(
        segy_files=segy_files,
        fb_files=fb_files,
        transform=transform,
        fbgate=fbgate,
        plan=plan,
        subset_traces=int(subset_traces),
        primary_keys=primary_keys,
        secondary_key_fixed=bool(secondary_key_fixed),
        verbose=bool(verbose),
        progress=bool(progress),
        max_trials=int(max_trials),
        use_header_cache=bool(use_header_cache),
    )
