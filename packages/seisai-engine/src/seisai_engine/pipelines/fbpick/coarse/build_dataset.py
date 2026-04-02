from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from seisai_dataset import (
    BuildPlan,
    FirstBreakGate,
    FirstBreakGateConfig,
    InputOnlyPlan,
    SegyGatherPipelineDataset,
)
from seisai_transforms.augment import (
    DeterministicCropOrPad,
    PerTraceStandardize,
    RandomCropOrPad,
    ViewCompose,
)
from seisai_utils.config import (
    optional_bool,
    optional_float,
    optional_str,
    require_dict,
    require_int,
)
from seisai_utils.fs import validate_files_exist
from torch.utils.data import Dataset

from seisai_engine.pipelines.common.augment import build_train_augment_ops
from seisai_engine.pipelines.common.config_keys import raise_if_deprecated_time_len_keys
from seisai_engine.pipelines.common.noise_add import maybe_build_noise_add_op

from .config import CoarseInputCfg

__all__ = [
    'CoarseDataset',
    'build_dataset',
    'build_fbgate',
    'build_infer_transform',
    'build_train_transform',
]


def _raise_if_deprecated_time_len_keys(*, cfg: dict) -> None:
    raise_if_deprecated_time_len_keys(
        train_cfg=cfg.get('train'),
        transform_cfg=cfg.get('transform'),
    )


def _resolve_time_len(cfg: dict) -> tuple[dict, int]:
    _raise_if_deprecated_time_len_keys(cfg=cfg)
    transform_cfg = require_dict(cfg, 'transform')
    return transform_cfg, int(require_int(transform_cfg, 'time_len'))


def build_fbgate(fbgate_cfg: dict | None) -> FirstBreakGate:
    if fbgate_cfg is None:
        return FirstBreakGate(
            FirstBreakGateConfig(
                apply_on='off',
                min_pick_ratio=0.0,
                verbose=False,
            )
        )
    if not isinstance(fbgate_cfg, dict):
        msg = 'fbgate must be dict'
        raise TypeError(msg)

    apply_on = optional_str(fbgate_cfg, 'apply_on', 'off').lower()
    if apply_on == 'on':
        apply_on = 'any'
    if apply_on not in ('any', 'super_only', 'off'):
        msg = 'fbgate.apply_on must be "any", "super_only", or "off"'
        raise ValueError(msg)

    min_pick_ratio = optional_float(fbgate_cfg, 'min_pick_ratio', 0.0)
    verbose = optional_bool(fbgate_cfg, 'verbose', default=False)
    return FirstBreakGate(
        FirstBreakGateConfig(
            apply_on=apply_on,
            min_pick_ratio=float(min_pick_ratio),
            verbose=bool(verbose),
        )
    )


def build_train_transform(
    cfg: dict,
    *,
    noise_provider_ctx: dict[str, object] | None = None,
) -> ViewCompose:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    transform_cfg, time_len = _resolve_time_len(cfg)
    augment_cfg = cfg.get('augment')
    standardize_eps = optional_float(transform_cfg, 'standardize_eps', 1.0e-8)
    geom_ops, post_ops = build_train_augment_ops(augment_cfg)
    noise_op = None
    if isinstance(augment_cfg, dict) and augment_cfg.get('noise_add') is not None:
        if noise_provider_ctx is None:
            msg = 'noise_provider_ctx is required when augment.noise_add is set'
            raise ValueError(msg)
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

    ops: list = [
        *geom_ops,
        RandomCropOrPad(target_len=int(time_len)),
        *post_ops,
    ]
    if noise_op is not None:
        ops.append(noise_op)
    ops.append(PerTraceStandardize(eps=float(standardize_eps)))
    return ViewCompose(ops)


def build_infer_transform(cfg: dict) -> ViewCompose:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    transform_cfg, time_len = _resolve_time_len(cfg)
    standardize_eps = optional_float(transform_cfg, 'standardize_eps', 1.0e-8)
    return ViewCompose(
        [
            DeterministicCropOrPad(target_len=int(time_len)),
            PerTraceStandardize(eps=float(standardize_eps)),
        ]
    )


def _to_1d_torch(
    value: Any,
    *,
    name: str,
    dtype: torch.dtype,
    expected_len: int,
) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if int(tensor.ndim) != 1:
        msg = f'{name} must be 1D, got shape={tuple(tensor.shape)}'
        raise ValueError(msg)
    if int(tensor.shape[0]) != int(expected_len):
        msg = f'{name} length {int(tensor.shape[0])} != expected {int(expected_len)}'
        raise ValueError(msg)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _to_3d_input(value: Any, *, expected_c: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        msg = 'sample["input"] must be torch.Tensor'
        raise TypeError(msg)
    if int(value.ndim) != 3:
        msg = f'sample["input"] must have shape (C,H,W), got {tuple(value.shape)}'
        raise ValueError(msg)
    if int(value.shape[0]) != int(expected_c):
        msg = (
            f'sample["input"] channel dim {int(value.shape[0])} '
            f'!= expected {int(expected_c)}'
        )
        raise ValueError(msg)
    return value


def _to_3d_target(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        msg = 'sample["target"] must be torch.Tensor'
        raise TypeError(msg)
    if int(value.ndim) != 3:
        msg = f'sample["target"] must have shape (1,H,W), got {tuple(value.shape)}'
        raise ValueError(msg)
    if int(value.shape[0]) != 1:
        msg = f'sample["target"] channel dim must be 1, got {int(value.shape[0])}'
        raise ValueError(msg)
    return value


class CoarseDataset(Dataset):
    def __init__(
        self,
        base_dataset: SegyGatherPipelineDataset,
        *,
        input_cfg: CoarseInputCfg,
        require_target: bool,
    ) -> None:
        if not isinstance(base_dataset, SegyGatherPipelineDataset):
            msg = 'base_dataset must be SegyGatherPipelineDataset'
            raise TypeError(msg)
        if not isinstance(input_cfg, CoarseInputCfg):
            msg = 'input_cfg must be CoarseInputCfg'
            raise TypeError(msg)
        self._base_dataset = base_dataset
        self._input_cfg = input_cfg
        self._require_target = bool(require_target)

    def __len__(self) -> int:
        return len(self._base_dataset)

    @property
    def _rng(self) -> np.random.Generator:
        return self._base_dataset._rng

    @_rng.setter
    def _rng(self, value: np.random.Generator) -> None:
        self._base_dataset._rng = value

    def close(self) -> None:
        self._base_dataset.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_dataset, name)

    def _normalize_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(sample, dict):
            msg = f'dataset sample must be dict, got {type(sample).__name__}'
            raise TypeError(msg)
        if 'meta' not in sample or not isinstance(sample['meta'], dict):
            msg = 'dataset sample must contain dict meta'
            raise KeyError(msg)

        input_tensor = _to_3d_input(
            sample.get('input'),
            expected_c=len(self._input_cfg.stack_keys),
        )
        x_view = input_tensor[0]
        h = int(x_view.shape[0])
        w = int(x_view.shape[1])
        if h <= 0 or w <= 0:
            msg = f'x_view must have positive shape, got {(h, w)}'
            raise ValueError(msg)

        trace_valid = _to_1d_torch(
            sample.get('trace_valid'),
            name='trace_valid',
            dtype=torch.bool,
            expected_len=h,
        )

        meta = sample['meta']
        meta_trace_valid = _to_1d_torch(
            meta.get('trace_valid'),
            name='meta.trace_valid',
            dtype=torch.bool,
            expected_len=h,
        )
        if not torch.equal(trace_valid.cpu(), meta_trace_valid.cpu()):
            msg = 'trace_valid and meta.trace_valid must match exactly'
            raise ValueError(msg)

        offsets_view = _to_1d_torch(
            meta.get('offsets_view'),
            name='meta.offsets_view',
            dtype=torch.float32,
            expected_len=h,
        )
        fb_idx_view = _to_1d_torch(
            meta.get('fb_idx_view'),
            name='meta.fb_idx_view',
            dtype=torch.int64,
            expected_len=h,
        )
        time_view = _to_1d_torch(
            meta.get('time_view'),
            name='meta.time_view',
            dtype=torch.float32,
            expected_len=w,
        )

        if torch.any(fb_idx_view < -1):
            msg = 'fb_idx_view must contain only -1 or valid view indices'
            raise ValueError(msg)
        if torch.any(fb_idx_view == 0):
            msg = 'fb_idx_view must use -1 for invalid traces; value 0 is not allowed'
            raise ValueError(msg)
        if torch.any(fb_idx_view >= int(w)):
            msg = f'fb_idx_view must be < W={w}'
            raise ValueError(msg)
        if torch.any((~trace_valid) & (fb_idx_view > 0)):
            msg = 'fb_idx_view must be invalid (-1) on trace_valid=false rows'
            raise ValueError(msg)

        label_valid = trace_valid & (fb_idx_view > 0)

        sample['x_view'] = x_view
        sample['trace_valid'] = trace_valid
        sample['label_valid'] = label_valid
        sample['offsets_view'] = offsets_view
        sample['fb_idx_view'] = fb_idx_view
        sample['time_view'] = time_view

        if self._require_target:
            target = _to_3d_target(sample.get('target'))
            if int(target.shape[1]) != h or int(target.shape[2]) != w:
                msg = f'target shape {tuple(target.shape)} must match (1,{h},{w})'
                raise ValueError(msg)
        elif 'target' in sample:
            msg = 'input-only coarse dataset must not produce target'
            raise ValueError(msg)

        return sample

    def __getitem__(self, index: int | None = None) -> dict[str, Any]:
        sample = self._base_dataset[index]
        return self._normalize_sample(sample)


def build_dataset(
    *,
    segy_files: list[str],
    fb_files: list[str] | None,
    sampling_overrides: list[dict[str, object] | None] | None,
    transform: ViewCompose,
    fbgate: FirstBreakGate,
    plan: BuildPlan | InputOnlyPlan,
    subset_traces: int,
    trace_decimate_prob: float,
    trace_decimate_stride_range: tuple[int, int],
    primary_keys: Sequence[str],
    secondary_key_fixed: bool,
    verbose: bool,
    progress: bool,
    max_trials: int,
    use_header_cache: bool,
    waveform_mode: str,
    segy_endian: str,
    input_cfg: CoarseInputCfg,
    require_target: bool,
) -> CoarseDataset:
    if len(segy_files) == 0:
        msg = 'segy_files must be non-empty'
        raise ValueError(msg)
    if fb_files is None:
        if bool(require_target):
            msg = 'fb_files must be provided when require_target=True'
            raise ValueError(msg)
        msg = (
            'coarse build_dataset requires fb_files even when require_target=False '
            'to validate fb_idx_view; raw SEG-Y inference must use infer_segy2npz'
        )
        raise ValueError(msg)
    if len(fb_files) == 0:
        msg = 'fb_files must be non-empty'
        raise ValueError(msg)
    if len(segy_files) != len(fb_files):
        msg = 'segy_files and fb_files must have same length'
        raise ValueError(msg)
    if sampling_overrides is not None and len(sampling_overrides) != len(segy_files):
        msg = 'sampling_overrides length must match segy_files length'
        raise ValueError(msg)
    if not callable(transform):
        msg = 'transform must be callable'
        raise TypeError(msg)
    if not isinstance(fbgate, FirstBreakGate):
        msg = 'fbgate must be FirstBreakGate'
        raise TypeError(msg)
    if not isinstance(plan, (BuildPlan, InputOnlyPlan)):
        msg = 'plan must be BuildPlan or InputOnlyPlan'
        raise TypeError(msg)
    if bool(require_target):
        if not isinstance(plan, BuildPlan):
            msg = 'require_target=True requires BuildPlan'
            raise TypeError(msg)
    else:
        if not isinstance(plan, InputOnlyPlan):
            msg = 'require_target=False requires InputOnlyPlan'
            raise TypeError(msg)

    validate_files_exist(list(segy_files) + list(fb_files))
    base_dataset = SegyGatherPipelineDataset(
        segy_files=list(segy_files),
        fb_files=list(fb_files),
        sampling_overrides=sampling_overrides,
        transform=transform,
        fbgate=fbgate,
        plan=plan,
        subset_traces=int(subset_traces),
        trace_decimate_prob=float(trace_decimate_prob),
        trace_decimate_stride_range=tuple(trace_decimate_stride_range),
        primary_keys=tuple(primary_keys),
        secondary_key_fixed=bool(secondary_key_fixed),
        waveform_mode=str(waveform_mode),
        segy_endian=str(segy_endian),
        verbose=bool(verbose),
        progress=bool(progress),
        max_trials=int(max_trials),
        use_header_cache=bool(use_header_cache),
    )
    return CoarseDataset(
        base_dataset,
        input_cfg=input_cfg,
        require_target=bool(require_target),
    )
