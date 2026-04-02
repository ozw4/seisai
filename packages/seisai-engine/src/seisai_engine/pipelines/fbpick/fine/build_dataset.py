from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
from seisai_dataset import BuildPlan
from seisai_dataset.local_window_dataset import (
    LocalWindowDataset,
    LocalWindowDatasetConfig,
)
from seisai_utils.fs import validate_files_exist
from torch.utils.data import Dataset

from .config import FineInputCfg, FineWindowCfg

__all__ = [
    'FineLocalDataset',
    'build_dataset',
]


def _require_input_cfg(cfg: FineInputCfg) -> FineInputCfg:
    if not isinstance(cfg, FineInputCfg):
        msg = 'input_cfg must be FineInputCfg'
        raise TypeError(msg)
    return cfg


def _require_window_cfg(cfg: FineWindowCfg) -> FineWindowCfg:
    if not isinstance(cfg, FineWindowCfg):
        msg = 'window_cfg must be FineWindowCfg'
        raise TypeError(msg)
    return cfg


def _to_1d_torch(
    value: Any,
    *,
    name: str,
    dtype: torch.dtype,
    expected_len: int,
) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if int(tensor.ndim) == 0:
        tensor = tensor.reshape(1)
    if int(tensor.ndim) != 1:
        msg = f'{name} must be 1D, got shape={tuple(tensor.shape)}'
        raise ValueError(msg)
    if int(tensor.shape[0]) != int(expected_len):
        msg = f'{name} length {int(tensor.shape[0])} != expected {int(expected_len)}'
        raise ValueError(msg)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _to_2d_float(
    value: Any,
    *,
    name: str,
    expected_h: int,
    expected_w: int | None = None,
) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if int(tensor.ndim) != 2:
        msg = f'{name} must have shape (H,W), got {tuple(tensor.shape)}'
        raise ValueError(msg)
    if int(tensor.shape[0]) != int(expected_h):
        msg = f'{name} height {int(tensor.shape[0])} != expected {int(expected_h)}'
        raise ValueError(msg)
    if expected_w is not None and int(tensor.shape[1]) != int(expected_w):
        msg = f'{name} width {int(tensor.shape[1])} != expected {int(expected_w)}'
        raise ValueError(msg)
    if tensor.dtype != torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


def _to_3d_float(
    value: Any,
    *,
    name: str,
    expected_c: int,
    expected_h: int,
    expected_w: int | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        msg = f'{name} must be torch.Tensor'
        raise TypeError(msg)
    if int(value.ndim) != 3:
        msg = f'{name} must have shape (C,H,W), got {tuple(value.shape)}'
        raise ValueError(msg)
    if int(value.shape[0]) != int(expected_c):
        msg = f'{name} channels {int(value.shape[0])} != expected {int(expected_c)}'
        raise ValueError(msg)
    if int(value.shape[1]) != int(expected_h):
        msg = f'{name} height {int(value.shape[1])} != expected {int(expected_h)}'
        raise ValueError(msg)
    if expected_w is not None and int(value.shape[2]) != int(expected_w):
        msg = f'{name} width {int(value.shape[2])} != expected {int(expected_w)}'
        raise ValueError(msg)
    if value.dtype != torch.float32:
        value = value.to(dtype=torch.float32)
    return value


class FineLocalDataset(Dataset):
    def __init__(
        self,
        base_dataset: LocalWindowDataset,
        *,
        input_cfg: FineInputCfg,
        require_target: bool,
        expected_mode: Literal['train', 'eval', 'infer'],
        expected_seed_source: Literal['gt', 'coarse'],
    ) -> None:
        if not isinstance(base_dataset, LocalWindowDataset):
            msg = 'base_dataset must be LocalWindowDataset'
            raise TypeError(msg)
        self._base_dataset = base_dataset
        self._input_cfg = _require_input_cfg(input_cfg)
        self._require_target = bool(require_target)
        self._expected_mode = str(expected_mode)
        self._expected_seed_source = str(expected_seed_source)

    def __len__(self) -> int:
        return len(self._base_dataset)

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

        meta = dict(sample['meta'])

        x_view_local = _to_2d_float(
            sample.get('x_view_local'),
            name='x_view_local',
            expected_h=1,
        )
        w_local = int(x_view_local.shape[1])

        amplitude_value = sample.get(self._input_cfg.amplitude_key)
        if amplitude_value is None:
            msg = f'plan must set sample[{self._input_cfg.amplitude_key!r}]'
            raise KeyError(msg)
        amplitude = _to_2d_float(
            amplitude_value,
            name=self._input_cfg.amplitude_key,
            expected_h=1,
            expected_w=w_local,
        )
        if not torch.equal(amplitude, x_view_local):
            msg = 'fine amplitude input must match x_view_local exactly in Phase 5'
            raise ValueError(msg)

        input_tensor = _to_3d_float(
            sample.get('input'),
            name='input',
            expected_c=len(self._input_cfg.stack_keys),
            expected_h=1,
            expected_w=w_local,
        )
        if not torch.equal(input_tensor[0], amplitude):
            msg = 'sample["input"][0] must match the amplitude channel exactly'
            raise ValueError(msg)

        trace_valid = _to_1d_torch(
            sample.get('trace_valid'),
            name='trace_valid',
            dtype=torch.bool,
            expected_len=1,
        )
        label_valid = _to_1d_torch(
            sample.get('label_valid'),
            name='label_valid',
            dtype=torch.bool,
            expected_len=1,
        )
        raw_trace_idx = _to_1d_torch(
            sample.get('raw_trace_idx'),
            name='raw_trace_idx',
            dtype=torch.int64,
            expected_len=1,
        )
        raw_seed_idx = _to_1d_torch(
            sample.get('raw_seed_idx'),
            name='raw_seed_idx',
            dtype=torch.int64,
            expected_len=1,
        )
        raw_pick_idx = _to_1d_torch(
            sample.get('raw_pick_idx'),
            name='raw_pick_idx',
            dtype=torch.int64,
            expected_len=1,
        )
        local_window_start_idx = _to_1d_torch(
            sample.get('local_window_start_idx'),
            name='local_window_start_idx',
            dtype=torch.int64,
            expected_len=1,
        )
        local_window_end_idx = _to_1d_torch(
            sample.get('local_window_end_idx'),
            name='local_window_end_idx',
            dtype=torch.int64,
            expected_len=1,
        )
        local_seed_idx = _to_1d_torch(
            sample.get('local_seed_idx'),
            name='local_seed_idx',
            dtype=torch.int64,
            expected_len=1,
        )
        local_pick_idx = _to_1d_torch(
            sample.get('local_pick_idx'),
            name='local_pick_idx',
            dtype=torch.int64,
            expected_len=1,
        )

        meta_mode = meta.get('mode')
        if meta_mode != self._expected_mode:
            msg = f'meta.mode must be {self._expected_mode!r}, got {meta_mode!r}'
            raise ValueError(msg)
        meta_seed_source = meta.get('seed_source')
        if meta_seed_source != self._expected_seed_source:
            msg = (
                f'meta.seed_source must be {self._expected_seed_source!r}, '
                f'got {meta_seed_source!r}'
            )
            raise ValueError(msg)

        meta_trace_valid = _to_1d_torch(
            meta.get('trace_valid'),
            name='meta.trace_valid',
            dtype=torch.bool,
            expected_len=1,
        )
        meta_label_valid = _to_1d_torch(
            meta.get('label_valid'),
            name='meta.label_valid',
            dtype=torch.bool,
            expected_len=1,
        )
        meta_raw_trace_idx = _to_1d_torch(
            meta.get('raw_trace_idx'),
            name='meta.raw_trace_idx',
            dtype=torch.int64,
            expected_len=1,
        )
        meta_raw_seed_idx = _to_1d_torch(
            meta.get('raw_seed_idx'),
            name='meta.raw_seed_idx',
            dtype=torch.int64,
            expected_len=1,
        )
        meta_raw_pick_idx = _to_1d_torch(
            meta.get('raw_pick_idx'),
            name='meta.raw_pick_idx',
            dtype=torch.int64,
            expected_len=1,
        )
        meta_local_window_start_idx = _to_1d_torch(
            meta.get('local_window_start_idx'),
            name='meta.local_window_start_idx',
            dtype=torch.int64,
            expected_len=1,
        )
        meta_local_window_end_idx = _to_1d_torch(
            meta.get('local_window_end_idx'),
            name='meta.local_window_end_idx',
            dtype=torch.int64,
            expected_len=1,
        )
        meta_local_seed_idx = _to_1d_torch(
            meta.get('local_seed_idx'),
            name='meta.local_seed_idx',
            dtype=torch.int64,
            expected_len=1,
        )
        meta_local_pick_idx = _to_1d_torch(
            meta.get('local_pick_idx'),
            name='meta.local_pick_idx',
            dtype=torch.int64,
            expected_len=1,
        )
        offsets_view = _to_1d_torch(
            meta.get('offsets_view'),
            name='meta.offsets_view',
            dtype=torch.float32,
            expected_len=1,
        )
        time_view = _to_1d_torch(
            meta.get('time_view'),
            name='meta.time_view',
            dtype=torch.float32,
            expected_len=w_local,
        )
        sample_raw_sample_idx_local = _to_1d_torch(
            sample.get('raw_sample_idx_local'),
            name='raw_sample_idx_local',
            dtype=torch.int64,
            expected_len=w_local,
        )
        raw_sample_idx_local = _to_1d_torch(
            meta.get('raw_sample_idx_local'),
            name='meta.raw_sample_idx_local',
            dtype=torch.int64,
            expected_len=w_local,
        )

        for name, lhs, rhs in (
            ('trace_valid', trace_valid, meta_trace_valid),
            ('label_valid', label_valid, meta_label_valid),
            ('raw_trace_idx', raw_trace_idx, meta_raw_trace_idx),
            ('raw_seed_idx', raw_seed_idx, meta_raw_seed_idx),
            ('raw_pick_idx', raw_pick_idx, meta_raw_pick_idx),
            ('local_window_start_idx', local_window_start_idx, meta_local_window_start_idx),
            ('local_window_end_idx', local_window_end_idx, meta_local_window_end_idx),
            ('local_seed_idx', local_seed_idx, meta_local_seed_idx),
            ('local_pick_idx', local_pick_idx, meta_local_pick_idx),
        ):
            if not torch.equal(lhs, rhs):
                msg = f'{name} and meta.{name} must match exactly'
                raise ValueError(msg)

        if not bool(trace_valid[0]):
            msg = 'LocalWindowDataset must emit trace_valid=true for enumerated fine samples'
            raise ValueError(msg)
        if int(local_window_start_idx[0]) < 0:
            msg = 'local_window_start_idx must be >= 0'
            raise ValueError(msg)
        if int(local_window_end_idx[0]) <= int(local_window_start_idx[0]):
            msg = 'local_window_end_idx must be > local_window_start_idx'
            raise ValueError(msg)
        if int(local_window_end_idx[0] - local_window_start_idx[0]) > int(w_local):
            msg = 'raw local coverage must not exceed the fixed local window length'
            raise ValueError(msg)
        if int(local_seed_idx[0]) < 0 or int(local_seed_idx[0]) >= int(w_local):
            msg = f'local_seed_idx must satisfy 0 <= idx < {w_local}'
            raise ValueError(msg)
        if int(raw_seed_idx[0]) != int(local_window_start_idx[0] + local_seed_idx[0]):
            msg = 'raw_seed_idx must map from local_window_start_idx + local_seed_idx'
            raise ValueError(msg)

        valid_local_count = int(torch.count_nonzero(raw_sample_idx_local >= 0).item())
        if valid_local_count != int(local_window_end_idx[0] - local_window_start_idx[0]):
            msg = (
                'raw_sample_idx_local valid coverage must match '
                'local_window_end_idx - local_window_start_idx'
            )
            raise ValueError(msg)
        if int(raw_sample_idx_local[int(local_seed_idx[0])]) != int(raw_seed_idx[0]):
            msg = 'raw_sample_idx_local must map local_seed_idx back to raw_seed_idx'
            raise ValueError(msg)
        if not torch.equal(sample_raw_sample_idx_local, raw_sample_idx_local):
            msg = 'raw_sample_idx_local and meta.raw_sample_idx_local must match exactly'
            raise ValueError(msg)

        sample['input'] = input_tensor
        sample['x_view_local'] = x_view_local
        sample[self._input_cfg.amplitude_key] = amplitude
        sample['raw_sample_idx_local'] = raw_sample_idx_local
        sample['trace_valid'] = trace_valid
        sample['label_valid'] = label_valid
        sample['raw_trace_idx'] = raw_trace_idx
        sample['raw_seed_idx'] = raw_seed_idx
        sample['raw_pick_idx'] = raw_pick_idx
        sample['local_window_start_idx'] = local_window_start_idx
        sample['local_window_end_idx'] = local_window_end_idx
        sample['local_seed_idx'] = local_seed_idx
        sample['local_pick_idx'] = local_pick_idx

        meta['trace_valid'] = trace_valid
        meta['label_valid'] = label_valid
        meta['offsets_view'] = offsets_view
        meta['time_view'] = time_view
        meta['raw_sample_idx_local'] = raw_sample_idx_local
        meta['raw_trace_idx'] = raw_trace_idx
        meta['raw_seed_idx'] = raw_seed_idx
        meta['raw_pick_idx'] = raw_pick_idx
        meta['local_window_start_idx'] = local_window_start_idx
        meta['local_window_end_idx'] = local_window_end_idx
        meta['local_seed_idx'] = local_seed_idx
        meta['local_pick_idx'] = local_pick_idx
        sample['meta'] = meta

        if self._require_target:
            target = _to_3d_float(
                sample.get('target'),
                name='target',
                expected_c=1,
                expected_h=1,
                expected_w=w_local,
            )
            if not bool(label_valid[0]):
                msg = 'train/eval fine samples must have label_valid=true'
                raise ValueError(msg)
            if int(local_pick_idx[0]) < 0 or int(local_pick_idx[0]) >= int(w_local):
                msg = f'local_pick_idx must satisfy 0 <= idx < {w_local} in train/eval mode'
                raise ValueError(msg)
            if int(local_seed_idx[0]) != int(local_pick_idx[0]):
                msg = 'train/eval fine samples must use the GT pick as the local seed'
                raise ValueError(msg)
            if int(raw_seed_idx[0]) != int(raw_pick_idx[0]):
                msg = 'train/eval fine samples must use the GT pick as the raw seed'
                raise ValueError(msg)
            if int(raw_pick_idx[0]) != int(local_window_start_idx[0] + local_pick_idx[0]):
                msg = 'raw_pick_idx must map from local_window_start_idx + local_pick_idx'
                raise ValueError(msg)
            if int(raw_sample_idx_local[int(local_pick_idx[0])]) != int(raw_pick_idx[0]):
                msg = 'raw_sample_idx_local must map local_pick_idx back to raw_pick_idx'
                raise ValueError(msg)
            sample['target'] = target
        else:
            if 'target' in sample:
                msg = 'input-only fine dataset must not produce target'
                raise ValueError(msg)
            if bool(label_valid[0]):
                msg = 'infer fine samples must have label_valid=false'
                raise ValueError(msg)
            if int(local_pick_idx[0]) != -1:
                msg = 'infer fine samples must have local_pick_idx=-1'
                raise ValueError(msg)
            if int(raw_pick_idx[0]) != -1:
                msg = 'infer fine samples must have raw_pick_idx=-1'
                raise ValueError(msg)

        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._base_dataset[index]
        return self._normalize_sample(sample)


def build_dataset(
    *,
    segy_files: Sequence[str],
    fb_files: Sequence[str],
    transform,
    plan: BuildPlan,
    input_cfg: FineInputCfg,
    window_cfg: FineWindowCfg,
    mode: Literal['train', 'eval'] = 'train',
    use_header_cache: bool = True,
    header_cache_dir: str | None = None,
    waveform_mode: str = 'mmap',
    segy_endian: str = 'big',
) -> FineLocalDataset:
    input_cfg_checked = _require_input_cfg(input_cfg)
    window_cfg_checked = _require_window_cfg(window_cfg)

    if str(mode) not in ('train', 'eval'):
        msg = 'mode must be "train" or "eval"'
        raise ValueError(msg)
    if len(segy_files) == 0:
        msg = 'segy_files must be non-empty'
        raise ValueError(msg)
    if len(fb_files) == 0:
        msg = 'fb_files must be non-empty'
        raise ValueError(msg)
    if len(segy_files) != len(fb_files):
        msg = 'segy_files and fb_files must have same length'
        raise ValueError(msg)
    if not isinstance(plan, BuildPlan):
        msg = 'train/eval fine build_dataset requires BuildPlan'
        raise TypeError(msg)
    if transform is not None and not callable(transform):
        msg = 'transform must be callable or None'
        raise TypeError(msg)

    validate_files_exist(list(segy_files) + list(fb_files))

    base_dataset = LocalWindowDataset(
        list(segy_files),
        cfg=LocalWindowDatasetConfig(
            local_window_len=int(window_cfg_checked.local_window_len),
            mode=str(mode),
        ),
        fb_files=list(fb_files),
        plan=plan,
        transform=transform,
        use_header_cache=bool(use_header_cache),
        header_cache_dir=header_cache_dir,
        waveform_mode=str(waveform_mode),
        segy_endian=str(segy_endian),
    )
    return FineLocalDataset(
        base_dataset,
        input_cfg=input_cfg_checked,
        require_target=True,
        expected_mode=str(mode),
        expected_seed_source='gt',
    )
