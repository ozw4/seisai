from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from seisai_dataset import (
    BuildPlan,
    InferenceGatherWindowsConfig,
    InferenceGatherWindowsDataset,
    InputOnlyPlan,
    SegyGatherPipelineDataset,
)
from seisai_dataset.config import LoaderConfig
from seisai_dataset.segy_gather_base import SampleTransformer
from seisai_dataset.trace_subset_preproc import TraceSubsetLoader
from seisai_dataset.transform_flow_utils import (
    apply_transform_2d_with_meta,
    pad_indices_offsets_fb,
)
from seisai_transforms.augment import PerTraceStandardize, ViewCompose
from seisai_utils.fs import validate_files_exist

from seisai_engine.pipelines.fbpick.common import load_robust_npz
from seisai_engine.pipelines.fbpick.fine.config import FineCenterAugmentCfg

__all__ = [
    'FineInferenceGatherWindowsDataset',
    'FineLocalWindowSampleTransformer',
    'LocalWindowView',
    'build_infer_transform',
    'build_labeled_infer_dataset',
    'build_raw_infer_dataset',
    'build_train_dataset',
    'build_train_transform',
    'collate_input_meta_list',
    'extract_local_windowed_view',
    'restore_local_pick_to_raw',
    'sample_center_jitter',
]


@dataclass(frozen=True)
class LocalWindowView:
    x_view: np.ndarray
    fb_idx_view: np.ndarray | None
    window_start_i: np.ndarray
    window_end_i: np.ndarray
    center_raw_i: np.ndarray


def _pad_int_vector(values: np.ndarray, *, H: int, fill_value: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim != 1:
        msg = 'values must be 1D'
        raise ValueError(msg)
    if int(arr.shape[0]) > int(H):
        msg = f'values length {int(arr.shape[0])} > H {int(H)}'
        raise ValueError(msg)
    if int(arr.shape[0]) == int(H):
        return arr
    pad = np.full((int(H) - int(arr.shape[0]),), int(fill_value), dtype=np.int64)
    return np.concatenate([arr, pad], axis=0)


def _pad_bool_vector(values: np.ndarray, *, H: int, fill_value: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.bool_)
    if arr.ndim != 1:
        msg = 'values must be 1D'
        raise ValueError(msg)
    if int(arr.shape[0]) > int(H):
        msg = f'values length {int(arr.shape[0])} > H {int(H)}'
        raise ValueError(msg)
    if int(arr.shape[0]) == int(H):
        return arr
    pad = np.full((int(H) - int(arr.shape[0]),), bool(fill_value), dtype=np.bool_)
    return np.concatenate([arr, pad], axis=0)


def _build_local_time_view(*, time_len: int, dt_sec: float) -> np.ndarray:
    return np.arange(int(time_len), dtype=np.float32) * np.float32(dt_sec)


def _validate_local_transform_meta(meta: dict) -> None:
    hflip = bool(meta.get('hflip', False))
    if hflip:
        msg = 'fine local-window transform must not hflip traces'
        raise ValueError(msg)

    factor_h = float(meta.get('factor_h', 1.0))
    if not np.isclose(factor_h, 1.0):
        msg = 'fine local-window transform must keep factor_h == 1.0'
        raise ValueError(msg)

    factor = float(meta.get('factor', 1.0))
    if not np.isclose(factor, 1.0):
        msg = 'fine local-window transform must keep factor == 1.0'
        raise ValueError(msg)

    start = int(meta.get('start', 0))
    if start != 0:
        msg = 'fine local-window transform must keep start == 0'
        raise ValueError(msg)

    meta['hflip'] = False
    meta['factor_h'] = 1.0
    meta['factor'] = 1.0
    meta['start'] = 0


def _select_center_vector(
    robust: dict[str, np.ndarray],
    *,
    npz_key: str,
    fallback_npz_key: str | None,
) -> tuple[str, np.ndarray]:
    center_key = str(npz_key)
    fallback_key = None if fallback_npz_key is None else str(fallback_npz_key)
    if center_key in robust:
        return center_key, np.asarray(robust[center_key])
    if fallback_key is not None and fallback_key in robust:
        return fallback_key, np.asarray(robust[fallback_key])
    msg = f'robust npz missing requested fine window center key: {center_key}'
    raise KeyError(msg)


def _load_robust_centers_for_info(
    robust_npz_path: str,
    info,
    *,
    npz_key: str = 'robust_pick_i',
    fallback_npz_key: str | None = None,
    valid_mask_npz_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    robust = load_robust_npz(robust_npz_path)
    n_traces = int(np.asarray(robust['n_traces']).item())
    n_samples_orig = int(np.asarray(robust['n_samples_orig']).item())
    if n_traces != int(info['n_traces']):
        msg = f'robust n_traces {n_traces} != info.n_traces {int(info["n_traces"])}'
        raise ValueError(msg)
    if n_samples_orig != int(info['n_samples']):
        msg = (
            f'robust n_samples_orig {n_samples_orig} != '
            f'info.n_samples {int(info["n_samples"])}'
        )
        raise ValueError(msg)

    trace_indices = np.asarray(robust['trace_indices'], dtype=np.int64)
    expected_trace_indices = np.arange(n_traces, dtype=np.int64)
    if not np.array_equal(trace_indices, expected_trace_indices):
        msg = 'robust trace_indices must equal np.arange(n_traces)'
        raise ValueError(msg)

    dt_sec = float(np.asarray(robust['dt_sec']).item())
    if not np.isclose(dt_sec, float(info['dt_sec'])):
        msg = f'robust dt_sec {dt_sec} != info.dt_sec {float(info["dt_sec"])}'
        raise ValueError(msg)

    selected_key, selected = _select_center_vector(
        robust,
        npz_key=npz_key,
        fallback_npz_key=fallback_npz_key,
    )
    if selected.ndim != 1 or int(selected.shape[0]) != n_traces:
        msg = f'{selected_key} must be 1D with length n_traces'
        raise ValueError(msg)
    if not np.issubdtype(selected.dtype, np.integer):
        msg = f'{selected_key} must be an integer center vector'
        raise ValueError(msg)

    valid_mask: np.ndarray | None = None
    if valid_mask_npz_key is not None:
        mask_key = str(valid_mask_npz_key)
        if mask_key not in robust:
            msg = f'robust npz missing requested fine window valid mask key: {mask_key}'
            raise KeyError(msg)
        mask_arr = np.asarray(robust[mask_key])
        if mask_arr.ndim != 1 or int(mask_arr.shape[0]) != n_traces:
            msg = f'{mask_key} must be 1D with length n_traces'
            raise ValueError(msg)
        if mask_arr.dtype != np.dtype(np.bool_):
            msg = f'{mask_key} must be a bool valid mask'
            raise ValueError(msg)
        valid_mask = mask_arr.astype(np.bool_, copy=False)

    centers = selected.astype(np.int64, copy=False)
    center_validate_mask = (
        np.ones((n_traces,), dtype=np.bool_) if valid_mask is None else valid_mask
    )
    centers_to_validate = centers[center_validate_mask]
    if np.any(centers_to_validate < 0) or np.any(centers_to_validate >= n_samples_orig):
        msg = f'{selected_key} must lie in [0, n_samples_orig)'
        raise ValueError(msg)
    return centers, valid_mask


def sample_center_jitter(
    *,
    size: int,
    cfg: FineCenterAugmentCfg,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample integer center offsets; configured weights are normalized internally."""
    n = int(size)
    if n < 0:
        msg = 'size must be non-negative'
        raise ValueError(msg)
    out = np.zeros((n,), dtype=np.int64)
    if n == 0 or not bool(cfg.enabled):
        return out

    weights = np.asarray(
        [float(cfg.p_no_jitter)]
        + [float(component.prob) for component in cfg.uniform_jitter_samples],
        dtype=np.float64,
    )
    if np.any(weights < 0.0):
        msg = 'center jitter probabilities must be non-negative'
        raise ValueError(msg)
    total = float(weights.sum())
    if total <= 0.0:
        msg = 'center jitter probabilities must sum to > 0'
        raise ValueError(msg)
    weights = weights / total

    choices = rng.choice(int(weights.shape[0]), size=n, p=weights)
    for component_idx, component in enumerate(cfg.uniform_jitter_samples, start=1):
        mask = choices == component_idx
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        out[mask] = rng.integers(
            int(component.lo),
            int(component.hi) + 1,
            size=count,
            dtype=np.int64,
        )
    return out


def extract_local_windowed_view(
    x: np.ndarray,
    *,
    center_raw_i: np.ndarray,
    trace_valid: np.ndarray,
    fb_raw_i: np.ndarray | None,
    time_len: int,
    center_index: int,
    require_fb_inside: bool,
) -> LocalWindowView | None:
    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim != 2:
        msg = 'x must be 2D (H, W0)'
        raise ValueError(msg)

    H, W0 = x_arr.shape
    if H <= 0 or W0 <= 0:
        msg = 'x must be non-empty'
        raise ValueError(msg)
    if int(time_len) <= 0:
        msg = 'time_len must be positive'
        raise ValueError(msg)
    if int(center_index) < 0 or int(center_index) >= int(time_len):
        msg = 'center_index must satisfy 0 <= center_index < time_len'
        raise ValueError(msg)

    center_arr = np.asarray(center_raw_i, dtype=np.int64)
    trace_valid_arr = np.asarray(trace_valid, dtype=np.bool_)
    if center_arr.shape != (H,):
        msg = f'center_raw_i shape {center_arr.shape} != ({H},)'
        raise ValueError(msg)
    if trace_valid_arr.shape != (H,):
        msg = f'trace_valid shape {trace_valid_arr.shape} != ({H},)'
        raise ValueError(msg)

    if fb_raw_i is None:
        fb_arr = None
    else:
        fb_arr = np.asarray(fb_raw_i, dtype=np.int64)
        if fb_arr.shape != (H,):
            msg = f'fb_raw_i shape {fb_arr.shape} != ({H},)'
            raise ValueError(msg)

    out = np.zeros((H, int(time_len)), dtype=np.float32)
    window_start_i = np.full((H,), -1, dtype=np.int32)
    window_end_i = np.full((H,), -1, dtype=np.int32)
    center_out = np.full((H,), -1, dtype=np.int32)
    fb_idx_view = None if fb_arr is None else np.full((H,), -1, dtype=np.int64)

    for row_idx in range(H):
        if not bool(trace_valid_arr[row_idx]):
            continue
        center_value = int(center_arr[row_idx])
        if center_value < 0 or center_value >= W0:
            msg = f'center_raw_i out of range for row {row_idx}: {center_value}'
            raise ValueError(msg)

        start = center_value - int(center_index)
        end = start + int(time_len)
        src_start = max(start, 0)
        src_end = min(end, W0)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)

        out[row_idx, dst_start:dst_end] = x_arr[row_idx, src_start:src_end]
        window_start_i[row_idx] = np.int32(start)
        window_end_i[row_idx] = np.int32(end)
        center_out[row_idx] = np.int32(center_value)

        if fb_idx_view is not None and fb_arr is not None:
            fb_value = int(fb_arr[row_idx])
            if fb_value >= 0:
                local_idx = fb_value - start
                if require_fb_inside and (local_idx <= 0 or local_idx >= int(time_len)):
                    return None
                fb_idx_view[row_idx] = np.int64(local_idx)

    return LocalWindowView(
        x_view=out,
        fb_idx_view=fb_idx_view,
        window_start_i=window_start_i,
        window_end_i=window_end_i,
        center_raw_i=center_out,
    )


def restore_local_pick_to_raw(
    local_pick: np.ndarray,
    *,
    window_start_i: np.ndarray,
    n_samples_orig: int,
) -> np.ndarray:
    local_arr = np.asarray(local_pick, dtype=np.float32)
    start_arr = np.asarray(window_start_i, dtype=np.float32)
    if local_arr.shape != start_arr.shape:
        msg = 'local_pick and window_start_i must have the same shape'
        raise ValueError(msg)
    n_samples = int(n_samples_orig)
    if n_samples <= 0:
        msg = 'n_samples_orig must be positive'
        raise ValueError(msg)
    raw = local_arr + start_arr
    raw = np.clip(raw, 0.0, float(n_samples - 1))
    return raw.astype(np.float32, copy=False)


class FineLocalWindowSampleTransformer(SampleTransformer):
    def __init__(
        self,
        subsetloader: TraceSubsetLoader,
        transform,
        *,
        time_len: int,
        center_index: int,
    ) -> None:
        super().__init__(subsetloader, transform)
        self.time_len = int(time_len)
        self.center_index = int(center_index)
        if self.time_len <= 0:
            msg = 'time_len must be positive'
            raise ValueError(msg)
        if self.center_index < 0 or self.center_index >= self.time_len:
            msg = 'center_index must satisfy 0 <= center_index < time_len'
            raise ValueError(msg)

    def load_transform_or_reject(
        self,
        info,
        indices: np.ndarray,
        fb_subset: np.ndarray,
        center_subset: np.ndarray,
        center_valid_subset: np.ndarray | None,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        x = self.subsetloader.load(info.mmap, indices)
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            msg = 'TraceSubsetLoader must return 2D numpy array'
            raise TypeError(msg)

        H = int(x.shape[0])
        offsets = info.offsets[indices].astype(np.float32, copy=False)
        indices_pad, offsets_pad, fb_subset_pad, trace_valid, _pad = pad_indices_offsets_fb(
            indices=indices,
            offsets=offsets,
            fb_subset=fb_subset,
            H=H,
        )
        if fb_subset_pad is None:
            msg = 'fb_subset is required for fine local-window training'
            raise RuntimeError(msg)

        center_pad = _pad_int_vector(center_subset, H=H, fill_value=-1)
        if center_valid_subset is not None:
            trace_valid = trace_valid & _pad_bool_vector(
                center_valid_subset,
                H=H,
                fill_value=False,
            )
        local = extract_local_windowed_view(
            x,
            center_raw_i=center_pad,
            trace_valid=trace_valid,
            fb_raw_i=fb_subset_pad,
            time_len=self.time_len,
            center_index=self.center_index,
            require_fb_inside=True,
        )
        if local is None:
            return None
        if local.fb_idx_view is None:
            msg = 'local fb_idx_view is required for fine local-window training'
            raise RuntimeError(msg)

        meta: dict = {}
        x_view = local.x_view
        if self.transform is not None:
            x_view, post_meta = apply_transform_2d_with_meta(
                self.transform,
                x_view,
                rng,
                msg_bad_out='fine local-window transform must return 2D numpy or (2D, meta)',
                msg_bad_meta='fine local-window transform meta must be dict, got {type}',
                exc_bad_out=ValueError,
                exc_bad_meta=TypeError,
            )
            if x_view.shape != local.x_view.shape:
                msg = 'fine local-window transform must keep the local window shape'
                raise ValueError(msg)
            meta.update(post_meta)
        _validate_local_transform_meta(meta)

        meta['trace_valid'] = trace_valid
        meta['fb_idx_view'] = local.fb_idx_view
        meta['offsets_view'] = offsets_pad
        meta['time_view'] = _build_local_time_view(
            time_len=self.time_len,
            dt_sec=float(info.dt_sec),
        )
        meta['window_start_i'] = local.window_start_i
        meta['window_end_i'] = local.window_end_i
        meta['center_raw_i'] = local.center_raw_i
        meta['dt_sec'] = np.float32(info.dt_sec)
        meta['dt_eff_sec'] = np.float32(info.dt_sec)

        return x_view, meta, offsets_pad, fb_subset_pad, indices_pad, trace_valid


class _FineLocalWindowTrainDataset(SegyGatherPipelineDataset):
    def __init__(
        self,
        *,
        robust_npz_files: list[str],
        window_center_npz_key: str = 'robust_pick_i',
        window_center_fallback_npz_key: str | None = None,
        window_center_valid_mask_npz_key: str | None = None,
        center_augment: FineCenterAugmentCfg | None = None,
        **kwargs,
    ) -> None:
        self._robust_npz_files = list(robust_npz_files)
        self._window_center_npz_key = str(window_center_npz_key)
        self._window_center_fallback_npz_key = window_center_fallback_npz_key
        self._window_center_valid_mask_npz_key = window_center_valid_mask_npz_key
        self._center_augment = center_augment
        super().__init__(**kwargs)
        if len(self._robust_npz_files) != len(self.file_infos):
            msg = 'robust_npz_files length must match indexed training files'
            raise ValueError(msg)
        self._center_pick_by_path: dict[str, np.ndarray] = {}
        self._center_valid_by_path: dict[str, np.ndarray] = {}
        for info, robust_npz_path in zip(self.file_infos, self._robust_npz_files, strict=True):
            centers, valid_mask = _load_robust_centers_for_info(
                robust_npz_path,
                info,
                npz_key=self._window_center_npz_key,
                fallback_npz_key=self._window_center_fallback_npz_key,
                valid_mask_npz_key=self._window_center_valid_mask_npz_key,
            )
            self._center_pick_by_path[str(info.path)] = centers
            if valid_mask is not None:
                self._center_valid_by_path[str(info.path)] = valid_mask

    def _init_rejection_counters(self) -> dict[str, int]:
        counters = super()._init_rejection_counters()
        counters['local_window'] = 0
        counters['center_jitter'] = 0
        return counters

    def _format_max_trials_error(self, counters: dict[str, int]) -> str:
        parts = []
        for key in ('empty', 'min_pick', 'fblc', 'local_window', 'center_jitter'):
            if key in counters:
                parts.append(f'{key}={counters[key]}')
        rej = ', '.join(parts)
        return (
            f'failed to draw a valid sample within max_trials={self.max_trials}; '
            f'rejections: {rej}, files={len(self.file_infos)}'
        )

    def _load_fb_subset(
        self,
        info,
        indices: np.ndarray,
        sample: dict,
        counters: dict[str, int],
    ) -> tuple[np.ndarray, dict] | None:
        _ = (sample, counters)
        if info.fb is None:
            msg = 'fine local-window training requires info.fb'
            raise RuntimeError(msg)
        fb_subset = np.asarray(info.fb[indices], dtype=np.int64)
        centers = self._center_pick_by_path[str(info.path)]
        center_subset = np.asarray(centers[indices], dtype=np.int64)
        center_valid = self._center_valid_by_path.get(str(info.path))
        center_valid_subset = (
            None if center_valid is None else np.asarray(center_valid[indices], dtype=np.bool_)
        )
        center_jitter_nonzero = False
        if self._center_augment is not None and bool(self._center_augment.enabled):
            jitter = sample_center_jitter(
                size=int(center_subset.shape[0]),
                cfg=self._center_augment,
                rng=self._rng,
            )
            center_jitter_nonzero = bool(np.any(jitter != 0))
            center_subset = center_subset + jitter
            if bool(self._center_augment.clip_to_record):
                center_subset = np.clip(
                    center_subset,
                    0,
                    int(info.n_samples) - 1,
                ).astype(np.int64, copy=False)
        return fb_subset, {
            'apply_fb_gates': False,
            'center_subset': center_subset,
            'center_valid_subset': center_valid_subset,
            'center_jitter_nonzero': center_jitter_nonzero,
        }

    def _try_build_sample(self, info, counters: dict[str, int]) -> dict | None:
        sample = self._draw_sample(info)
        indices = sample['indices']

        label = self._load_fb_subset(info, indices, sample, counters)
        if label is None:
            return None
        fb_subset, label_state = label
        center_subset = np.asarray(label_state['center_subset'], dtype=np.int64)

        transformer = self.sample_transformer
        if not isinstance(transformer, FineLocalWindowSampleTransformer):
            msg = 'sample_transformer must be FineLocalWindowSampleTransformer'
            raise TypeError(msg)
        transformed = transformer.load_transform_or_reject(
            info,
            indices,
            fb_subset,
            center_subset,
            label_state.get('center_valid_subset'),
            self._rng,
        )
        if transformed is None:
            if bool(label_state.get('center_jitter_nonzero', False)):
                self._count_rejection(counters, 'center_jitter')
            else:
                self._count_rejection(counters, 'local_window')
            return None
        x_view, meta, offsets, fb_subset_pad, indices_pad, trace_valid = transformed
        view_shape = self._get_view_shape(x_view, label_state)

        sample['indices'] = indices_pad
        sample['trace_valid'] = trace_valid
        sample['x_view'] = x_view
        meta['key_name'] = sample['key_name']
        meta['primary_unique'] = sample['primary_unique']

        self._post_transform_meta(
            meta=meta,
            label_state=label_state,
            trace_valid=trace_valid,
            view_shape=view_shape,
        )
        self._set_dt_eff_meta_no_gate(meta, info=info)

        sample_for_plan = self.sample_flow.build_plan_input_base(
            meta=meta,
            dt_sec=float(meta['dt_eff_sec']),
            offsets=offsets,
            indices=sample['indices'],
            key_name=sample['key_name'],
            secondary_key=sample['secondary_key'],
            primary_unique=sample['primary_unique'],
            extra=self._build_plan_extras_common(
                info=info,
                sample=sample,
                fb_subset=fb_subset_pad,
                trace_valid=trace_valid,
            ),
        )
        self.sample_flow.run_plan(sample_for_plan, rng=self._rng)
        self._post_plan_validation(sample_for_plan, label_state, view_shape)

        out = self.sample_flow.build_output_base(
            sample_for_plan,
            meta=meta,
            dt_sec=float(meta['dt_eff_sec']),
            offsets=offsets,
            indices=sample['indices'],
            key_name=sample['key_name'],
            secondary_key=sample['secondary_key'],
            primary_unique=sample['primary_unique'],
            extra=self._build_output_extras_common(
                info=info,
                sample=sample,
                fb_subset=fb_subset_pad,
            ),
        )
        self._add_trace_valid_output(out, trace_valid)
        self._add_mask_bool_output(out, sample_for_plan)
        self._finalize_output(out, sample_for_plan, label_state, view_shape)
        return out


class FineInferenceGatherWindowsDataset(InferenceGatherWindowsDataset):
    def __init__(
        self,
        *,
        robust_npz_files: Sequence[str],
        center_index: int,
        window_center_npz_key: str = 'robust_pick_i',
        window_center_fallback_npz_key: str | None = None,
        window_center_valid_mask_npz_key: str | None = None,
        **kwargs,
    ) -> None:
        self._robust_npz_files = list(robust_npz_files)
        self._center_index = int(center_index)
        self._window_center_npz_key = str(window_center_npz_key)
        self._window_center_fallback_npz_key = window_center_fallback_npz_key
        self._window_center_valid_mask_npz_key = window_center_valid_mask_npz_key
        super().__init__(**kwargs)
        if len(self._robust_npz_files) != len(self.file_infos):
            msg = 'robust_npz_files length must match indexed inference files'
            raise ValueError(msg)
        self._center_pick_by_path: dict[str, np.ndarray] = {}
        self._center_valid_by_path: dict[str, np.ndarray] = {}
        for info, robust_npz_path in zip(self.file_infos, self._robust_npz_files, strict=True):
            centers, valid_mask = _load_robust_centers_for_info(
                robust_npz_path,
                info,
                npz_key=self._window_center_npz_key,
                fallback_npz_key=self._window_center_fallback_npz_key,
                valid_mask_npz_key=self._window_center_valid_mask_npz_key,
            )
            self._center_pick_by_path[str(info['path'])] = centers
            if valid_mask is not None:
                self._center_valid_by_path[str(info['path'])] = valid_mask

    def fine_window_valid_mask_for_info(self, info) -> np.ndarray | None:
        return self._center_valid_by_path.get(str(info['path']))

    def __getitem__(self, i: int) -> dict:
        gidx, s, e = self.items[i]
        fi, dom, pk, idxs_sorted, Htot = self.groups[gidx]
        info = self.file_infos[fi]

        idx_win = idxs_sorted[s:e]
        H0 = int(idx_win.size)
        if H0 <= 0:
            msg = 'empty window'
            raise RuntimeError(msg)

        x = self._subsetloader.load(info['mmap'], idx_win.astype(np.int64, copy=False))
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            msg = 'TraceSubsetLoader must return 2D numpy array'
            raise TypeError(msg)
        H = int(x.shape[0])
        if int(self.cfg.win_size_traces) != H:
            msg = f'loaded H {H} != win_size_traces {int(self.cfg.win_size_traces)}'
            raise ValueError(msg)
        if H0 > H:
            msg = f'window size {H0} > loaded H {H}'
            raise ValueError(msg)

        off = info['offsets'][idx_win].astype(np.float32, copy=False)
        indices_pad, off, _fb_pad, trace_valid, _pad = pad_indices_offsets_fb(
            indices=idx_win.astype(np.int64, copy=False),
            offsets=off,
            fb_subset=None,
            H=H,
        )
        centers = self._center_pick_by_path[str(info['path'])]
        center_subset = np.asarray(centers[idx_win], dtype=np.int64)
        center_pad = _pad_int_vector(center_subset, H=H, fill_value=-1)
        center_valid = self._center_valid_by_path.get(str(info['path']))
        if center_valid is not None:
            center_valid_subset = np.asarray(center_valid[idx_win], dtype=np.bool_)
            trace_valid = trace_valid & _pad_bool_vector(
                center_valid_subset,
                H=H,
                fill_value=False,
            )

        local = extract_local_windowed_view(
            x,
            center_raw_i=center_pad,
            trace_valid=trace_valid,
            fb_raw_i=None,
            time_len=int(self.cfg.target_len),
            center_index=self._center_index,
            require_fb_inside=False,
        )
        if local is None:
            msg = 'fine inference local-window extraction unexpectedly rejected sample'
            raise RuntimeError(msg)

        x_view, meta = apply_transform_2d_with_meta(
            self.transform,
            local.x_view,
            self._rng,
            msg_bad_out='transform must return 2D numpy or (2D, meta)',
            msg_bad_meta='transform meta must be dict, got {type}',
            exc_bad_out=ValueError,
            exc_bad_meta=TypeError,
            allow_non_dict_meta=True,
        )
        if x_view.shape != local.x_view.shape:
            msg = 'fine inference transform must keep the local window shape'
            raise ValueError(msg)
        _validate_local_transform_meta(meta)

        raw_idx_global = np.full((H,), -1, dtype=np.int64)
        raw_idx_global[:H0] = self._file_base[fi] + idx_win.astype(np.int64, copy=False)

        abs_h = np.full((H,), -1, dtype=np.int64)
        abs_h[:H0] = np.arange(s, s + H0, dtype=np.int64)

        meta['raw_idx_global'] = raw_idx_global
        meta['abs_h'] = abs_h
        meta['gather_len'] = int(Htot)
        meta['domain'] = dom
        meta['primary_key'] = int(pk)
        meta['group_id'] = f'{fi}:{dom}:{int(pk)}'
        meta['file_idx'] = int(fi)
        meta['file_path'] = str(info['path'])
        meta['n_total'] = int(self.n_total)
        meta['dt_sec'] = np.float32(info['dt_sec'])
        meta['dt_eff_sec'] = np.float32(info['dt_sec'])
        meta['trace_valid'] = trace_valid
        meta['fb_idx_view'] = np.full((H,), -1, dtype=np.int64)
        meta['offsets_view'] = off
        meta['time_view'] = _build_local_time_view(
            time_len=int(self.cfg.target_len),
            dt_sec=float(info['dt_sec']),
        )
        meta['window_start_i'] = local.window_start_i
        meta['window_end_i'] = local.window_end_i
        meta['center_raw_i'] = local.center_raw_i
        meta['trace_slice_start'] = int(s)
        meta['trace_slice_end'] = int(e)
        meta['indices_pad'] = indices_pad

        sample = {
            'x_view': x_view,
            'meta': meta,
        }
        self.plan.run(sample, rng=self._rng)
        if 'input' not in sample:
            msg = "plan must set sample['input']"
            raise KeyError(msg)

        x_in = sample['input']
        if not isinstance(x_in, torch.Tensor):
            msg = "sample['input'] must be torch.Tensor"
            raise TypeError(msg)
        if x_in.ndim != 3:
            msg = f"sample['input'] must be (C,H,W), got {tuple(x_in.shape)}"
            raise ValueError(msg)
        if int(x_in.shape[1]) != H or int(x_in.shape[2]) != int(self.cfg.target_len):
            msg = 'input shape must match local window (H,W)'
            raise ValueError(msg)

        return {
            'input': x_in,
            'meta': meta,
        }


def build_train_transform(*, standardize_eps: float) -> ViewCompose:
    return ViewCompose([PerTraceStandardize(eps=float(standardize_eps))])


def build_infer_transform(*, standardize_eps: float) -> ViewCompose:
    return ViewCompose([PerTraceStandardize(eps=float(standardize_eps))])


def _build_labeled_dataset(
    *,
    segy_files: list[str],
    fb_files: list[str],
    robust_npz_files: list[str],
    sampling_overrides: list[dict[str, object] | None] | None,
    plan: BuildPlan,
    fbgate,
    trace_len: int,
    time_len: int,
    center_index: int,
    standardize_eps: float,
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
    window_center_npz_key: str = 'robust_pick_i',
    window_center_fallback_npz_key: str | None = None,
    window_center_valid_mask_npz_key: str | None = None,
    center_augment: FineCenterAugmentCfg | None = None,
) -> SegyGatherPipelineDataset:
    if sampling_overrides is not None and len(sampling_overrides) != len(segy_files):
        msg = 'sampling_overrides length must match segy_files length'
        raise ValueError(msg)
    if len(segy_files) != len(fb_files):
        msg = 'segy_files and fb_files must have the same length'
        raise ValueError(msg)
    if len(segy_files) != len(robust_npz_files):
        msg = 'segy_files and robust_npz_files must have the same length'
        raise ValueError(msg)
    validate_files_exist(list(segy_files) + list(fb_files) + list(robust_npz_files))

    transform = build_train_transform(standardize_eps=float(standardize_eps))
    subsetloader = TraceSubsetLoader(LoaderConfig(pad_traces_to=int(trace_len)))
    sample_transformer = FineLocalWindowSampleTransformer(
        subsetloader,
        transform,
        time_len=int(time_len),
        center_index=int(center_index),
    )
    return _FineLocalWindowTrainDataset(
        segy_files=list(segy_files),
        fb_files=list(fb_files),
        robust_npz_files=list(robust_npz_files),
        window_center_npz_key=str(window_center_npz_key),
        window_center_fallback_npz_key=window_center_fallback_npz_key,
        window_center_valid_mask_npz_key=window_center_valid_mask_npz_key,
        center_augment=center_augment,
        sampling_overrides=sampling_overrides,
        transform=transform,
        sample_transformer=sample_transformer,
        fbgate=fbgate,
        plan=plan,
        subset_traces=int(trace_len),
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


def build_train_dataset(
    *,
    segy_files: list[str],
    fb_files: list[str],
    robust_npz_files: list[str],
    sampling_overrides: list[dict[str, object] | None] | None,
    plan: BuildPlan,
    fbgate,
    trace_len: int,
    time_len: int,
    center_index: int,
    standardize_eps: float,
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
    window_center_npz_key: str = 'robust_pick_i',
    window_center_fallback_npz_key: str | None = None,
    window_center_valid_mask_npz_key: str | None = None,
    center_augment: FineCenterAugmentCfg | None = None,
) -> SegyGatherPipelineDataset:
    return _build_labeled_dataset(
        segy_files=segy_files,
        fb_files=fb_files,
        robust_npz_files=robust_npz_files,
        sampling_overrides=sampling_overrides,
        plan=plan,
        fbgate=fbgate,
        trace_len=trace_len,
        time_len=time_len,
        center_index=center_index,
        window_center_npz_key=window_center_npz_key,
        window_center_fallback_npz_key=window_center_fallback_npz_key,
        window_center_valid_mask_npz_key=window_center_valid_mask_npz_key,
        center_augment=center_augment,
        standardize_eps=standardize_eps,
        trace_decimate_prob=trace_decimate_prob,
        trace_decimate_stride_range=trace_decimate_stride_range,
        primary_keys=primary_keys,
        secondary_key_fixed=secondary_key_fixed,
        verbose=verbose,
        progress=progress,
        max_trials=max_trials,
        use_header_cache=use_header_cache,
        waveform_mode=waveform_mode,
        segy_endian=segy_endian,
    )


def build_labeled_infer_dataset(
    *,
    segy_files: list[str],
    fb_files: list[str],
    robust_npz_files: list[str],
    sampling_overrides: list[dict[str, object] | None] | None,
    plan: BuildPlan,
    fbgate,
    trace_len: int,
    time_len: int,
    center_index: int,
    standardize_eps: float,
    primary_keys: Sequence[str],
    secondary_key_fixed: bool,
    verbose: bool,
    progress: bool,
    max_trials: int,
    use_header_cache: bool,
    waveform_mode: str,
    segy_endian: str,
    window_center_npz_key: str = 'robust_pick_i',
    window_center_fallback_npz_key: str | None = None,
    window_center_valid_mask_npz_key: str | None = None,
) -> SegyGatherPipelineDataset:
    return _build_labeled_dataset(
        segy_files=segy_files,
        fb_files=fb_files,
        robust_npz_files=robust_npz_files,
        sampling_overrides=sampling_overrides,
        plan=plan,
        fbgate=fbgate,
        trace_len=trace_len,
        time_len=time_len,
        center_index=center_index,
        window_center_npz_key=window_center_npz_key,
        window_center_fallback_npz_key=window_center_fallback_npz_key,
        window_center_valid_mask_npz_key=window_center_valid_mask_npz_key,
        standardize_eps=standardize_eps,
        trace_decimate_prob=0.0,
        trace_decimate_stride_range=(1, 1),
        primary_keys=primary_keys,
        secondary_key_fixed=secondary_key_fixed,
        verbose=verbose,
        progress=progress,
        max_trials=max_trials,
        use_header_cache=use_header_cache,
        waveform_mode=waveform_mode,
        segy_endian=segy_endian,
    )


def build_raw_infer_dataset(
    *,
    segy_files: list[str],
    robust_npz_files: list[str],
    plan: BuildPlan | InputOnlyPlan,
    trace_len: int,
    overlap_h: int,
    time_len: int,
    center_index: int,
    standardize_eps: float,
    waveform_mode: str,
    segy_endian: str,
    use_header_cache: bool,
    window_center_npz_key: str = 'robust_pick_i',
    window_center_fallback_npz_key: str | None = None,
    window_center_valid_mask_npz_key: str | None = None,
) -> FineInferenceGatherWindowsDataset:
    stride_traces = int(trace_len) - int(overlap_h)
    if stride_traces <= 0:
        msg = 'trace_len - overlap_h must be positive'
        raise ValueError(msg)
    if len(segy_files) != len(robust_npz_files):
        msg = 'segy_files and robust_npz_files must have the same length'
        raise ValueError(msg)
    validate_files_exist(list(segy_files) + list(robust_npz_files))

    cfg = InferenceGatherWindowsConfig(
        domains=('shot',),
        secondary_sort={'shot': 'chno', 'recv': 'ffid', 'cmp': 'offset'},
        win_size_traces=int(trace_len),
        stride_traces=int(stride_traces),
        pad_last=True,
        target_len=int(time_len),
    )
    return FineInferenceGatherWindowsDataset(
        segy_files=list(segy_files),
        fb_files=None,
        robust_npz_files=list(robust_npz_files),
        center_index=int(center_index),
        window_center_npz_key=str(window_center_npz_key),
        window_center_fallback_npz_key=window_center_fallback_npz_key,
        window_center_valid_mask_npz_key=window_center_valid_mask_npz_key,
        plan=plan,
        cfg=cfg,
        transform=build_infer_transform(standardize_eps=float(standardize_eps)),
        waveform_mode=str(waveform_mode),
        segy_endian=str(segy_endian),
        use_header_cache=bool(use_header_cache),
    )


def collate_input_meta_list(batch: Sequence[dict]) -> tuple[torch.Tensor, list[dict]]:
    if len(batch) == 0:
        msg = 'empty batch'
        raise ValueError(msg)
    xs = [item['input'] for item in batch]
    metas = [item['meta'] for item in batch]
    if not all(isinstance(x, torch.Tensor) for x in xs):
        msg = "batch['input'] must be torch.Tensor"
        raise TypeError(msg)
    x0_shape = tuple(xs[0].shape)
    for x in xs:
        if tuple(x.shape) != x0_shape:
            msg = 'all fine infer inputs must share the same shape for stack-collate'
            raise ValueError(msg)
    return torch.stack(xs, dim=0), metas
