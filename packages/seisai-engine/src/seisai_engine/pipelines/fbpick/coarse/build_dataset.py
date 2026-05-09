from __future__ import annotations

import contextlib
import random
from collections.abc import Sequence

import numpy as np
import segyio
import torch
from seisai_dataset import (
    BuildPlan,
    FirstBreakGate,
    FirstBreakGateConfig,
    InputOnlyPlan,
    SegyGatherPipelineDataset,
)
from seisai_dataset.config import LoaderConfig, TraceSubsetSamplerConfig
from seisai_dataset.file_info import build_file_info
from seisai_dataset.geometry_headers import validate_coord_unit_scale_to_m
from seisai_dataset.trace_subset_preproc import TraceSubsetLoader
from seisai_dataset.trace_subset_sampler import TraceSubsetSampler
from seisai_dataset.transform_flow_utils import (
    apply_transform_2d_with_meta,
)
from seisai_transforms.augment import PerTraceStandardize, ViewCompose
from seisai_utils.fs import validate_files_exist
from torch.utils.data import Dataset

from .config import COARSE_TIME_LEN, COARSE_TRACE_LEN
from .time_axis import (
    build_coarse_fb_labels_for_anchors,
    build_coarse_time_grid,
    resample_waveform_time_axis,
)
from .trace_anchor import select_trace_anchors

__all__ = [
    'GlobalAnchorCoarseDataset',
    'GlobalAnchorCoarseRawInferDataset',
    'build_fbgate',
    'build_infer_transform',
    'build_labeled_infer_dataset',
    'build_raw_infer_dataset',
    'build_train_dataset',
    'build_train_transform',
    'collate_input_meta_list',
]


class _NoRandRNG:
    def random(self, *args, **kwargs):
        msg = 'random() is not allowed in global-anchor coarse inference'
        raise RuntimeError(msg)

    def uniform(self, *args, **kwargs):
        msg = 'uniform() is not allowed in global-anchor coarse inference'
        raise RuntimeError(msg)

    def integers(self, *args, **kwargs):
        msg = 'integers() is not allowed in global-anchor coarse inference'
        raise RuntimeError(msg)


class GlobalAnchorCoarseDataset(SegyGatherPipelineDataset):
    def __init__(
        self,
        *,
        segy_files: list[str],
        fb_files: list[str],
        sampling_overrides: list[dict[str, object] | None] | None,
        plan: BuildPlan,
        fbgate: FirstBreakGate,
        trace_len: int,
        time_len: int,
        standardize_eps: float,
        anchor_mode: str,
        gap_ratio: float,
        min_gap_m: float | None,
        primary_keys: Sequence[str],
        secondary_key_fixed: bool,
        verbose: bool,
        progress: bool,
        max_trials: int,
        use_header_cache: bool,
        waveform_mode: str,
        segy_endian: str,
    ) -> None:
        trace_len_int = int(trace_len)
        time_len_int = int(time_len)
        if trace_len_int != COARSE_TRACE_LEN:
            msg = f'trace_len must be {COARSE_TRACE_LEN} for global-anchor coarse'
            raise ValueError(msg)
        if time_len_int != COARSE_TIME_LEN:
            msg = f'time_len must be {COARSE_TIME_LEN} for global-anchor coarse'
            raise ValueError(msg)

        mode = str(anchor_mode).strip().lower()
        if mode not in ('random', 'center'):
            msg = 'anchor_mode must be "random" or "center"'
            raise ValueError(msg)

        self.trace_len = trace_len_int
        self.time_len = time_len_int
        self.anchor_mode = mode
        self.gap_ratio = float(gap_ratio)
        self.min_gap_m = None if min_gap_m is None else float(min_gap_m)
        self._standardize_transform = build_train_transform(
            standardize_eps=float(standardize_eps)
        )

        super().__init__(
            segy_files=list(segy_files),
            fb_files=list(fb_files),
            sampling_overrides=sampling_overrides,
            transform=self._standardize_transform,
            fbgate=fbgate,
            plan=plan,
            subset_traces=trace_len_int,
            trace_decimate_prob=0.0,
            trace_decimate_stride_range=(1, 1),
            primary_keys=tuple(primary_keys),
            secondary_key_fixed=bool(secondary_key_fixed),
            waveform_mode=str(waveform_mode),
            segy_endian=str(segy_endian),
            verbose=bool(verbose),
            progress=bool(progress),
            max_trials=int(max_trials),
            use_header_cache=bool(use_header_cache),
        )

    def __getitem__(self, index: int | None = None) -> dict:
        if self.anchor_mode != 'center':
            return self._sample_with_retries()
        idx = 0 if index is None else int(index)
        rejections = self._init_rejection_counters()
        for attempt in range(self.max_trials):
            info = self._choose_file_info_for_index(idx + attempt)
            out = self._try_build_sample_for_index(info, idx + attempt, rejections)
            if out is not None:
                return out
        raise RuntimeError(self._format_max_trials_error(rejections))

    def _choose_file_info_for_index(self, index: int):
        if not self.file_infos:
            msg = 'file_infos is empty'
            raise RuntimeError(msg)
        return self.file_infos[int(index) % len(self.file_infos)]

    def _draw_full_gather_random(self, info) -> dict:
        seed = int(self._rng.integers(0, 2**31 - 1))
        return self.sampler.draw_full_gather(info, py_random=random.Random(seed))

    def _draw_full_gather_for_index(self, info, index: int) -> dict:
        seed = int(index) // max(1, len(self.file_infos))
        return self.sampler.draw_full_gather(info, py_random=random.Random(seed))

    def _try_build_sample(self, info, counters: dict[str, int]) -> dict | None:
        sample = self._draw_full_gather_random(info)
        return self._build_global_anchor_sample(
            info=info,
            sample=sample,
            anchor_rng=self._rng,
            counters=counters,
        )

    def _try_build_sample_for_index(
        self,
        info,
        index: int,
        counters: dict[str, int],
    ) -> dict | None:
        sample = self._draw_full_gather_for_index(info, index)
        return self._build_global_anchor_sample(
            info=info,
            sample=sample,
            anchor_rng=None,
            counters=counters,
        )

    def _build_global_anchor_sample(
        self,
        *,
        info,
        sample: dict,
        anchor_rng: np.random.Generator | None,
        counters: dict[str, int],
    ) -> dict | None:
        indices_full = np.asarray(sample['indices'], dtype=np.int64)
        if indices_full.ndim != 1 or indices_full.size == 0:
            self._count_rejection(counters, 'empty')
            return None
        if info.fb is None:
            msg = 'fb labels are required for global-anchor coarse dataset'
            raise ValueError(msg)

        fb_full = np.asarray(info.fb[indices_full], dtype=np.int64)
        apply_fb_gates = self._should_apply_fb_gates({})
        if apply_fb_gates and (not self.gate_evaluator.min_pick_accept(fb_full)):
            self._count_rejection(counters, 'min_pick')
            return None

        offsets_full = np.asarray(info.offsets[indices_full], dtype=np.float32)
        selection = select_trace_anchors(
            raw_indices=indices_full,
            offsets_m=offsets_full,
            trace_len=self.trace_len,
            mode=self.anchor_mode,
            gap_ratio=self.gap_ratio,
            min_gap_m=self.min_gap_m,
            rng=anchor_rng,
        )
        trace_valid = np.asarray(selection.trace_valid, dtype=np.bool_)
        anchor_raw_indices = np.asarray(selection.anchor_raw_indices, dtype=np.int64)
        anchor_source_pos = np.asarray(selection.anchor_source_pos, dtype=np.int64)
        anchor_offsets_m = np.asarray(selection.anchor_offsets_m, dtype=np.float32)

        valid_indices = anchor_raw_indices[trace_valid]
        subset_loader = TraceSubsetLoader(
            LoaderConfig(pad_traces_to=int(valid_indices.size))
        )
        x_valid = subset_loader.load_traces(info.mmap, valid_indices)
        x_resampled_valid = resample_waveform_time_axis(
            x_valid,
            out_time_len=self.time_len,
        )
        x_view = np.zeros((self.trace_len, self.time_len), dtype=np.float32)
        x_view[trace_valid] = x_resampled_valid
        x_view, transform_meta = apply_transform_2d_with_meta(
            self._standardize_transform,
            x_view,
            self._rng,
            msg_bad_out='global-anchor transform must return 2D numpy or (2D, meta)',
            msg_bad_meta='global-anchor transform meta must be dict, got {type}',
            exc_bad_out=ValueError,
            exc_bad_meta=TypeError,
        )
        if x_view.shape != (self.trace_len, self.time_len):
            msg = (
                'global-anchor transform must keep fixed shape '
                f'{(self.trace_len, self.time_len)}, got {x_view.shape}'
            )
            raise ValueError(msg)
        x_view = np.ascontiguousarray(x_view, dtype=np.float32)
        x_view[~trace_valid] = 0.0

        fb_idx_raw_for_anchors = np.full((self.trace_len,), -1, dtype=np.int64)
        valid_source_pos = np.asarray(
            anchor_source_pos[trace_valid],
            dtype=np.int64,
        )
        fb_idx_raw_for_anchors[trace_valid] = fb_full[valid_source_pos]
        fb_idx_coarse_for_anchors = build_coarse_fb_labels_for_anchors(
            fb_idx_raw_for_anchors,
            trace_valid,
            raw_time_len=int(info.n_samples),
            coarse_time_len=self.time_len,
        )

        grid = build_coarse_time_grid(
            raw_time_len=int(info.n_samples),
            coarse_time_len=self.time_len,
            dt_sec=float(info.dt_sec),
        )
        meta = {
            'hflip': False,
            'factor_h': 1.0,
            'factor': float(grid.raw_to_coarse_factor),
            'start': 0,
            'trace_valid': trace_valid,
            'fb_idx': fb_idx_raw_for_anchors,
            'fb_idx_view': fb_idx_coarse_for_anchors,
            'offsets_view': anchor_offsets_m,
            'time_view': np.asarray(grid.time_view_sec, dtype=np.float32),
            'dt_sec': np.float32(grid.dt_sec),
            'dt_eff_sec': np.float32(grid.dt_eff_sec),
            'raw_time_len': int(grid.raw_time_len),
            'coarse_time_len': int(grid.coarse_time_len),
            'raw_to_coarse_factor': np.float32(grid.raw_to_coarse_factor),
            'coarse_to_raw_factor': np.float32(grid.coarse_to_raw_factor),
            'segment_id': np.asarray(selection.segment_id, dtype=np.int64),
            'anchor_bin_start_pos': np.asarray(
                selection.anchor_bin_start_pos,
                dtype=np.int64,
            ),
            'anchor_bin_stop_pos': np.asarray(
                selection.anchor_bin_stop_pos,
                dtype=np.int64,
            ),
            'fb_idx_raw_for_anchors': fb_idx_raw_for_anchors,
            'fb_idx_coarse_for_anchors': fb_idx_coarse_for_anchors,
            'key_name': sample['key_name'],
            'primary_value': int(sample['primary_value']),
            'primary_unique': sample['primary_unique'],
            'secondary_key': sample['secondary_key'],
        }
        meta.update(transform_meta)

        if apply_fb_gates:
            if not self.gate_evaluator.apply_gates(
                meta,
                did_super=bool(sample['did_super']),
                info=info,
            ):
                if self.gate_evaluator.last_reject == 'fblc':
                    self._count_rejection(counters, 'fblc')
                return None

        sample_for_plan = self.sample_flow.build_plan_input_base(
            meta=meta,
            dt_sec=float(grid.dt_eff_sec),
            offsets=anchor_offsets_m,
            indices=anchor_raw_indices,
            key_name=str(sample['key_name']),
            secondary_key=str(sample['secondary_key']),
            primary_unique=str(sample['primary_unique']),
            extra={
                'x_view': x_view,
                'fb_idx': fb_idx_raw_for_anchors,
                'file_path': info.path,
                'trace_valid': trace_valid,
            },
        )
        self.sample_flow.run_plan(sample_for_plan, rng=self._rng)

        # ``indices`` is the generic dataset convention for selected raw trace ids.
        # Keep ``anchor_raw_indices`` as the explicit global-anchor debug alias.
        out = self.sample_flow.build_output_base(
            sample_for_plan,
            meta=meta,
            dt_sec=float(grid.dt_eff_sec),
            offsets=anchor_offsets_m,
            indices=anchor_raw_indices,
            key_name=str(sample['key_name']),
            secondary_key=str(sample['secondary_key']),
            primary_unique=str(sample['primary_unique']),
            extra={
                'fb_idx': torch.from_numpy(fb_idx_raw_for_anchors),
                'fb_idx_coarse': torch.from_numpy(fb_idx_coarse_for_anchors),
                'file_path': info.path,
                'did_superwindow': bool(sample['did_super']),
                'anchor_raw_indices': torch.from_numpy(anchor_raw_indices),
                'anchor_source_pos': torch.from_numpy(anchor_source_pos),
                'anchor_offsets_m': torch.from_numpy(anchor_offsets_m),
            },
        )
        self._add_trace_valid_output(out, trace_valid)
        self._add_mask_bool_output(out, sample_for_plan)
        self._finalize_output(out, sample_for_plan, {}, x_view.shape)
        return out


class GlobalAnchorCoarseRawInferDataset(Dataset):
    def __init__(
        self,
        *,
        segy_files: Sequence[str],
        plan: BuildPlan | InputOnlyPlan,
        trace_len: int,
        time_len: int,
        standardize_eps: float,
        gap_ratio: float,
        min_gap_m: float | None,
        primary_keys: Sequence[str],
        waveform_mode: str,
        segy_endian: str,
        use_header_cache: bool,
        coord_unit_scale_to_m: float = 1.0,
    ) -> None:
        trace_len_int = int(trace_len)
        time_len_int = int(time_len)
        if trace_len_int != COARSE_TRACE_LEN:
            msg = f'trace_len must be {COARSE_TRACE_LEN} for global-anchor coarse'
            raise ValueError(msg)
        if time_len_int != COARSE_TIME_LEN:
            msg = f'time_len must be {COARSE_TIME_LEN} for global-anchor coarse'
            raise ValueError(msg)

        primary_keys_tuple = tuple(str(key) for key in primary_keys)
        if len(primary_keys_tuple) != 1:
            msg = 'global-anchor coarse raw inference requires exactly one primary key'
            raise ValueError(msg)

        if isinstance(plan, BuildPlan):
            self.plan = InputOnlyPlan.from_build_plan(plan, include_label_ops=False)
        elif isinstance(plan, InputOnlyPlan):
            self.plan = plan
        else:
            msg = 'plan must be BuildPlan or InputOnlyPlan'
            raise TypeError(msg)

        self.segy_files = list(segy_files)
        if not self.segy_files:
            msg = 'segy_files must be non-empty'
            raise ValueError(msg)
        self.trace_len = trace_len_int
        self.time_len = time_len_int
        self.gap_ratio = float(gap_ratio)
        self.min_gap_m = None if min_gap_m is None else float(min_gap_m)
        self.primary_key = primary_keys_tuple[0]
        self.coord_unit_scale_to_m = validate_coord_unit_scale_to_m(
            coord_unit_scale_to_m
        )
        self.transform = build_infer_transform(standardize_eps=float(standardize_eps))
        self._rng = _NoRandRNG()
        self._subset_loader = TraceSubsetLoader(
            LoaderConfig(pad_traces_to=self.trace_len)
        )
        self.sampler = TraceSubsetSampler(
            TraceSubsetSamplerConfig(
                primary_keys=(self.primary_key,),
                primary_key_weights=None,
                use_superwindow=False,
                sw_halfspan=0,
                sw_prob=0.0,
                secondary_key_fixed=True,
                subset_traces=self.trace_len,
                trace_decimate_prob=0.0,
                trace_decimate_stride_range=(1, 1),
            )
        )

        self.file_infos: list[dict] = [
            build_file_info(
                segy_path,
                ffid_byte=segyio.TraceField.FieldRecord,
                chno_byte=segyio.TraceField.TraceNumber,
                cmp_byte=segyio.TraceField.CDP,
                header_cache_dir=None,
                use_header_cache=bool(use_header_cache),
                include_centroids=False,
                include_geometry_arrays=True,
                coord_unit_scale_to_m=self.coord_unit_scale_to_m,
                waveform_mode=str(waveform_mode),
                segy_endian=str(segy_endian),
            )
            for segy_path in self.segy_files
        ]
        self.items: list[tuple[int, str, int]] = []
        self._build_items()

    def _build_items(self) -> None:
        for file_idx, info in enumerate(self.file_infos):
            unique_keys = info.get(f'{self.primary_key}_unique_keys')
            if not unique_keys:
                msg = f'no {self.primary_key} gathers found in {info["path"]}'
                raise RuntimeError(msg)
            for primary_value in sorted(int(value) for value in unique_keys):
                self.items.append((int(file_idx), self.primary_key, int(primary_value)))
        if not self.items:
            msg = 'global-anchor coarse raw inference found no gathers'
            raise RuntimeError(msg)

    def close(self) -> None:
        for info in self.file_infos:
            segy_obj = info.get('segy_obj')
            if segy_obj is not None:
                with contextlib.suppress(Exception):
                    segy_obj.close()
        self.file_infos.clear()
        self.items.clear()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def __len__(self) -> int:
        return len(self.items)

    def _draw_item_full_gather(
        self,
        *,
        info: dict,
        key_name: str,
        primary_value: int,
    ) -> dict:
        info_for_sample = dict(info)
        info_for_sample['sampling_override'] = {
            'primary_keys': [key_name],
            'primary_ranges': {key_name: [int(primary_value), int(primary_value)]},
            'secondary_key_fixed': True,
        }
        return self.sampler.draw_full_gather(
            info_for_sample,
            py_random=random.Random(0),
        )

    def __getitem__(self, index: int) -> dict:
        file_idx, key_name, primary_value = self.items[int(index)]
        info = self.file_infos[file_idx]
        sample = self._draw_item_full_gather(
            info=info,
            key_name=key_name,
            primary_value=primary_value,
        )
        return self._build_sample(info=info, file_idx=file_idx, sample=sample)

    def _build_sample(self, *, info: dict, file_idx: int, sample: dict) -> dict:
        raw_trace_indices = np.asarray(sample['indices'], dtype=np.int64)
        if raw_trace_indices.ndim != 1 or raw_trace_indices.size == 0:
            msg = 'draw_full_gather returned an empty raw gather'
            raise RuntimeError(msg)

        offsets_full = np.asarray(info['offsets'][raw_trace_indices], dtype=np.float32)
        selection = select_trace_anchors(
            raw_indices=raw_trace_indices,
            offsets_m=offsets_full,
            trace_len=self.trace_len,
            mode='center',
            gap_ratio=self.gap_ratio,
            min_gap_m=self.min_gap_m,
            rng=None,
        )

        trace_valid = np.asarray(selection.trace_valid, dtype=np.bool_)
        anchor_raw_indices = np.asarray(selection.anchor_raw_indices, dtype=np.int64)
        anchor_source_pos = np.asarray(selection.anchor_source_pos, dtype=np.int64)
        anchor_offsets_m = np.asarray(selection.anchor_offsets_m, dtype=np.float32)

        valid_indices = anchor_raw_indices[trace_valid]
        x_valid = self._subset_loader.load_traces(info['mmap'], valid_indices)
        x_resampled_valid = resample_waveform_time_axis(
            x_valid,
            out_time_len=self.time_len,
        )
        x_view = np.zeros((self.trace_len, self.time_len), dtype=np.float32)
        x_view[trace_valid] = x_resampled_valid
        x_view, transform_meta = apply_transform_2d_with_meta(
            self.transform,
            x_view,
            self._rng,
            msg_bad_out=(
                'global-anchor raw infer transform must return 2D numpy or (2D, meta)'
            ),
            msg_bad_meta=(
                'global-anchor raw infer transform meta must be dict, got {type}'
            ),
            exc_bad_out=ValueError,
            exc_bad_meta=TypeError,
        )
        if x_view.shape != (self.trace_len, self.time_len):
            msg = (
                'global-anchor raw infer transform must keep fixed shape '
                f'{(self.trace_len, self.time_len)}, got {x_view.shape}'
            )
            raise ValueError(msg)
        x_view = np.ascontiguousarray(x_view, dtype=np.float32)
        x_view[~trace_valid] = 0.0

        grid = build_coarse_time_grid(
            raw_time_len=int(info['n_samples']),
            coarse_time_len=self.time_len,
            dt_sec=float(info['dt_sec']),
        )
        meta = {
            'source_file': str(info['path']),
            'file_path': str(info['path']),
            'file_idx': int(file_idx),
            'key_name': str(sample['key_name']),
            'primary_value': int(sample['primary_value']),
            'primary_unique': str(sample['primary_unique']),
            'secondary_key': str(sample['secondary_key']),
            'raw_trace_indices': raw_trace_indices,
            'anchor_raw_indices': anchor_raw_indices,
            'anchor_source_pos': anchor_source_pos,
            'anchor_offsets_m': anchor_offsets_m,
            'trace_valid': trace_valid,
            'segment_id': np.asarray(selection.segment_id, dtype=np.int64),
            'segments': selection.segments,
            'anchor_bin_start_pos': np.asarray(
                selection.anchor_bin_start_pos,
                dtype=np.int64,
            ),
            'anchor_bin_stop_pos': np.asarray(
                selection.anchor_bin_stop_pos,
                dtype=np.int64,
            ),
            'raw_time_len': int(grid.raw_time_len),
            'coarse_time_len': int(grid.coarse_time_len),
            'dt_sec': np.float32(grid.dt_sec),
            'dt_eff_sec': np.float32(grid.dt_eff_sec),
            'raw_to_coarse_factor': np.float32(grid.raw_to_coarse_factor),
            'coarse_to_raw_factor': np.float32(grid.coarse_to_raw_factor),
            'offsets_view': anchor_offsets_m,
            'time_view': np.asarray(grid.time_view_sec, dtype=np.float32),
            'offsets_m': offsets_full,
            'ffid_values': np.asarray(info['ffid_values'], dtype=np.int32),
            'chno_values': np.asarray(info['chno_values'], dtype=np.int32),
            'n_traces': int(info['n_traces']),
            'n_samples_orig': int(info['n_samples']),
            'hflip': False,
            'factor_h': 1.0,
            'factor': float(grid.raw_to_coarse_factor),
            'start': 0,
        }
        meta.update(transform_meta)

        sample_for_plan = {
            'x_view': x_view,
            'meta': meta,
            'dt_sec': float(grid.dt_eff_sec),
            'offsets': anchor_offsets_m,
            'indices': anchor_raw_indices,
            'key_name': str(sample['key_name']),
            'secondary_key': str(sample['secondary_key']),
            'primary_unique': str(sample['primary_unique']),
            'file_path': str(info['path']),
            'trace_valid': trace_valid,
        }
        self.plan.run(sample_for_plan, rng=self._rng)
        if 'input' not in sample_for_plan:
            msg = "plan must populate 'input'"
            raise KeyError(msg)
        x_in = sample_for_plan['input']
        if not isinstance(x_in, torch.Tensor):
            msg = "sample['input'] must be torch.Tensor"
            raise TypeError(msg)
        if tuple(x_in.shape) != (3, self.trace_len, self.time_len):
            msg = (
                'global-anchor raw infer input must have shape '
                f'{(3, self.trace_len, self.time_len)}, got {tuple(x_in.shape)}'
            )
            raise ValueError(msg)
        return {
            'input': x_in,
            'meta': meta,
        }


def build_train_transform(*, standardize_eps: float) -> ViewCompose:
    return ViewCompose([PerTraceStandardize(eps=float(standardize_eps))])


def build_infer_transform(*, standardize_eps: float) -> ViewCompose:
    return ViewCompose([PerTraceStandardize(eps=float(standardize_eps))])


def build_fbgate(
    *,
    apply_on: str,
    min_pick_ratio: float,
    verbose: bool,
) -> FirstBreakGate:
    ap = str(apply_on).lower()
    if ap == 'on':
        ap = 'any'
    if ap not in ('any', 'super_only', 'off'):
        msg = 'fbgate.apply_on must be "any", "super_only", or "off"'
        raise ValueError(msg)
    return FirstBreakGate(
        FirstBreakGateConfig(
            apply_on=ap,
            min_pick_ratio=float(min_pick_ratio),
            verbose=bool(verbose),
        )
    )


def build_train_dataset(
    *,
    segy_files: list[str],
    fb_files: list[str],
    sampling_overrides: list[dict[str, object] | None] | None,
    plan: BuildPlan,
    fbgate: FirstBreakGate,
    trace_len: int,
    time_len: int,
    standardize_eps: float,
    anchor_mode: str,
    gap_ratio: float,
    min_gap_m: float | None,
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
) -> GlobalAnchorCoarseDataset:
    if float(trace_decimate_prob) != 0.0:
        msg = 'train.trace_decimation is not supported for global-anchor fbpick coarse'
        raise ValueError(msg)
    if tuple(trace_decimate_stride_range) != (1, 1):
        msg = 'train.trace_decimation.stride_range must be [1, 1] for fbpick coarse'
        raise ValueError(msg)
    if str(anchor_mode) != 'random':
        msg = 'train global-anchor coarse dataset requires anchor_mode="random"'
        raise ValueError(msg)
    validate_files_exist(list(segy_files) + list(fb_files))
    return GlobalAnchorCoarseDataset(
        segy_files=segy_files,
        fb_files=fb_files,
        sampling_overrides=sampling_overrides,
        plan=plan,
        fbgate=fbgate,
        trace_len=trace_len,
        time_len=time_len,
        standardize_eps=standardize_eps,
        anchor_mode=anchor_mode,
        gap_ratio=gap_ratio,
        min_gap_m=min_gap_m,
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
    sampling_overrides: list[dict[str, object] | None] | None,
    plan: BuildPlan,
    fbgate: FirstBreakGate,
    trace_len: int,
    time_len: int,
    standardize_eps: float,
    anchor_mode: str,
    gap_ratio: float,
    min_gap_m: float | None,
    primary_keys: Sequence[str],
    secondary_key_fixed: bool,
    verbose: bool,
    progress: bool,
    max_trials: int,
    use_header_cache: bool,
    waveform_mode: str,
    segy_endian: str,
) -> GlobalAnchorCoarseDataset:
    if str(anchor_mode) != 'center':
        msg = 'validation global-anchor coarse dataset requires anchor_mode="center"'
        raise ValueError(msg)
    validate_files_exist(list(segy_files) + list(fb_files))
    return GlobalAnchorCoarseDataset(
        segy_files=segy_files,
        fb_files=fb_files,
        sampling_overrides=sampling_overrides,
        plan=plan,
        fbgate=fbgate,
        trace_len=trace_len,
        time_len=time_len,
        standardize_eps=standardize_eps,
        anchor_mode=anchor_mode,
        gap_ratio=gap_ratio,
        min_gap_m=min_gap_m,
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
    plan: BuildPlan | InputOnlyPlan,
    trace_len: int,
    time_len: int,
    standardize_eps: float,
    gap_ratio: float,
    min_gap_m: float | None,
    primary_keys: Sequence[str],
    waveform_mode: str,
    segy_endian: str,
    use_header_cache: bool,
    coord_unit_scale_to_m: float = 1.0,
) -> GlobalAnchorCoarseRawInferDataset:
    validate_files_exist(list(segy_files))
    return GlobalAnchorCoarseRawInferDataset(
        segy_files=list(segy_files),
        plan=plan,
        trace_len=int(trace_len),
        time_len=int(time_len),
        standardize_eps=float(standardize_eps),
        gap_ratio=float(gap_ratio),
        min_gap_m=min_gap_m,
        primary_keys=tuple(primary_keys),
        waveform_mode=str(waveform_mode),
        segy_endian=str(segy_endian),
        use_header_cache=bool(use_header_cache),
        coord_unit_scale_to_m=coord_unit_scale_to_m,
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
            msg = 'all global-anchor coarse infer inputs must share shape'
            raise ValueError(msg)
    return torch.stack(xs, dim=0), metas
