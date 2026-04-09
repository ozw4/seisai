from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from seisai_dataset import (
    BuildPlan,
    FirstBreakGate,
    FirstBreakGateConfig,
    InferenceGatherWindowsConfig,
    InferenceGatherWindowsDataset,
    InputOnlyPlan,
    SegyGatherPipelineDataset,
)
from seisai_dataset.config import LoaderConfig
from seisai_dataset.sample_flow import SampleFlow
from seisai_dataset.segy_gather_base import SampleTransformer
from seisai_dataset.trace_subset_preproc import TraceSubsetLoader
from seisai_dataset.transform_flow_utils import (
    add_view_projection_meta,
    apply_transform_2d_with_meta,
    pad_indices_offsets_fb,
)
from seisai_transforms.augment import PerTraceStandardize, ViewCompose
from seisai_utils.fs import validate_files_exist

__all__ = [
    'PickAwareCropSampleTransformer',
    'build_fbgate',
    'build_infer_transform',
    'build_labeled_infer_dataset',
    'build_raw_infer_dataset',
    'build_train_dataset',
    'build_train_transform',
]


class PickAwareCropSampleTransformer(SampleTransformer):
    def __init__(
        self,
        subsetloader: TraceSubsetLoader,
        transform,
        *,
        target_len: int,
        start_mode: str,
    ) -> None:
        super().__init__(subsetloader, transform)
        self.target_len = int(target_len)
        if self.target_len <= 0:
            msg = 'target_len must be positive'
            raise ValueError(msg)
        mode = str(start_mode).strip().lower()
        if mode not in ('random', 'midpoint'):
            msg = 'start_mode must be "random" or "midpoint"'
            raise ValueError(msg)
        self.start_mode = mode

    def _choose_start(
        self,
        *,
        fb_subset: np.ndarray,
        trace_valid: np.ndarray,
        W0: int,
        rng: np.random.Generator,
    ) -> int | None:
        valid = np.asarray(trace_valid, dtype=np.bool_) & (fb_subset > 0) & (fb_subset < W0)
        if not np.any(valid):
            return None

        picks = np.asarray(fb_subset[valid], dtype=np.int64)
        start_lo = max(0, int(picks.max()) - (self.target_len - 1))
        start_hi = min(int(picks.min()) - 1, int(W0) - self.target_len)
        if start_lo > start_hi:
            return None
        if self.start_mode == 'random':
            return int(rng.integers(start_lo, start_hi + 1))
        return int((start_lo + start_hi) // 2)

    def _apply_post_transform(
        self,
        x_view: np.ndarray,
        meta: dict,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict]:
        if self.transform is None:
            return x_view, meta

        x_post, post_meta = apply_transform_2d_with_meta(
            self.transform,
            x_view,
            rng,
            msg_bad_out='pick-aware transform must return 2D numpy or (2D, meta)',
            msg_bad_meta='pick-aware transform meta must be dict, got {type}',
            exc_bad_out=ValueError,
            exc_bad_meta=TypeError,
        )
        if x_post.shape != x_view.shape:
            msg = 'pick-aware post transform must keep the cropped/padded shape'
            raise ValueError(msg)
        meta.update(post_meta)
        return x_post, meta

    def load_transform_or_reject(
        self,
        info,
        indices: np.ndarray,
        fb_subset: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        x = self.subsetloader.load(info.mmap, indices)
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            msg = 'TraceSubsetLoader must return 2D numpy array'
            raise TypeError(msg)

        H = int(x.shape[0])
        W0 = int(x.shape[1])
        offsets = info.offsets[indices].astype(np.float32, copy=False)
        indices_pad, offsets, fb_subset_pad, trace_valid, _pad = pad_indices_offsets_fb(
            indices=indices,
            offsets=offsets,
            fb_subset=fb_subset,
            H=H,
        )
        if fb_subset_pad is None:
            msg = 'fb_subset must be provided for pick-aware crop'
            raise RuntimeError(msg)

        meta = {
            'hflip': False,
            'factor_h': 1.0,
            'factor': 1.0,
            'start': 0,
        }

        if W0 > self.target_len:
            start = self._choose_start(
                fb_subset=fb_subset_pad,
                trace_valid=trace_valid,
                W0=W0,
                rng=rng,
            )
            if start is None:
                return None
            meta['start'] = int(start)
            x_view = x[:, start : start + self.target_len]
        else:
            x_view = np.zeros((H, self.target_len), dtype=x.dtype)
            x_view[:, :W0] = x

        x_view, meta = self._apply_post_transform(x_view, meta, rng)
        add_view_projection_meta(
            meta,
            trace_valid=trace_valid,
            fb_idx=fb_subset_pad,
            offsets=offsets,
            dt_sec=float(info.dt_sec),
            W0=W0,
            H=H,
            W=int(x_view.shape[1]),
        )
        return x_view, meta, offsets, fb_subset_pad, indices_pad, trace_valid


class _PickAwareSegyGatherPipelineDataset(SegyGatherPipelineDataset):
    def _init_rejection_counters(self) -> dict[str, int]:
        counters = super()._init_rejection_counters()
        counters['pick_crop'] = 0
        return counters

    def _format_max_trials_error(self, counters: dict[str, int]) -> str:
        parts = []
        if 'empty' in counters:
            parts.append(f'empty={counters["empty"]}')
        if 'min_pick' in counters:
            parts.append(f'min_pick={counters["min_pick"]}')
        if 'fblc' in counters:
            parts.append(f'fblc={counters["fblc"]}')
        if 'pick_crop' in counters:
            parts.append(f'pick_crop={counters["pick_crop"]}')
        rej = ', '.join(parts)
        return (
            f'failed to draw a valid sample within max_trials={self.max_trials}; '
            f'rejections: {rej}, files={len(self.file_infos)}'
        )

    def _try_build_sample(self, info, counters: dict[str, int]) -> dict | None:
        sample = self._draw_sample(info)
        indices = sample['indices']
        did_super = sample['did_super']

        label = self._load_fb_subset(info, indices, sample, counters)
        if label is None:
            return None
        fb_subset, label_state = label

        apply_fb_gates = self._should_apply_fb_gates(label_state)
        if apply_fb_gates and (not self.gate_evaluator.min_pick_accept(fb_subset)):
            self._count_rejection(counters, 'min_pick')
            return None

        transformer = self.sample_transformer
        if not isinstance(transformer, PickAwareCropSampleTransformer):
            msg = 'sample_transformer must be PickAwareCropSampleTransformer'
            raise TypeError(msg)
        transformed = transformer.load_transform_or_reject(
            info,
            indices,
            fb_subset,
            self._rng,
        )
        if transformed is None:
            self._count_rejection(counters, 'pick_crop')
            return None
        x_view, meta, offsets, fb_subset, indices_pad, trace_valid = transformed
        view_shape = self._get_view_shape(x_view, label_state)

        sample['indices'] = indices_pad
        sample['trace_valid'] = trace_valid
        meta['key_name'] = sample['key_name']
        meta['primary_unique'] = sample['primary_unique']
        sample['x_view'] = x_view

        self._post_transform_meta(
            meta=meta,
            label_state=label_state,
            trace_valid=trace_valid,
            view_shape=view_shape,
        )

        if apply_fb_gates:
            if not self.gate_evaluator.apply_gates(
                meta,
                did_super=did_super,
                info=info,
            ):
                if self.gate_evaluator.last_reject == 'fblc':
                    self._count_rejection(counters, 'fblc')
                return None
        else:
            self._set_dt_eff_meta_no_gate(meta, info=info)

        dt_eff_sec = float(meta.get('dt_eff_sec', info.dt_sec))
        sample_for_plan = self.sample_flow.build_plan_input_base(
            meta=meta,
            dt_sec=dt_eff_sec,
            offsets=offsets,
            indices=sample['indices'],
            key_name=sample['key_name'],
            secondary_key=sample['secondary_key'],
            primary_unique=sample['primary_unique'],
            extra=self._build_plan_extras(
                info=info,
                sample=sample,
                fb_subset=fb_subset,
                trace_valid=trace_valid,
                label_state=label_state,
            ),
        )
        self.sample_flow.run_plan(sample_for_plan, rng=self._rng)
        self._post_plan_validation(sample_for_plan, label_state, view_shape)

        out = self.sample_flow.build_output_base(
            sample_for_plan,
            meta=meta,
            dt_sec=dt_eff_sec,
            offsets=offsets,
            indices=sample['indices'],
            key_name=sample['key_name'],
            secondary_key=sample['secondary_key'],
            primary_unique=sample['primary_unique'],
            extra=self._build_output_extras(
                info=info,
                sample=sample,
                fb_subset=fb_subset,
                label_state=label_state,
            ),
        )
        self._add_trace_valid_output(out, trace_valid)
        self._add_mask_bool_output(out, sample_for_plan)
        self._finalize_output(out, sample_for_plan, label_state, view_shape)
        return out


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


def _build_pick_aware_dataset(
    *,
    segy_files: list[str],
    fb_files: list[str],
    sampling_overrides: list[dict[str, object] | None] | None,
    plan: BuildPlan,
    fbgate: FirstBreakGate,
    subset_traces: int,
    target_len: int,
    standardize_eps: float,
    start_mode: str,
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
) -> _PickAwareSegyGatherPipelineDataset:
    if sampling_overrides is not None and len(sampling_overrides) != len(segy_files):
        msg = 'sampling_overrides length must match segy_files length'
        raise ValueError(msg)
    if len(segy_files) != len(fb_files):
        msg = 'segy_files and fb_files must have the same length'
        raise ValueError(msg)
    validate_files_exist(list(segy_files) + list(fb_files))

    transform = build_train_transform(standardize_eps=float(standardize_eps))
    subsetloader = TraceSubsetLoader(LoaderConfig(pad_traces_to=int(subset_traces)))
    sample_transformer = PickAwareCropSampleTransformer(
        subsetloader,
        transform,
        target_len=int(target_len),
        start_mode=str(start_mode),
    )
    return _PickAwareSegyGatherPipelineDataset(
        segy_files=list(segy_files),
        fb_files=list(fb_files),
        sampling_overrides=sampling_overrides,
        transform=transform,
        sample_transformer=sample_transformer,
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


def build_train_dataset(
    *,
    segy_files: list[str],
    fb_files: list[str],
    sampling_overrides: list[dict[str, object] | None] | None,
    plan: BuildPlan,
    fbgate: FirstBreakGate,
    subset_traces: int,
    time_len: int,
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
) -> SegyGatherPipelineDataset:
    return _build_pick_aware_dataset(
        segy_files=segy_files,
        fb_files=fb_files,
        sampling_overrides=sampling_overrides,
        plan=plan,
        fbgate=fbgate,
        subset_traces=subset_traces,
        target_len=time_len,
        standardize_eps=standardize_eps,
        start_mode='random',
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
    sampling_overrides: list[dict[str, object] | None] | None,
    plan: BuildPlan,
    fbgate: FirstBreakGate,
    subset_traces: int,
    time_len: int,
    standardize_eps: float,
    primary_keys: Sequence[str],
    secondary_key_fixed: bool,
    verbose: bool,
    progress: bool,
    max_trials: int,
    use_header_cache: bool,
    waveform_mode: str,
    segy_endian: str,
) -> SegyGatherPipelineDataset:
    return _build_pick_aware_dataset(
        segy_files=segy_files,
        fb_files=fb_files,
        sampling_overrides=sampling_overrides,
        plan=plan,
        fbgate=fbgate,
        subset_traces=subset_traces,
        target_len=time_len,
        standardize_eps=standardize_eps,
        start_mode='midpoint',
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
    plan: BuildPlan | InputOnlyPlan,
    subset_traces: int,
    overlap_h: int,
    time_len: int,
    standardize_eps: float,
    waveform_mode: str,
    segy_endian: str,
    use_header_cache: bool,
) -> InferenceGatherWindowsDataset:
    stride_traces = int(subset_traces) - int(overlap_h)
    if stride_traces <= 0:
        msg = 'subset_traces - overlap_h must be positive'
        raise ValueError(msg)
    validate_files_exist(list(segy_files))
    cfg = InferenceGatherWindowsConfig(
        domains=('shot',),
        secondary_sort={'shot': 'chno', 'recv': 'ffid', 'cmp': 'offset'},
        win_size_traces=int(subset_traces),
        stride_traces=int(stride_traces),
        pad_last=True,
        target_len=int(time_len),
    )
    return InferenceGatherWindowsDataset(
        segy_files=list(segy_files),
        fb_files=None,
        plan=plan,
        cfg=cfg,
        transform=build_infer_transform(standardize_eps=float(standardize_eps)),
        waveform_mode=str(waveform_mode),
        segy_endian=str(segy_endian),
        use_header_cache=bool(use_header_cache),
    )
