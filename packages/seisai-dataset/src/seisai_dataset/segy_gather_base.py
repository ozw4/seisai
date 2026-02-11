"""Shared base utilities for SEG-Y gather datasets."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
import torch
from seisai_transforms.view_projection import (
    project_fb_idx_view,
    project_offsets_view,
    project_time_view,
)
from torch.utils.data import Dataset

from .config import LoaderConfig, TraceSubsetSamplerConfig
from .sample_flow import SampleFlow
from .trace_subset_preproc import TraceSubsetLoader
from .trace_subset_sampler import TraceSubsetSampler

if TYPE_CHECKING:
    from .file_info import FileInfo
    from .gate_fblc import FirstBreakGate


class SampleTransformer:
    """Load a subset of SEG-Y traces and apply the configured transform.

    This helper encapsulates:
    - loading a fixed-size trace subset via `TraceSubsetLoader` (with padding as needed),
    - aligning indices/offsets/first-break picks to the loaded height (H),
    - applying the user transform and projecting metadata to the transformed view.

    Parameters
    ----------
    subsetloader : TraceSubsetLoader
            Loader used to read and (optionally) pad the trace subset.
    transform
            Callable applied to the loaded gather. Expected signature:
            `transform(x: np.ndarray, rng: np.random.Generator, return_meta: bool=True)
            -> np.ndarray | tuple[np.ndarray, dict]`.

    Notes
    -----
    `load_transform` returns `(x_view, meta, offsets, fb_subset, indices, trace_valid)`,
    where `meta` is augmented with view-projected fields like `fb_idx_view`,
    `offsets_view`, and `time_view`.

    """

    def __init__(
        self,
        subsetloader: TraceSubsetLoader,
        transform,
    ) -> None:
        """Initialize the transformer with a subset loader and transform callable."""
        self.subsetloader = subsetloader
        self.transform = transform

    def load_transform(
        self,
        info: FileInfo,
        indices: np.ndarray,
        fb_subset: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mmap = info.mmap

        # 波形読み出し(subset_traces に満たない場合は loader が H 方向にパッドする)
        x = self.subsetloader.load(mmap, indices)  # (H,W0)
        H = int(x.shape[0])
        W0 = int(x.shape[1])

        # ここで offsets/fb/indices を H に合わせる (仕様)
        offsets = info.offsets[indices].astype(np.float32, copy=False)
        indices, offsets, fb_subset, trace_valid, _pad = (
            SampleFlow.pad_indices_offsets_fb(
                indices=indices,
                offsets=offsets,
                fb_subset=fb_subset,
                H=H,
            )
        )

        # 変換 (Crop/Pad / TimeStretch 等)
        out = self.transform(x, rng=rng, return_meta=True)
        x_view, meta = out if isinstance(out, tuple) else (out, {})
        if not isinstance(meta, dict):
            msg = f'transform meta must be dict, got {type(meta).__name__}'
            raise TypeError(msg)
        if not isinstance(x_view, np.ndarray) or x_view.ndim != 2:
            msg = 'transform は 2D numpy または (2D, meta) を返す必要があります'
            raise ValueError(msg)

        Hv, W = x_view.shape
        if Hv != H:
            msg = f'transform must keep H: got Hv={Hv}, expected H={H}'
            raise ValueError(msg)

        t_raw = np.arange(W0, dtype=np.float32) * float(info.dt_sec)

        meta['trace_valid'] = trace_valid
        meta['fb_idx_view'] = project_fb_idx_view(fb_subset, H, W, meta)
        meta['offsets_view'] = project_offsets_view(offsets, H, meta)
        meta['time_view'] = project_time_view(t_raw, H, W, meta)

        return x_view, meta, offsets, fb_subset, indices, trace_valid


class GateEvaluator:
    """Evaluate first-break quality gates for a gather.

    Attributes
    ----------
    fbgate : FirstBreakGate
            Gate implementation providing min-pick and FBLC checks.
    verbose : bool
            Whether to print rejection details.
    last_reject : str | None
            Most recent rejection reason ("min_pick" or "fblc").

    """

    def __init__(self, fbgate: FirstBreakGate, *, verbose: bool = False) -> None:
        """Initialize the gate evaluator with a gate implementation and verbosity."""
        self.fbgate = fbgate
        self.verbose = bool(verbose)
        self.last_reject: str | None = None

    def min_pick_accept(self, fb_subset: np.ndarray) -> bool:
        """Check whether the gather passes the minimum pick-count quality gate.

        Parameters
        ----------
        fb_subset : np.ndarray
                First-break pick indices for the currently sampled trace subset.

        Returns
        -------
        bool
                True if the minimum pick criterion is satisfied; otherwise False, with
                `last_reject` set to "min_pick".

        """
        ok_pick, _, _ = self.fbgate.min_pick_accept(fb_subset)
        if not ok_pick:
            self.last_reject = 'min_pick'
            return False
        self.last_reject = None
        return True

    def apply_gates(self, meta: dict, *, did_super: bool, info: FileInfo) -> bool:
        """Apply post-transform first-break quality gates to a gather.

        This currently evaluates the FBLC (first-break linearity/consistency) gate
        using first-break indices projected into the transformed view.

        Parameters
        ----------
        meta : dict
                Metadata dictionary produced/augmented during transformation; must
                include `fb_idx_view` and may include `factor` for effective dt scaling.
        did_super : bool
                Whether the sample was drawn using the superwindow logic.
        info : FileInfo
                File-level metadata (e.g., `dt_sec`, `path`) for the current gather.

        Returns
        -------
        bool
                True if the gather passes all gates; otherwise False, with `last_reject`
                set to the rejection reason (currently "fblc").

        """
        # FBLC gate (After transform)
        factor = float(meta.get('factor', 1.0))
        dt_eff_sec = info.dt_sec / max(factor, 1e-9)
        meta['dt_eff_sec'] = float(dt_eff_sec)
        ok_fblc, p_ms, valid_pairs = self.fbgate.fblc_accept(
            meta['fb_idx_view'], dt_eff_sec=dt_eff_sec, did_super=did_super
        )
        if not ok_fblc:
            # if self.verbose:
            #    print(
            #        f'Rejecting gather {info.path} key={meta.get("key_name", "")}:{meta.get("primary_unique", "")} '
            #        f'(FBLC gate; pairs={valid_pairs}, p_ms={p_ms})'
            #    )
            self.last_reject = 'fblc'
            return False

        self.last_reject = None
        return True


class BaseRandomSegyDataset(Dataset, abc.ABC):
    """Shared sampling loop and utilities for randomized SEG-Y datasets."""

    def __init__(
        self,
        transform,
        plan,
        *,
        max_trials: int,
        verbose: bool = False,
    ) -> None:
        self.transform = transform
        self.plan = plan
        self.verbose = bool(verbose)
        self._rng = np.random.default_rng()
        self.max_trials = int(max_trials)
        self.sample_flow = SampleFlow(transform, plan)
        self.file_infos: list = []

    def __len__(self) -> int:
        """Return the nominal dataset length used for randomized sampling."""
        return 1024

    def _init_header_config(
        self,
        *,
        ffid_byte: int,
        chno_byte: int,
        cmp_byte: int,
        use_header_cache: bool,
        header_cache_dir: str | None,
    ) -> None:
        self.ffid_byte = ffid_byte
        self.chno_byte = chno_byte
        self.cmp_byte = cmp_byte
        self.use_header_cache = bool(use_header_cache)
        self.header_cache_dir = header_cache_dir

    def _init_sampler_config(
        self,
        *,
        primary_keys: tuple[str, ...] | None,
        primary_key_weights: tuple[float, ...] | None,
        use_superwindow: bool,
        sw_halfspan: int,
        sw_prob: float,
        secondary_key_fixed: bool,
        subset_traces: int,
    ) -> TraceSubsetSampler:
        self.primary_keys = tuple(primary_keys) if primary_keys else None
        self.primary_key_weights = (
            tuple(primary_key_weights) if primary_key_weights else None
        )
        self.use_superwindow = bool(use_superwindow)
        self.sw_halfspan = int(sw_halfspan)
        self.sw_prob = float(sw_prob)
        self.secondary_key_fixed = bool(secondary_key_fixed)
        self.subset_traces = int(subset_traces)
        return TraceSubsetSampler(
            TraceSubsetSamplerConfig(
                primary_keys=self.primary_keys,
                primary_key_weights=self.primary_key_weights,
                use_superwindow=self.use_superwindow,
                sw_halfspan=self.sw_halfspan,
                sw_prob=self.sw_prob,
                secondary_key_fixed=self.secondary_key_fixed,
                subset_traces=int(self.subset_traces),
            )
        )

    @staticmethod
    def _build_subset_loader(subset_traces: int) -> TraceSubsetLoader:
        return TraceSubsetLoader(LoaderConfig(pad_traces_to=int(subset_traces)))

    def _choose_file_info(self):
        if not self.file_infos:
            msg = 'file_infos is empty'
            raise RuntimeError(msg)
        fidx = int(self._rng.integers(0, len(self.file_infos)))
        return self.file_infos[fidx]

    def _draw_sample(self, info) -> dict:
        return self.sample_flow.draw_sample(info, self._rng, sampler=self.sampler)

    def _sample_with_retries(self) -> dict:
        rejections = self._init_rejection_counters()
        for _attempt in range(self.max_trials):
            info = self._choose_file_info()
            out = self._try_build_sample(info, rejections)
            if out is not None:
                return out
        raise RuntimeError(self._format_max_trials_error(rejections))

    def _init_rejection_counters(self) -> dict[str, int]:
        return {}

    def _count_rejection(self, counters: dict[str, int], key: str) -> None:
        counters[key] = counters.get(key, 0) + 1

    def _format_max_trials_error(self, counters: dict[str, int]) -> str:
        return (
            f'failed to draw a valid sample within max_trials={self.max_trials}; '
            f'files={len(self.file_infos)}'
        )

    def close(self) -> None:
        """Close any open SEG-Y file handles and release cached file metadata."""
        infos = getattr(self, 'file_infos', None)
        if not infos:
            return
        for info in infos:
            self._close_file_info(info)
        infos.clear()

    def __del__(self) -> None:
        """Finalize the dataset by closing any open SEG-Y resources."""
        self.close()

    @abc.abstractmethod
    def _try_build_sample(self, info, counters: dict[str, int]) -> dict | None:
        """Build a single sample or return None to retry."""

    @abc.abstractmethod
    def _close_file_info(self, info) -> None:
        """Close any resources held by a file info object."""


class BaseSegyGatherPipelineDataset(BaseRandomSegyDataset, abc.ABC):
    """Base class for pipeline datasets that sample a single SEG-Y gather."""

    def __init__(
        self,
        segy_files: list[str],
        transform,
        fbgate: FirstBreakGate,
        plan,
        *,
        ffid_byte: int,
        chno_byte: int,
        cmp_byte: int,
        primary_keys: tuple[str, ...] | None,
        primary_key_weights: tuple[float, ...] | None,
        use_superwindow: bool,
        sw_halfspan: int,
        sw_prob: float,
        use_header_cache: bool,
        header_cache_dir: str | None,
        subset_traces: int,
        secondary_key_fixed: bool,
        verbose: bool,
        max_trials: int,
        sample_transformer: SampleTransformer | None = None,
        gate_evaluator: GateEvaluator | None = None,
    ) -> None:
        super().__init__(transform, plan, max_trials=max_trials, verbose=verbose)

        self.segy_files = list(segy_files)
        self.fbgate = fbgate

        self._init_header_config(
            ffid_byte=ffid_byte,
            chno_byte=chno_byte,
            cmp_byte=cmp_byte,
            use_header_cache=use_header_cache,
            header_cache_dir=header_cache_dir,
        )

        self.sampler = self._init_sampler_config(
            primary_keys=primary_keys,
            primary_key_weights=primary_key_weights,
            use_superwindow=use_superwindow,
            sw_halfspan=sw_halfspan,
            sw_prob=sw_prob,
            secondary_key_fixed=secondary_key_fixed,
            subset_traces=subset_traces,
        )

        if sample_transformer is None:
            subsetloader = self._build_subset_loader(self.subset_traces)
            sample_transformer = SampleTransformer(subsetloader, self.transform)
        if gate_evaluator is None:
            gate_evaluator = GateEvaluator(self.fbgate, verbose=self.verbose)

        self.sample_transformer = sample_transformer
        self.gate_evaluator = gate_evaluator

    def _init_rejection_counters(self) -> dict[str, int]:
        return {'min_pick': 0, 'fblc': 0}

    def _format_max_trials_error(self, counters: dict[str, int]) -> str:
        parts = []
        if 'empty' in counters:
            parts.append(f'empty={counters["empty"]}')
        if 'min_pick' in counters:
            parts.append(f'min_pick={counters["min_pick"]}')
        if 'fblc' in counters:
            parts.append(f'fblc={counters["fblc"]}')
        if parts:
            rej = ', '.join(parts)
            return (
                f'failed to draw a valid sample within max_trials={self.max_trials}; '
                f'rejections: {rej}, files={len(self.file_infos)}'
            )
        return super()._format_max_trials_error(counters)

    def _should_apply_fb_gates(self, label_state: dict | None) -> bool:
        if isinstance(label_state, dict):
            return bool(label_state.get('apply_fb_gates', True))
        return True

    def _set_dt_eff_meta_no_gate(self, meta: dict, *, info: FileInfo) -> None:
        factor = float(meta.get('factor', 1.0))
        meta['dt_eff_sec'] = float(info.dt_sec / max(factor, 1e-9))

    def _build_plan_extras_common(
        self,
        *,
        info: FileInfo,
        sample: dict,
        fb_subset: np.ndarray,
        trace_valid: np.ndarray | None,
    ) -> dict:
        return {
            'x_view': sample['x_view'],
            'fb_idx': fb_subset,
            'file_path': info.path,
            'trace_valid': trace_valid,
        }

    def _build_output_extras_common(
        self,
        *,
        info: FileInfo,
        sample: dict,
        fb_subset: np.ndarray,
    ) -> dict:
        return {
            'fb_idx': torch.from_numpy(fb_subset),
            'file_path': info.path,
            'did_superwindow': sample['did_super'],
        }

    def _get_view_shape(
        self, x_view, label_state: dict | None
    ) -> tuple[int, int] | None:
        if not hasattr(x_view, 'ndim') or not hasattr(x_view, 'shape'):
            return None
        if int(x_view.ndim) != 2:
            return None
        shape = x_view.shape
        if shape is None or len(shape) < 2:
            return None
        return int(shape[0]), int(shape[1])

    def _post_transform_meta(
        self,
        *,
        meta: dict,
        label_state: dict | None,
        trace_valid: np.ndarray,
        view_shape: tuple[int, int] | None,
    ) -> None:
        return None

    def _post_plan_validation(
        self,
        sample_for_plan: dict,
        label_state: dict | None,
        view_shape: tuple[int, int] | None,
    ) -> None:
        return None

    def _add_trace_valid_output(
        self, out: dict, trace_valid: np.ndarray | None
    ) -> None:
        if trace_valid is not None:
            out['trace_valid'] = torch.from_numpy(trace_valid)

    def _add_mask_bool_output(self, out: dict, sample_for_plan: dict) -> None:
        mask_bool = sample_for_plan.get('mask_bool')
        if mask_bool is not None:
            out['mask_bool'] = mask_bool

    def _finalize_output(
        self,
        out: dict,
        sample_for_plan: dict,
        label_state: dict | None,
        view_shape: tuple[int, int] | None,
    ) -> None:
        return None

    def _try_build_sample(
        self, info: FileInfo, counters: dict[str, int]
    ) -> dict | None:
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

        x_view, meta, offsets, fb_subset, indices_pad, trace_valid = (
            self.sample_transformer.load_transform(info, indices, fb_subset, self._rng)
        )
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
                meta, did_super=did_super, info=info
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

    @abc.abstractmethod
    def _load_fb_subset(
        self,
        info: FileInfo,
        indices: np.ndarray,
        sample: dict,
        counters: dict[str, int],
    ) -> tuple[np.ndarray, dict] | None:
        """Return (fb_subset, label_state) or None to retry."""

    @abc.abstractmethod
    def _build_plan_extras(
        self,
        *,
        info: FileInfo,
        sample: dict,
        fb_subset: np.ndarray,
        trace_valid: np.ndarray | None,
        label_state: dict | None,
    ) -> dict:
        """Build extra keys to pass into build_plan_input_base."""

    @abc.abstractmethod
    def _build_output_extras(
        self,
        *,
        info: FileInfo,
        sample: dict,
        fb_subset: np.ndarray,
        label_state: dict | None,
    ) -> dict:
        """Build extra keys to include in the output dict."""
