"""SEG-Y gather dataset pipeline (phase pick CSR variant).

This dataset mirrors `SegyGatherPipelineDataset` but reads phase picks from CSR
`.npz` files and makes them available to label producers (e.g. `PhasePSNMap`)
without breaking the existing FB-only pipeline.

Key points:
- Keeps the existing load -> sample -> transform -> gates -> BuildPlan flow.
- Uses P-first as the legacy `fb_idx` (compatibility).
- Passes subset/padded CSR picks to the plan (not returned in output).
- Returns fixed-size convenience keys: `p_idx`, `s_idx`, `label_valid` (and `fb_idx`).
"""

import contextlib
from pathlib import Path

import numpy as np
import segyio
import torch
from tqdm import tqdm
from seisai_transforms.view_projection import (
    project_fb_idx_view,
)

from .builder.builder import BuildPlan
from .file_info import (
    FileInfo,
    build_file_info_dataclass,
    normalize_segy_endian,
    normalize_waveform_mode,
)
from .gate_fblc import FirstBreakGate
from .phase_pick_io import load_phase_pick_csr_npz, subset_pad_first_invalidate
from .segy_gather_base import (
    BaseSegyGatherPipelineDataset,
    GateEvaluator,
    SampleTransformer,
)


class SegyGatherPhasePipelineDataset(BaseSegyGatherPipelineDataset):
    """SEG-Y gather dataset that consumes phase picks in CSR (.npz) format.

    This dataset is designed to coexist with `SegyGatherPipelineDataset` without
    changing its behavior. It uses the same sampler/transform/gate conventions.

    Additional inputs
    -----------------
    phase_pick_files : list[str]
            CSR `.npz` files aligned 1:1 with `segy_files`.
    include_empty_gathers : bool
            If False, rejects samples where both P and S picks are absent in the sampled
            trace subset (after CSR invalidation). If True, returns such samples and
            skips FB-based quality gates for those empty samples.
    """

    def __init__(
        self,
        segy_files: list[str],
        phase_pick_files: list[str],
        transform,
        fbgate: FirstBreakGate,
        plan: BuildPlan,
        *,
        include_empty_gathers: bool = False,
        ffid_byte: int = segyio.TraceField.FieldRecord,
        chno_byte: int = segyio.TraceField.TraceNumber,
        cmp_byte: int = segyio.TraceField.CDP,
        primary_keys: tuple[str, ...] | None = None,
        primary_key_weights: tuple[float, ...] | None = None,
        use_superwindow: bool = False,
        sw_halfspan: int = 0,
        sw_prob: float = 0.3,
        use_header_cache: bool = True,
        header_cache_dir: str | None = None,
        waveform_mode: str = 'eager',
        segy_endian: str = 'big',
        subset_traces: int = 128,
        secondary_key_fixed: bool = False,
        sampling_overrides: list[dict[str, object] | None] | None = None,
        verbose: bool = False,
        progress: bool | None = None,
        max_trials: int = 2048,
        sample_transformer: SampleTransformer | None = None,
        gate_evaluator: GateEvaluator | None = None,
    ) -> None:
        if len(segy_files) == 0 or len(phase_pick_files) == 0:
            msg = 'segy_files / phase_pick_files は空であってはならない'
            raise ValueError(msg)
        if len(segy_files) != len(phase_pick_files):
            msg = 'segy_files と phase_pick_files の長さが一致していません'
            raise ValueError(msg)

        self.phase_pick_files = list(phase_pick_files)
        self.include_empty_gathers = bool(include_empty_gathers)
        if sampling_overrides is not None and len(sampling_overrides) != len(segy_files):
            msg = 'sampling_overrides length must match segy_files length'
            raise ValueError(msg)
        self.sampling_overrides = (
            list(sampling_overrides) if sampling_overrides is not None else None
        )

        super().__init__(
            segy_files=segy_files,
            transform=transform,
            fbgate=fbgate,
            plan=plan,
            ffid_byte=ffid_byte,
            chno_byte=chno_byte,
            cmp_byte=cmp_byte,
            primary_keys=primary_keys,
            primary_key_weights=primary_key_weights,
            use_superwindow=use_superwindow,
            sw_halfspan=sw_halfspan,
            sw_prob=sw_prob,
            use_header_cache=use_header_cache,
            header_cache_dir=header_cache_dir,
            subset_traces=subset_traces,
            secondary_key_fixed=secondary_key_fixed,
            verbose=verbose,
            max_trials=max_trials,
            sample_transformer=sample_transformer,
            gate_evaluator=gate_evaluator,
        )
        self.progress = self.verbose if progress is None else bool(progress)
        self.waveform_mode = normalize_waveform_mode(waveform_mode)
        self.segy_endian = normalize_segy_endian(segy_endian)
        # Build per-file metadata and attach CSR arrays.
        self.file_infos: list[FileInfo] = []

        total_files = int(len(self.segy_files))
        total_traces = 0
        total_p_nnz = 0
        total_s_nnz = 0
        total_p_valid = 0
        total_s_valid = 0

        it = tqdm(
            zip(self.segy_files, self.phase_pick_files, strict=True),
            total=total_files,
            desc='Indexing SEG-Y + picks',
            unit='file',
            disable=not self.progress,
        )
        for file_idx, (segy_path, pick_path) in enumerate(it):
            it.set_description_str(f'Index {Path(segy_path).name}', refresh=False)
            info = build_file_info_dataclass(
                segy_path,
                ffid_byte=self.ffid_byte,
                chno_byte=self.chno_byte,
                cmp_byte=self.cmp_byte,
                header_cache_dir=self.header_cache_dir,
                use_header_cache=self.use_header_cache,
                include_centroids=True,
                waveform_mode=self.waveform_mode,
                segy_endian=self.segy_endian,
            )
            raw_override = (
                None
                if self.sampling_overrides is None
                else self.sampling_overrides[int(file_idx)]
            )
            info.sampling_override = self.sampler.normalize_sampling_override(
                raw_override
            )

            picks = load_phase_pick_csr_npz(pick_path)
            if int(picks.n_traces) != int(info.n_traces):
                if info.segy_obj is not None:
                    info.segy_obj.close()
                msg = (
                    f'phase picks n_traces mismatch: {pick_path}={picks.n_traces}, '
                    f'{segy_path}={info.n_traces}'
                )
                raise ValueError(msg)

            info.p_indptr = picks.p_indptr
            info.p_data = picks.p_data
            info.s_indptr = picks.s_indptr
            info.s_data = picks.s_data
            self.file_infos.append(info)

            total_traces += int(info.n_traces)
            p_nnz = int(picks.p_data.size)
            s_nnz = int(picks.s_data.size)
            total_p_nnz += p_nnz
            total_s_nnz += s_nnz
            total_p_valid += int((picks.p_data > 0).sum())
            total_s_valid += int((picks.s_data > 0).sum())
            it.set_postfix(
                traces=total_traces,
                p=f'{total_p_valid}/{total_p_nnz}',
                s=f'{total_s_valid}/{total_s_nnz}',
            )

    def _close_file_info(self, info: FileInfo) -> None:
        if info.segy_obj is not None:
            with contextlib.suppress(Exception):
                info.segy_obj.close()

    def __getitem__(self, _: int | None = None) -> dict:
        """Draw and return a single training/evaluation sample from the dataset.

        This method performs a randomized retry loop (up to ``self.max_trials``) to
        sample a gather subset from a randomly chosen file, apply padding/subsetting,
        optionally reject invalid/undesired samples, run the configured processing
        plan, and finally build the output dictionary consumed by downstream code.

        High-level steps
        ----------------
        1. Randomly select a file (``FileInfo``) and draw a sample via
                ``self.sample_flow.draw_sample`` (provides trace ``indices`` and metadata).
        2. Subset/pad phase-pick CSR data (P/S first-arrival picks) into the selected
                gather window.
        3. Optionally reject:
                - empty gather subsets (no P/S picks) when ``include_empty_gathers=False``
                - samples failing minimum-pick acceptance
                - samples failing additional "FB legacy" gates (e.g., FBL/C gates)
        4. Load/transform the seismic window via
                ``self.sample_transformer.load_transform`` and validate ``x_view`` shape:
                - Accept (H, W) or channels-first (C, H, W)
                - Reject channels-last (H, W, C)
                - Require ``H == self.subset_traces``
        5. Populate view-space pick indices in ``meta`` (P as ``p_idx_view``, S as
                ``s_idx_view``), respecting per-trace validity (invalid traces set to -1).
        6. Build plan input, run the plan (``self.sample_flow.run_plan``), and require
                that it produces ``label_valid`` of shape (H,).
        7. Build and return the final output dictionary.

        Parameters
        ----------
        _ : int | None, optional
                Ignored. Present to satisfy a Dataset-style indexing signature.

        Returns
        -------
        dict
                A sample dictionary containing at least:
                - ``x_view``: transformed window (2D or 3D channels-first tensor/array)
                - ``fb_idx``/``p_idx``: P-first picks as a torch tensor
                - ``s_idx``: S-first picks as a torch tensor (invalid traces set to -1)
                - ``trace_valid``: per-trace validity mask as a torch tensor (H,)
                - ``label_valid``: per-trace label validity mask as a torch tensor (H,)
                Plus metadata keys added by the plan and the output builder.

        Raises
        ------
        RuntimeError
                If required CSR pick arrays are not attached to ``FileInfo`` or if no valid
                sample can be drawn within ``max_trials``.
        ValueError
                If ``x_view`` has an unsupported dimensionality/layout, if the transformed
                height does not match ``subset_traces``, or if pick/label shapes mismatch.
        KeyError
                If the processing plan does not populate ``label_valid``.

        """
        return self._sample_with_retries()

    def _init_rejection_counters(self) -> dict[str, int]:
        counters = super()._init_rejection_counters()
        counters['empty'] = 0
        return counters

    def _load_fb_subset(
        self,
        info: FileInfo,
        indices: np.ndarray,
        sample: dict,
        counters: dict[str, int],
    ) -> tuple[np.ndarray, dict] | None:
        if (
            info.p_indptr is None
            or info.p_data is None
            or info.s_indptr is None
            or info.s_data is None
        ):
            msg = 'phase pick CSR is not attached to FileInfo'
            raise RuntimeError(msg)

        H0 = int(indices.size)
        if H0 == 0:
            return None

        win = subset_pad_first_invalidate(
            p_indptr=info.p_indptr,
            p_data=info.p_data,
            s_indptr=info.s_indptr,
            s_data=info.s_data,
            indices=indices,
            subset_traces=int(self.subset_traces),
        )

        p_first = win.p_first[:H0]
        s_first = win.s_first[:H0]
        is_empty = (not np.any(p_first > 0)) and (not np.any(s_first > 0))

        # Reject empty gather subsets unless explicitly allowed.
        if (not self.include_empty_gathers) and is_empty:
            self._count_rejection(counters, 'empty')
            return None

        # Legacy gates operate on P-first picks (fb semantics).
        # If empty-gathers are explicitly allowed, skip FB gates for empty samples.
        apply_fb_gates = not (self.include_empty_gathers and is_empty)
        label_state = {'win': win, 'apply_fb_gates': apply_fb_gates}
        return p_first, label_state

    def _get_view_shape(
        self, x_view, label_state: dict | None
    ) -> tuple[int, int] | None:
        # Accept 2D (H,W) and 3D channels-first (C,H,W). Reject channels-last (H,W,C).
        if not hasattr(x_view, 'shape') or not hasattr(x_view, 'ndim'):
            msg = f'x_view must have shape/ndim, got {type(x_view).__name__}'
            raise ValueError(msg)
        ndim = int(x_view.ndim)
        shape = tuple(int(s) for s in x_view.shape)
        if ndim == 2:
            H, W = shape[0], shape[1]
        elif ndim == 3:
            st = int(self.subset_traces)
            if shape[1] == st:
                # channels-first: (C,H,W)
                H, W = shape[1], shape[2]
            elif shape[0] == st:
                # channels-last: (H,W,C) is not supported
                msg = f'x_view must be channels-first (C,H,W); got channels-last shape={shape}'
                raise ValueError(msg)
            else:
                msg = (
                    f'x_view 3D must be channels-first (C,H,W) with H==subset_traces at axis=1; '
                    f'got shape={shape}, subset_traces={st}'
                )
                raise ValueError(msg)
        else:
            msg = f'x_view must be 2D or 3D, got shape={shape}'
            raise ValueError(msg)
        if int(self.subset_traces) != H:
            msg = f'loader/transform must keep H=subset_traces: got H={H}, subset_traces={self.subset_traces}'
            raise ValueError(msg)
        return H, W

    def _post_transform_meta(
        self,
        *,
        meta: dict,
        label_state: dict | None,
        trace_valid: np.ndarray,
        view_shape: tuple[int, int] | None,
    ) -> None:
        if label_state is None or view_shape is None:
            msg = 'phase label_state/view_shape is required'
            raise ValueError(msg)
        H, W = view_shape
        win = label_state['win']

        meta['trace_valid'] = trace_valid

        # Add phase-first picks in view space.
        meta['p_idx_view'] = meta['fb_idx_view'].copy()

        s_idx = win.s_first.astype(np.int64, copy=True)
        if s_idx.shape != (H,):
            msg = f's_first shape mismatch: {s_idx.shape} != ({H},)'
            raise ValueError(msg)
        s_idx[~trace_valid] = -1
        meta['s_idx_view'] = project_fb_idx_view(s_idx, H, W, meta)
        label_state['s_idx'] = s_idx

    def _post_plan_validation(
        self,
        sample_for_plan: dict,
        label_state: dict | None,
        view_shape: tuple[int, int] | None,
    ) -> None:
        # label_valid must be produced by the plan (e.g., PhasePSNMap).
        if 'label_valid' not in sample_for_plan:
            msg = "plan must populate 'label_valid' for SegyGatherPhasePipelineDataset"
            raise KeyError(msg)

    def _build_plan_extras(
        self,
        *,
        info: FileInfo,
        sample: dict,
        fb_subset: np.ndarray,
        trace_valid: np.ndarray | None,
        label_state: dict | None,
    ) -> dict:
        extra = self._build_plan_extras_common(
            info=info,
            sample=sample,
            fb_subset=fb_subset,
            trace_valid=trace_valid,
        )
        win = label_state['win'] if label_state is not None else None
        if win is None:
            msg = 'phase label_state is required for plan extras'
            raise ValueError(msg)
        extra.update(
            {
                # CSR picks for label producers (do not return in output)
                'p_indptr': win.p_indptr,
                'p_data': win.p_data,
                's_indptr': win.s_indptr,
                's_data': win.s_data,
            }
        )
        return extra

    def _build_output_extras(
        self,
        *,
        info: FileInfo,
        sample: dict,
        fb_subset: np.ndarray,
        label_state: dict | None,
    ) -> dict:
        extra = self._build_output_extras_common(
            info=info,
            sample=sample,
            fb_subset=fb_subset,
        )
        s_idx = label_state.get('s_idx') if isinstance(label_state, dict) else None
        if s_idx is None:
            msg = 'phase label_state.s_idx is required for output'
            raise ValueError(msg)
        extra.update(
            {
                'p_idx': torch.from_numpy(fb_subset),
                's_idx': torch.from_numpy(s_idx),
            }
        )
        return extra

    def _finalize_output(
        self,
        out: dict,
        sample_for_plan: dict,
        label_state: dict | None,
        view_shape: tuple[int, int] | None,
    ) -> None:
        if view_shape is None:
            msg = 'label_valid requires view shape'
            raise ValueError(msg)
        H, _W = view_shape
        label_valid = sample_for_plan['label_valid']

        # label_valid: (H,) bool
        if isinstance(label_valid, torch.Tensor):
            out['label_valid'] = label_valid
        else:
            lv = np.asarray(label_valid, dtype=np.bool_)
            if lv.shape != (H,):
                msg = f'label_valid shape {lv.shape} != ({H},)'
                raise ValueError(msg)
            out['label_valid'] = torch.from_numpy(lv)
