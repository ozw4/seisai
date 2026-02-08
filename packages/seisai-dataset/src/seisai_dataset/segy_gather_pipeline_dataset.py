"""SEG-Y gather dataset pipeline.

This module provides a `torch.utils.data.Dataset` implementation that:
- loads SEG-Y gathers and corresponding first-break (FB) picks,
- samples a subset of traces (optionally with superwindow logic),
- applies a user-provided transform and projects metadata to the view,
- applies first-break quality gates (min-pick and FBLC),
- optionally builds model inputs/targets via a `BuildPlan`.
"""

# %%
import contextlib

import numpy as np
import segyio

from .builder.builder import (
    BuildPlan,
)
from .file_info import FileInfo, build_file_info_dataclass
from .gate_fblc import FirstBreakGate
from .segy_gather_base import (
    BaseSegyGatherPipelineDataset,
    GateEvaluator,
    SampleTransformer,
)


class SegyGatherPipelineDataset(BaseSegyGatherPipelineDataset):
    """SEG-Y ギャザー読み込み → サンプリング → 変換) → FBLC ゲート → (任意) BuildPlanで入出力生成.

    期待する transform:  x(H,W) -> x_view  もしくは  (x_view, meta)
    - meta は少なくとも { 'hflip':bool, 'factor':float, 'start':int, 'did_space':bool, 'factor_h':float } の任意サブセット
    期待する fbgate: FirstBreakGate (min_pick_accept と fblc_accept を持つ)
    期待する plan:  BuildPlan (任意)。与えれば sample に 'input' / 'target' などを組み立てる
    """

    def __init__(
        self,
        segy_files: list[str],
        fb_files: list[str],
        transform,
        fbgate: FirstBreakGate,
        plan: BuildPlan,
        *,
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
        subset_traces: int = 128,
        secondary_key_fixed: bool = False,
        verbose: bool = False,
        max_trials: int = 2048,
        sample_transformer: SampleTransformer | None = None,
        gate_evaluator: GateEvaluator | None = None,
    ) -> None:
        if len(segy_files) == 0 or len(fb_files) == 0:
            msg = 'segy_files / fb_files は空であってはならない'
            raise ValueError(msg)
        if len(segy_files) != len(fb_files):
            msg = 'segy_files と fb_files の長さが一致していません'
            raise ValueError(msg)

        self.fb_files = list(fb_files)

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

        # ファイルごとのインデックス辞書等を構築
        self.file_infos: list[FileInfo] = []
        for segy_path, fb_path in zip(self.segy_files, self.fb_files, strict=True):
            info = build_file_info_dataclass(
                segy_path,
                ffid_byte=self.ffid_byte,
                chno_byte=self.chno_byte,
                cmp_byte=self.cmp_byte,
                header_cache_dir=self.header_cache_dir,
                use_header_cache=self.use_header_cache,
                include_centroids=True,  # or False
            )
            fb = np.load(fb_path)
            info.fb = fb
            self.file_infos.append(info)

    def _close_file_info(self, info: FileInfo) -> None:
        if info.segy_obj is not None:
            with contextlib.suppress(Exception):
                info.segy_obj.close()

    def __getitem__(self, _: int | None = None) -> dict:
        """Return a single sampled and processed gather from the dataset."""
        return self._sample_with_retries()

    def _load_fb_subset(
        self,
        info: FileInfo,
        indices: np.ndarray,
        sample: dict,
        counters: dict[str, int],
    ) -> tuple[np.ndarray, dict] | None:
        fb_subset = info.fb[indices]
        return fb_subset, {}

    def _build_plan_extras(
        self,
        *,
        info: FileInfo,
        sample: dict,
        fb_subset: np.ndarray,
        trace_valid: np.ndarray | None,
        label_state: dict | None,
    ) -> dict:
        return self._build_plan_extras_common(
            info=info,
            sample=sample,
            fb_subset=fb_subset,
            trace_valid=trace_valid,
        )

    def _build_output_extras(
        self,
        *,
        info: FileInfo,
        sample: dict,
        fb_subset: np.ndarray,
        label_state: dict | None,
    ) -> dict:
        return self._build_output_extras_common(
            info=info,
            sample=sample,
            fb_subset=fb_subset,
        )
