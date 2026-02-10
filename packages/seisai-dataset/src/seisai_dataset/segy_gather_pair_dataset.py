# %%
import contextlib
from pathlib import Path
from typing import cast

import numpy as np
import segyio
from tqdm import tqdm

from .builder.builder import BuildPlan
from .file_info import (
    PairFileInfo,
    build_file_info_dataclass,
    normalize_waveform_mode,
)
from .segy_gather_base import BaseRandomSegyDataset


class SegyGatherPairDataset(BaseRandomSegyDataset):
    """SEG-Y 対応ペアから同期 transform で input/target を生成する Dataset."""

    def __init__(
        self,
        input_segy_files: list[str],
        target_segy_files: list[str],
        transform,
        plan: BuildPlan,
        *,
        ffid_byte: int = segyio.TraceField.FieldRecord,
        chno_byte: int = segyio.TraceField.TraceNumber,
        cmp_byte: int = segyio.TraceField.CDP,
        primary_keys: tuple[str, ...] | None = ('ffid',),
        primary_key_weights: tuple[float, ...] | None = None,
        use_superwindow: bool = False,
        sw_halfspan: int = 0,
        sw_prob: float = 0.3,
        use_header_cache: bool = True,
        header_cache_dir: str | None = None,
        waveform_mode: str = 'eager',
        subset_traces: int = 128,
        secondary_key_fixed: bool = False,
        verbose: bool = False,
        progress: bool | None = None,
        max_trials: int = 2048,
    ) -> None:
        if len(input_segy_files) == 0 or len(target_segy_files) == 0:
            msg = 'input_segy_files / target_segy_files は空であってはならない'
            raise ValueError(msg)
        if len(input_segy_files) != len(target_segy_files):
            msg = 'input_segy_files と target_segy_files の長さが一致していません'
            raise ValueError(msg)

        self.input_segy_files = list(input_segy_files)
        self.target_segy_files = list(target_segy_files)

        super().__init__(transform, plan, max_trials=max_trials, verbose=verbose)

        self.progress = self.verbose if progress is None else bool(progress)
        self.waveform_mode = normalize_waveform_mode(waveform_mode)

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
        self.subsetloader = self._build_subset_loader(self.subset_traces)
        self.file_infos: list[PairFileInfo] = []

        total_files = int(len(self.input_segy_files))
        total_traces = 0

        it = tqdm(
            zip(self.input_segy_files, self.target_segy_files, strict=True),
            total=total_files,
            desc='Indexing SEG-Y pairs',
            unit='file',
            disable=not self.progress,
        )
        for input_path, target_path in it:
            it.set_description_str(f'Index {Path(input_path).name}', refresh=False)
            input_info = build_file_info_dataclass(
                input_path,
                ffid_byte=self.ffid_byte,
                chno_byte=self.chno_byte,
                cmp_byte=self.cmp_byte,
                header_cache_dir=self.header_cache_dir,
                use_header_cache=self.use_header_cache,
                include_centroids=True,
                waveform_mode=self.waveform_mode,
            )
            target_obj = segyio.open(target_path, 'r', ignore_geometry=True)
            if self.waveform_mode == 'mmap':
                target_obj.mmap()
                target_mmap = target_obj.trace.raw
            else:
                target_mmap = target_obj.trace.raw[:]
            target_n_samples = (
                int(target_obj.samples.size) if target_obj.samples is not None else 0
            )
            target_n_traces = int(target_obj.tracecount)
            target_dt_us = int(cast('int', target_obj.bin[segyio.BinField.Interval]))
            target_dt_sec = target_dt_us * 1e-6

            if input_info.n_samples != target_n_samples:
                if input_info.segy_obj is not None:
                    input_info.segy_obj.close()
                target_obj.close()
                msg = (
                    'nsamples mismatch: '
                    f'{input_path}={input_info.n_samples}, '
                    f'{target_path}={target_n_samples}'
                )
                raise ValueError(msg)
            if input_info.n_traces != target_n_traces:
                if input_info.segy_obj is not None:
                    input_info.segy_obj.close()
                target_obj.close()
                msg = (
                    'trace count mismatch: '
                    f'{input_path}={input_info.n_traces}, '
                    f'{target_path}={target_n_traces}'
                )
                raise ValueError(msg)
            if not np.isclose(input_info.dt_sec, target_dt_sec, rtol=0.0, atol=1e-12):
                if input_info.segy_obj is not None:
                    input_info.segy_obj.close()
                target_obj.close()
                msg = (
                    'dt mismatch: '
                    f'{input_path}={input_info.dt_sec}, '
                    f'{target_path}={target_dt_sec}'
                )
                raise ValueError(msg)

            self.file_infos.append(
                PairFileInfo(
                    input_info=input_info,
                    target_path=str(target_path),
                    target_mmap=target_mmap,
                    target_segy_obj=target_obj,
                    target_n_samples=target_n_samples,
                    target_n_traces=target_n_traces,
                    target_dt_sec=float(target_dt_sec),
                )
            )

            total_traces += int(input_info.n_traces)
            it.set_postfix(traces=total_traces)

    def _close_file_info(self, info: PairFileInfo) -> None:
        if info.input_info.segy_obj is not None:
            with contextlib.suppress(Exception):
                info.input_info.segy_obj.close()
        if info.target_segy_obj is not None:
            with contextlib.suppress(Exception):
                info.target_segy_obj.close()

    def __getitem__(self, idx: int) -> dict:
        """Return one randomized input/target sample pair.

        Args:
                idx: Unused index value (sampling is randomized internally).

        Returns:
                A dict containing the dataset output produced by the configured sample flow.

        """
        _ = idx
        return self._sample_with_retries()

    def _try_build_sample(
        self, info: PairFileInfo, counters: dict[str, int]
    ) -> dict | None:
        input_info = info.input_info

        sample = self._draw_sample(input_info)
        indices = sample['indices']
        if indices.size == 0:
            return None

        x_in = self.subsetloader.load(input_info.mmap, indices)
        x_tg = self.subsetloader.load(info.target_mmap, indices)

        H = int(x_in.shape[0])
        offsets = input_info.offsets[indices].astype(np.float32, copy=False)
        indices, offsets, _fb_subset, _trace_valid, _pad = (
            self.sample_flow.pad_indices_offsets_fb(
                indices=indices,
                offsets=offsets,
                fb_subset=None,
                H=H,
            )
        )

        seed = int(self._rng.integers(0, 2**31 - 1))
        rng_in = np.random.default_rng(seed)
        rng_tg = np.random.default_rng(seed)
        x_view_input, meta = self.sample_flow.apply_transform(
            x_in,
            rng_in,
            name='input',
        )
        x_view_target, _meta_tg = self.sample_flow.apply_transform(
            x_tg,
            rng_tg,
            name='target',
        )
        if x_view_input.shape != x_view_target.shape:
            msg = (
                'input/target transform shape mismatch: '
                f'{x_view_input.shape} vs {x_view_target.shape}'
            )
            raise ValueError(msg)
        did_superwindow = bool(sample['did_super'])

        dt_sec = float(input_info.dt_sec)
        sample_for_plan = self.sample_flow.build_plan_input_base(
            meta=meta,
            dt_sec=dt_sec,
            offsets=offsets,
            indices=indices,
            key_name=sample['key_name'],
            secondary_key=sample['secondary_key'],
            primary_unique=sample['primary_unique'],
            extra={
                'x_view_input': x_view_input,
                'x_view_target': x_view_target,
                'file_path_input': input_info.path,
                'file_path_target': info.target_path,
                'did_superwindow': did_superwindow,
            },
        )

        self.sample_flow.run_plan(sample_for_plan, rng=self._rng)

        return self.sample_flow.build_output_base(
            sample_for_plan,
            meta=meta,
            dt_sec=dt_sec,
            offsets=offsets,
            indices=indices,
            key_name=sample['key_name'],
            secondary_key=sample['secondary_key'],
            primary_unique=sample['primary_unique'],
            extra={
                'file_path_input': input_info.path,
                'file_path_target': info.target_path,
                'did_superwindow': did_superwindow,
            },
        )
