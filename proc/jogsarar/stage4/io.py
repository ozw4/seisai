from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import segyio
from common.segy_io import (
    read_trace_field,
    require_expected_samples,
    require_matching_tracecount,
)


@dataclass(frozen=True)
class Stage4Inputs:
    n_traces: int
    n_samples_raw: int
    n_samples_win: int
    dt_us_raw: int
    dt_us_win: int
    dt_sec_raw: float
    dt_sec_win: float
    ffid_values: np.ndarray
    chno_values: np.ndarray
    offsets: np.ndarray
    window_start_i: np.ndarray


@contextmanager
def open_and_load_stage4_inputs(
    *,
    raw_path: Path,
    win_path: Path,
    sidecar_path: Path,
    cfg,
    load_sidecar_window_start_fn: Callable[..., np.ndarray],
) -> Iterator[tuple[Stage4Inputs, segyio.SegyFile, segyio.SegyFile]]:
    with (
        segyio.open(str(raw_path), 'r', ignore_geometry=True) as raw,
        segyio.open(str(win_path), 'r', ignore_geometry=True) as win,
    ):
        n_traces = require_matching_tracecount(
            raw,
            win,
            raw_path=raw_path,
            win_path=win_path,
        )
        if n_traces <= 0:
            msg = f'no traces in raw segy: {raw_path}'
            raise ValueError(msg)

        n_samples_raw = int(raw.samples.size)
        n_samples_win = int(win.samples.size)
        if n_samples_raw <= 0 or n_samples_win <= 0:
            msg = (
                f'invalid n_samples raw={n_samples_raw} win={n_samples_win} '
                f'raw={raw_path} win={win_path}'
            )
            raise ValueError(msg)
        require_expected_samples(
            win,
            expected=int(cfg.tile_w),
            win_path=win_path,
        )

        dt_us_raw = int(raw.bin[segyio.BinField.Interval])
        dt_us_win = int(win.bin[segyio.BinField.Interval])
        if dt_us_raw <= 0 or dt_us_win <= 0:
            msg = f'invalid dt_us raw={dt_us_raw} win={dt_us_win}'
            raise ValueError(msg)
        dt_sec_raw = float(dt_us_raw) * 1.0e-6
        dt_sec_win = float(dt_us_win) * 1.0e-6

        ffid_values = read_trace_field(
            raw,
            segyio.TraceField.FieldRecord,
            dtype=np.int32,
            name='raw ffid_values',
        )
        chno_values = read_trace_field(
            raw,
            segyio.TraceField.TraceNumber,
            dtype=np.int32,
            name='raw chno_values',
        )
        offsets = read_trace_field(
            raw,
            segyio.TraceField.offset,
            dtype=np.float32,
            name='raw offsets',
        )

        ffid_win = read_trace_field(
            win,
            segyio.TraceField.FieldRecord,
            dtype=np.int32,
            name='win ffid_values',
        )
        chno_win = read_trace_field(
            win,
            segyio.TraceField.TraceNumber,
            dtype=np.int32,
            name='win chno_values',
        )

        if not np.array_equal(ffid_values, ffid_win):
            msg = f'raw/win ffid arrays differ (index mapping would break): {raw_path}'
            raise ValueError(msg)
        if not np.array_equal(chno_values, chno_win):
            msg = f'raw/win chno arrays differ (index mapping would break): {raw_path}'
            raise ValueError(msg)

        window_start_i = load_sidecar_window_start_fn(
            sidecar_path=sidecar_path,
            n_traces=n_traces,
            n_samples_in=n_samples_raw,
            n_samples_out=n_samples_win,
            dt_sec_in=dt_sec_raw,
            dt_sec_out=dt_sec_win,
            cfg=cfg,
        )

        inputs = Stage4Inputs(
            n_traces=n_traces,
            n_samples_raw=n_samples_raw,
            n_samples_win=n_samples_win,
            dt_us_raw=dt_us_raw,
            dt_us_win=dt_us_win,
            dt_sec_raw=dt_sec_raw,
            dt_sec_win=dt_sec_win,
            ffid_values=ffid_values,
            chno_values=chno_values,
            offsets=offsets,
            window_start_i=window_start_i,
        )
        yield inputs, raw, win


__all__ = ['Stage4Inputs', 'open_and_load_stage4_inputs']
