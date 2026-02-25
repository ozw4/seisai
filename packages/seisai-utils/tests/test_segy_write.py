from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import segyio

from seisai_utils.segy_write import (
    _line80_to_content76,
    write_segy_float32_like_input_with_text0_append,
)


def _create_dummy_src_segy(
    path: Path,
    *,
    n_traces: int,
    n_samples: int,
    dt_us: int,
) -> None:
    if n_traces <= 0 or n_samples <= 0:
        msg = 'n_traces and n_samples must be positive'
        raise ValueError(msg)

    spec = segyio.spec()
    spec.iline = 189
    spec.xline = 193
    spec.format = 1
    spec.sorting = 2
    spec.samples = np.arange(n_samples, dtype=np.int32)
    spec.tracecount = int(n_traces)

    text_lines = {
        1: 'SRC HEADER LINE 1',
        2: 'SRC HEADER LINE 2',
        37: 'SRC RESERVED C37',
        38: 'SRC RESERVED C38',
        39: 'SRC RESERVED C39',
        40: 'SRC RESERVED C40',
    }

    with segyio.create(str(path), spec) as f:
        f.text[0] = segyio.tools.create_text_header(text_lines)
        f.bin[segyio.BinField.Interval] = int(dt_us)
        for tr_idx in range(n_traces):
            f.header[tr_idx] = {
                segyio.TraceField.TRACE_SEQUENCE_LINE: int(tr_idx + 1),
                segyio.TraceField.FieldRecord: 1001,
                segyio.TraceField.TraceNumber: int(tr_idx + 11),
                segyio.TraceField.CDP: 2001,
                segyio.TraceField.offset: int((tr_idx + 1) * 25),
                segyio.TraceField.SourceX: int(1000 + tr_idx * 3),
                segyio.TraceField.SourceY: int(2000 + tr_idx * 5),
                segyio.TraceField.GroupX: int(3000 + tr_idx * 7),
                segyio.TraceField.GroupY: int(4000 + tr_idx * 9),
                segyio.TraceField.SourceGroupScalar: 1,
            }
            wave = (
                np.arange(n_samples, dtype=np.float32) + np.float32(tr_idx * 10.0)
            ) / np.float32(10.0)
            f.trace[tr_idx] = wave


def test_write_segy_float32_like_input_with_text0_append(
    tmp_path: Path,
) -> None:
    src_path = tmp_path / 'src.sgy'
    dst_path = tmp_path / 'dst.sgy'
    n_traces = 5
    n_samples = 12
    dt_us = 2000
    _create_dummy_src_segy(
        src_path,
        n_traces=int(n_traces),
        n_samples=int(n_samples),
        dt_us=int(dt_us),
    )

    data_hw = (
        np.arange(n_traces * n_samples, dtype=np.float32).reshape(n_traces, n_samples)
    ) / np.float32(7.0)
    append_lines = [
        'pipeline=blindtrace',
        'task=infer',
        'model=encdec2d',
        'note=unit-test',
    ]

    out_path = write_segy_float32_like_input_with_text0_append(
        src_path=src_path,
        dst_path=dst_path,
        data_hw_float32=data_hw,
        text0_append_lines=append_lines,
        overwrite=False,
    )
    assert out_path == dst_path
    assert dst_path.is_file()

    with (
        segyio.open(str(src_path), 'r', ignore_geometry=True) as src,
        segyio.open(str(dst_path), 'r', ignore_geometry=True) as dst,
    ):
        assert int(len(src.trace)) == int(len(dst.trace)) == int(n_traces)
        assert int(src.samples.size) == int(dst.samples.size) == int(n_samples)
        assert int(dst.bin[segyio.BinField.Format]) == 5
        assert int(dst.bin[segyio.BinField.Interval]) == int(src.bin[segyio.BinField.Interval])

        for tr_idx in range(n_traces):
            src_header = dict(src.header[tr_idx])
            dst_header = dict(dst.header[tr_idx])
            assert src_header == dst_header

            dst_trace = np.asarray(dst.trace.raw[tr_idx])
            assert dst_trace.dtype == np.float32
            np.testing.assert_allclose(
                dst_trace.astype(np.float32, copy=False),
                data_hw[tr_idx],
                rtol=0.0,
                atol=0.0,
            )

        src_lines = segyio.tools.wrap(src.text[0], 80).splitlines()
        dst_lines = segyio.tools.wrap(dst.text[0], 80).splitlines()
        assert len(src_lines) == 40
        assert len(dst_lines) == 40
        for line_idx in range(36):
            assert dst_lines[line_idx] == src_lines[line_idx]

        assert dst_lines[36][4:].rstrip() == 'pipeline=blindtrace'
        assert dst_lines[37][4:].rstrip() == 'task=infer'
        assert dst_lines[38][4:].rstrip() == 'model=encdec2d'
        assert dst_lines[39][4:].rstrip() == 'note=unit-test'


def test_write_segy_float32_like_input_with_text0_append_rejects_too_many_lines(
    tmp_path: Path,
) -> None:
    src_path = tmp_path / 'src.sgy'
    dst_path = tmp_path / 'dst.sgy'
    _create_dummy_src_segy(src_path, n_traces=2, n_samples=8, dt_us=2000)

    data_hw = np.zeros((2, 8), dtype=np.float32)
    append_lines = ['l1', 'l2', 'l3', 'l4', 'l5']

    with pytest.raises(ValueError, match='at most 4 lines'):
        write_segy_float32_like_input_with_text0_append(
            src_path=src_path,
            dst_path=dst_path,
            data_hw_float32=data_hw,
            text0_append_lines=append_lines,
            overwrite=False,
        )


def test_line80_to_content76_accepts_c01_and_cspace_prefixes() -> None:
    line_no = 1
    content = 'SRC LINE 01'.ljust(76)
    line_c01 = f'C01 {content}'
    line_cspace = f'C 1 {content}'

    assert _line80_to_content76(line_c01, line_no=line_no) == content
    assert _line80_to_content76(line_cspace, line_no=line_no) == content
