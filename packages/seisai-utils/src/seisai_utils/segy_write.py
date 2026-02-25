"""SEGY writing utilities.

This module provides helpers to write SEGY files by copying an input SEGY and
overwriting only the trace sample data while preserving text/binary/trace
headers and the destination sample format.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import segyio

_TEXT0_TOTAL_LINES = 40
_TEXT0_LINE_WIDTH = 80
_TEXT0_CONTENT_WIDTH = 76
_TEXT0_APPEND_MAX_LINES = 4
_TEXT0_APPEND_START_LINE_NO = 37


def _cast_like_dtype(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    dt = np.dtype(dtype)

    if np.issubdtype(dt, np.floating):
        return np.asarray(x, dtype=dt)

    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        y = np.rint(np.asarray(x, dtype=np.float64))
        y = np.clip(y, info.min, info.max)
        return y.astype(dt, copy=False)

    msg = f'unsupported dtype: {dt}'
    raise TypeError(msg)


def _normalize_text0_append_lines(lines: list[str]) -> list[str]:
    if not isinstance(lines, list):
        msg = 'text0_append_lines must be list[str]'
        raise TypeError(msg)
    if len(lines) > _TEXT0_APPEND_MAX_LINES:
        msg = f'text0_append_lines must contain at most {_TEXT0_APPEND_MAX_LINES} lines'
        raise ValueError(msg)

    normalized: list[str] = []
    for idx, line in enumerate(lines):
        if not isinstance(line, str):
            msg = f'text0_append_lines[{idx}] must be str'
            raise TypeError(msg)
        one_line = line.replace('\r', ' ').replace('\n', ' ')
        normalized.append(one_line[:_TEXT0_CONTENT_WIDTH])
    return normalized


def _text0_to_lines80(text0: bytearray | bytes | str) -> list[str]:
    if isinstance(text0, (bytearray, bytes)):
        raw = bytes(text0)
    elif isinstance(text0, str):
        raw = text0.encode('ascii', errors='ignore')
    else:
        msg = f'text[0] must be bytearray/bytes/str, got {type(text0)}'
        raise TypeError(msg)

    total = _TEXT0_TOTAL_LINES * _TEXT0_LINE_WIDTH
    if len(raw) < total:
        raw = raw + (b' ' * (total - len(raw)))
    elif len(raw) > total:
        raw = raw[:total]

    out: list[str] = []
    for line_idx in range(_TEXT0_TOTAL_LINES):
        start = line_idx * _TEXT0_LINE_WIDTH
        end = start + _TEXT0_LINE_WIDTH
        out.append(raw[start:end].decode('ascii', errors='ignore'))
    return out


def _line80_to_content76(line80: str, *, line_no: int) -> str:
    if not isinstance(line80, str):
        msg = 'line80 must be str'
        raise TypeError(msg)
    if not isinstance(line_no, int) or isinstance(line_no, bool):
        msg = 'line_no must be int'
        raise TypeError(msg)
    if line_no < 1 or line_no > _TEXT0_TOTAL_LINES:
        msg = f'line_no must be in [1,{_TEXT0_TOTAL_LINES}]'
        raise ValueError(msg)

    line_fixed = line80
    if len(line_fixed) < _TEXT0_LINE_WIDTH:
        line_fixed = line_fixed.ljust(_TEXT0_LINE_WIDTH)
    elif len(line_fixed) > _TEXT0_LINE_WIDTH:
        line_fixed = line_fixed[:_TEXT0_LINE_WIDTH]

    expected_prefixes = (
        f'C{line_no:>2} ',
        f'C{line_no:02d} ',
    )
    if line_fixed.startswith(expected_prefixes):
        return line_fixed[4 : 4 + _TEXT0_CONTENT_WIDTH]
    return line_fixed[:_TEXT0_CONTENT_WIDTH]


def _build_text0_with_appended_tail(
    *,
    src_text0: bytearray | bytes | str,
    text0_append_lines: list[str],
) -> str:
    src_lines80 = _text0_to_lines80(src_text0)
    line_dict: dict[int, str] = {}
    for line_idx, line80 in enumerate(src_lines80):
        line_no = line_idx + 1
        line_dict[line_no] = _line80_to_content76(line80, line_no=line_no)

    append_lines = _normalize_text0_append_lines(text0_append_lines)
    for idx, line in enumerate(append_lines):
        line_no = _TEXT0_APPEND_START_LINE_NO + idx
        line_dict[line_no] = line

    return segyio.tools.create_text_header(line_dict)


def _copy_binary_header_all_fields(
    *, src_bin, dst_bin, force_format_5: bool
) -> None:
    for key in src_bin:
        dst_bin[int(key)] = src_bin[key]
    if force_format_5:
        dst_bin[int(segyio.BinField.Format)] = 5


def _copy_trace_header_all_fields(*, src_header) -> dict[int, int]:
    src_dict = dict(src_header)
    dst_dict: dict[int, int] = {}
    for key, value in src_dict.items():
        dst_dict[int(key)] = int(value)
    return dst_dict


def write_segy_like_input(
    *,
    src_path: str | Path,
    out_dir: str | Path,
    out_suffix: str,
    data_hw: np.ndarray,
    overwrite: bool = False,
    trace_indices: list[int] | None = None,
) -> Path:
    """入力SEGYをコピーし、traceサンプルだけ差し替えたSEGYを出力する。.

    - text/binary/trace header は変更しない(srcを丸ごとコピー→traceのみ上書き)
    - sample format は「コピーされたdstの形式」に従う(=維持)
    - data_hw は (n_traces, n_samples) または trace_indices 指定時 (len(indices), n_samples)

    Args:
            src_path: 入力SEGY
            out_dir: 出力ディレクトリ
            out_suffix: 出力ファイル名の末尾(例: "_pred.sgy")
            data_hw: 上書きしたい trace データ(H=trace, W=sample)
            overwrite: 出力ファイルが存在する場合に上書きするか
            trace_indices: 上書き対象のtrace index(0-based)。Noneなら全traceを上書き。

    Returns:
            dst_path: 出力SEGYパス

    """
    src_path = Path(src_path)
    if not src_path.is_file():
        msg = f'src segy not found: {src_path}'
        raise FileNotFoundError(msg)

    if not isinstance(out_suffix, str) or len(out_suffix) == 0:
        msg = 'out_suffix must be non-empty str'
        raise ValueError(msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dst_path = out_dir / f'{src_path.stem}{out_suffix}'
    if dst_path.exists() and not overwrite:
        msg = f'output already exists: {dst_path}'
        raise FileExistsError(msg)

    data_hw = np.asarray(data_hw)
    if data_hw.ndim != 2:
        msg = f'data_hw must be (H,W), got {data_hw.shape}'
        raise ValueError(msg)

    # ヘッダ含め完全コピー
    shutil.copyfile(src_path, dst_path)

    in_f = segyio.open(str(src_path), 'r', ignore_geometry=True)
    out_f = segyio.open(str(dst_path), 'r+', ignore_geometry=True)

    n_traces = len(in_f.trace)
    if n_traces <= 0:
        msg = 'input segy has no traces'
        raise ValueError(msg)

    # 出力側のトレース dtype(sample format維持)
    out_sample_dtype = np.asarray(out_f.trace[0]).dtype

    # 入力のサンプル数(trace長)を基準に整合チェック
    tr0_in = np.asarray(in_f.trace.raw[0])
    n_samples = int(tr0_in.shape[0])
    if n_samples <= 0:
        msg = 'input segy has invalid trace length'
        raise ValueError(msg)

    if int(data_hw.shape[1]) != n_samples:
        msg = f'data_hw W must equal n_samples={n_samples}, got {int(data_hw.shape[1])}'
        raise ValueError(msg)

    if trace_indices is None:
        if int(data_hw.shape[0]) != n_traces:
            msg = (
                f'data_hw H must equal n_traces={n_traces}, got {int(data_hw.shape[0])}'
            )
            raise ValueError(msg)
        indices = list(range(n_traces))
    else:
        if not isinstance(trace_indices, list) or not all(
            isinstance(i, int) for i in trace_indices
        ):
            msg = 'trace_indices must be list[int]'
            raise TypeError(msg)
        if len(trace_indices) == 0:
            msg = 'trace_indices must be non-empty'
            raise ValueError(msg)
        if len(set(trace_indices)) != len(trace_indices):
            msg = 'trace_indices must be unique'
            raise ValueError(msg)
        if min(trace_indices) < 0 or max(trace_indices) >= n_traces:
            msg = f'trace_indices out of range: valid=[0,{n_traces - 1}]'
            raise ValueError(msg)
        if int(data_hw.shape[0]) != len(trace_indices):
            msg = f'data_hw H must equal len(trace_indices)={len(trace_indices)}, got {int(data_hw.shape[0])}'
            raise ValueError(msg)
        indices = trace_indices

    # 書き込み(ヘッダは触らない。traceサンプルだけ差し替え)
    for j, tr_idx in enumerate(indices):
        tr = _cast_like_dtype(data_hw[j], out_sample_dtype)
        out_f.trace[int(tr_idx)] = tr

    in_f.close()
    out_f.close()

    return dst_path


def write_segy_float32_like_input_with_text0_append(
    *,
    src_path: str | Path,
    dst_path: str | Path,
    data_hw_float32: np.ndarray,
    text0_append_lines: list[str],
    overwrite: bool = False,
) -> Path:
    """Write float32 SEG-Y with trace headers copied from input and text[0] tail appended.

    - 出力 format は IEEE float32(format=5) に固定
    - trace header は全trace・全フィールドを src から dst へコピー
    - text[0] は src をベースに C37-C40 を append lines で上書き
    """
    src_path = Path(src_path)
    if not src_path.is_file():
        msg = f'src segy not found: {src_path}'
        raise FileNotFoundError(msg)

    dst_path = Path(dst_path)
    dst_parent = dst_path.parent
    dst_parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        if not overwrite:
            msg = f'output already exists: {dst_path}'
            raise FileExistsError(msg)
        if not dst_path.is_file():
            msg = f'output path exists and is not a file: {dst_path}'
            raise ValueError(msg)
        dst_path.unlink()

    data_hw = np.asarray(data_hw_float32, dtype=np.float32)
    if data_hw.ndim != 2:
        msg = f'data_hw_float32 must be (H,W), got {data_hw.shape}'
        raise ValueError(msg)
    _normalize_text0_append_lines(text0_append_lines)

    with segyio.open(str(src_path), 'r', ignore_geometry=True) as src:
        n_traces = len(src.trace)
        if n_traces <= 0:
            msg = 'input segy has no traces'
            raise ValueError(msg)

        tr0 = np.asarray(src.trace.raw[0])
        n_samples = int(tr0.shape[0])
        if n_samples <= 0:
            msg = 'input segy has invalid trace length'
            raise ValueError(msg)

        if tuple(data_hw.shape) != (n_traces, n_samples):
            msg = (
                f'data_hw_float32 shape must be ({n_traces},{n_samples}), '
                f'got {tuple(data_hw.shape)}'
            )
            raise ValueError(msg)

        spec = segyio.spec()
        spec.tracecount = int(n_traces)
        spec.samples = np.asarray(src.samples, dtype=np.int32)
        if int(spec.samples.shape[0]) != n_samples:
            msg = (
                'spec.samples length must match trace sample length: '
                f'len(samples)={int(spec.samples.shape[0])}, n_samples={n_samples}'
            )
            raise ValueError(msg)
        spec.format = 5

        sorting = getattr(src, 'sorting')
        if sorting is None:
            pass
        elif isinstance(sorting, bool) or not isinstance(sorting, (int, np.integer)):
            msg = f'input segy sorting must be int or None, got {type(sorting)}'
            raise TypeError(msg)
        else:
            spec.sorting = int(sorting)

        with segyio.create(str(dst_path), spec) as dst:
            _copy_binary_header_all_fields(
                src_bin=src.bin,
                dst_bin=dst.bin,
                force_format_5=True,
            )

            if len(src.text) <= 0:
                msg = 'input segy has no textual file header (text[0])'
                raise ValueError(msg)
            dst.text[0] = _build_text0_with_appended_tail(
                src_text0=src.text[0],
                text0_append_lines=text0_append_lines,
            )

            for tr_idx in range(n_traces):
                dst.header[tr_idx] = _copy_trace_header_all_fields(
                    src_header=src.header[tr_idx],
                )
                dst.trace[tr_idx] = data_hw[tr_idx]

            dst.flush()

    return dst_path


__all__ = [
    'write_segy_float32_like_input_with_text0_append',
    'write_segy_like_input',
]
