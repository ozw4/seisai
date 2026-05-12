import datetime
import textwrap
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, NamedTuple, Optional

import numpy as np

SENTINEL: int = -9999

GatherRange = Optional[tuple[int, int] | Iterable[int]]
Mode = Literal['trace', 'gather']
GrstatOutputFormat = Literal['legacy', 'recno_channel_range']


class GrstatMatrix(NamedTuple):
    """Parsed grstat first-break matrix in sample-index domain.

    Attributes:
        record_numbers: 1D array of rec.no./FFID values in file order.
        samples: 2D array with shape ``(n_records, n_channels)``. Values are
            sample indices computed as ``floor(raw_values / dt_multiplier)``.
            Invalid/no-pick values are stored as 0.
        raw_values: 2D array with the original grstat numeric values before
            sample conversion. Missing/no-pick cells are stored as 0 or the
            value present in the file, such as -9999.
    """

    record_numbers: np.ndarray
    samples: np.ndarray
    raw_values: np.ndarray


def _parse_int_or_none(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    if s[0] == '-':
        return int(s) if s[1:].isdigit() else None
    return int(s) if s.isdigit() else None


def _parse_float_or_none(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _iter_fixed_width(payload: str, width: int) -> list[str]:
    return [payload[i : i + width] for i in range(0, len(payload), width)]


def _add_record_if_needed(
    *,
    records_by_rec: dict[int, dict[int, float]],
    record_order: list[int],
    expected_next_start_by_rec: dict[int, int],
    rec_no: int,
) -> None:
    if rec_no in records_by_rec:
        return
    records_by_rec[rec_no] = {}
    record_order.append(rec_no)
    expected_next_start_by_rec[rec_no] = 1


def _parse_legacy_fb_line(line: str, line_no: int) -> tuple[int, int, list[float]]:
    if len(line) < 20:
        msg = f'fb line too short at line {line_no}: {line}'
        raise ValueError(msg)

    start = _parse_int_or_none(line[2:15])
    end = _parse_int_or_none(line[15:20])
    if start is None or end is None or end < start:
        msg = f'invalid fb start/end at line {line_no}: {line}'
        raise ValueError(msg)

    chunks = _iter_fixed_width(line[20:], 5)
    expected_count = end - start + 1
    if len(chunks) != expected_count:
        msg = (
            f'fb value count mismatch at line {line_no}: '
            f'{len(chunks)} != {expected_count}'
        )
        raise ValueError(msg)

    values: list[float] = []
    for chunk in chunks:
        value = _parse_float_or_none(chunk)
        if value is None:
            msg = f'invalid fb value at line {line_no}: {chunk!r}'
            raise ValueError(msg)
        values.append(value)
    return start, end, values


def _parse_recno_channel_range_fb_line(
    line: str, line_no: int
) -> tuple[int, int, int, list[float]]:
    # New compact format:
    #   fb:      recno  start_ch  end_ch  fb(start_ch) ... fb(end_ch)
    # Example:
    #   fb:          1       1       5    92.000    82.000 ...
    parts = line.replace(':', ' ', 1).split()
    if len(parts) < 5 or parts[0] != 'fb':
        msg = f'invalid recno-channel-range fb line at line {line_no}: {line!r}'
        raise ValueError(msg)

    rec_no = _parse_int_or_none(parts[1])
    start = _parse_int_or_none(parts[2])
    end = _parse_int_or_none(parts[3])
    if rec_no is None or start is None or end is None or end < start:
        msg = f'invalid recno/start/end at line {line_no}: {line!r}'
        raise ValueError(msg)

    expected_count = end - start + 1
    raw_values = parts[4:]
    if len(raw_values) != expected_count:
        msg = (
            f'fb value count mismatch at line {line_no}: '
            f'{len(raw_values)} != {expected_count}'
        )
        raise ValueError(msg)

    values: list[float] = []
    for token in raw_values:
        value = _parse_float_or_none(token)
        if value is None:
            msg = f'invalid fb value at line {line_no}: {token!r}'
            raise ValueError(msg)
        values.append(value)
    return rec_no, start, end, values


def load_grstat_matrix(
    fb_file: str | Path,
    *,
    dt_multiplier: float,
    strict_blocks: bool = True,
    strict_channel_count: bool = False,
) -> GrstatMatrix:
    """Load grstat first-break text into a 2D matrix while preserving rec.no.

    Both supported grstat layouts are accepted automatically.

    Legacy layout::

        * rec.no.=    1
        fb            1    5   20   40-9999   80  100

    New recno-channel-range layout::

        fb:          1       1       5    92.000    82.000 ...

    Args:
        fb_file: Path to CRD text file.
        dt_multiplier: Sampling interval multiplier used to convert grstat
            numeric values to sample indices. Samples are computed as
            ``floor(raw_value / dt_multiplier)``.
        strict_blocks: If True, validates contiguous channel blocks within each
            record and rejects duplicate channels.
        strict_channel_count: If True, validates that all records have the same
            number of channels and no missing channels up to each record's max
            channel. This reproduces the strict behavior of
            :func:`load_fb_irasformat`.

    Returns:
        :class:`GrstatMatrix` containing record numbers, sample-index matrix,
        and raw grstat values.

    Raises:
        ValueError: If ``dt_multiplier <= 0`` or the file content is invalid.
        FileNotFoundError: If ``fb_file`` does not exist.

    """
    if dt_multiplier <= 0:
        msg = 'dt_multiplier must be > 0.'
        raise ValueError(msg)

    path = Path(fb_file)
    if not path.is_file():
        raise FileNotFoundError(path)

    records_by_rec: dict[int, dict[int, float]] = {}
    record_order: list[int] = []
    expected_next_start_by_rec: dict[int, int] = {}
    current_rec: int | None = None

    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip('\n')

            if line.startswith('* rec.no.='):
                rec = _parse_int_or_none(line[len('* rec.no.=') :])
                if rec is None:
                    msg = f'invalid rec.no. at line {line_no}: {line!r}'
                    raise ValueError(msg)
                current_rec = rec
                _add_record_if_needed(
                    records_by_rec=records_by_rec,
                    record_order=record_order,
                    expected_next_start_by_rec=expected_next_start_by_rec,
                    rec_no=rec,
                )
                continue

            if line.startswith('fb:'):
                rec, start, end, values = _parse_recno_channel_range_fb_line(
                    line, line_no
                )
                _add_record_if_needed(
                    records_by_rec=records_by_rec,
                    record_order=record_order,
                    expected_next_start_by_rec=expected_next_start_by_rec,
                    rec_no=rec,
                )
            elif line.startswith('fb'):
                if current_rec is None:
                    msg = f'fb line appears before first record at line {line_no}.'
                    raise ValueError(msg)
                rec = current_rec
                start, end, values = _parse_legacy_fb_line(line, line_no)
            else:
                continue

            if strict_blocks and start != expected_next_start_by_rec[rec]:
                msg = (
                    f'non-contiguous fb block at line {line_no}: start={start}, '
                    f'expected={expected_next_start_by_rec[rec]}'
                )
                raise ValueError(msg)
            expected_next_start_by_rec[rec] = end + 1

            record_values = records_by_rec[rec]
            for offset, value in enumerate(values):
                chno = start + offset
                if chno < 1:
                    msg = f'invalid channel number rec={rec}, chno={chno}'
                    raise ValueError(msg)
                if strict_blocks and chno in record_values:
                    msg = f'duplicate channel rec={rec}, chno={chno} at line {line_no}'
                    raise ValueError(msg)
                record_values[chno] = float(value)

    if not record_order:
        msg = f'no records found in file: {path}'
        raise ValueError(msg)

    record_channel_counts = [max(records_by_rec[rec]) for rec in record_order]
    channel_count = max(record_channel_counts)

    if strict_channel_count:
        for rec in record_order:
            values = records_by_rec[rec]
            max_ch = max(values)
            if len(values) != max_ch:
                missing = sorted(set(range(1, max_ch + 1)).difference(values))
                msg = (
                    f'record {rec}: value count mismatch: {len(values)} != {max_ch}; '
                    f'first missing channel={missing[0] if missing else "unknown"}'
                )
                raise ValueError(msg)
    if strict_channel_count and any(cc != channel_count for cc in record_channel_counts):
        msg = f'channel count differs across records: {record_channel_counts}'
        raise ValueError(msg)

    raw_values = np.zeros((len(record_order), channel_count), dtype=np.float64)
    for row_i, rec in enumerate(record_order):
        for chno, value in records_by_rec[rec].items():
            raw_values[row_i, chno - 1] = float(value)

    fb_sample = np.floor(raw_values / float(dt_multiplier)).astype(np.int32)
    fb_sample[raw_values <= 0] = 0
    fb_sample[np.isclose(raw_values, float(SENTINEL))] = 0
    fb_sample[fb_sample < 0] = 0

    return GrstatMatrix(
        record_numbers=np.asarray(record_order, dtype=np.int32),
        samples=fb_sample,
        raw_values=raw_values,
    )


def load_fb_irasformat(
    fb_file: str,
    dt: float,
    *,
    strict: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Load grstat first-break time dump into a 1D sample-index array.

    Both supported grstat layouts are accepted automatically.

    Legacy layout::

        * rec.no.=    1
        fb            1    5   20   40-9999   80  100

    New recno-channel-range layout::

        fb:          1       1       5    92.000    82.000 ...

    The new layout has columns ``recno, start_ch, end_ch, fb(start_ch) ...``
    on every ``fb:`` line and does not require separate ``* rec.no.=`` lines.

    Args:
        fb_file: Path to CRD text file.
        dt: Sampling interval multiplier used when the file was created.
            Output samples are computed as floor(fb_time / dt).
        strict: If True, validates contiguous channel blocks per record and
            equal channel count across records.
        verbose: If True, prints record count and estimated channel count.

    Returns:
        Flattened 1D numpy array in sample-index domain. Invalid/sentinel picks
        are converted to 0.

    Raises:
        ValueError: If dt <= 0, file format is broken, or strict checks fail.
        FileNotFoundError: If fb_file does not exist.

    """
    parsed = load_grstat_matrix(
        fb_file,
        dt_multiplier=dt,
        strict_blocks=strict,
        strict_channel_count=strict,
    )
    if verbose:
        print(f'推定チャンネル数: {parsed.samples.shape[1]}')
        print(f'レコード数: {parsed.samples.shape[0]}')
    return parsed.samples.reshape(-1)


def _normalize_gather_numbers(gather_range: GatherRange, n_gathers: int) -> list[int]:
    """Normalize gather numbers for labeling output.

    Args:
        gather_range: Range specification for gather numbers.
            - None: Uses [1, 2, ..., n_gathers].
            - tuple(start, end): Uses [start, ..., end] (end inclusive).
            - list/Iterable[int]: Uses explicit values as-is.
        n_gathers: Number of gathers (rows) in fb array.

    Returns:
        A list of gather numbers with length exactly `n_gathers`.

    Raises:
        TypeError: If (start, end) is given but not integers.
        ValueError: If end < start or resulting length != n_gathers.

    """
    if gather_range is None:
        numbers = list(range(1, n_gathers + 1))
    elif isinstance(gather_range, tuple) and len(gather_range) == 2:
        start, end = gather_range
        if not (isinstance(start, int) and isinstance(end, int)):
            msg = 'gather_range (start, end) must be integers.'
            raise TypeError(msg)
        if end < start:
            msg = 'gather_range end must be >= start.'
            raise ValueError(msg)
        numbers = list(range(start, end + 1))
    else:
        numbers = list(gather_range)

    if len(numbers) != n_gathers:
        msg = (
            f'gather_range length ({len(numbers)}) must match fbnum rows ({n_gathers}).'
        )
        raise ValueError(msg)
    return numbers


def _apply_original_picks(
    fb_pred: np.ndarray, original_path: str | None, mode: Mode
) -> np.ndarray:
    """Apply original picks to prediction array.

    Args:
        fb_pred: Prediction array of shape (n_gathers, n_traces).
        original_path: Path to original pick .npy file.
            If None, returns `fb_pred` unchanged.
        mode: Replacement strategy.
            - "trace": Replace per-trace where original pick exists (ori != 0).
            - "gather": Replace indices specified by `npy/pick_gather_index.npy`.

    Returns:
        A new ndarray with original picks applied when requested.

    Raises:
        FileNotFoundError: If original_path or pick_gather_index.npy is missing.
        ValueError: If shapes mismatch or invalid pick index mask shape.
        ValueError: If mode is not one of ("trace", "gather").

    """
    if original_path is None:
        return fb_pred

    p = Path(original_path)
    if not p.is_file():
        msg = f'original file not found: {original_path}'
        raise FileNotFoundError(msg)

    ori = np.load(p)
    if ori.shape != fb_pred.shape:
        msg = f'shape mismatch: fbnum{fb_pred.shape} vs original{ori.shape}'
        raise ValueError(msg)

    if mode == 'trace':
        mask = ori != 0
        out = fb_pred.copy()
        out[mask] = ori[mask]
        return out

    if mode == 'gather':
        idx_path = Path('npy/pick_gather_index.npy')
        if not idx_path.is_file():
            msg = 'pick_gather_index.npy not found: npy/pick_gather_index.npy'
            raise FileNotFoundError(msg)

        pick_index = np.load(idx_path)

        if pick_index.dtype == bool:
            valid_shapes = {(fb_pred.shape[0],), fb_pred.shape}
            if pick_index.shape not in valid_shapes:
                msg = (
                    f'invalid boolean pick_index shape: {pick_index.shape} '
                    f'(expected {(fb_pred.shape[0],)} or {fb_pred.shape})'
                )
                raise ValueError(msg)

        out = fb_pred.copy()
        out[pick_index] = ori[pick_index]
        return out

    msg = "mode must be 'trace' or 'gather'."
    raise ValueError(msg)


def _build_grstat_header(now: datetime.datetime, header_comment: str) -> str:
    """Build legacy grstat header text with customizable comment."""
    # Keep same style as the original script: 72-char star lines + framed comment lines.
    star_line = '*' * 72 + '\n'
    title_line = (
        '*****< grstat first-break time dump : ver.dec96 >***********************\n'
    )

    prefix = '****'
    suffix = '*****'
    inner_width = 72 - len(prefix) - len(suffix)

    def frame(text: str) -> str:
        t = text[:inner_width]
        return f'{prefix}{t.center(inner_width)}{suffix}\n'

    wrapped = textwrap.wrap(header_comment, width=inner_width) or ['']

    header = [
        title_line,
        star_line,
        star_line,
        frame(''),
        *[frame(line) for line in wrapped],
        frame(''),
        star_line,
        f'*****************************< print DATE {now.ctime()} >****\n',
    ]
    return ''.join(header)


def _build_grstat_range_header(now: datetime.datetime, header_comment: str) -> str:
    """Build 80-column header for recno-channel-range grstat output."""

    def frame(content: str = '') -> str:
        return f'*{content[:78].ljust(78)}*\n'

    comment = (header_comment or 'first-break time dump')[:70]
    date_s = now.strftime('%Y-%m-%d')
    time_s = now.strftime('%H:%M:%S%z')
    return ''.join(
        [
            '** GRSTAT ver.dec96a : first-break time dump ***********************************\n',
            frame(),
            frame(f'  AREA: {comment}'),
            frame('  LINE: generated by seisai_pick.pickio.io_grstat'),
            frame(f'  DATE: {date_s}    TIME: {time_s}'),
            frame(),
            '*' * 80 + '\n',
        ]
    )


def numpy2fbcrd(
    dt: float,
    fbnum: np.ndarray | Sequence[Sequence[float]],
    gather_range: GatherRange = None,
    output_name: str = 'fb_ml_prediction.txt',
    original: str | None = None,
    mode: Mode = 'gather',
    header_comment: str = 'machine learning fb pick',
    output_format: GrstatOutputFormat = 'recno_channel_range',
    values_per_line: int | None = None,
) -> np.ndarray:
    """Convert numpy first-break picks to grstat-style text dump.

    Supports both legacy ``* rec.no.=`` blocks and the new compact format whose
    ``fb:`` lines contain ``recno, start_ch, end_ch, fb(start_ch) ...``.

    Args:
        dt: Sampling interval multiplier. The output value is computed as
            round(fb * dt).
        fbnum: 2D prediction array of shape (n_gathers, n_traces).
        gather_range: Controls gather numbering in the output.
        output_name: Output text file path.
        original: Path to original picks .npy. If None, no overwrite is applied.
        mode: Overwrite mode, either "trace" or "gather".
        header_comment: Comment text in the header.
        output_format: ``"recno_channel_range"`` for the new format or
            ``"legacy"`` for the historical ``* rec.no.=`` format.
        values_per_line: Number of channel values per ``fb`` line. Defaults to
            5 for the new format and 10 for legacy format.

    Returns:
        A 2D numpy array of dtype int32 containing the values written to file.
        Shape is (n_gathers, n_traces). Sentinel values are -9999.

    """
    print('start process numpy2fbcrd')

    if output_format not in ('legacy', 'recno_channel_range'):
        msg = "output_format must be 'legacy' or 'recno_channel_range'"
        raise ValueError(msg)

    if values_per_line is None:
        values_per_line = 10 if output_format == 'legacy' else 5
    if not isinstance(values_per_line, int) or values_per_line <= 0:
        msg = 'values_per_line must be a positive integer'
        raise ValueError(msg)

    fb_pred = np.asarray(fbnum)
    if fb_pred.ndim != 2:
        msg = f'fbnum must be 2D array, got shape {fb_pred.shape}'
        raise ValueError(msg)

    fb_pred = _apply_original_picks(fb_pred, original, mode)

    nopick = fb_pred == 0

    fb_time = np.rint(fb_pred * dt).astype(np.int32)
    fb_time[nopick] = SENTINEL
    fb_time[fb_time <= 0] = SENTINEL

    n_gathers, n_traces = fb_time.shape
    gather_numbers = _normalize_gather_numbers(gather_range, n_gathers)

    now = datetime.datetime.now()
    header = (
        _build_grstat_header(now, header_comment)
        if output_format == 'legacy'
        else _build_grstat_range_header(now, header_comment)
    )

    with open(output_name, 'w', encoding='utf-8') as f:
        f.write(header)

        for rec_no, row in zip(gather_numbers, fb_time, strict=False):
            if output_format == 'legacy':
                f.write(f'* rec.no.={rec_no:5d}\n')
                for start in range(0, n_traces, values_per_line):
                    end = min(start + values_per_line, n_traces)
                    f.write(f'fb{start + 1:13d}{end:5d}')
                    f.write(''.join(f'{int(v):5d}' for v in row[start:end]))
                    f.write('\n')
            else:
                for start in range(0, n_traces, values_per_line):
                    end = min(start + values_per_line, n_traces)
                    f.write(f'fb:{rec_no:11d}{start + 1:8d}{end:8d}')
                    f.write(''.join(f'{float(v):10.3f}' for v in row[start:end]))
                    f.write('\n')

        f.write('*\n')

    print('save txt file: ' + str(output_name))
    print('finish process numpy2fbcrd')
    return fb_time
