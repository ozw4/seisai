import datetime
import textwrap
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, Optional

import numpy as np

SENTINEL: int = -9999

GatherRange = Optional[tuple[int, int] | Iterable[int]]
Mode = Literal['trace', 'gather']
from typing import Optional

SENTINEL: int = -9999


def _parse_int_or_none(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    if s[0] == '-':
        return int(s) if s[1:].isdigit() else None
    return int(s) if s.isdigit() else None


def _iter_fixed_width(payload: str, width: int) -> list[str]:
    return [payload[i : i + width] for i in range(0, len(payload), width)]


def load_fb_irasformat(
    fb_file: str,
    dt: float,
    *,
    strict: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Load grstat first-break time dump (IRAS-like) into numpy array.

    This parser expects lines like:
        * rec.no.=    1
        fb            1   10    6   20 ...
    i.e.:
        - "fb" at columns [0:2]
        - start channel field at [2:15] (width=13)
        - end channel field at [15:20] (width=5)
        - values start at [20:], each width=5

    Args:
        fb_file: Path to CRD text file.
        dt: Sampling interval multiplier used when the file was created.
            Output samples are computed as floor(fb_time / dt).
        strict: If True, validates:
            - fb blocks are contiguous per record (start == previous_end + 1)
            - number of values per record equals record channel count
            - all records have the same channel count
        verbose: If True, prints record count and estimated channel count.

    Returns:
        1D numpy array of length (record_count * channel_count) in sample index domain.
        Sentinel (-9999) is converted to 0.

    Raises:
        ValueError: If dt <= 0, file format is broken, or strict checks fail.
        FileNotFoundError: If fb_file does not exist.

    """
    if dt <= 0:
        msg = 'dt must be > 0.'
        raise ValueError(msg)

    fb_values: list[int] = []
    record_count = 0

    record_value_counts: list[int] = []
    record_channel_counts: list[int] = []

    current_value_count = 0
    current_max_channel = 0
    expected_next_start = 1
    in_record = False

    with open(fb_file) as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip('\n')

            if line.startswith('* rec.no.='):
                if in_record:
                    record_value_counts.append(current_value_count)
                    record_channel_counts.append(current_max_channel)

                record_count += 1
                in_record = True
                current_value_count = 0
                current_max_channel = 0
                expected_next_start = 1
                continue

            if not line.startswith('fb'):
                continue

            if not in_record:
                msg = f'fb line appears before first record at line {line_no}.'
                raise ValueError(msg)

            if len(line) < 20:
                msg = f'fb line too short at line {line_no}: {line}'
                raise ValueError(msg)

            start = _parse_int_or_none(line[2:15])
            end = _parse_int_or_none(line[15:20])
            if start is None or end is None:
                msg = f'invalid fb start/end at line {line_no}: {line}'
                raise ValueError(msg)

            if strict and start != expected_next_start:
                msg = (
                    f'non-contiguous fb block at line {line_no}: start={start}, '
                    f'expected={expected_next_start}'
                )
                raise ValueError(msg)
            expected_next_start = end + 1
            current_max_channel = max(current_max_channel, end)

            payload = line[20:]
            chunks = _iter_fixed_width(payload, 5)

            for c in chunks:
                v = _parse_int_or_none(c)
                if v is None or v == SENTINEL:
                    fb_values.append(0)
                else:
                    fb_values.append(v)
                current_value_count += 1

    if in_record:
        record_value_counts.append(current_value_count)
        record_channel_counts.append(current_max_channel)

    if record_count == 0:
        msg = 'no records found in file.'
        raise ValueError(msg)

    # 期待するチャンネル数（レコードごとの最大 end）
    channel_count = max(record_channel_counts)
    if verbose:
        print(f'推定チャンネル数: {channel_count}')
        print(f'レコード数: {record_count}')

    if strict:
        # 各レコードは「そのレコードのmax_channel個」の値を持つはず
        for i, (vc, cc) in enumerate(
            zip(record_value_counts, record_channel_counts, strict=False), start=1
        ):
            if vc != cc:
                msg = f'record {i}: value count mismatch: {vc} != channel_count_in_record {cc}'
                raise ValueError(msg)
        # すべてのレコードでチャンネル数が同じ（運用前提がそうなら）
        if any(cc != channel_count for cc in record_channel_counts):
            msg = f'channel count differs across records: {record_channel_counts}'
            raise ValueError(msg)

    fb_time = np.asarray(fb_values, dtype=np.int32)

    # 元コード互換: fb_time // dt だが dt が float でも安全に floor で整数化
    fb_sample = np.floor(fb_time / dt).astype(np.int32)
    fb_sample[fb_sample < 0] = 0

    expected_len = record_count * channel_count
    if fb_sample.size != expected_len:
        msg = f'record/channel mismatch: {fb_sample.size} != {record_count} * {channel_count}'
        raise ValueError(msg)

    return fb_sample


def _normalize_gather_numbers(gather_range: GatherRange, n_gathers: int) -> list[int]:
    """Normalize gather numbers for labeling output.

    Args:
        gather_range: Range specification for gather numbers.
            - None: Uses [1, 2, ..., n_gathers].
            - (start, end): Uses [start, ..., end] (end inclusive).
            - Iterable[int]: Uses list(gather_range) as-is.
        n_gathers: Number of gathers (rows) in fb array.

    Returns:
        A list of gather numbers with length exactly `n_gathers`.

    Raises:
        TypeError: If (start, end) is given but not integers.
        ValueError: If end < start or resulting length != n_gathers.

    """
    if gather_range is None:
        numbers = list(range(1, n_gathers + 1))
    elif isinstance(gather_range, (tuple, list)) and len(gather_range) == 2:
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
    """Build grstat header text with customizable comment.

    Args:
        now: Timestamp used in DATE line.
        header_comment: Comment text shown in the framed header area.
            Long text is wrapped automatically.

    Returns:
        Header string for output file.

    """
    # Keep same style as the original script: 72-char star lines + framed comment lines.
    star_line = '*' * 72 + '\n'

    # The first title line is traditionally this exact text.
    title_line = (
        '*****< grstat first-break time dump : ver.dec96 >***********************\n'
    )

    prefix = '****'
    suffix = '*****'
    inner_width = 72 - len(prefix) - len(suffix)

    def frame(text: str) -> str:
        # clip just in case (wrap should prevent overflow)
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


def numpy2fbcrd(
    dt: float,
    fbnum: np.ndarray | Sequence[Sequence[float]],
    gather_range: GatherRange = None,
    output_name: str = 'fb_ml_prediction.txt',
    original: str | None = None,
    mode: Mode = 'gather',
    header_comment: str = 'machine learning fb pick',
) -> np.ndarray:
    """Convert numpy first-break picks to grstat-style text dump.

    This function converts predicted first-break picks to a
    "grstat first-break time dump" text format.

    Notes:
        - If `original` is provided, original picks can overwrite predictions.
          Behavior depends on `mode`:
            - "trace": For each element where original (ori) is non-zero, replace.
            - "gather": For indices specified by `npy/pick_gather_index.npy`, replace.
        - Values equal to 0 in the (post-overwrite) array are treated as "no pick".
        - Output values are integer values after multiplying by `dt` and rounding
          to nearest integer (np.rint).
        - Non-positive values are set to sentinel (-9999).
        - Header comment can be customized via `header_comment`.

    Args:
        dt: Sampling interval multiplier. The output value is computed as
            round(fb * dt) and written as integers.
        fbnum: 2D prediction array of shape (n_gathers, n_traces).
            Accepts a numpy array or array-like.
        gather_range: Controls gather numbering in the output.
            - None: labels rec.no as 1..n_gathers.
            - (start, end): labels rec.no as start..end (inclusive).
            - Iterable[int]: uses given values directly.
            Length must match n_gathers.
        output_name: Output text file path.
        original: Path to original picks .npy. If None, no overwrite is applied.
        mode: Overwrite mode, either "trace" or "gather".
        header_comment: Comment text in the header. Default is
            "machine learning fb pick". Long text is wrapped automatically.

    Returns:
        A 2D numpy array of dtype int32 containing the values written to file.
        Shape is (n_gathers, n_traces). Sentinel values are -9999.

    Raises:
        ValueError: If fbnum is not 2D or gather_range length mismatch.
        FileNotFoundError: If required .npy files are missing.
        ValueError: If original and fbnum shapes mismatch.
        ValueError: If mode is invalid.

    """
    print('start process numpy2fbcrd')

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
    header = _build_grstat_header(now, header_comment)

    with open(output_name, 'w') as f:
        f.write(header)

        for rec_no, row in zip(gather_numbers, fb_time, strict=False):
            f.write(f'* rec.no.={rec_no:5d}\n')

            for start in range(0, n_traces, 10):
                end = min(start + 10, n_traces)
                f.write(f'fb{start + 1:13d}{end:5d}')
                f.write(''.join(f'{int(v):5d}' for v in row[start:end]))
                f.write('\n')

        f.write('*\n')

    print('save txt file: ' + str(output_name))
    print('finish process numpy2fbcrd')
    return fb_time
