from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

__all__ = ['compute_ref_stats', 'compute_ref_stats_from_records']


def _coerce_values(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)
    if int(arr.size) <= 0:
        msg = f'{name} must be non-empty'
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = f'{name} must contain only finite values'
        raise ValueError(msg)
    return arr


def _concat_record_values(
    records: Iterable[Mapping[str, Any]],
    *,
    key: str,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    count = 0
    for idx, record in enumerate(records):
        if not isinstance(record, Mapping):
            msg = f'records[{idx}] must be a mapping'
            raise TypeError(msg)
        if key not in record:
            msg = f'records[{idx}] is missing key {key!r}'
            raise KeyError(msg)
        chunks.append(_coerce_values(f'records[{idx}][{key!r}]', record[key]))
        count += 1
    if count <= 0:
        msg = 'records must be non-empty'
        raise ValueError(msg)
    return np.concatenate(chunks, axis=0)


def _positive_p99(name: str, values: np.ndarray) -> float:
    ref = float(np.percentile(values, 99))
    if not np.isfinite(ref) or ref <= 0.0:
        msg = f'{name} p99 must be > 0'
        raise ValueError(msg)
    return ref


def compute_ref_stats(*, time_abs_sec: Any, offset_m: Any) -> dict[str, float]:
    time_values = _coerce_values('time_abs_sec', time_abs_sec)
    if np.any(time_values < 0.0):
        msg = 'time_abs_sec must be >= 0'
        raise ValueError(msg)

    offset_values = _coerce_values('offset_m', offset_m)
    offset_abs = np.abs(offset_values)

    return {
        'time_ref_sec': _positive_p99('time_abs_sec', time_values),
        'offset_ref_m': _positive_p99('abs(offset_m)', offset_abs),
    }


def compute_ref_stats_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    time_key: str = 'time_abs_sec',
    offset_key: str = 'offset_m',
) -> dict[str, float]:
    if not isinstance(time_key, str) or not time_key:
        msg = 'time_key must be non-empty str'
        raise TypeError(msg)
    if not isinstance(offset_key, str) or not offset_key:
        msg = 'offset_key must be non-empty str'
        raise TypeError(msg)

    cached_records = tuple(records)
    return compute_ref_stats(
        time_abs_sec=_concat_record_values(cached_records, key=time_key),
        offset_m=_concat_record_values(cached_records, key=offset_key),
    )
