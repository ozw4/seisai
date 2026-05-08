from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    'iter_qc_gathers',
    'sort_gather_indices_for_qc',
]

SUPPORTED_QC_PRIMARY_KEYS = ('ffid', 'chno', 'cmp')


def sort_gather_indices_for_qc(
    info: Mapping[str, Any],
    *,
    primary_key: str,
    indices: np.ndarray,
) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return idx
    if primary_key == 'ffid':
        secondary = np.asarray(info['chno_values'], dtype=np.int64)[idx]
    elif primary_key == 'chno':
        secondary = np.asarray(info['ffid_values'], dtype=np.int64)[idx]
    elif primary_key == 'cmp':
        secondary = np.asarray(info['offsets'], dtype=np.float32)[idx]
    else:
        msg = f'unsupported dataset.primary_keys value: {primary_key}'
        raise ValueError(msg)
    order = np.argsort(secondary, kind='mergesort')
    return idx[order]


def iter_qc_gathers(
    info: Mapping[str, Any],
    *,
    primary_keys: Sequence[str],
    max_gathers: int,
    skip_gather_keys: Mapping[str, set[int] | frozenset[int]],
    max_traces_per_gather: int | None,
    segy_path: str | Path | None = None,
) -> Iterator[tuple[str, int, np.ndarray]]:
    yielded = 0
    max_gathers_int = int(max_gathers)
    if max_gathers_int <= 0:
        return

    for primary_key in primary_keys:
        if primary_key not in SUPPORTED_QC_PRIMARY_KEYS:
            msg = f'unsupported dataset.primary_keys value: {primary_key}'
            raise ValueError(msg)
        key_to_indices = info.get(f'{primary_key}_key_to_indices')
        if key_to_indices is None:
            continue
        skip_for_primary = skip_gather_keys.get(primary_key, set())
        for gather_key in sorted(key_to_indices):
            if yielded >= max_gathers_int:
                return
            gather_key_i = int(gather_key)
            raw_indices = key_to_indices[gather_key]
            n_traces = int(len(raw_indices))
            if gather_key_i in skip_for_primary:
                print(
                    f'skip gather by key: file={segy_path} '
                    f'primary={primary_key} key={gather_key_i} traces={n_traces}'
                )
                continue
            if (
                max_traces_per_gather is not None
                and n_traces > int(max_traces_per_gather)
            ):
                print(
                    f'skip oversized gather: file={segy_path} '
                    f'primary={primary_key} key={gather_key_i} '
                    f'traces={n_traces} limit={int(max_traces_per_gather)}'
                )
                continue
            indices = np.asarray(raw_indices, dtype=np.int64)
            trace_indices = sort_gather_indices_for_qc(
                info,
                primary_key=primary_key,
                indices=indices,
            )
            if int(trace_indices.size) <= 0:
                continue
            yield primary_key, gather_key_i, trace_indices
            yielded += 1
