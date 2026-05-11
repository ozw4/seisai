from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    'iter_qc_gathers',
    'sort_gather_indices_for_qc',
]

SUPPORTED_QC_GATHER_SELECTIONS = ('first', 'even')

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


def _select_evenly_spaced_gather_candidates(
    candidates: list[tuple[int, np.ndarray]],
    *,
    max_gathers: int,
) -> list[tuple[int, np.ndarray]]:
    if max_gathers <= 0 or not candidates:
        return []
    if len(candidates) <= max_gathers:
        return list(candidates)

    keys = np.asarray([key for key, _indices in candidates], dtype=np.float64)
    targets = np.linspace(float(keys[0]), float(keys[-1]), int(max_gathers))
    selected: list[tuple[int, np.ndarray]] = []
    selected_indices: set[int] = set()
    original_positions = np.arange(len(candidates), dtype=np.int64)

    for target in targets:
        # Prefer the nearest gather key. When the target is exactly between two
        # integer keys, choose the lower key so 1..300 with count=3 gives
        # 1,150,300 rather than 1,151,300.
        order = np.lexsort((original_positions, keys, np.abs(keys - target)))
        for candidate_index in order:
            idx = int(candidate_index)
            if idx in selected_indices:
                continue
            selected_indices.add(idx)
            selected.append(candidates[idx])
            break

    return selected


def iter_qc_gathers(
    info: Mapping[str, Any],
    *,
    primary_keys: Sequence[str],
    max_gathers: int,
    skip_gather_keys: Mapping[str, set[int] | frozenset[int]],
    max_traces_per_gather: int | None,
    segy_path: str | Path | None = None,
    gather_selection: str = 'first',
) -> Iterator[tuple[str, int, np.ndarray]]:
    yielded = 0
    max_gathers_int = int(max_gathers)
    if max_gathers_int <= 0:
        return

    selection = str(gather_selection)
    if selection not in SUPPORTED_QC_GATHER_SELECTIONS:
        msg = (
            'gather_selection must be one of: '
            + ', '.join(SUPPORTED_QC_GATHER_SELECTIONS)
        )
        raise ValueError(msg)

    for primary_key in primary_keys:
        if primary_key not in SUPPORTED_QC_PRIMARY_KEYS:
            msg = f'unsupported dataset.primary_keys value: {primary_key}'
            raise ValueError(msg)
        key_to_indices = info.get(f'{primary_key}_key_to_indices')
        if key_to_indices is None:
            continue
        skip_for_primary = skip_gather_keys.get(primary_key, set())
        candidates: list[tuple[int, np.ndarray]] = []
        for gather_key in sorted(key_to_indices):
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
            candidates.append((gather_key_i, trace_indices))

        remaining = max_gathers_int - yielded
        if remaining <= 0:
            return
        if selection == 'even':
            selected = _select_evenly_spaced_gather_candidates(
                candidates,
                max_gathers=remaining,
            )
        else:
            selected = candidates[:remaining]

        for gather_key_i, trace_indices in selected:
            if yielded >= max_gathers_int:
                return
            yield primary_key, gather_key_i, trace_indices
            yielded += 1
