from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ['COARSE_REQUIRED_KEYS', 'load_coarse_npz', 'save_coarse_npz']


COARSE_REQUIRED_KEYS = (
    'dt_sec',
    'n_samples_orig',
    'n_traces',
    'ffid_values',
    'chno_values',
    'offsets_m',
    'trace_indices',
    'coarse_pick_i',
    'coarse_pick_t_sec',
    'coarse_pmax',
    'coarse_prob_summary',
    'lineage',
)


def _coerce_scalar(name: str, value, *, dtype) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 0:
        msg = f'{name} must be scalar'
        raise ValueError(msg)
    return arr


def _coerce_vector(name: str, value, *, dtype, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 1:
        msg = f'{name} must be 1D'
        raise ValueError(msg)
    if int(arr.shape[0]) != int(length):
        msg = f'{name} length {int(arr.shape[0])} != n_traces {int(length)}'
        raise ValueError(msg)
    return arr


def _coerce_lineage(lineage) -> np.ndarray:
    if isinstance(lineage, np.ndarray):
        arr = lineage
    else:
        arr = np.asarray(lineage)
    if arr.ndim != 0:
        msg = 'lineage must be scalar'
        raise ValueError(msg)
    return arr


def save_coarse_npz(
    path: str | Path,
    *,
    dt_sec: float,
    n_samples_orig: int,
    n_traces: int,
    ffid_values,
    chno_values,
    offsets_m,
    trace_indices,
    coarse_pick_i,
    coarse_pick_t_sec,
    coarse_pmax,
    coarse_prob_summary,
    lineage,
) -> Path:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_traces_int = int(n_traces)
    n_samples_orig_int = int(n_samples_orig)
    if n_traces_int <= 0:
        msg = 'n_traces must be positive'
        raise ValueError(msg)
    if n_samples_orig_int <= 0:
        msg = 'n_samples_orig must be positive'
        raise ValueError(msg)

    arrays = {
        'dt_sec': _coerce_scalar('dt_sec', dt_sec, dtype=np.float32),
        'n_samples_orig': _coerce_scalar(
            'n_samples_orig',
            n_samples_orig_int,
            dtype=np.int32,
        ),
        'n_traces': _coerce_scalar('n_traces', n_traces_int, dtype=np.int32),
        'ffid_values': _coerce_vector(
            'ffid_values',
            ffid_values,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'chno_values': _coerce_vector(
            'chno_values',
            chno_values,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'offsets_m': _coerce_vector(
            'offsets_m',
            offsets_m,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'trace_indices': _coerce_vector(
            'trace_indices',
            trace_indices,
            dtype=np.int64,
            length=n_traces_int,
        ),
        'coarse_pick_i': _coerce_vector(
            'coarse_pick_i',
            coarse_pick_i,
            dtype=np.int32,
            length=n_traces_int,
        ),
        'coarse_pick_t_sec': _coerce_vector(
            'coarse_pick_t_sec',
            coarse_pick_t_sec,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'coarse_pmax': _coerce_vector(
            'coarse_pmax',
            coarse_pmax,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'coarse_prob_summary': _coerce_vector(
            'coarse_prob_summary',
            coarse_prob_summary,
            dtype=np.float32,
            length=n_traces_int,
        ),
        'lineage': _coerce_lineage(lineage),
    }

    coarse_pick_i_arr = arrays['coarse_pick_i']
    if np.any(coarse_pick_i_arr < 0) or np.any(coarse_pick_i_arr >= n_samples_orig_int):
        msg = 'coarse_pick_i must lie in [0, n_samples_orig)'
        raise ValueError(msg)

    np.savez_compressed(out_path, **arrays)
    return out_path


def load_coarse_npz(path: str | Path) -> dict[str, np.ndarray]:
    npz_path = Path(path).expanduser().resolve()
    if not npz_path.is_file():
        msg = f'coarse npz not found: {npz_path}'
        raise FileNotFoundError(msg)

    with np.load(npz_path, allow_pickle=False) as z:
        missing = [key for key in COARSE_REQUIRED_KEYS if key not in z.files]
        if missing:
            msg = f'coarse npz missing keys: {missing}'
            raise KeyError(msg)
        out = {key: z[key] for key in z.files}

    n_traces = int(np.asarray(out['n_traces']).item())
    n_samples_orig = int(np.asarray(out['n_samples_orig']).item())
    if n_traces <= 0:
        msg = 'n_traces must be positive'
        raise ValueError(msg)
    if n_samples_orig <= 0:
        msg = 'n_samples_orig must be positive'
        raise ValueError(msg)
    for key in (
        'ffid_values',
        'chno_values',
        'offsets_m',
        'trace_indices',
        'coarse_pick_i',
        'coarse_pick_t_sec',
        'coarse_pmax',
        'coarse_prob_summary',
    ):
        arr = np.asarray(out[key])
        if arr.ndim != 1 or int(arr.shape[0]) != n_traces:
            msg = f'{key} must be 1D with length n_traces'
            raise ValueError(msg)
    lineage = np.asarray(out['lineage'])
    if lineage.ndim != 0:
        msg = 'lineage must be scalar'
        raise ValueError(msg)
    return out
