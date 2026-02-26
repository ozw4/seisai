from __future__ import annotations

import numpy as np


def require_npz_key(
    z: np.lib.npyio.NpzFile,
    key: str,
    *,
    context: str | None = None,
) -> np.ndarray:
    """Load a required key from npz, raising a clear KeyError if missing."""
    if key not in z.files:
        prefix = f'{context}: ' if context else ''
        msg = f'{prefix}npz missing key={key!r}. available={sorted(z.files)}'
        raise KeyError(msg)
    return np.asarray(z[key])


def _require_scalar(
    z: np.lib.npyio.NpzFile,
    key: str,
    *,
    context: str,
) -> np.ndarray:
    arr = require_npz_key(z, key, context=context)
    if arr.ndim != 0:
        msg = f'{context}: {key} must be scalar, got shape={arr.shape}'
        raise ValueError(msg)
    return arr


def npz_scalar_int(
    z: np.lib.npyio.NpzFile,
    key: str,
    *,
    context: str,
) -> int:
    arr = _require_scalar(z, key, context=context)
    if not np.issubdtype(arr.dtype, np.integer):
        msg = f'{context}: {key} must be int scalar, got dtype={arr.dtype}'
        raise TypeError(msg)
    return int(arr.item())


def npz_scalar_float(
    z: np.lib.npyio.NpzFile,
    key: str,
    *,
    context: str,
) -> float:
    arr = _require_scalar(z, key, context=context)
    if not np.issubdtype(arr.dtype, np.floating):
        msg = f'{context}: {key} must be float scalar, got dtype={arr.dtype}'
        raise TypeError(msg)
    return float(arr.item())


def npz_scalar_str(
    z: np.lib.npyio.NpzFile,
    key: str,
    *,
    context: str,
) -> str:
    arr = _require_scalar(z, key, context=context)
    val = arr.item()
    if isinstance(val, str):
        return val
    if isinstance(val, bytes):
        return val.decode()
    msg = f'{context}: {key} must be string scalar, got dtype={arr.dtype}'
    raise TypeError(msg)


def npz_1d(
    z: np.lib.npyio.NpzFile,
    key: str,
    *,
    context: str,
    n: int | None = None,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    arr = require_npz_key(z, key, context=context)
    if arr.ndim != 1:
        msg = f'{context}: {key} must be 1D, got shape={arr.shape}'
        raise ValueError(msg)
    if n is not None and arr.shape[0] != int(n):
        msg = f'{context}: {key} must have length {int(n)}, got {arr.shape[0]}'
        raise ValueError(msg)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


__all__ = [
    'npz_1d',
    'npz_scalar_float',
    'npz_scalar_int',
    'npz_scalar_str',
    'require_npz_key',
]
