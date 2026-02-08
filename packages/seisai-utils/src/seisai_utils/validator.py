# validators.py
from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
from torch import Tensor

Backend = Literal['numpy', 'torch', 'auto']

# 例示表記(必要に応じて拡張)
_SHAPE_EXAMPLES = {1: '(W,)', 2: '(H,W)', 3: '(C,H,W)', 4: '(B,C,H,W)'}


def _allowed_shapes_text(allowed_ndims: Iterable[int]) -> str:
    items: list[str] = [
        _SHAPE_EXAMPLES.get(int(d), f'(ndim={int(d)})') for d in allowed_ndims
    ]
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' or ' + items[-1]


# ---------- NumPy ----------
def require_numpy(x, *, name: str = 'x') -> None:
    if not isinstance(x, np.ndarray):
        msg = f'{name} must be numpy.ndarray'
        raise TypeError(msg)


def require_ndim_numpy(
    x: np.ndarray,
    *,
    allowed_ndims: Iterable[int],
    name: str = 'x',
    shape_hint: str | None = None,
) -> None:
    allowed = tuple(int(d) for d in allowed_ndims)
    if x.ndim not in allowed:
        hint = shape_hint if shape_hint is not None else _allowed_shapes_text(allowed)
        msg = f'{name} must be {hint}'
        raise ValueError(msg)


def require_non_empty_numpy(x: np.ndarray, *, name: str = 'x') -> None:
    if x.size == 0:
        msg = f'{name} must be non-empty'
        raise ValueError(msg)


def validate_numpy(
    x,
    *,
    allowed_ndims: Iterable[int] = (1, 2, 3, 4),
    name: str = 'x',
    shape_hint: str | None = None,
) -> None:
    require_numpy(x, name=name)
    require_ndim_numpy(x, allowed_ndims=allowed_ndims, name=name, shape_hint=shape_hint)
    require_non_empty_numpy(x, name=name)


# ---------- Torch ----------
def require_torch_tensor(x, *, name: str = 'x') -> None:
    if not isinstance(x, torch.Tensor):
        msg = f'{name} must be torch.Tensor'
        raise TypeError(msg)


def require_ndim_torch(
    x: Tensor,
    *,
    allowed_ndims: Iterable[int],
    name: str = 'x',
    shape_hint: str | None = None,
) -> None:
    allowed = tuple(int(d) for d in allowed_ndims)
    if int(x.ndim) not in allowed:
        hint = shape_hint if shape_hint is not None else _allowed_shapes_text(allowed)
        msg = f'{name} must be {hint}'
        raise ValueError(msg)


def require_non_empty_torch(x: Tensor, *, name: str = 'x') -> None:
    if int(x.numel()) == 0:
        msg = f'{name} must be non-empty'
        raise ValueError(msg)


def validate_torch(
    x,
    *,
    allowed_ndims: Iterable[int] = (1, 2, 3, 4),
    name: str = 'x',
    shape_hint: str | None = None,
) -> None:
    require_torch_tensor(x, name=name)
    require_ndim_torch(x, allowed_ndims=allowed_ndims, name=name, shape_hint=shape_hint)
    require_non_empty_torch(x, name=name)


# ---------- 汎用(自動/明示バックエンド) ----------
def validate_array(
    x,
    *,
    allowed_ndims: Iterable[int] = (1, 2, 3, 4),
    name: str = 'x',
    backend: Backend = 'auto',
    shape_hint: str | None = None,
) -> None:
    if backend == 'numpy':
        validate_numpy(x, allowed_ndims=allowed_ndims, name=name, shape_hint=shape_hint)
        return
    if backend == 'torch':
        validate_torch(x, allowed_ndims=allowed_ndims, name=name, shape_hint=shape_hint)
        return
    if isinstance(x, np.ndarray):
        validate_numpy(x, allowed_ndims=allowed_ndims, name=name, shape_hint=shape_hint)
        return
    if isinstance(x, torch.Tensor):
        validate_torch(x, allowed_ndims=allowed_ndims, name=name, shape_hint=shape_hint)
        return
    msg = f'{name} must be numpy.ndarray or torch.Tensor'
    raise TypeError(msg)


def _require_array_like(x, *, name: str, backend: Backend) -> None:
    if backend == 'numpy':
        if not isinstance(x, np.ndarray):
            msg = f'{name} must be numpy.ndarray'
            raise TypeError(msg)
        return
    if backend == 'torch':
        if not isinstance(x, torch.Tensor):
            msg = f'{name} must be torch.Tensor'
            raise TypeError(msg)
        return
    if not (isinstance(x, (np.ndarray, torch.Tensor))):
        msg = f'{name} must be numpy.ndarray or torch.Tensor'
        raise TypeError(msg)


def require_float_array(x, *, name='x', backend: Backend = 'auto') -> None:
    if backend == 'numpy' or (backend == 'auto' and isinstance(x, np.ndarray)):
        require_numpy(x, name=name)
        if not np.issubdtype(x.dtype, np.floating):
            msg = f'{name} must be a floating array'
            raise TypeError(msg)
        return
    if backend == 'torch' or (backend == 'auto' and isinstance(x, torch.Tensor)):
        require_torch_tensor(x, name=name)
        if not torch.is_floating_point(x):
            msg = f'{name} must be a floating tensor'
            raise TypeError(msg)
        return
    msg = f'{name} must be numpy.ndarray or torch.Tensor'
    raise TypeError(msg)


def require_boolint_array(x, *, name='x', backend: Backend = 'auto') -> None:
    if backend == 'numpy' or (backend == 'auto' and isinstance(x, np.ndarray)):
        require_numpy(x, name=name)
        if not (np.issubdtype(x.dtype, np.bool_) or np.issubdtype(x.dtype, np.integer)):
            msg = f'{name} must be bool/int array'
            raise TypeError(msg)
        return
    if backend == 'torch' or (backend == 'auto' and isinstance(x, torch.Tensor)):
        require_torch_tensor(x, name=name)
        if x.dtype not in (
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            msg = f'{name} must be bool/int tensor'
            raise TypeError(msg)
        return
    msg = f'{name} must be numpy.ndarray or torch.Tensor'
    raise TypeError(msg)


def require_non_negative(x, *, name='x', backend: Backend = 'auto') -> None:
    if backend == 'numpy' or (backend == 'auto' and isinstance(x, np.ndarray)):
        if (x < 0).any():
            msg = f'{name} must be >= 0'
            raise ValueError(msg)
        return
    if backend == 'torch' or (backend == 'auto' and isinstance(x, torch.Tensor)):
        if (x < 0).any().item():
            msg = f'{name} must be >= 0'
            raise ValueError(msg)
        return
    msg = f'{name} must be numpy.ndarray or torch.Tensor'
    raise TypeError(msg)


def require_all_finite(x, *, name='x', backend: Backend = 'auto') -> None:
    if backend == 'numpy' or (backend == 'auto' and isinstance(x, np.ndarray)):
        if not np.isfinite(x).all():
            msg = f'{name} must contain only finite values'
            raise ValueError(msg)
        return
    if backend == 'torch' or (backend == 'auto' and isinstance(x, torch.Tensor)):
        # torch.isfinite: NaN/Inf を False にする
        if (~torch.isfinite(x)).any().item():
            msg = f'{name} must contain only finite values'
            raise ValueError(msg)
        return
    msg = f'{name} must be numpy.ndarray or torch.Tensor'
    raise TypeError(msg)


def require_all_numpy(*xs) -> bool:
    for x in xs:
        if x is None:
            continue
        if not isinstance(x, np.ndarray):
            return False
    return True


def require_same_shape_and_backend(
    a,
    b,
    *others,
    name_a: str = 'a',
    name_b: str = 'b',
    other_names: list[str] | None = None,
    backend: Backend = 'auto',
    shape_hint: str = '(B,H)',
) -> None:
    _require_array_like(a, name=name_a, backend=backend)
    _require_array_like(b, name=name_b, backend=backend)
    if type(a) is not type(b):
        msg = f'{name_a} and {name_b} must use the same backend'
        raise TypeError(msg)

    sa = tuple(a.shape)
    sb = tuple(b.shape)
    if sa != sb:
        msg = (
            f'{name_a} and {name_b} must have the same shape (e.g., {shape_hint}). '
            f'Got {sa} vs {sb}'
        )
        raise ValueError(
            msg
        )

    if other_names is not None and len(other_names) != len(others):
        msg = f'other_names length must equal number of extra arrays ({len(others)})'
        raise ValueError(
            msg
        )

    # 以降の追加引数も検証
    base_type = type(a)
    for i, x in enumerate(others):
        name_x = other_names[i] if other_names is not None else f'arg{i + 3}'
        _require_array_like(x, name=name_x, backend=backend)
        if type(x) is not base_type:
            msg = f'{name_x} must use the same backend as {name_a}'
            raise TypeError(msg)
        sx = tuple(x.shape)
        if sx != sa:
            msg = (
                f'{name_x} must have the same shape as {name_a} (e.g., {shape_hint}). '
                f'Got {sx} vs {sa}'
            )
            raise ValueError(
                msg
            )
