from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import threading

import torch

__all__ = [
    'ViewerModelBundle',
    'clear_model_cache',
    'get_or_create_model_bundle',
]

CacheKey = tuple[Path, int, str]


@dataclass
class ViewerModelBundle:
    model: torch.nn.Module
    in_chans: int
    out_chans: int
    softmax_axis: str
    output_ids: tuple[str, ...]


_MAX_CACHE_SIZE = 8
_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[CacheKey, ViewerModelBundle] = {}
_CACHE_ORDER: list[CacheKey] = []


def _make_cache_key(*, ckpt_path: str | Path, device_str: str) -> CacheKey:
    ckpt = Path(ckpt_path).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    device_key = str(device_str).strip()
    if not device_key:
        msg = 'device_str must be non-empty'
        raise ValueError(msg)

    return ckpt, int(ckpt.stat().st_mtime_ns), device_key


def clear_model_cache() -> None:
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
        _CACHE_ORDER.clear()


def get_or_create_model_bundle(
    *,
    ckpt_path: str | Path,
    device_str: str,
    loader: Callable[[Path], ViewerModelBundle],
) -> ViewerModelBundle:
    key = _make_cache_key(ckpt_path=ckpt_path, device_str=device_str)

    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

    bundle = loader(key[0])
    if not isinstance(bundle, ViewerModelBundle):
        msg = 'loader must return ViewerModelBundle'
        raise TypeError(msg)

    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        _MODEL_CACHE[key] = bundle
        _CACHE_ORDER.append(key)
        while len(_CACHE_ORDER) > _MAX_CACHE_SIZE:
            old_key = _CACHE_ORDER.pop(0)
            _MODEL_CACHE.pop(old_key, None)

    return bundle
