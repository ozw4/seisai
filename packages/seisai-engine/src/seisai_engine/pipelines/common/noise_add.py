from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import fields
from typing import Any

import numpy as np
import torch
from seisai_dataset.config import LoaderConfig, TraceSubsetSamplerConfig
from seisai_dataset.noise_dataset import NoiseTraceSubsetDataset
from seisai_dataset.noise_decider import EventDetectConfig
from seisai_transforms import AdditiveNoiseMix, RandomCropOrPad, ViewCompose
from seisai_utils.config import (
    optional_float,
    optional_int,
    optional_tuple2_float,
    require_list_str,
)
from seisai_utils.fs import validate_files_exist

__all__ = ['NoiseTraceSubsetProvider', 'maybe_build_noise_add_op']

_EVENT_DETECT_FIELDS = {f.name for f in fields(EventDetectConfig)}


def _validate_prob_01(*, value: float, name: str) -> float:
    out = float(value)
    if out < 0.0 or out > 1.0:
        msg = f'{name} must be within [0, 1]'
        raise ValueError(msg)
    return out


def _validate_ordered_range(
    *,
    value: tuple[float, float],
    name: str,
) -> tuple[float, float]:
    lo, hi = float(value[0]), float(value[1])
    if lo > hi:
        msg = f'{name} must satisfy lo <= hi'
        raise ValueError(msg)
    return (lo, hi)


def _parse_optional_int(raw: Any, *, name: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        msg = f'{name} must be int'
        raise TypeError(msg)
    if isinstance(raw, float) and not raw.is_integer():
        msg = f'{name} must be int'
        raise ValueError(msg)
    return int(raw)


class NoiseTraceSubsetProvider:
    def __init__(
        self,
        *,
        segy_files: list[str],
        subset_traces: int,
        primary_keys: Sequence[str],
        secondary_key_fixed: bool,
        waveform_mode: str,
        segy_endian: str,
        header_cache_dir: str | None,
        use_header_cache: bool,
        detect_cfg_overrides: dict | None,
        max_redraw: int = 32,
        seed: int | None = None,
    ) -> None:
        if not isinstance(segy_files, list) or len(segy_files) == 0:
            msg = 'segy_files must be non-empty list[str]'
            raise ValueError(msg)
        if not all(isinstance(p, str) for p in segy_files):
            msg = 'segy_files must be list[str]'
            raise TypeError(msg)
        if (
            not isinstance(primary_keys, Sequence)
            or isinstance(primary_keys, (str, bytes))
            or len(primary_keys) == 0
            or not all(isinstance(k, str) for k in primary_keys)
        ):
            msg = 'primary_keys must be non-empty sequence[str]'
            raise TypeError(msg)
        if isinstance(subset_traces, bool) or int(subset_traces) <= 0:
            msg = 'subset_traces must be positive int'
            raise ValueError(msg)
        if isinstance(max_redraw, bool) or int(max_redraw) <= 0:
            msg = 'max_redraw must be positive int'
            raise ValueError(msg)
        wm = str(waveform_mode).lower()
        if wm not in ('eager', 'mmap'):
            msg = 'waveform_mode must be "eager" or "mmap"'
            raise ValueError(msg)
        endian = str(segy_endian).lower()
        if endian not in ('big', 'little'):
            msg = 'segy_endian must be "big" or "little"'
            raise ValueError(msg)
        if detect_cfg_overrides is not None and not isinstance(detect_cfg_overrides, dict):
            msg = 'detect_cfg_overrides must be dict or None'
            raise TypeError(msg)
        if detect_cfg_overrides is not None:
            unknown = set(detect_cfg_overrides.keys()).difference(_EVENT_DETECT_FIELDS)
            if unknown:
                msg = (
                    'detect_cfg_overrides has unknown keys: '
                    + ', '.join(sorted(str(k) for k in unknown))
                )
                raise ValueError(msg)

        self.segy_files = list(str(p) for p in segy_files)
        self.subset_traces = int(subset_traces)
        self.primary_keys = tuple(str(k) for k in primary_keys)
        self.secondary_key_fixed = bool(secondary_key_fixed)
        self.waveform_mode = wm
        self.segy_endian = endian
        self.header_cache_dir = (
            None if header_cache_dir is None else str(header_cache_dir)
        )
        self.use_header_cache = bool(use_header_cache)
        self.detect_cfg_overrides = (
            None if detect_cfg_overrides is None else dict(detect_cfg_overrides)
        )
        # keep config key name max_redraw for backward compatibility,
        # but use a single retry budget in dataset.max_retries.
        self.max_retries = int(max_redraw)
        self.seed = None if seed is None else int(seed)

        self._dataset: NoiseTraceSubsetDataset | None = None
        self._dataset_pid: int | None = None
        self._dataset_time_len: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state['_dataset'] = None
        state['_dataset_pid'] = None
        state['_dataset_time_len'] = None
        return state

    def _build_detect_cfg(self) -> EventDetectConfig:
        if self.detect_cfg_overrides is None:
            return EventDetectConfig()
        params: dict[str, Any] = {}
        for key in _EVENT_DETECT_FIELDS:
            if key in self.detect_cfg_overrides:
                params[key] = self.detect_cfg_overrides[key]
        return EventDetectConfig(**params)

    def _build_dataset(self, *, target_len: int) -> NoiseTraceSubsetDataset:
        transform = ViewCompose([RandomCropOrPad(int(target_len))])
        return NoiseTraceSubsetDataset(
            segy_files=list(self.segy_files),
            loader_cfg=LoaderConfig(pad_traces_to=int(self.subset_traces)),
            sampler_cfg=TraceSubsetSamplerConfig(
                primary_keys=tuple(self.primary_keys),
                secondary_key_fixed=bool(self.secondary_key_fixed),
                subset_traces=int(self.subset_traces),
            ),
            detect_cfg=self._build_detect_cfg(),
            transform=transform,
            header_cache_dir=self.header_cache_dir,
            use_header_cache=bool(self.use_header_cache),
            max_retries=int(self.max_retries),
            secondary_key_fixed=bool(self.secondary_key_fixed),
            waveform_mode=str(self.waveform_mode),
            segy_endian=str(self.segy_endian),
            seed=self.seed,
        )

    def _ensure_dataset(self, *, target_len: int) -> None:
        cur_pid = int(os.getpid())
        needs_rebuild = (
            self._dataset is None
            or self._dataset_pid != cur_pid
            or self._dataset_time_len != int(target_len)
        )
        if not needs_rebuild:
            return
        self.close()
        self._dataset = self._build_dataset(target_len=int(target_len))
        self._dataset_pid = cur_pid
        self._dataset_time_len = int(target_len)

    def sample(
        self,
        shape: tuple[int, int],
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or isinstance(shape[0], bool)
            or isinstance(shape[1], bool)
        ):
            msg = 'shape must be tuple[int, int]'
            raise TypeError(msg)
        h_req, w_req = int(shape[0]), int(shape[1])
        if h_req <= 0 or w_req <= 0:
            msg = 'shape values must be positive'
            raise ValueError(msg)
        if h_req != int(self.subset_traces):
            msg = (
                'shape[0] must match subset_traces; '
                f'got shape[0]={h_req}, subset_traces={self.subset_traces}'
            )
            raise ValueError(msg)

        self._ensure_dataset(target_len=int(w_req))
        if self._dataset is None:
            msg = 'internal error: dataset is not initialized'
            raise RuntimeError(msg)

        _rng = rng or np.random.default_rng()
        sample = self._dataset.sample(rng=_rng)
        x = sample['x']
        if torch.is_tensor(x):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.asarray(x)
        if arr.shape != (h_req, w_req):
            msg = (
                'sampled noise shape mismatch: '
                f'got={tuple(arr.shape)} expected={(h_req, w_req)}'
            )
            raise ValueError(msg)
        return np.asarray(arr, dtype=np.float32)

    def close(self) -> None:
        dataset = getattr(self, '_dataset', None)
        if dataset is not None:
            dataset.close()
        self._dataset = None
        self._dataset_pid = None
        self._dataset_time_len = None

    def __del__(self) -> None:
        self.close()


def maybe_build_noise_add_op(
    *,
    augment_cfg: dict | None,
    subset_traces: int,
    primary_keys: Sequence[str],
    secondary_key_fixed: bool,
    waveform_mode: str,
    segy_endian: str,
    header_cache_dir: str | None,
    use_header_cache: bool,
):
    if augment_cfg is None:
        return None
    if not isinstance(augment_cfg, dict):
        msg = 'augment must be dict'
        raise TypeError(msg)

    noise_cfg = augment_cfg.get('noise_add')
    if noise_cfg is None:
        return None
    if not isinstance(noise_cfg, dict):
        msg = 'augment.noise_add must be dict'
        raise TypeError(msg)

    segy_files = require_list_str(noise_cfg, 'segy_files')
    if len(segy_files) == 0:
        msg = 'augment.noise_add.segy_files must be non-empty'
        raise ValueError(msg)
    validate_files_exist(segy_files)

    prob = _validate_prob_01(
        value=optional_float(noise_cfg, 'prob', 1.0),
        name='augment.noise_add.prob',
    )
    gain_range = (
        _validate_ordered_range(
            value=optional_tuple2_float(noise_cfg, 'gain_range', (0.0, 0.0)),
            name='augment.noise_add.gain_range',
        )
        if 'gain_range' in noise_cfg
        else None
    )
    snr_db_range = (
        _validate_ordered_range(
            value=optional_tuple2_float(noise_cfg, 'snr_db_range', (0.0, 0.0)),
            name='augment.noise_add.snr_db_range',
        )
        if 'snr_db_range' in noise_cfg
        else None
    )
    if gain_range is not None and snr_db_range is not None:
        msg = (
            'augment.noise_add.gain_range and augment.noise_add.snr_db_range '
            'cannot be set at the same time'
        )
        raise ValueError(msg)
    if gain_range is not None and gain_range[0] < 0.0:
        msg = 'augment.noise_add.gain_range must be non-negative'
        raise ValueError(msg)

    max_redraw = optional_int(noise_cfg, 'max_redraw', 32)
    if int(max_redraw) <= 0:
        msg = 'augment.noise_add.max_redraw must be positive'
        raise ValueError(msg)
    seed = _parse_optional_int(noise_cfg.get('seed'), name='augment.noise_add.seed')

    detect_cfg_overrides = noise_cfg.get('detect')
    if detect_cfg_overrides is not None and not isinstance(detect_cfg_overrides, dict):
        msg = 'augment.noise_add.detect must be dict'
        raise TypeError(msg)

    provider = NoiseTraceSubsetProvider(
        segy_files=list(segy_files),
        subset_traces=int(subset_traces),
        primary_keys=tuple(primary_keys),
        secondary_key_fixed=bool(secondary_key_fixed),
        waveform_mode=str(waveform_mode),
        segy_endian=str(segy_endian),
        header_cache_dir=header_cache_dir,
        use_header_cache=bool(use_header_cache),
        detect_cfg_overrides=detect_cfg_overrides,
        max_redraw=int(max_redraw),
        seed=seed,
    )
    return AdditiveNoiseMix(
        provider=provider,
        prob=float(prob),
        gain_range=gain_range,
        snr_db_range=snr_db_range,
    )
