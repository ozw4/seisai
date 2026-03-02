from __future__ import annotations

import random

import numpy as np
import segyio
import torch
from seisai_dataset.config import LoaderConfig, TraceSubsetSamplerConfig
from seisai_dataset.file_info import build_file_info
from seisai_dataset.noise_decider import EventDetectConfig, decide_noise
from seisai_dataset.trace_subset_preproc import TraceSubsetLoader
from seisai_dataset.trace_subset_sampler import TraceSubsetSampler
from torch.utils.data import Dataset

__all__ = ['PsnNoiseDataset']


class PsnNoiseDataset(Dataset):
    def __init__(
        self,
        *,
        segy_files: list[str],
        subset_traces: int,
        transform,
        detect_cfg: EventDetectConfig,
        sampler_cfg: TraceSubsetSamplerConfig,
        loader_cfg: LoaderConfig,
        max_retries: int,
        secondary_key_fixed: bool,
        use_header_cache: bool,
        header_cache_dir: str | None,
        waveform_mode: str,
        segy_endian: str,
    ) -> None:
        super().__init__()

        if not isinstance(segy_files, list) or len(segy_files) == 0:
            msg = 'segy_files must be non-empty list[str]'
            raise ValueError(msg)
        if not all(isinstance(path, str) for path in segy_files):
            msg = 'segy_files must be list[str]'
            raise TypeError(msg)
        if isinstance(subset_traces, bool) or int(subset_traces) <= 0:
            msg = 'subset_traces must be positive int'
            raise ValueError(msg)
        if not callable(transform):
            msg = 'transform must be callable'
            raise TypeError(msg)
        if not isinstance(detect_cfg, EventDetectConfig):
            msg = 'detect_cfg must be EventDetectConfig'
            raise TypeError(msg)
        if not isinstance(sampler_cfg, TraceSubsetSamplerConfig):
            msg = 'sampler_cfg must be TraceSubsetSamplerConfig'
            raise TypeError(msg)
        if not isinstance(loader_cfg, LoaderConfig):
            msg = 'loader_cfg must be LoaderConfig'
            raise TypeError(msg)
        if isinstance(max_retries, bool) or int(max_retries) <= 0:
            msg = 'max_retries must be positive int'
            raise ValueError(msg)

        waveform_mode_norm = str(waveform_mode).lower()
        if waveform_mode_norm not in ('eager', 'mmap'):
            msg = 'waveform_mode must be "eager" or "mmap"'
            raise ValueError(msg)
        segy_endian_norm = str(segy_endian).lower()
        if segy_endian_norm not in ('big', 'little'):
            msg = 'segy_endian must be "big" or "little"'
            raise ValueError(msg)

        H = int(subset_traces)
        if int(loader_cfg.pad_traces_to) != H:
            msg = (
                'loader_cfg.pad_traces_to must match subset_traces; '
                f'got pad_traces_to={loader_cfg.pad_traces_to}, subset_traces={H}'
            )
            raise ValueError(msg)

        self.segy_files = [str(path) for path in segy_files]
        self.subset_traces = H
        self.transform = transform
        self.detect_cfg = detect_cfg
        self.max_retries = int(max_retries)
        self.secondary_key_fixed = bool(secondary_key_fixed)
        self.use_header_cache = bool(use_header_cache)
        self.header_cache_dir = (
            None if header_cache_dir is None else str(header_cache_dir)
        )
        self.waveform_mode = waveform_mode_norm
        self.segy_endian = segy_endian_norm
        self._rng = np.random.default_rng()

        self.sampler_cfg = TraceSubsetSamplerConfig(
            primary_keys=sampler_cfg.primary_keys,
            primary_key_weights=sampler_cfg.primary_key_weights,
            use_superwindow=bool(sampler_cfg.use_superwindow),
            sw_halfspan=int(sampler_cfg.sw_halfspan),
            sw_prob=float(sampler_cfg.sw_prob),
            secondary_key_fixed=bool(self.secondary_key_fixed),
            subset_traces=int(self.subset_traces),
        )
        self.loader_cfg = LoaderConfig(pad_traces_to=int(self.subset_traces))

        self.sampler = TraceSubsetSampler(self.sampler_cfg)
        self.loader = TraceSubsetLoader(self.loader_cfg)
        self.file_infos: list[dict] = [
            build_file_info(
                segy_path,
                ffid_byte=segyio.TraceField.FieldRecord,
                chno_byte=segyio.TraceField.TraceNumber,
                cmp_byte=segyio.TraceField.CDP,
                header_cache_dir=self.header_cache_dir,
                use_header_cache=self.use_header_cache,
                include_centroids=False,
                waveform_mode=self.waveform_mode,
                segy_endian=self.segy_endian,
            )
            for segy_path in self.segy_files
        ]

    def __len__(self) -> int:
        return 1_000_000

    def sample(self) -> dict:
        for _attempt in range(1, self.max_retries + 1):
            fidx = int(self._rng.integers(0, len(self.file_infos)))
            info = self.file_infos[fidx]

            seed = int(self._rng.integers(0, 2**31 - 1))
            draw = self.sampler.draw(info, py_random=random.Random(seed))
            indices = np.asarray(draw['indices'], dtype=np.int64)
            pad_len = int(draw['pad_len'])
            key_name = str(draw['key_name'])
            primary_unique = str(draw.get('primary_unique', ''))

            if pad_len < 0 or pad_len > self.subset_traces:
                msg = f'pad_len out of range: {pad_len}'
                raise ValueError(msg)
            if int(indices.size) + pad_len != int(self.subset_traces):
                msg = (
                    'sampler output mismatch: '
                    f'indices={indices.size}, pad_len={pad_len}, subset_traces={self.subset_traces}'
                )
                raise ValueError(msg)

            x_hw = self.loader.load(info['mmap'], indices).astype(np.float32, copy=False)
            if x_hw.ndim != 2:
                msg = f'loaded traces must be 2D (H,W), got shape={x_hw.shape}'
                raise ValueError(msg)

            out = self.transform(x_hw, rng=self._rng, return_meta=True)
            if not isinstance(out, tuple) or len(out) != 2:
                msg = 'transform must return (x_hw, meta) when return_meta=True'
                raise ValueError(msg)
            x_view, meta = out
            if not isinstance(x_view, np.ndarray) or x_view.ndim != 2:
                msg = 'transform x_hw must be 2D numpy array'
                raise ValueError(msg)
            if not isinstance(meta, dict):
                msg = f'transform meta must be dict, got {type(meta).__name__}'
                raise TypeError(msg)

            H, W = int(x_view.shape[0]), int(x_view.shape[1])
            if H != int(self.subset_traces):
                msg = (
                    'transform must keep H=subset_traces; '
                    f'got H={H}, subset_traces={self.subset_traces}'
                )
                raise ValueError(msg)

            dec = decide_noise(x_view, float(info['dt_sec']), self.detect_cfg)
            if not dec.is_noise:
                continue

            trace_valid = np.ones((H,), dtype=np.bool_)
            if pad_len > 0:
                trace_valid[-pad_len:] = False
            trace_valid_tensor = torch.from_numpy(trace_valid)

            input_tensor = torch.from_numpy(x_view[None, :, :].copy()).to(
                dtype=torch.float32
            )
            target_tensor = torch.zeros((3, H, W), dtype=torch.float32)
            target_tensor[2].fill_(1.0)

            out_meta = {
                'is_noise': True,
                'file_path': str(info['path']),
                'primary_key': key_name,
                'primary_unique': primary_unique,
                'secondary_key': str(draw.get('secondary_key', '')),
                'did_super': bool(draw.get('did_super', False)),
                'pad_len': int(pad_len),
                'start_t': int(meta.get('start', 0)),
            }

            return {
                'input': input_tensor,
                'target': target_tensor,
                'trace_valid': trace_valid_tensor,
                'label_valid': trace_valid_tensor.clone(),
                'meta': out_meta,
            }

        msg = f'Failed to sample noise-only window within max_retries={self.max_retries}'
        raise RuntimeError(msg)

    def __getitem__(self, _index: int | None = None) -> dict:
        return self.sample()

    def close(self) -> None:
        infos = getattr(self, 'file_infos', None)
        if not infos:
            return
        for info in infos:
            segy_obj = info.get('segy_obj')
            if segy_obj is not None:
                segy_obj.close()
        infos.clear()

    def __del__(self) -> None:
        self.close()
