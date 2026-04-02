"""Deterministic fixed-length local-window dataset for fine first-break work."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import segyio
import torch
from torch.utils.data import Dataset

from .builder.builder import BuildPlan, InputOnlyPlan
from .config import LoaderConfig
from .file_info import build_file_info, normalize_segy_endian, normalize_waveform_mode
from .trace_subset_preproc import TraceSubsetLoader
from .transform_flow_utils import apply_transform_2d_with_meta

LocalWindowMode = Literal['train', 'eval', 'infer']

__all__ = ['LocalWindowDataset', 'LocalWindowDatasetConfig']

_COARSE_ARTIFACT_VERSION = 1

_COARSE_META_REQUIRED_KEYS = (
    'artifact_version',
    'stage',
    'survey_id',
    'npz_filename',
    'source_refs',
    'dimensions',
)

_COARSE_FIELD_SPECS = {
    'prob': ('float32', 2),
    'pick_idx': ('int32', 1),
    'confidence': ('float32', 1),
    'trace_valid': ('bool', 1),
    'raw_trace_idx': ('int64', 1),
    'offsets': ('float32', 1),
    'time_axis': ('float32', 1),
}


@dataclass(frozen=True)
class LocalWindowDatasetConfig:
    """Configuration for deterministic local-window enumeration.

    ``local_window_len`` is the returned fixed length on the local time axis.
    The dataset centers the seed at ``local_window_len // 2`` when the raw trace
    is long enough, otherwise it shifts the window to stay inside the recorded
    range and pads only the right tail when the trace itself is shorter than the
    requested local window length.
    """

    local_window_len: int
    mode: LocalWindowMode = 'train'


@dataclass(frozen=True)
class _LocalWindowItem:
    file_idx: int
    trace_idx_in_file: int
    raw_trace_idx: int
    raw_seed_idx: int
    raw_pick_idx: int
    local_window_start_idx: int
    local_window_end_idx: int
    local_seed_idx: int
    local_pick_idx: int
    seed_source: Literal['gt', 'coarse']


@dataclass(frozen=True)
class _LoadedCoarseArtifact:
    pick_idx: np.ndarray
    confidence: np.ndarray
    trace_valid: np.ndarray
    raw_trace_idx: np.ndarray
    offsets: np.ndarray
    time_axis: np.ndarray
    n_traces: int
    n_samples: int


def _validate_local_window_cfg(cfg: LocalWindowDatasetConfig) -> None:
    if not isinstance(cfg, LocalWindowDatasetConfig):
        msg = 'cfg must be LocalWindowDatasetConfig'
        raise TypeError(msg)
    if int(cfg.local_window_len) <= 0:
        msg = 'local_window_len must be positive'
        raise ValueError(msg)
    if cfg.mode not in ('train', 'eval', 'infer'):
        msg = 'mode must be one of "train", "eval", or "infer"'
        raise ValueError(msg)


def _load_label_array(label_path: str | Path, *, expected_len: int) -> np.ndarray:
    raw = np.load(str(label_path), allow_pickle=False)
    if not isinstance(raw, np.ndarray):
        with contextlib.suppress(Exception):
            raw.close()
        msg = f'GT pick file must contain a single ndarray: {label_path}'
        raise TypeError(msg)
    arr = np.asarray(raw)
    if arr.ndim != 1:
        msg = f'GT pick file must be 1D, got shape {arr.shape}: {label_path}'
        raise ValueError(msg)
    if int(arr.shape[0]) != int(expected_len):
        msg = (
            f'GT pick length {int(arr.shape[0])} != expected {int(expected_len)} '
            f'for {label_path}'
        )
        raise ValueError(msg)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)
    if np.issubdtype(arr.dtype, np.floating):
        finite = np.isfinite(arr)
        if not np.all(finite):
            msg = f'GT pick file contains non-finite values: {label_path}'
            raise ValueError(msg)
        rounded = np.rint(arr)
        if not np.array_equal(arr, rounded):
            msg = f'GT pick file must contain integer-valued picks: {label_path}'
            raise ValueError(msg)
        return rounded.astype(np.int64, copy=False)
    msg = f'unsupported GT pick dtype {arr.dtype} in {label_path}'
    raise TypeError(msg)


def _validate_coarse_meta(*, meta_path: Path, npz_path: Path) -> dict[str, int]:
    raw = json.loads(meta_path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        msg = f'coarse artifact meta must be a JSON object: {meta_path}'
        raise TypeError(msg)
    missing = set(_COARSE_META_REQUIRED_KEYS).difference(raw.keys())
    if missing:
        msg = f'coarse artifact meta missing required keys {sorted(missing)}: {meta_path}'
        raise ValueError(msg)
    artifact_version = raw['artifact_version']
    if isinstance(artifact_version, bool) or not isinstance(artifact_version, int):
        msg = f'coarse artifact meta artifact_version must be int: {meta_path}'
        raise TypeError(msg)
    if int(artifact_version) != int(_COARSE_ARTIFACT_VERSION):
        msg = (
            f'unsupported coarse artifact version {artifact_version}; '
            f'expected {_COARSE_ARTIFACT_VERSION}'
        )
        raise ValueError(msg)
    if not isinstance(raw['stage'], str):
        msg = f'coarse artifact meta stage must be str: {meta_path}'
        raise TypeError(msg)
    if raw['stage'] != 'coarse':
        msg = f'coarse artifact meta stage must be "coarse", got {raw["stage"]!r}'
        raise ValueError(msg)
    if not isinstance(raw['npz_filename'], str) or raw['npz_filename'] == '':
        msg = f'coarse artifact meta npz_filename must be a non-empty string: {meta_path}'
        raise ValueError(msg)
    if raw['npz_filename'] != npz_path.name:
        msg = (
            'coarse artifact meta npz_filename mismatch: '
            f'expected {npz_path.name}, got {raw["npz_filename"]}'
        )
        raise ValueError(msg)
    if not isinstance(raw['survey_id'], str) or raw['survey_id'] == '':
        msg = f'coarse artifact meta survey_id must be a non-empty string: {meta_path}'
        raise ValueError(msg)
    if not isinstance(raw['source_refs'], dict):
        msg = f'coarse artifact meta source_refs must be dict[str, str]: {meta_path}'
        raise TypeError(msg)
    for key, value in raw['source_refs'].items():
        if not isinstance(key, str) or not isinstance(value, str):
            msg = f'coarse artifact meta source_refs must be dict[str, str]: {meta_path}'
            raise TypeError(msg)
    dims = raw['dimensions']
    if not isinstance(dims, dict):
        msg = f'coarse artifact meta dimensions must be dict[str, int]: {meta_path}'
        raise TypeError(msg)
    out: dict[str, int] = {}
    for key, value in dims.items():
        if not isinstance(key, str):
            msg = f'coarse artifact meta dimension names must be str: {meta_path}'
            raise TypeError(msg)
        if isinstance(value, bool) or not isinstance(value, int):
            msg = f'coarse artifact meta dimension values must be int: {meta_path}'
            raise TypeError(msg)
        if value <= 0:
            msg = f'coarse artifact meta dimensions must be positive: {meta_path}'
            raise ValueError(msg)
        out[key] = int(value)
    return out


def _load_coarse_artifact(
    *,
    npz_path: str | Path,
    meta_path: str | Path,
) -> _LoadedCoarseArtifact:
    npz_p = Path(npz_path).expanduser().resolve()
    meta_p = Path(meta_path).expanduser().resolve()
    if not npz_p.exists():
        msg = f'coarse artifact npz not found: {npz_p}'
        raise FileNotFoundError(msg)
    if not meta_p.exists():
        msg = f'coarse artifact meta not found: {meta_p}'
        raise FileNotFoundError(msg)

    dims_from_meta = _validate_coarse_meta(meta_path=meta_p, npz_path=npz_p)
    with np.load(npz_p, allow_pickle=False) as z:
        keys = tuple(z.files)
        expected = set(_COARSE_FIELD_SPECS.keys())
        actual = set(keys)
        missing = expected.difference(actual)
        extra = actual.difference(expected)
        if missing:
            msg = f'coarse artifact missing required keys: {sorted(missing)}'
            raise ValueError(msg)
        if extra:
            msg = f'coarse artifact has unsupported keys: {sorted(extra)}'
            raise ValueError(msg)

        arrays: dict[str, np.ndarray] = {}
        dims: dict[str, int] = {}
        for key, (dtype_name, ndim) in _COARSE_FIELD_SPECS.items():
            arr = np.asarray(z[key])
            expected_dtype = np.dtype(dtype_name)
            if arr.dtype != expected_dtype:
                msg = (
                    f'coarse artifact key "{key}" must have dtype '
                    f'{expected_dtype.name}, got {arr.dtype.name}'
                )
                raise TypeError(msg)
            if int(arr.ndim) != int(ndim):
                msg = (
                    f'coarse artifact key "{key}" must have {ndim} dims, '
                    f'got shape {arr.shape}'
                )
                raise ValueError(msg)
            if any(int(size) <= 0 for size in arr.shape):
                msg = f'coarse artifact key "{key}" has non-positive shape {arr.shape}'
                raise ValueError(msg)
            arrays[key] = arr

        n_traces = int(arrays['pick_idx'].shape[0])
        n_samples = int(arrays['prob'].shape[1])
        dims['n_traces'] = n_traces
        dims['n_samples'] = n_samples
        if arrays['prob'].shape != (n_traces, n_samples):
            msg = 'coarse artifact prob shape is internally inconsistent'
            raise ValueError(msg)
        for key in ('pick_idx', 'confidence', 'trace_valid', 'raw_trace_idx', 'offsets'):
            if arrays[key].shape != (n_traces,):
                msg = f'coarse artifact key "{key}" shape {arrays[key].shape} != ({n_traces},)'
                raise ValueError(msg)
        if arrays['time_axis'].shape != (n_samples,):
            msg = (
                f'coarse artifact key "time_axis" shape {arrays["time_axis"].shape} '
                f'!= ({n_samples},)'
            )
            raise ValueError(msg)
        if dims_from_meta != dims:
            msg = (
                f'coarse artifact meta dimensions {dims_from_meta} do not match '
                f'array dimensions {dims}'
            )
            raise ValueError(msg)

        return _LoadedCoarseArtifact(
            pick_idx=arrays['pick_idx'],
            confidence=arrays['confidence'],
            trace_valid=arrays['trace_valid'],
            raw_trace_idx=arrays['raw_trace_idx'],
            offsets=arrays['offsets'],
            time_axis=arrays['time_axis'],
            n_traces=n_traces,
            n_samples=n_samples,
        )


def _resolve_local_window(
    *,
    raw_seed_idx: int,
    n_samples: int,
    local_window_len: int,
) -> tuple[int, int, int]:
    if int(n_samples) <= 0:
        msg = 'n_samples must be positive'
        raise ValueError(msg)
    if int(local_window_len) <= 0:
        msg = 'local_window_len must be positive'
        raise ValueError(msg)
    if int(raw_seed_idx) < 0 or int(raw_seed_idx) >= int(n_samples):
        msg = f'raw_seed_idx {raw_seed_idx} must satisfy 0 <= raw_seed_idx < {n_samples}'
        raise ValueError(msg)

    half = int(local_window_len) // 2
    max_start = max(int(n_samples) - int(local_window_len), 0)
    start = int(raw_seed_idx) - half
    if start < 0:
        start = 0
    if start > max_start:
        start = max_start
    end = min(start + int(local_window_len), int(n_samples))
    local_seed_idx = int(raw_seed_idx) - start

    if end <= start:
        msg = f'local window must have positive raw coverage, got start={start}, end={end}'
        raise RuntimeError(msg)
    if local_seed_idx < 0 or local_seed_idx >= (end - start):
        msg = (
            'resolved local seed index is outside the raw coverage: '
            f'seed={raw_seed_idx}, start={start}, end={end}, local_seed_idx={local_seed_idx}'
        )
        raise RuntimeError(msg)

    return start, end, local_seed_idx


def _extract_fixed_window(
    trace: np.ndarray,
    *,
    start_idx: int,
    end_idx: int,
    local_window_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    if trace.ndim != 1:
        msg = f'trace must be 1D, got shape {trace.shape}'
        raise ValueError(msg)
    if start_idx < 0 or end_idx < 0 or end_idx > int(trace.shape[0]) or end_idx <= start_idx:
        msg = f'invalid local window bounds start={start_idx}, end={end_idx}, n_samples={trace.shape[0]}'
        raise ValueError(msg)
    raw_len = int(end_idx) - int(start_idx)
    if raw_len > int(local_window_len):
        msg = f'raw window length {raw_len} exceeds local_window_len {local_window_len}'
        raise ValueError(msg)

    out = np.zeros((int(local_window_len),), dtype=np.float32)
    out[:raw_len] = trace[int(start_idx) : int(end_idx)].astype(np.float32, copy=False)

    raw_sample_idx_local = np.full((int(local_window_len),), -1, dtype=np.int64)
    raw_sample_idx_local[:raw_len] = np.arange(int(start_idx), int(end_idx), dtype=np.int64)
    return out, raw_sample_idx_local


def _build_time_view_local(
    *,
    raw_sample_idx_local: np.ndarray,
    dt_sec: float,
) -> np.ndarray:
    if raw_sample_idx_local.ndim != 1:
        msg = f'raw_sample_idx_local must be 1D, got shape {raw_sample_idx_local.shape}'
        raise ValueError(msg)
    out = np.zeros(raw_sample_idx_local.shape, dtype=np.float32)
    valid = raw_sample_idx_local >= 0
    out[valid] = (
        raw_sample_idx_local[valid].astype(np.float32, copy=False)
        * np.float32(float(dt_sec))
    )
    return out


class LocalWindowDataset(Dataset):
    """Enumerate fixed-length local windows for fine first-break work.

    The dataset works on a single-trace local time axis. Each item corresponds to
    one raw trace and returns ``x_view_local`` with shape ``(1, W_local)``.

    Modes
    -----
    ``train`` / ``eval``
        Use GT first-break picks from ``fb_files`` as both the local-window seed and
        the local label source. GT picks follow the existing first-break dataset
        convention: values ``<= 0`` are treated as invalid and are not enumerated.
    ``infer``
        Use the frozen coarse artifact ``pick_idx`` as the local-window seed. In
        this mode ``raw_pick_idx`` and ``local_pick_idx`` are returned as ``-1`` and
        ``label_valid`` is ``False``.

    Coordinate contract
    -------------------
    - ``local_window_start_idx`` / ``local_window_end_idx`` are raw sample indices.
    - ``local_window_end_idx`` is exclusive.
    - ``local_seed_idx`` / ``local_pick_idx`` are indices on the local window axis.
    - ``raw_sample_idx_local`` maps each local sample back to the raw sample axis and
      uses ``-1`` for right-padded positions.

    Edge handling
    -------------
    The window is centered on the seed at ``local_window_len // 2`` when possible.
    Near the record edges the window is shifted to stay within the recorded range.
    Zero-padding is used only on the right when the raw trace is shorter than the
    requested fixed window length.

    The dataset always returns auxiliary offset/time metadata, but Phase 4 keeps the
    fine-v1 model input policy out of this class. Future plans may choose to consume
    only the amplitude channel.
    """

    def __init__(
        self,
        segy_files: Sequence[str],
        *,
        cfg: LocalWindowDatasetConfig,
        fb_files: Sequence[str] | None = None,
        coarse_artifact_npz_path: str | Path | None = None,
        coarse_artifact_meta_path: str | Path | None = None,
        plan: BuildPlan | InputOnlyPlan | None = None,
        transform=None,
        ffid_byte=segyio.TraceField.FieldRecord,
        chno_byte=segyio.TraceField.TraceNumber,
        cmp_byte=segyio.TraceField.CDP,
        use_header_cache: bool = True,
        header_cache_dir: str | None = None,
        waveform_mode: str = 'mmap',
        segy_endian: str = 'big',
    ) -> None:
        if len(segy_files) == 0:
            msg = 'segy_files must be non-empty'
            raise ValueError(msg)
        _validate_local_window_cfg(cfg)
        if plan is not None and not isinstance(plan, (BuildPlan, InputOnlyPlan)):
            msg = 'plan must be BuildPlan, InputOnlyPlan, or None'
            raise TypeError(msg)
        if cfg.mode == 'infer' and isinstance(plan, BuildPlan):
            msg = 'infer mode does not support BuildPlan because GT local labels are absent'
            raise TypeError(msg)

        self.cfg = cfg
        self.plan = plan
        self.transform = transform
        self._rng = np.random.default_rng()
        self._subsetloader = TraceSubsetLoader(LoaderConfig(pad_traces_to=1))
        self._waveform_mode = normalize_waveform_mode(waveform_mode)
        self._segy_endian = normalize_segy_endian(segy_endian)

        if self.cfg.mode in ('train', 'eval'):
            if fb_files is None:
                msg = f'fb_files is required in {self.cfg.mode} mode'
                raise ValueError(msg)
            if len(fb_files) != len(segy_files):
                msg = 'segy_files and fb_files must have the same length'
                raise ValueError(msg)
            if coarse_artifact_npz_path is not None or coarse_artifact_meta_path is not None:
                msg = f'coarse artifact paths are not used in {self.cfg.mode} mode'
                raise ValueError(msg)
        else:
            if fb_files is not None:
                msg = 'fb_files must be omitted in infer mode'
                raise ValueError(msg)
            if coarse_artifact_npz_path is None or coarse_artifact_meta_path is None:
                msg = 'infer mode requires coarse_artifact_npz_path and coarse_artifact_meta_path'
                raise ValueError(msg)

        self.segy_files = list(segy_files)
        self.file_infos: list[dict[str, Any]] = []
        for segy_path in self.segy_files:
            self.file_infos.append(
                build_file_info(
                    segy_path,
                    ffid_byte=ffid_byte,
                    chno_byte=chno_byte,
                    cmp_byte=cmp_byte,
                    header_cache_dir=header_cache_dir,
                    use_header_cache=bool(use_header_cache),
                    include_centroids=False,
                    waveform_mode=self._waveform_mode,
                    segy_endian=self._segy_endian,
                )
            )

        self._file_bases: list[int] = []
        n_total = 0
        for info in self.file_infos:
            self._file_bases.append(n_total)
            n_total += int(info['n_traces'])
        if n_total <= 0:
            msg = 'no traces were loaded from segy_files'
            raise ValueError(msg)
        self.n_total = int(n_total)

        self._items: list[_LocalWindowItem] = []
        if self.cfg.mode in ('train', 'eval'):
            self._build_gt_items(fb_files=list(fb_files or []))
        else:
            coarse = _load_coarse_artifact(
                npz_path=coarse_artifact_npz_path,
                meta_path=coarse_artifact_meta_path,
            )
            self._validate_coarse_alignment(coarse)
            self._build_infer_items(coarse)

        if len(self._items) == 0:
            msg = f'LocalWindowDataset produced no items in {self.cfg.mode} mode'
            raise ValueError(msg)

    def close(self) -> None:
        for info in self.file_infos:
            segy_obj = info.get('segy_obj')
            if segy_obj is not None:
                with contextlib.suppress(Exception):
                    segy_obj.close()
        self.file_infos.clear()
        self._items.clear()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def __len__(self) -> int:
        return len(self._items)

    def _build_gt_items(self, *, fb_files: list[str]) -> None:
        for file_idx, (info, fb_path) in enumerate(zip(self.file_infos, fb_files, strict=True)):
            gt_pick_idx = _load_label_array(
                fb_path,
                expected_len=int(info['n_traces']),
            )
            n_samples = int(info['n_samples'])
            file_base = int(self._file_bases[file_idx])
            for trace_idx_in_file, raw_pick_idx in enumerate(gt_pick_idx.tolist()):
                raw_pick_idx_int = int(raw_pick_idx)
                if raw_pick_idx_int <= 0:
                    continue
                if raw_pick_idx_int >= n_samples:
                    msg = (
                        f'GT pick {raw_pick_idx_int} is outside raw sample range '
                        f'[0, {n_samples}) for {info["path"]} trace {trace_idx_in_file}'
                    )
                    raise ValueError(msg)
                start_idx, end_idx, local_seed_idx = _resolve_local_window(
                    raw_seed_idx=raw_pick_idx_int,
                    n_samples=n_samples,
                    local_window_len=int(self.cfg.local_window_len),
                )
                local_pick_idx = raw_pick_idx_int - start_idx
                if local_pick_idx < 0 or local_pick_idx >= int(self.cfg.local_window_len):
                    msg = (
                        'local GT pick index is outside the fixed local window: '
                        f'raw_pick_idx={raw_pick_idx_int}, start={start_idx}, '
                        f'local_pick_idx={local_pick_idx}, W_local={int(self.cfg.local_window_len)}'
                    )
                    raise RuntimeError(msg)
                self._items.append(
                    _LocalWindowItem(
                        file_idx=int(file_idx),
                        trace_idx_in_file=int(trace_idx_in_file),
                        raw_trace_idx=file_base + int(trace_idx_in_file),
                        raw_seed_idx=raw_pick_idx_int,
                        raw_pick_idx=raw_pick_idx_int,
                        local_window_start_idx=start_idx,
                        local_window_end_idx=end_idx,
                        local_seed_idx=local_seed_idx,
                        local_pick_idx=int(local_pick_idx),
                        seed_source='gt',
                    )
                )

    def _validate_coarse_alignment(self, coarse: _LoadedCoarseArtifact) -> None:
        if coarse.n_traces != int(self.n_total):
            msg = f'coarse artifact n_traces {coarse.n_traces} != loaded raw traces {self.n_total}'
            raise ValueError(msg)
        expected_raw_trace_idx = np.arange(int(self.n_total), dtype=np.int64)
        if not np.array_equal(coarse.raw_trace_idx, expected_raw_trace_idx):
            msg = 'coarse artifact raw_trace_idx must be exactly arange(n_traces)'
            raise ValueError(msg)

        expected_offsets = np.concatenate(
            [np.asarray(info['offsets'], dtype=np.float32) for info in self.file_infos],
            axis=0,
        ).astype(np.float32, copy=False)
        if not np.array_equal(coarse.offsets, expected_offsets):
            msg = 'coarse artifact offsets must match the raw SEG-Y offsets exactly'
            raise ValueError(msg)

        for info in self.file_infos:
            n_samples = int(info['n_samples'])
            if n_samples != int(coarse.n_samples):
                msg = (
                    f'raw SEG-Y n_samples {n_samples} does not match coarse artifact '
                    f'n_samples {coarse.n_samples} for {info["path"]}'
                )
                raise ValueError(msg)
            expected_time = (
                np.arange(n_samples, dtype=np.float32)
                * np.float32(float(info['dt_sec']))
            )
            if not np.array_equal(coarse.time_axis, expected_time):
                msg = (
                    'coarse artifact time_axis must match the raw SEG-Y absolute time axis '
                    f'for {info["path"]}'
                )
                raise ValueError(msg)

    def _build_infer_items(self, coarse: _LoadedCoarseArtifact) -> None:
        for file_idx, info in enumerate(self.file_infos):
            n_samples = int(info['n_samples'])
            file_base = int(self._file_bases[file_idx])
            for trace_idx_in_file in range(int(info['n_traces'])):
                raw_trace_idx = file_base + int(trace_idx_in_file)
                trace_valid = bool(coarse.trace_valid[raw_trace_idx])
                raw_seed_idx = int(coarse.pick_idx[raw_trace_idx])
                if not trace_valid:
                    if raw_seed_idx != -1:
                        msg = (
                            'coarse artifact invalid trace must have pick_idx=-1, got '
                            f'pick_idx={raw_seed_idx} at raw_trace_idx={raw_trace_idx}'
                        )
                        raise ValueError(msg)
                    continue
                if raw_seed_idx < 0 or raw_seed_idx >= n_samples:
                    msg = (
                        f'coarse seed {raw_seed_idx} is outside raw sample range '
                        f'[0, {n_samples}) for {info["path"]} trace {trace_idx_in_file}'
                    )
                    raise ValueError(msg)

                start_idx, end_idx, local_seed_idx = _resolve_local_window(
                    raw_seed_idx=raw_seed_idx,
                    n_samples=n_samples,
                    local_window_len=int(self.cfg.local_window_len),
                )
                self._items.append(
                    _LocalWindowItem(
                        file_idx=int(file_idx),
                        trace_idx_in_file=int(trace_idx_in_file),
                        raw_trace_idx=int(raw_trace_idx),
                        raw_seed_idx=int(raw_seed_idx),
                        raw_pick_idx=-1,
                        local_window_start_idx=start_idx,
                        local_window_end_idx=end_idx,
                        local_seed_idx=local_seed_idx,
                        local_pick_idx=-1,
                        seed_source='coarse',
                    )
                )

    def _load_trace(self, item: _LocalWindowItem) -> np.ndarray:
        info = self.file_infos[item.file_idx]
        trace_2d = self._subsetloader.load(
            info['mmap'],
            np.asarray([item.trace_idx_in_file], dtype=np.int64),
        )
        if not isinstance(trace_2d, np.ndarray):
            msg = 'TraceSubsetLoader must return numpy.ndarray'
            raise TypeError(msg)
        if trace_2d.shape != (1, int(info['n_samples'])):
            msg = (
                f'expected single-trace load shape (1, {int(info["n_samples"])}), '
                f'got {trace_2d.shape}'
            )
            raise ValueError(msg)
        return trace_2d[0].astype(np.float32, copy=False)

    def _apply_transform(
        self,
        x_view_local: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.transform is None:
            return x_view_local, {}

        x_out, transform_meta = apply_transform_2d_with_meta(
            self.transform,
            x_view_local,
            self._rng,
            msg_bad_out='local transform must return 2D numpy or (2D, meta)',
            msg_bad_meta='local transform meta must be dict, got {type}',
            exc_bad_out=ValueError,
            exc_bad_meta=TypeError,
            allow_non_dict_meta=True,
        )
        if x_out.shape != x_view_local.shape:
            msg = (
                f'local transform must preserve shape {x_view_local.shape}, '
                f'got {x_out.shape}'
            )
            raise ValueError(msg)
        if bool(transform_meta.get('hflip', False)):
            msg = 'local transform must not flip the local window axis'
            raise ValueError(msg)
        if float(transform_meta.get('factor_h', 1.0)) != 1.0:
            msg = 'local transform must keep the trace axis unchanged'
            raise ValueError(msg)
        if float(transform_meta.get('factor', 1.0)) != 1.0:
            msg = 'local transform must not rescale the local time axis'
            raise ValueError(msg)
        if int(transform_meta.get('start', 0)) != 0:
            msg = 'local transform must not shift the local time axis'
            raise ValueError(msg)
        return x_out.astype(np.float32, copy=False), transform_meta

    def _validate_plan_output(self, sample: dict[str, Any]) -> None:
        if 'input' not in sample:
            msg = "plan must set sample['input']"
            raise KeyError(msg)
        x_in = sample['input']
        if not isinstance(x_in, torch.Tensor):
            msg = "sample['input'] must be torch.Tensor"
            raise TypeError(msg)
        if x_in.ndim != 3:
            msg = f"sample['input'] must be (C,H,W), got {tuple(x_in.shape)}"
            raise ValueError(msg)
        if int(x_in.shape[1]) != 1 or int(x_in.shape[2]) != int(self.cfg.local_window_len):
            msg = (
                'sample["input"] must preserve the local dataset geometry: '
                f'expected (*,1,{int(self.cfg.local_window_len)}), got {tuple(x_in.shape)}'
            )
            raise ValueError(msg)

        if isinstance(self.plan, BuildPlan):
            if 'target' not in sample:
                msg = "BuildPlan must set sample['target']"
                raise KeyError(msg)
            target = sample['target']
            if not isinstance(target, torch.Tensor):
                msg = "sample['target'] must be torch.Tensor"
                raise TypeError(msg)
            if target.ndim != 3:
                msg = f"sample['target'] must be (C,H,W), got {tuple(target.shape)}"
                raise ValueError(msg)
            if int(target.shape[1]) != 1 or int(target.shape[2]) != int(self.cfg.local_window_len):
                msg = (
                    'sample["target"] must preserve the local dataset geometry: '
                    f'expected (*,1,{int(self.cfg.local_window_len)}), got {tuple(target.shape)}'
                )
                raise ValueError(msg)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._items[index]
        info = self.file_infos[item.file_idx]
        trace = self._load_trace(item)
        x_local_1d, raw_sample_idx_local = _extract_fixed_window(
            trace,
            start_idx=item.local_window_start_idx,
            end_idx=item.local_window_end_idx,
            local_window_len=int(self.cfg.local_window_len),
        )
        x_view_local = x_local_1d[None, :]
        x_view_local, transform_meta = self._apply_transform(x_view_local)

        if raw_sample_idx_local[item.local_seed_idx] != int(item.raw_seed_idx):
            msg = (
                'raw/local seed index mapping is inconsistent: '
                f'raw_sample_idx_local[{item.local_seed_idx}]={raw_sample_idx_local[item.local_seed_idx]} '
                f'!= raw_seed_idx={item.raw_seed_idx}'
            )
            raise RuntimeError(msg)
        if item.raw_pick_idx >= 0:
            if item.local_pick_idx < 0:
                msg = 'raw_pick_idx is present but local_pick_idx is invalid'
                raise RuntimeError(msg)
            if raw_sample_idx_local[item.local_pick_idx] != int(item.raw_pick_idx):
                msg = (
                    'raw/local pick index mapping is inconsistent: '
                    f'raw_sample_idx_local[{item.local_pick_idx}]={raw_sample_idx_local[item.local_pick_idx]} '
                    f'!= raw_pick_idx={item.raw_pick_idx}'
                )
                raise RuntimeError(msg)

        valid_local_count = int(np.count_nonzero(raw_sample_idx_local >= 0))
        if valid_local_count != int(item.local_window_end_idx - item.local_window_start_idx):
            msg = (
                'raw/local local-window coverage is inconsistent: '
                f'valid_local_count={valid_local_count}, '
                f'raw_coverage={int(item.local_window_end_idx - item.local_window_start_idx)}'
            )
            raise RuntimeError(msg)

        trace_valid = np.asarray([True], dtype=np.bool_)
        label_valid = np.asarray([item.local_pick_idx >= 0], dtype=np.bool_)
        offsets_view_local = np.asarray(
            [abs(float(info['offsets'][item.trace_idx_in_file]))],
            dtype=np.float32,
        )
        time_view_local = _build_time_view_local(
            raw_sample_idx_local=raw_sample_idx_local,
            dt_sec=float(info['dt_sec']),
        )

        raw_trace_idx = np.asarray([item.raw_trace_idx], dtype=np.int64)
        raw_seed_idx = np.asarray([item.raw_seed_idx], dtype=np.int64)
        raw_pick_idx = np.asarray([item.raw_pick_idx], dtype=np.int64)
        local_window_start_idx = np.asarray([item.local_window_start_idx], dtype=np.int64)
        local_window_end_idx = np.asarray([item.local_window_end_idx], dtype=np.int64)
        local_seed_idx = np.asarray([item.local_seed_idx], dtype=np.int64)
        local_pick_idx = np.asarray([item.local_pick_idx], dtype=np.int64)

        meta: dict[str, Any] = {
            'mode': self.cfg.mode,
            'seed_source': item.seed_source,
            'file_idx': int(item.file_idx),
            'file_path': str(info['path']),
            'trace_idx_in_file': int(item.trace_idx_in_file),
            'dt_sec': np.float32(float(info['dt_sec'])),
            'local_window_len': int(self.cfg.local_window_len),
            'trace_valid': trace_valid,
            'label_valid': label_valid,
            'offsets_view': offsets_view_local,
            'time_view': time_view_local,
            'raw_sample_idx_local': raw_sample_idx_local,
            'raw_trace_idx': raw_trace_idx,
            'raw_seed_idx': raw_seed_idx,
            'raw_pick_idx': raw_pick_idx,
            'local_window_start_idx': local_window_start_idx,
            'local_window_end_idx': local_window_end_idx,
            'local_seed_idx': local_seed_idx,
            'local_pick_idx': local_pick_idx,
        }
        if transform_meta:
            meta['transform_meta'] = transform_meta

        sample: dict[str, Any] = {
            'x_view': x_view_local,
            'x_view_local': x_view_local,
            'offsets_view_local': offsets_view_local,
            'time_view_local': time_view_local,
            'raw_sample_idx_local': raw_sample_idx_local,
            'raw_trace_idx': raw_trace_idx,
            'raw_seed_idx': raw_seed_idx,
            'local_window_start_idx': local_window_start_idx,
            'local_window_end_idx': local_window_end_idx,
            'local_seed_idx': local_seed_idx,
            'local_pick_idx': local_pick_idx,
            'raw_pick_idx': raw_pick_idx,
            'trace_valid': trace_valid,
            'label_valid': label_valid,
            'meta': meta,
        }

        if self.plan is not None:
            self.plan.run(sample, rng=self._rng)
            self._validate_plan_output(sample)

        return sample
