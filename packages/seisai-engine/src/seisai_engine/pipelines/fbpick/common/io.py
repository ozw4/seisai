from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import (
    ARTIFACT_META_REQUIRED_KEYS,
    ARTIFACT_VERSION,
    ArtifactMeta,
    ArtifactSpec,
    get_artifact_spec,
)
from .config import FbpickPathsCfg

__all__ = [
    'ArtifactPaths',
    'LoadedArtifact',
    'load_artifact',
    'load_artifact_from_paths',
    'load_coarse_artifact',
    'load_coarse_artifact_from_paths',
    'load_fine_artifact',
    'load_global_qc_artifact',
    'require_artifact_files',
    'resolve_artifact_paths',
    'resolve_stage_dir',
    'resolve_survey_dir',
    'save_artifact',
    'save_coarse_artifact',
    'save_fine_artifact',
    'save_global_qc_artifact',
]


@dataclass(frozen=True)
class ArtifactPaths:
    survey_dir: Path
    stage_dir: Path
    npz_path: Path
    meta_path: Path


@dataclass(frozen=True)
class LoadedArtifact:
    meta: ArtifactMeta
    arrays: dict[str, np.ndarray]
    paths: ArtifactPaths


def _require_paths_cfg(paths_cfg: FbpickPathsCfg) -> FbpickPathsCfg:
    if not isinstance(paths_cfg, FbpickPathsCfg):
        msg = 'paths_cfg must be FbpickPathsCfg'
        raise TypeError(msg)
    return paths_cfg


def resolve_survey_dir(paths_cfg: FbpickPathsCfg) -> Path:
    cfg = _require_paths_cfg(paths_cfg)
    out_dir = Path(cfg.out_dir).expanduser()
    if cfg.survey_id == '':
        msg = 'paths_cfg.survey_id must not be empty'
        raise ValueError(msg)
    return out_dir / cfg.survey_id


def resolve_stage_dir(paths_cfg: FbpickPathsCfg, *, stage: str) -> Path:
    _ = get_artifact_spec(stage)
    return resolve_survey_dir(paths_cfg) / stage


def resolve_artifact_paths(paths_cfg: FbpickPathsCfg, *, stage: str) -> ArtifactPaths:
    spec = get_artifact_spec(stage)
    survey_dir = resolve_survey_dir(paths_cfg)
    stage_dir = survey_dir / spec.stage
    return ArtifactPaths(
        survey_dir=survey_dir,
        stage_dir=stage_dir,
        npz_path=stage_dir / spec.npz_filename,
        meta_path=stage_dir / spec.meta_filename,
    )


def require_artifact_files(paths_cfg: FbpickPathsCfg, *, stage: str) -> ArtifactPaths:
    paths = resolve_artifact_paths(paths_cfg, stage=stage)
    if not paths.npz_path.exists():
        msg = f'required artifact npz not found: {paths.npz_path}'
        raise FileNotFoundError(msg)
    if not paths.meta_path.exists():
        msg = f'required artifact meta json not found: {paths.meta_path}'
        raise FileNotFoundError(msg)
    return paths


def _require_explicit_artifact_paths(
    *,
    npz_path: str | Path,
    meta_path: str | Path,
) -> tuple[Path, Path]:
    npz = Path(npz_path).expanduser().resolve()
    meta = Path(meta_path).expanduser().resolve()
    if not npz.exists():
        msg = f'required artifact npz not found: {npz}'
        raise FileNotFoundError(msg)
    if not meta.exists():
        msg = f'required artifact meta json not found: {meta}'
        raise FileNotFoundError(msg)
    return npz, meta


def _expected_keys(spec: ArtifactSpec) -> Sequence[str]:
    return tuple(field.key for field in spec.fields if field.required)


def _check_array_keys(spec: ArtifactSpec, arrays: Mapping[str, Any]) -> None:
    expected = set(_expected_keys(spec))
    actual = set(arrays.keys())
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if missing:
        msg = f'{spec.stage} artifact missing required keys: {sorted(missing)}'
        raise ValueError(msg)
    if extra:
        msg = f'{spec.stage} artifact has unsupported keys: {sorted(extra)}'
        raise ValueError(msg)


def _coerce_save_array(*, key: str, value: Any, dtype_name: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.dtype(dtype_name))
    except TypeError as exc:
        msg = f'failed to convert key "{key}" to dtype {dtype_name}'
        raise TypeError(msg) from exc
    except ValueError as exc:
        msg = f'failed to convert key "{key}" to dtype {dtype_name}'
        raise ValueError(msg) from exc


def _validate_loaded_dtype(*, key: str, arr: np.ndarray, dtype_name: str) -> None:
    expected = np.dtype(dtype_name)
    if arr.dtype != expected:
        msg = f'artifact key "{key}" must have dtype {expected.name}, got {arr.dtype.name}'
        raise TypeError(msg)


def _validate_shapes(spec: ArtifactSpec, arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    dims: dict[str, int] = {}
    for field in spec.fields:
        arr = arrays[field.key]
        if arr.ndim != len(field.shape):
            msg = (
                f'artifact key "{field.key}" must have {len(field.shape)} dims, '
                f'got shape {arr.shape}'
            )
            raise ValueError(msg)
        for axis, dim_name in enumerate(field.shape):
            axis_size = int(arr.shape[axis])
            if axis_size <= 0:
                msg = f'artifact key "{field.key}" has non-positive axis size: {arr.shape}'
                raise ValueError(msg)
            if dim_name not in dims:
                dims[dim_name] = axis_size
                continue
            if dims[dim_name] != axis_size:
                msg = (
                    f'artifact dimension mismatch for "{dim_name}": '
                    f'expected {dims[dim_name]}, got {axis_size} from key "{field.key}"'
                )
                raise ValueError(msg)
    return dims


def _normalize_save_arrays(
    spec: ArtifactSpec,
    arrays: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    _check_array_keys(spec, arrays)
    normalized: dict[str, np.ndarray] = {}
    for field in spec.fields:
        normalized[field.key] = _coerce_save_array(
            key=field.key,
            value=arrays[field.key],
            dtype_name=field.dtype_name,
        )
    dims = _validate_shapes(spec, normalized)
    return normalized, dims


def _normalize_loaded_arrays(
    spec: ArtifactSpec,
    arrays: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    _check_array_keys(spec, arrays)
    normalized: dict[str, np.ndarray] = {}
    for field in spec.fields:
        arr = np.asarray(arrays[field.key])
        _validate_loaded_dtype(key=field.key, arr=arr, dtype_name=field.dtype_name)
        normalized[field.key] = arr
    dims = _validate_shapes(spec, normalized)
    return normalized, dims


def _normalize_source_refs(source_refs: Mapping[str, str] | None) -> dict[str, str]:
    if source_refs is None:
        return {}
    if not isinstance(source_refs, Mapping):
        msg = 'source_refs must be mapping[str, str]'
        raise TypeError(msg)
    normalized: dict[str, str] = {}
    for key, value in source_refs.items():
        if not isinstance(key, str):
            msg = 'source_refs keys must be str'
            raise TypeError(msg)
        if not isinstance(value, str):
            msg = 'source_refs values must be str'
            raise TypeError(msg)
        normalized[key] = value
    return normalized


def _meta_to_payload(meta: ArtifactMeta) -> dict[str, Any]:
    return {
        'artifact_version': int(meta.artifact_version),
        'stage': str(meta.stage),
        'survey_id': str(meta.survey_id),
        'npz_filename': str(meta.npz_filename),
        'source_refs': dict(meta.source_refs),
        'dimensions': dict(meta.dimensions),
    }


def _load_meta(*, meta_path: Path, spec: ArtifactSpec, survey_id: str, npz_filename: str) -> ArtifactMeta:
    raw = json.loads(meta_path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        msg = f'artifact meta must be json object: {meta_path}'
        raise TypeError(msg)
    missing = set(ARTIFACT_META_REQUIRED_KEYS).difference(raw.keys())
    if missing:
        msg = f'artifact meta missing required keys {sorted(missing)}: {meta_path}'
        raise ValueError(msg)

    artifact_version = raw['artifact_version']
    stage = raw['stage']
    survey_id_raw = raw['survey_id']
    npz_filename_raw = raw['npz_filename']
    source_refs_raw = raw['source_refs']
    dimensions_raw = raw['dimensions']

    if artifact_version != ARTIFACT_VERSION:
        msg = (
            f'unsupported artifact version {artifact_version} for {meta_path}; '
            f'expected {ARTIFACT_VERSION}'
        )
        raise ValueError(msg)
    if stage != spec.stage:
        msg = f'artifact meta stage mismatch: expected {spec.stage}, got {stage}'
        raise ValueError(msg)
    if survey_id_raw != survey_id:
        msg = (
            f'artifact meta survey_id mismatch: expected {survey_id}, '
            f'got {survey_id_raw}'
        )
        raise ValueError(msg)
    if npz_filename_raw != npz_filename:
        msg = (
            f'artifact meta npz_filename mismatch: expected {npz_filename}, '
            f'got {npz_filename_raw}'
        )
        raise ValueError(msg)
    if not isinstance(source_refs_raw, dict):
        msg = 'artifact meta source_refs must be dict[str, str]'
        raise TypeError(msg)
    if not isinstance(dimensions_raw, dict):
        msg = 'artifact meta dimensions must be dict[str, int]'
        raise TypeError(msg)

    source_refs: dict[str, str] = {}
    for key, value in source_refs_raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            msg = 'artifact meta source_refs must be dict[str, str]'
            raise TypeError(msg)
        source_refs[key] = value

    dimensions: dict[str, int] = {}
    for key, value in dimensions_raw.items():
        if not isinstance(key, str):
            msg = 'artifact meta dimension names must be str'
            raise TypeError(msg)
        if isinstance(value, bool) or not isinstance(value, int):
            msg = 'artifact meta dimension values must be int'
            raise TypeError(msg)
        if value <= 0:
            msg = 'artifact meta dimension values must be positive'
            raise ValueError(msg)
        dimensions[key] = int(value)

    return ArtifactMeta(
        artifact_version=int(artifact_version),
        stage=str(stage),
        survey_id=str(survey_id_raw),
        npz_filename=str(npz_filename_raw),
        source_refs=source_refs,
        dimensions=dimensions,
    )


def _write_meta(path: Path, meta: ArtifactMeta) -> None:
    payload = json.dumps(
        _meta_to_payload(meta),
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
    )
    path.write_text(payload + '\n', encoding='utf-8')


def save_artifact(
    *,
    paths_cfg: FbpickPathsCfg,
    stage: str,
    arrays: Mapping[str, Any],
    source_refs: Mapping[str, str] | None = None,
) -> ArtifactPaths:
    cfg = _require_paths_cfg(paths_cfg)
    spec = get_artifact_spec(stage)
    normalized_arrays, dims = _normalize_save_arrays(spec, arrays)
    paths = resolve_artifact_paths(cfg, stage=spec.stage)
    paths.stage_dir.mkdir(parents=True, exist_ok=True)

    meta = ArtifactMeta(
        artifact_version=ARTIFACT_VERSION,
        stage=spec.stage,
        survey_id=cfg.survey_id,
        npz_filename=paths.npz_path.name,
        source_refs=_normalize_source_refs(source_refs),
        dimensions=dims,
    )
    np.savez_compressed(paths.npz_path, **normalized_arrays)
    _write_meta(paths.meta_path, meta)
    return paths


def load_artifact(*, paths_cfg: FbpickPathsCfg, stage: str) -> LoadedArtifact:
    cfg = _require_paths_cfg(paths_cfg)
    spec = get_artifact_spec(stage)
    paths = require_artifact_files(cfg, stage=spec.stage)
    meta = _load_meta(
        meta_path=paths.meta_path,
        spec=spec,
        survey_id=cfg.survey_id,
        npz_filename=paths.npz_path.name,
    )
    with np.load(paths.npz_path, allow_pickle=False) as z:
        raw_arrays = {key: z[key] for key in z.files}
    arrays, dims = _normalize_loaded_arrays(spec, raw_arrays)
    if dims != meta.dimensions:
        msg = (
            f'artifact dimensions mismatch between npz and meta for {paths.npz_path}: '
            f'npz={dims}, meta={meta.dimensions}'
        )
        raise ValueError(msg)
    return LoadedArtifact(meta=meta, arrays=arrays, paths=paths)


def load_artifact_from_paths(
    *,
    stage: str,
    npz_path: str | Path,
    meta_path: str | Path,
    survey_id: str,
) -> LoadedArtifact:
    spec = get_artifact_spec(stage)
    npz_resolved, meta_resolved = _require_explicit_artifact_paths(
        npz_path=npz_path,
        meta_path=meta_path,
    )
    if not isinstance(survey_id, str) or survey_id.strip() == '':
        msg = 'survey_id must be non-empty str'
        raise ValueError(msg)

    meta = _load_meta(
        meta_path=meta_resolved,
        spec=spec,
        survey_id=survey_id,
        npz_filename=npz_resolved.name,
    )
    with np.load(npz_resolved, allow_pickle=False) as z:
        raw_arrays = {key: z[key] for key in z.files}
    arrays, dims = _normalize_loaded_arrays(spec, raw_arrays)
    if dims != meta.dimensions:
        msg = (
            f'artifact dimensions mismatch between npz and meta for {npz_resolved}: '
            f'npz={dims}, meta={meta.dimensions}'
        )
        raise ValueError(msg)

    return LoadedArtifact(
        meta=meta,
        arrays=arrays,
        paths=ArtifactPaths(
            survey_dir=meta_resolved.parent.parent,
            stage_dir=npz_resolved.parent,
            npz_path=npz_resolved,
            meta_path=meta_resolved,
        ),
    )


def save_coarse_artifact(
    *,
    paths_cfg: FbpickPathsCfg,
    arrays: Mapping[str, Any],
    source_refs: Mapping[str, str] | None = None,
) -> ArtifactPaths:
    return save_artifact(
        paths_cfg=paths_cfg,
        stage='coarse',
        arrays=arrays,
        source_refs=source_refs,
    )


def save_fine_artifact(
    *,
    paths_cfg: FbpickPathsCfg,
    arrays: Mapping[str, Any],
    source_refs: Mapping[str, str] | None = None,
) -> ArtifactPaths:
    return save_artifact(
        paths_cfg=paths_cfg,
        stage='fine',
        arrays=arrays,
        source_refs=source_refs,
    )


def save_global_qc_artifact(
    *,
    paths_cfg: FbpickPathsCfg,
    arrays: Mapping[str, Any],
    source_refs: Mapping[str, str] | None = None,
) -> ArtifactPaths:
    return save_artifact(
        paths_cfg=paths_cfg,
        stage='global_qc',
        arrays=arrays,
        source_refs=source_refs,
    )


def load_coarse_artifact(*, paths_cfg: FbpickPathsCfg) -> LoadedArtifact:
    return load_artifact(paths_cfg=paths_cfg, stage='coarse')


def load_coarse_artifact_from_paths(
    *,
    npz_path: str | Path,
    meta_path: str | Path,
    survey_id: str,
) -> LoadedArtifact:
    return load_artifact_from_paths(
        stage='coarse',
        npz_path=npz_path,
        meta_path=meta_path,
        survey_id=survey_id,
    )


def load_fine_artifact(*, paths_cfg: FbpickPathsCfg) -> LoadedArtifact:
    return load_artifact(paths_cfg=paths_cfg, stage='fine')


def load_global_qc_artifact(*, paths_cfg: FbpickPathsCfg) -> LoadedArtifact:
    return load_artifact(paths_cfg=paths_cfg, stage='global_qc')
