from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from seisai_pick.global_qc import (
    ArrivalBand,
    InversionBackend,
    MissingInversionBackend,
    RepickResult,
    build_arrival_band,
    build_hard_arrival_mask,
    build_survey_geometry,
    compute_global_consistency,
    compute_global_qc_confidence,
    effective_trace_valid,
    normalize_survey_geometry,
    repick_with_arrival_band,
    require_inversion_backend,
    validate_backend_pick_idx,
)

from seisai_engine.pipelines.common import load_cfg_with_base_dir
from seisai_engine.pipelines.fbpick.common import (
    QC_STATUS_ADJUST,
    QC_STATUS_KEEP,
    QC_STATUS_REJECT,
)

from .build_candidates import build_global_qc_candidates
from .config import GlobalQcBackendCfg, GlobalQcConfig, load_global_qc_config
from .export import export_global_qc_result

__all__ = [
    'DEFAULT_CONFIG_PATH',
    'GlobalQcRunResult',
    'build_inversion_backend',
    'load_global_qc_geometry',
    'main',
    'run_global_qc',
]

DEFAULT_CONFIG_PATH = Path('examples/config_fbpick_global_qc.yaml')

_GEOMETRY_REQUIRED_KEYS = (
    'shot_x',
    'shot_y',
    'shot_z',
    'recv_x',
    'recv_y',
    'recv_z',
    'raw_trace_idx',
)
_GEOMETRY_OPTIONAL_KEYS = (
    'trace_valid',
    'shot_elevation',
    'recv_elevation',
    'shot_datum',
    'recv_datum',
)


@dataclass(frozen=True)
class GlobalQcRunResult:
    artifact_npz_path: Path
    artifact_meta_path: Path
    csv_path: Path | None
    pick_global: np.ndarray
    confidence_global: np.ndarray
    reject_flag: np.ndarray
    qc_status: np.ndarray
    raw_trace_idx: np.ndarray


def _require_mapping(payload: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        msg = f'{label} must be a mapping'
        raise TypeError(msg)
    return payload


def _scalar_string(value: Any, *, label: str) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            msg = f'{label} must be a scalar string'
            raise TypeError(msg)
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if not isinstance(value, str) or value.strip() == '':
        msg = f'{label} must be a non-empty string'
        raise ValueError(msg)
    return value


def _normalize_geometry_payload(
    payload: Mapping[str, Any],
    *,
    survey_id: str,
    source_label: str,
) -> dict[str, Any]:
    allowed = set(_GEOMETRY_REQUIRED_KEYS) | set(_GEOMETRY_OPTIONAL_KEYS) | {'survey_id'}
    actual = set(payload.keys())
    missing = set(_GEOMETRY_REQUIRED_KEYS).difference(actual)
    extra = actual.difference(allowed)
    if missing:
        msg = f'{source_label} is missing required geometry keys: {sorted(missing)}'
        raise ValueError(msg)
    if extra:
        msg = f'{source_label} has unsupported geometry keys: {sorted(extra)}'
        raise ValueError(msg)
    if 'survey_id' in payload:
        payload_survey_id = _scalar_string(payload['survey_id'], label=f'{source_label}.survey_id')
        if payload_survey_id != survey_id:
            msg = (
                f'{source_label}.survey_id mismatch: expected {survey_id!r}, '
                f'got {payload_survey_id!r}'
            )
            raise ValueError(msg)
    return dict(payload)


def _coerce_geometry_value(name: str, value: Any) -> np.ndarray:
    if name == 'raw_trace_idx':
        return np.asarray(value, dtype=np.int64)
    if name == 'trace_valid':
        return np.asarray(value, dtype=bool)
    return np.asarray(value, dtype=np.float32)


def _build_geometry_from_payload(
    payload: Mapping[str, Any],
    *,
    survey_id: str,
    source_label: str,
):
    normalized = _normalize_geometry_payload(
        payload,
        survey_id=survey_id,
        source_label=source_label,
    )
    kwargs: dict[str, Any] = {'survey_id': survey_id}
    for key in _GEOMETRY_REQUIRED_KEYS + _GEOMETRY_OPTIONAL_KEYS:
        if key not in normalized:
            continue
        kwargs[key] = _coerce_geometry_value(key, normalized[key])
    return build_survey_geometry(**kwargs)


def _load_geometry_payload_from_path(path: str, *, format_value: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        msg = f'geometry file not found: {resolved}'
        raise FileNotFoundError(msg)
    format_name = format_value.lower()
    if format_name == 'json':
        raw = json.loads(resolved.read_text(encoding='utf-8'))
        mapping = _require_mapping(raw, label=f'geometry json {resolved}')
        return dict(mapping)
    if format_name == 'npz':
        with np.load(resolved, allow_pickle=False) as z:
            return {key: z[key] for key in z.files}
    msg = f'unsupported geometry file format {format_name!r}: {resolved}'
    raise ValueError(msg)


def _reorder_geometry_like_candidates(geometry, raw_trace_idx: np.ndarray):
    geometry_raw_trace_idx = np.asarray(geometry.raw_trace_idx, dtype=np.int64)
    candidate_raw_trace_idx = np.asarray(raw_trace_idx, dtype=np.int64)
    if geometry_raw_trace_idx.shape != candidate_raw_trace_idx.shape:
        msg = (
            'geometry raw_trace_idx shape must match candidate raw_trace_idx shape exactly, got '
            f'{geometry_raw_trace_idx.shape} vs {candidate_raw_trace_idx.shape}'
        )
        raise ValueError(msg)
    if set(geometry_raw_trace_idx.tolist()) != set(candidate_raw_trace_idx.tolist()):
        msg = 'geometry raw_trace_idx must match candidate raw_trace_idx exactly'
        raise ValueError(msg)
    order = {int(value): idx for idx, value in enumerate(geometry_raw_trace_idx.tolist())}
    gather_idx = np.asarray([order[int(value)] for value in candidate_raw_trace_idx.tolist()], dtype=np.int64)

    kwargs: dict[str, Any] = {'survey_id': geometry.survey_id}
    for key in _GEOMETRY_REQUIRED_KEYS + _GEOMETRY_OPTIONAL_KEYS:
        value = getattr(geometry, key)
        if value is None:
            continue
        kwargs[key] = np.asarray(value)[gather_idx]
    return build_survey_geometry(**kwargs)


def load_global_qc_geometry(cfg: GlobalQcConfig, *, raw_trace_idx: np.ndarray):
    if not isinstance(cfg, GlobalQcConfig):
        msg = 'cfg must be GlobalQcConfig'
        raise TypeError(msg)
    if cfg.geometry.inline is not None:
        geometry = _build_geometry_from_payload(
            cfg.geometry.inline,
            survey_id=cfg.fbpick.paths.survey_id,
            source_label='inline geometry',
        )
    else:
        payload = _load_geometry_payload_from_path(
            str(cfg.geometry.path),
            format_value=cfg.geometry.format,
        )
        geometry = _build_geometry_from_payload(
            payload,
            survey_id=cfg.fbpick.paths.survey_id,
            source_label=f'geometry file {cfg.geometry.path}',
        )
    geometry = normalize_survey_geometry(geometry, normalization=cfg.geometry.normalization)
    return _reorder_geometry_like_candidates(geometry, raw_trace_idx=np.asarray(raw_trace_idx, dtype=np.int64))


def build_inversion_backend(cfg: GlobalQcBackendCfg) -> InversionBackend:
    if not isinstance(cfg, GlobalQcBackendCfg):
        msg = 'cfg must be GlobalQcBackendCfg'
        raise TypeError(msg)
    return MissingInversionBackend(name=cfg.name)


def _build_arrival_band_with_policy(
    cfg: GlobalQcConfig,
    *,
    expected,
    sample_axis_len: int,
    sample_interval_sec: float,
) -> ArrivalBand:
    arrival_band = build_arrival_band(
        expected,
        sample_axis_len=int(sample_axis_len),
        sample_interval_sec=float(sample_interval_sec),
        uncertainty_scale=float(cfg.arrival_band.uncertainty_scale),
        band_radius_sigma=float(cfg.arrival_band.band_radius_sigma),
        min_half_width_idx=int(cfg.arrival_band.min_half_width_idx),
        prior_floor=float(cfg.arrival_band.prior_floor),
    )
    if cfg.arrival_band.band_half_width_idx is None:
        return arrival_band

    explicit_half_width = np.full(
        (int(arrival_band.center_idx.shape[0]),),
        int(cfg.arrival_band.band_half_width_idx),
        dtype=np.int64,
    )
    feasible_mask = build_hard_arrival_mask(
        arrival_band.center_idx,
        explicit_half_width,
        sample_axis_len=int(arrival_band.sample_axis_len),
    )
    return ArrivalBand(
        survey_id=arrival_band.survey_id,
        backend_name=arrival_band.backend_name,
        sample_axis_len=int(arrival_band.sample_axis_len),
        center_idx=np.asarray(arrival_band.center_idx, dtype=np.float32),
        uncertainty_idx=np.asarray(arrival_band.uncertainty_idx, dtype=np.float32),
        band_half_width_idx=explicit_half_width,
        feasible_mask=np.asarray(feasible_mask, dtype=bool),
        prior=np.asarray(arrival_band.prior, dtype=np.float32),
        center_time_sec=None
        if arrival_band.center_time_sec is None
        else np.asarray(arrival_band.center_time_sec, dtype=np.float32),
        uncertainty_sec=None
        if arrival_band.uncertainty_sec is None
        else np.asarray(arrival_band.uncertainty_sec, dtype=np.float32),
        trace_valid=None
        if arrival_band.trace_valid is None
        else np.asarray(arrival_band.trace_valid, dtype=bool),
    )


def _slice_arrival_band(arrival_band: ArrivalBand, index: int) -> ArrivalBand:
    slice_obj = slice(index, index + 1)
    return ArrivalBand(
        survey_id=arrival_band.survey_id,
        backend_name=arrival_band.backend_name,
        sample_axis_len=int(arrival_band.sample_axis_len),
        center_idx=np.asarray(arrival_band.center_idx, dtype=np.float32)[slice_obj],
        uncertainty_idx=np.asarray(arrival_band.uncertainty_idx, dtype=np.float32)[slice_obj],
        band_half_width_idx=np.asarray(arrival_band.band_half_width_idx, dtype=np.int64)[slice_obj],
        feasible_mask=np.asarray(arrival_band.feasible_mask, dtype=bool)[slice_obj],
        prior=np.asarray(arrival_band.prior, dtype=np.float32)[slice_obj],
        center_time_sec=None
        if arrival_band.center_time_sec is None
        else np.asarray(arrival_band.center_time_sec, dtype=np.float32)[slice_obj],
        uncertainty_sec=None
        if arrival_band.uncertainty_sec is None
        else np.asarray(arrival_band.uncertainty_sec, dtype=np.float32)[slice_obj],
        trace_valid=None
        if arrival_band.trace_valid is None
        else np.asarray(arrival_band.trace_valid, dtype=bool)[slice_obj],
    )


def _zero_mass_error(exc: Exception) -> bool:
    return 'zero feasible mass' in str(exc)


def _repick_kwargs(
    cfg: GlobalQcConfig,
    *,
    prob: np.ndarray,
    arrival_band: ArrivalBand,
    invalid_trace_mask: np.ndarray,
    reject_trace_mask: np.ndarray,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        'prob': prob,
        'invalid_trace_mask': invalid_trace_mask,
        'reject_trace_mask': reject_trace_mask,
        'prior_power': float(cfg.arrival_band.prior_power),
        'return_reweighted_prob': True,
    }
    if cfg.arrival_band.use_hard_mask:
        kwargs['arrival_band'] = arrival_band
    else:
        kwargs['prior'] = np.asarray(arrival_band.prior, dtype=np.float32)
    return kwargs


def _repick_with_zero_mass_policy(
    cfg: GlobalQcConfig,
    *,
    prob: np.ndarray,
    arrival_band: ArrivalBand,
    invalid_trace_mask: np.ndarray,
    reject_trace_mask: np.ndarray,
) -> tuple[RepickResult, np.ndarray]:
    try:
        result = repick_with_arrival_band(
            **_repick_kwargs(
                cfg,
                prob=prob,
                arrival_band=arrival_band,
                invalid_trace_mask=invalid_trace_mask,
                reject_trace_mask=reject_trace_mask,
            )
        )
        zero_mass_mask = np.zeros((int(prob.shape[0]),), dtype=bool)
        return result, zero_mass_mask
    except ValueError as exc:
        if not cfg.reject_policy.allow_zero_mass_failure or not _zero_mass_error(exc):
            raise

    n_traces = int(prob.shape[0])
    pick_idx = np.full((n_traces,), -1, dtype=np.int64)
    confidence = np.zeros((n_traces,), dtype=np.float32)
    reject_mask = np.asarray(reject_trace_mask, dtype=bool).copy()
    reweighted_prob = np.zeros_like(prob, dtype=np.float32)
    zero_mass_mask = np.zeros((n_traces,), dtype=bool)

    for trace_idx in range(n_traces):
        row_invalid = np.asarray(invalid_trace_mask[trace_idx : trace_idx + 1], dtype=bool)
        row_reject = np.asarray(reject_trace_mask[trace_idx : trace_idx + 1], dtype=bool)
        if bool(row_invalid[0]) or bool(row_reject[0]):
            continue
        row_kwargs = _repick_kwargs(
            cfg,
            prob=np.asarray(prob[trace_idx : trace_idx + 1], dtype=np.float32),
            arrival_band=_slice_arrival_band(arrival_band, trace_idx),
            invalid_trace_mask=row_invalid,
            reject_trace_mask=row_reject,
        )
        try:
            row_result = repick_with_arrival_band(**row_kwargs)
        except ValueError as row_exc:
            if not _zero_mass_error(row_exc):
                raise
            zero_mass_mask[trace_idx] = True
            reject_mask[trace_idx] = True
            continue

        pick_idx[trace_idx] = int(np.asarray(row_result.pick_idx, dtype=np.int64)[0])
        confidence[trace_idx] = float(np.asarray(row_result.confidence, dtype=np.float32)[0])
        reject_mask[trace_idx] = bool(np.asarray(row_result.reject_mask, dtype=bool)[0])
        if row_result.reweighted_prob is not None:
            reweighted_prob[trace_idx] = np.asarray(
                row_result.reweighted_prob,
                dtype=np.float32,
            )[0]

    return (
        RepickResult(
            pick_idx=pick_idx,
            confidence=confidence,
            reject_mask=reject_mask,
            reweighted_prob=reweighted_prob,
        ),
        zero_mass_mask,
    )


def _source_refs(
    cfg: GlobalQcConfig,
    *,
    backend_name: str,
    config_path: str | Path | None,
    geometry_source: str,
) -> dict[str, str]:
    return {
        'coarse_artifact_npz_path': cfg.coarse_artifact.artifact_npz_path,
        'coarse_artifact_meta_path': cfg.coarse_artifact.artifact_meta_path,
        'fine_artifact_npz_path': cfg.fine_artifact.artifact_npz_path,
        'fine_artifact_meta_path': cfg.fine_artifact.artifact_meta_path,
        'geometry_source': geometry_source,
        'backend_name': backend_name,
        'backend_source_path': '' if cfg.backend.source_path is None else cfg.backend.source_path,
        'config_path': '' if config_path is None else str(Path(config_path).expanduser().resolve()),
        'zero_mass_policy': 'trace_reject'
        if cfg.reject_policy.allow_zero_mass_failure
        else 'survey_fail',
        'confidence_weights': json.dumps(
            {
                'probability_weight': float(cfg.confidence.probability_weight),
                'band_weight': float(cfg.confidence.band_weight),
                'trend_weight': float(cfg.confidence.trend_weight),
                'consistency_weight': float(cfg.confidence.consistency_weight),
            },
            sort_keys=True,
            separators=(',', ':'),
        ),
    }


def run_global_qc(
    cfg: GlobalQcConfig,
    *,
    backend: InversionBackend | None = None,
    config_path: str | Path | None = None,
) -> GlobalQcRunResult:
    """Run engine-side global QC orchestration.

    Notes
    -----
    - Fractional `center_idx` / `uncertainty_idx` are treated as continuous kernel inputs and
      are never rounded on the engine side.
    - Zero feasible mass is handled by engine policy: survey failure by default, or per-trace
      reject when `allow_zero_mass_failure=true`.
    - Confidence weights are always injected from config rather than relying on kernel defaults.
    """
    if not isinstance(cfg, GlobalQcConfig):
        msg = 'cfg must be GlobalQcConfig'
        raise TypeError(msg)

    candidates = build_global_qc_candidates(cfg)
    geometry = load_global_qc_geometry(cfg, raw_trace_idx=candidates.raw_trace_idx)

    geometry_trace_valid = np.asarray(effective_trace_valid(geometry), dtype=bool)
    stage_trace_valid = np.asarray(candidates.trace_valid, dtype=bool) & geometry_trace_valid
    base_pick_idx = np.asarray(candidates.base_pick_idx, dtype=np.int64)
    validate_backend_pick_idx(
        base_pick_idx,
        trace_count=int(candidates.n_traces),
        backend='numpy',
    )

    backend_impl = build_inversion_backend(cfg.backend) if backend is None else backend
    backend_impl = require_inversion_backend(backend_impl)
    summary = backend_impl.fit_travel_time_summary(
        geometry=geometry,
        pick_idx=base_pick_idx,
        sample_interval_sec=float(candidates.sample_interval_sec),
        trace_valid=stage_trace_valid,
        pick_confidence=np.asarray(candidates.base_confidence, dtype=np.float32),
    )
    expected = backend_impl.predict_expected_arrivals(
        geometry=geometry,
        sample_interval_sec=float(candidates.sample_interval_sec),
        summary=summary,
    )

    arrival_band = _build_arrival_band_with_policy(
        cfg,
        expected=expected,
        sample_axis_len=int(candidates.n_samples),
        sample_interval_sec=float(candidates.sample_interval_sec),
    )

    expected_trace_valid = stage_trace_valid.copy()
    if expected.trace_valid is not None:
        expected_trace_valid &= np.asarray(expected.trace_valid, dtype=bool)
    invalid_trace_mask = ~stage_trace_valid
    reject_trace_mask = stage_trace_valid & (~expected_trace_valid)

    repick_result, zero_mass_mask = _repick_with_zero_mass_policy(
        cfg,
        prob=np.asarray(candidates.base_prob, dtype=np.float32),
        arrival_band=arrival_band,
        invalid_trace_mask=invalid_trace_mask,
        reject_trace_mask=reject_trace_mask,
    )

    reweighted_prob = np.asarray(repick_result.reweighted_prob, dtype=np.float32)
    repicked_pick_idx = np.asarray(repick_result.pick_idx, dtype=np.int64)
    repick_reject_mask = np.asarray(repick_result.reject_mask, dtype=bool)
    consistency = compute_global_consistency(
        repicked_pick_idx,
        trace_valid=stage_trace_valid & (~zero_mass_mask),
        trend_center_idx=np.asarray(arrival_band.center_idx, dtype=np.float32),
        adjacent_radius=int(cfg.consistency.adjacent_radius),
        adjacent_sigma_idx=float(cfg.consistency.adjacent_sigma_idx),
        adjacent_min_count=int(cfg.consistency.adjacent_min_count),
        trend_sigma_idx=float(cfg.consistency.trend_sigma_idx),
        outlier_radius=int(cfg.consistency.outlier_radius),
        outlier_z_scale=float(cfg.consistency.outlier_z_scale),
        outlier_min_count=int(cfg.consistency.outlier_min_count),
        outlier_mad_floor_idx=float(cfg.consistency.outlier_mad_floor_idx),
        adjacent_weight=float(cfg.consistency.adjacent_weight),
        trend_weight=float(cfg.consistency.trend_weight),
        outlier_weight=float(cfg.consistency.outlier_weight),
    )
    global_confidence = compute_global_qc_confidence(
        repicked_pick_idx,
        prob=reweighted_prob,
        arrival_band=arrival_band,
        consistency=consistency,
        trend_center_idx=np.asarray(arrival_band.center_idx, dtype=np.float32),
        trace_valid=stage_trace_valid & (~zero_mass_mask),
        sample_interval_sec=float(candidates.sample_interval_sec),
        probability_weight=float(cfg.confidence.probability_weight),
        band_weight=float(cfg.confidence.band_weight),
        trend_weight=float(cfg.confidence.trend_weight),
        consistency_weight=float(cfg.confidence.consistency_weight),
        entropy_floor=float(cfg.confidence.entropy_floor),
        entropy_power=float(cfg.confidence.entropy_power),
        trend_sigma_idx=float(cfg.confidence.trend_sigma_idx),
    )

    final_confidence = np.asarray(global_confidence.confidence, dtype=np.float32)
    reject_flag = invalid_trace_mask | repick_reject_mask | zero_mass_mask
    reject_flag |= final_confidence < float(cfg.reject_policy.min_confidence)

    final_pick = repicked_pick_idx.copy()
    final_pick[reject_flag] = -1
    final_confidence = np.where(reject_flag, np.float32(0.0), final_confidence).astype(
        np.float32,
        copy=False,
    )

    qc_status = np.full((int(candidates.n_traces),), QC_STATUS_KEEP, dtype=np.int8)
    adjust_mask = (~reject_flag) & (final_pick != np.asarray(candidates.base_pick_idx, dtype=np.int64))
    qc_status[adjust_mask] = np.int8(QC_STATUS_ADJUST)
    qc_status[reject_flag] = np.int8(QC_STATUS_REJECT)

    export_result = export_global_qc_result(
        cfg,
        pick_global=final_pick.astype(np.int32, copy=False),
        confidence_global=final_confidence,
        reject_flag=reject_flag,
        qc_status=qc_status,
        raw_trace_idx=np.asarray(candidates.raw_trace_idx, dtype=np.int64),
        source_refs=_source_refs(
            cfg,
            backend_name=backend_impl.name,
            config_path=config_path,
            geometry_source='inline'
            if cfg.geometry.inline is not None
            else str(Path(str(cfg.geometry.path)).expanduser().resolve()),
        ),
    )
    return GlobalQcRunResult(
        artifact_npz_path=export_result.artifact_npz_path,
        artifact_meta_path=export_result.artifact_meta_path,
        csv_path=export_result.csv_path,
        pick_global=final_pick.astype(np.int32, copy=False),
        confidence_global=final_confidence,
        reject_flag=reject_flag,
        qc_status=qc_status,
        raw_trace_idx=np.asarray(candidates.raw_trace_idx, dtype=np.int64),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args(argv)

    cfg_dict, base_dir = load_cfg_with_base_dir(Path(args.config))
    typed_cfg = load_global_qc_config(cfg_dict, base_dir=base_dir)
    result = run_global_qc(
        typed_cfg,
        config_path=args.config,
    )
    print(str(result.artifact_npz_path))


if __name__ == '__main__':
    main()
