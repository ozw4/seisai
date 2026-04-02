from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seisai_pick.global_qc import GeometryNormalization
from seisai_utils.config import (
    optional_bool,
    optional_float,
    optional_int,
    optional_str,
    require_dict,
    require_value,
)

from seisai_engine.pipelines.common.config_io import resolve_relpath
from seisai_engine.pipelines.fbpick.common.config import (
    FbpickCommonConfig,
    FbpickPathsCfg,
    load_fbpick_common_config,
)
from seisai_engine.pipelines.fbpick.common.io import resolve_artifact_paths

__all__ = [
    'GlobalQcArtifactCfg',
    'GlobalQcArrivalBandCfg',
    'GlobalQcBackendCfg',
    'GlobalQcConfig',
    'GlobalQcConfidenceCfg',
    'GlobalQcConsistencyCfg',
    'GlobalQcExportCfg',
    'GlobalQcGeometryCfg',
    'GlobalQcRejectPolicyCfg',
    'load_global_qc_config',
    'resolve_default_stage_artifact_paths',
]


def _require_non_empty_str(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        msg = f'{label} must be str'
        raise TypeError(msg)
    out = value.strip()
    if out == '':
        msg = f'{label} must not be empty'
        raise ValueError(msg)
    return out


def _resolve_optional_path(
    *,
    base_dir: str | Path | None,
    value: str | None,
    label: str,
) -> str | None:
    if value is None:
        return None
    path_str = _require_non_empty_str(value, label=label)
    if base_dir is None:
        return str(Path(path_str).expanduser())
    return resolve_relpath(base_dir, path_str)


def _require_non_negative_float(value: float, *, label: str) -> float:
    out = float(value)
    if out < 0.0:
        msg = f'{label} must be >= 0'
        raise ValueError(msg)
    return out


def _require_unit_interval(value: float, *, label: str) -> float:
    out = float(value)
    if out < 0.0 or out > 1.0:
        msg = f'{label} must be in [0, 1]'
        raise ValueError(msg)
    return out


def resolve_default_stage_artifact_paths(
    paths_cfg: FbpickPathsCfg,
    *,
    stage: str,
) -> tuple[str, str]:
    if not isinstance(paths_cfg, FbpickPathsCfg):
        msg = 'paths_cfg must be FbpickPathsCfg'
        raise TypeError(msg)
    artifact_paths = resolve_artifact_paths(paths_cfg, stage=stage)
    return str(artifact_paths.npz_path), str(artifact_paths.meta_path)


@dataclass(frozen=True)
class GlobalQcArtifactCfg:
    artifact_npz_path: str
    artifact_meta_path: str

    def __post_init__(self) -> None:
        _require_non_empty_str(
            self.artifact_npz_path,
            label='artifact_npz_path',
        )
        _require_non_empty_str(
            self.artifact_meta_path,
            label='artifact_meta_path',
        )


@dataclass(frozen=True)
class GlobalQcGeometryCfg:
    format: str
    path: str | None
    inline: dict[str, Any] | None
    normalization: GeometryNormalization | None

    def __post_init__(self) -> None:
        format_value = _require_non_empty_str(
            self.format,
            label='config.global_qc.geometry.format',
        ).lower()
        if format_value not in {'npz', 'json', 'inline'}:
            msg = 'config.global_qc.geometry.format must be one of: npz, json, inline'
            raise ValueError(msg)
        object.__setattr__(self, 'format', format_value)

        has_path = self.path is not None
        has_inline = self.inline is not None
        if has_path == has_inline:
            msg = 'config.global_qc.geometry must specify exactly one of path or inline'
            raise ValueError(msg)
        if format_value == 'inline' and not has_inline:
            msg = 'config.global_qc.geometry.format="inline" requires geometry.inline'
            raise ValueError(msg)
        if format_value != 'inline' and not has_path:
            msg = 'config.global_qc.geometry.path is required for path-based geometry'
            raise ValueError(msg)
        if self.path is not None:
            _require_non_empty_str(self.path, label='config.global_qc.geometry.path')
        if self.inline is not None and not isinstance(self.inline, dict):
            msg = 'config.global_qc.geometry.inline must be dict'
            raise TypeError(msg)


@dataclass(frozen=True)
class GlobalQcBackendCfg:
    name: str
    source_path: str | None

    def __post_init__(self) -> None:
        _require_non_empty_str(self.name, label='config.global_qc.backend.name')
        if self.source_path is not None:
            _require_non_empty_str(
                self.source_path,
                label='config.global_qc.backend.source_path',
            )


@dataclass(frozen=True)
class GlobalQcArrivalBandCfg:
    use_hard_mask: bool
    band_half_width_idx: int | None
    uncertainty_scale: float
    band_radius_sigma: float
    min_half_width_idx: int
    prior_floor: float
    prior_power: float

    def __post_init__(self) -> None:
        if self.band_half_width_idx is not None and int(self.band_half_width_idx) < 0:
            msg = 'config.global_qc.arrival_band.band_half_width_idx must be >= 0'
            raise ValueError(msg)
        if float(self.uncertainty_scale) <= 0.0:
            msg = 'config.global_qc.arrival_band.uncertainty_scale must be > 0'
            raise ValueError(msg)
        if float(self.band_radius_sigma) <= 0.0:
            msg = 'config.global_qc.arrival_band.band_radius_sigma must be > 0'
            raise ValueError(msg)
        if int(self.min_half_width_idx) < 0:
            msg = 'config.global_qc.arrival_band.min_half_width_idx must be >= 0'
            raise ValueError(msg)
        if float(self.prior_floor) < 0.0:
            msg = 'config.global_qc.arrival_band.prior_floor must be >= 0'
            raise ValueError(msg)
        if float(self.prior_power) <= 0.0:
            msg = 'config.global_qc.arrival_band.prior_power must be > 0'
            raise ValueError(msg)


@dataclass(frozen=True)
class GlobalQcConsistencyCfg:
    adjacent_radius: int
    adjacent_sigma_idx: float
    adjacent_min_count: int
    trend_sigma_idx: float
    outlier_radius: int
    outlier_z_scale: float
    outlier_min_count: int
    outlier_mad_floor_idx: float
    adjacent_weight: float
    trend_weight: float
    outlier_weight: float

    def __post_init__(self) -> None:
        if int(self.adjacent_radius) < 1:
            msg = 'config.global_qc.consistency.adjacent_radius must be >= 1'
            raise ValueError(msg)
        if int(self.adjacent_min_count) < 1:
            msg = 'config.global_qc.consistency.adjacent_min_count must be >= 1'
            raise ValueError(msg)
        if float(self.adjacent_sigma_idx) <= 0.0:
            msg = 'config.global_qc.consistency.adjacent_sigma_idx must be > 0'
            raise ValueError(msg)
        if float(self.trend_sigma_idx) <= 0.0:
            msg = 'config.global_qc.consistency.trend_sigma_idx must be > 0'
            raise ValueError(msg)
        if int(self.outlier_radius) < 1:
            msg = 'config.global_qc.consistency.outlier_radius must be >= 1'
            raise ValueError(msg)
        if float(self.outlier_z_scale) <= 0.0:
            msg = 'config.global_qc.consistency.outlier_z_scale must be > 0'
            raise ValueError(msg)
        if int(self.outlier_min_count) < 1:
            msg = 'config.global_qc.consistency.outlier_min_count must be >= 1'
            raise ValueError(msg)
        if float(self.outlier_mad_floor_idx) <= 0.0:
            msg = 'config.global_qc.consistency.outlier_mad_floor_idx must be > 0'
            raise ValueError(msg)
        if (
            float(self.adjacent_weight) == 0.0
            and float(self.trend_weight) == 0.0
            and float(self.outlier_weight) == 0.0
        ):
            msg = 'config.global_qc.consistency requires at least one positive weight'
            raise ValueError(msg)
        for key in (
            'adjacent_weight',
            'trend_weight',
            'outlier_weight',
        ):
            _require_non_negative_float(
                getattr(self, key),
                label=f'config.global_qc.consistency.{key}',
            )


@dataclass(frozen=True)
class GlobalQcConfidenceCfg:
    probability_weight: float
    band_weight: float
    trend_weight: float
    consistency_weight: float
    entropy_floor: float
    entropy_power: float
    trend_sigma_idx: float

    def __post_init__(self) -> None:
        for key in (
            'probability_weight',
            'band_weight',
            'trend_weight',
            'consistency_weight',
        ):
            _require_non_negative_float(
                getattr(self, key),
                label=f'config.global_qc.confidence.{key}',
            )
        if (
            float(self.probability_weight) == 0.0
            and float(self.band_weight) == 0.0
            and float(self.trend_weight) == 0.0
            and float(self.consistency_weight) == 0.0
        ):
            msg = 'config.global_qc.confidence requires at least one positive weight'
            raise ValueError(msg)
        _require_unit_interval(
            self.entropy_floor,
            label='config.global_qc.confidence.entropy_floor',
        )
        if float(self.entropy_power) <= 0.0:
            msg = 'config.global_qc.confidence.entropy_power must be > 0'
            raise ValueError(msg)
        if float(self.trend_sigma_idx) <= 0.0:
            msg = 'config.global_qc.confidence.trend_sigma_idx must be > 0'
            raise ValueError(msg)


@dataclass(frozen=True)
class GlobalQcRejectPolicyCfg:
    min_confidence: float
    allow_zero_mass_failure: bool
    invalid_handling: str

    def __post_init__(self) -> None:
        _require_unit_interval(
            self.min_confidence,
            label='config.global_qc.reject_policy.min_confidence',
        )
        invalid_handling = _require_non_empty_str(
            self.invalid_handling,
            label='config.global_qc.reject_policy.invalid_handling',
        )
        if invalid_handling != 'reject_to_minus_one':
            msg = (
                'config.global_qc.reject_policy.invalid_handling must be '
                '"reject_to_minus_one"'
            )
            raise ValueError(msg)
        object.__setattr__(self, 'invalid_handling', invalid_handling)


@dataclass(frozen=True)
class GlobalQcExportCfg:
    write_csv: bool
    csv_path: str | None

    def __post_init__(self) -> None:
        if self.csv_path is not None:
            _require_non_empty_str(
                self.csv_path,
                label='config.global_qc.export.csv_path',
            )


@dataclass(frozen=True)
class GlobalQcConfig:
    fbpick: FbpickCommonConfig
    coarse_artifact: GlobalQcArtifactCfg
    fine_artifact: GlobalQcArtifactCfg
    geometry: GlobalQcGeometryCfg
    backend: GlobalQcBackendCfg
    arrival_band: GlobalQcArrivalBandCfg
    consistency: GlobalQcConsistencyCfg
    confidence: GlobalQcConfidenceCfg
    reject_policy: GlobalQcRejectPolicyCfg
    export: GlobalQcExportCfg


def _load_artifact_cfg(
    *,
    section_cfg: dict[str, Any],
    base_dir: str | Path | None,
    paths_cfg: FbpickPathsCfg,
    stage: str,
    label_prefix: str,
) -> GlobalQcArtifactCfg:
    default_npz_path, default_meta_path = resolve_default_stage_artifact_paths(
        paths_cfg,
        stage=stage,
    )
    npz_path = _resolve_optional_path(
        base_dir=base_dir,
        value=section_cfg.get('artifact_npz_path'),
        label=f'{label_prefix}.artifact_npz_path',
    )
    meta_path = _resolve_optional_path(
        base_dir=base_dir,
        value=section_cfg.get('artifact_meta_path'),
        label=f'{label_prefix}.artifact_meta_path',
    )
    return GlobalQcArtifactCfg(
        artifact_npz_path=default_npz_path if npz_path is None else npz_path,
        artifact_meta_path=default_meta_path if meta_path is None else meta_path,
    )


def _load_geometry_cfg(
    *,
    geometry_cfg: dict[str, Any],
    base_dir: str | Path | None,
) -> GlobalQcGeometryCfg:
    path_value = geometry_cfg.get('path')
    format_value = optional_str(geometry_cfg, 'format', '')
    inline_value = geometry_cfg.get('inline')
    normalization_cfg = geometry_cfg.get('normalization', {})
    if not isinstance(normalization_cfg, dict):
        msg = 'config.global_qc.geometry.normalization must be dict'
        raise TypeError(msg)

    path_resolved = _resolve_optional_path(
        base_dir=base_dir,
        value=path_value,
        label='config.global_qc.geometry.path',
    )

    if inline_value is not None and not isinstance(inline_value, dict):
        msg = 'config.global_qc.geometry.inline must be dict'
        raise TypeError(msg)

    if path_resolved is not None:
        inferred_format = Path(path_resolved).suffix.lower()
        if format_value == '':
            if inferred_format == '.npz':
                format_value = 'npz'
            elif inferred_format == '.json':
                format_value = 'json'
            else:
                msg = (
                    'config.global_qc.geometry.format is required when geometry.path '
                    f'has unsupported suffix: {path_resolved}'
                )
                raise ValueError(msg)
    elif inline_value is not None:
        format_value = 'inline'
    else:
        msg = 'config.global_qc.geometry requires either path or inline'
        raise ValueError(msg)

    normalization = None
    if normalization_cfg:
        normalization = GeometryNormalization(
            origin_x=float(optional_float(normalization_cfg, 'origin_x', 0.0)),
            origin_y=float(optional_float(normalization_cfg, 'origin_y', 0.0)),
            origin_z=float(optional_float(normalization_cfg, 'origin_z', 0.0)),
            xy_scale_to_m=float(optional_float(normalization_cfg, 'xy_scale_to_m', 1.0)),
            z_scale_to_m=float(optional_float(normalization_cfg, 'z_scale_to_m', 1.0)),
            flip_z_sign=bool(optional_bool(normalization_cfg, 'flip_z_sign', default=False)),
        )

    return GlobalQcGeometryCfg(
        format=str(format_value),
        path=path_resolved,
        inline=None if inline_value is None else dict(inline_value),
        normalization=normalization,
    )


def _load_backend_cfg(
    *,
    backend_cfg: dict[str, Any],
    base_dir: str | Path | None,
) -> GlobalQcBackendCfg:
    name = require_value(
        backend_cfg,
        'name',
        str,
        type_message='config.global_qc.backend.name must be str',
    )
    source_path = _resolve_optional_path(
        base_dir=base_dir,
        value=backend_cfg.get('source_path'),
        label='config.global_qc.backend.source_path',
    )
    return GlobalQcBackendCfg(
        name=str(name),
        source_path=source_path,
    )


def _load_arrival_band_cfg(arrival_band_cfg: dict[str, Any]) -> GlobalQcArrivalBandCfg:
    band_half_width_idx = None
    if 'band_half_width_idx' in arrival_band_cfg:
        band_half_width_idx = int(optional_int(arrival_band_cfg, 'band_half_width_idx', default=0))
    return GlobalQcArrivalBandCfg(
        use_hard_mask=bool(optional_bool(arrival_band_cfg, 'use_hard_mask', default=True)),
        band_half_width_idx=band_half_width_idx,
        uncertainty_scale=float(optional_float(arrival_band_cfg, 'uncertainty_scale', 1.0)),
        band_radius_sigma=float(optional_float(arrival_band_cfg, 'band_radius_sigma', 2.0)),
        min_half_width_idx=int(optional_int(arrival_band_cfg, 'min_half_width_idx', default=1)),
        prior_floor=float(optional_float(arrival_band_cfg, 'prior_floor', 0.0)),
        prior_power=float(optional_float(arrival_band_cfg, 'prior_power', 1.0)),
    )


def _load_consistency_cfg(consistency_cfg: dict[str, Any]) -> GlobalQcConsistencyCfg:
    return GlobalQcConsistencyCfg(
        adjacent_radius=int(optional_int(consistency_cfg, 'adjacent_radius', default=1)),
        adjacent_sigma_idx=float(optional_float(consistency_cfg, 'adjacent_sigma_idx', 4.0)),
        adjacent_min_count=int(optional_int(consistency_cfg, 'adjacent_min_count', default=1)),
        trend_sigma_idx=float(optional_float(consistency_cfg, 'trend_sigma_idx', 6.0)),
        outlier_radius=int(optional_int(consistency_cfg, 'outlier_radius', default=4)),
        outlier_z_scale=float(optional_float(consistency_cfg, 'outlier_z_scale', 3.0)),
        outlier_min_count=int(optional_int(consistency_cfg, 'outlier_min_count', default=3)),
        outlier_mad_floor_idx=float(
            optional_float(consistency_cfg, 'outlier_mad_floor_idx', 1.0)
        ),
        adjacent_weight=float(optional_float(consistency_cfg, 'adjacent_weight', 0.4)),
        trend_weight=float(optional_float(consistency_cfg, 'trend_weight', 0.35)),
        outlier_weight=float(optional_float(consistency_cfg, 'outlier_weight', 0.25)),
    )


def _load_confidence_cfg(confidence_cfg: dict[str, Any]) -> GlobalQcConfidenceCfg:
    return GlobalQcConfidenceCfg(
        probability_weight=float(optional_float(confidence_cfg, 'probability_weight', 0.5)),
        band_weight=float(optional_float(confidence_cfg, 'band_weight', 0.2)),
        trend_weight=float(optional_float(confidence_cfg, 'trend_weight', 0.15)),
        consistency_weight=float(optional_float(confidence_cfg, 'consistency_weight', 0.15)),
        entropy_floor=float(optional_float(confidence_cfg, 'entropy_floor', 0.2)),
        entropy_power=float(optional_float(confidence_cfg, 'entropy_power', 0.5)),
        trend_sigma_idx=float(optional_float(confidence_cfg, 'trend_sigma_idx', 6.0)),
    )


def _load_reject_policy_cfg(reject_policy_cfg: dict[str, Any]) -> GlobalQcRejectPolicyCfg:
    return GlobalQcRejectPolicyCfg(
        min_confidence=float(optional_float(reject_policy_cfg, 'min_confidence', 0.0)),
        allow_zero_mass_failure=bool(
            optional_bool(
                reject_policy_cfg,
                'allow_zero_mass_failure',
                default=False,
            )
        ),
        invalid_handling=optional_str(
            reject_policy_cfg,
            'invalid_handling',
            'reject_to_minus_one',
        ),
    )


def _load_export_cfg(
    *,
    export_cfg: dict[str, Any],
    base_dir: str | Path | None,
) -> GlobalQcExportCfg:
    csv_path = _resolve_optional_path(
        base_dir=base_dir,
        value=export_cfg.get('csv_path'),
        label='config.global_qc.export.csv_path',
    )
    return GlobalQcExportCfg(
        write_csv=bool(optional_bool(export_cfg, 'write_csv', default=True)),
        csv_path=csv_path,
    )


def load_global_qc_config(
    cfg: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> GlobalQcConfig:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    fbpick = load_fbpick_common_config(cfg, base_dir=base_dir)
    global_qc_cfg = require_dict(cfg, 'global_qc')

    coarse_artifact_cfg = global_qc_cfg.get('coarse_artifact', {})
    fine_artifact_cfg = global_qc_cfg.get('fine_artifact', {})
    arrival_band_cfg = global_qc_cfg.get('arrival_band', {})
    consistency_cfg = global_qc_cfg.get('consistency', {})
    confidence_cfg = global_qc_cfg.get('confidence', {})
    reject_policy_cfg = global_qc_cfg.get('reject_policy', {})
    export_cfg = global_qc_cfg.get('export', {})

    for value, label in (
        (coarse_artifact_cfg, 'config.global_qc.coarse_artifact'),
        (fine_artifact_cfg, 'config.global_qc.fine_artifact'),
        (arrival_band_cfg, 'config.global_qc.arrival_band'),
        (consistency_cfg, 'config.global_qc.consistency'),
        (confidence_cfg, 'config.global_qc.confidence'),
        (reject_policy_cfg, 'config.global_qc.reject_policy'),
        (export_cfg, 'config.global_qc.export'),
    ):
        if not isinstance(value, dict):
            msg = f'{label} must be dict'
            raise TypeError(msg)

    geometry_cfg = require_dict(global_qc_cfg, 'geometry')
    backend_cfg = require_dict(global_qc_cfg, 'backend')

    return GlobalQcConfig(
        fbpick=fbpick,
        coarse_artifact=_load_artifact_cfg(
            section_cfg=coarse_artifact_cfg,
            base_dir=base_dir,
            paths_cfg=fbpick.paths,
            stage='coarse',
            label_prefix='config.global_qc.coarse_artifact',
        ),
        fine_artifact=_load_artifact_cfg(
            section_cfg=fine_artifact_cfg,
            base_dir=base_dir,
            paths_cfg=fbpick.paths,
            stage='fine',
            label_prefix='config.global_qc.fine_artifact',
        ),
        geometry=_load_geometry_cfg(
            geometry_cfg=geometry_cfg,
            base_dir=base_dir,
        ),
        backend=_load_backend_cfg(
            backend_cfg=backend_cfg,
            base_dir=base_dir,
        ),
        arrival_band=_load_arrival_band_cfg(arrival_band_cfg),
        consistency=_load_consistency_cfg(consistency_cfg),
        confidence=_load_confidence_cfg(confidence_cfg),
        reject_policy=_load_reject_policy_cfg(reject_policy_cfg),
        export=_load_export_cfg(
            export_cfg=export_cfg,
            base_dir=base_dir,
        ),
    )
