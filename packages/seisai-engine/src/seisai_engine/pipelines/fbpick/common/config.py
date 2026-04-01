from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from seisai_utils.config import optional_float, require_dict, require_value

from seisai_engine.pipelines.common.config_io import resolve_relpath

__all__ = [
    'FbpickCommonConfig',
    'FbpickPathsCfg',
    'FbpickThresholdsCfg',
    'load_fbpick_common_config',
]


@dataclass(frozen=True)
class FbpickPathsCfg:
    out_dir: str
    survey_id: str


@dataclass(frozen=True)
class FbpickThresholdsCfg:
    confidence_min: float
    trace_valid_min_fraction: float
    qc_reject_confidence_below: float


@dataclass(frozen=True)
class FbpickCommonConfig:
    paths: FbpickPathsCfg
    thresholds: FbpickThresholdsCfg


def _normalize_survey_id(raw: str) -> str:
    if not isinstance(raw, str):
        msg = 'config.paths.survey_id must be str'
        raise TypeError(msg)
    survey_id = raw.strip()
    if survey_id == '':
        msg = 'config.paths.survey_id must not be empty'
        raise ValueError(msg)
    if '/' in survey_id or '\\' in survey_id:
        msg = 'config.paths.survey_id must be a single path segment'
        raise ValueError(msg)
    if survey_id in {'.', '..'}:
        msg = 'config.paths.survey_id must not be "." or ".."'
        raise ValueError(msg)
    return survey_id


def _require_unit_interval(value: float, *, key_name: str) -> float:
    value_f = float(value)
    if value_f < 0.0 or value_f > 1.0:
        msg = f'{key_name} must be in [0, 1]'
        raise ValueError(msg)
    return value_f


def load_fbpick_common_config(
    cfg: dict,
    *,
    base_dir: str | Path | None = None,
) -> FbpickCommonConfig:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    paths_cfg = require_dict(cfg, 'paths')
    thresholds_cfg = cfg.get('thresholds', {})
    if not isinstance(thresholds_cfg, dict):
        msg = 'config.thresholds must be dict'
        raise TypeError(msg)

    out_dir_raw = require_value(
        paths_cfg,
        'out_dir',
        str,
        type_message='config.paths.out_dir must be str',
    )
    survey_id_raw = require_value(
        paths_cfg,
        'survey_id',
        str,
        type_message='config.paths.survey_id must be str',
    )

    if base_dir is None:
        out_dir = str(Path(out_dir_raw).expanduser())
    else:
        out_dir = resolve_relpath(base_dir, out_dir_raw)

    confidence_min = _require_unit_interval(
        optional_float(thresholds_cfg, 'confidence_min', 0.0),
        key_name='config.thresholds.confidence_min',
    )
    trace_valid_min_fraction = _require_unit_interval(
        optional_float(thresholds_cfg, 'trace_valid_min_fraction', 0.0),
        key_name='config.thresholds.trace_valid_min_fraction',
    )
    qc_reject_confidence_below = _require_unit_interval(
        optional_float(thresholds_cfg, 'qc_reject_confidence_below', 0.0),
        key_name='config.thresholds.qc_reject_confidence_below',
    )

    return FbpickCommonConfig(
        paths=FbpickPathsCfg(
            out_dir=str(out_dir),
            survey_id=_normalize_survey_id(survey_id_raw),
        ),
        thresholds=FbpickThresholdsCfg(
            confidence_min=float(confidence_min),
            trace_valid_min_fraction=float(trace_valid_min_fraction),
            qc_reject_confidence_below=float(qc_reject_confidence_below),
        ),
    )
