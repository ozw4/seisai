from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    'TrackingConfig',
    'load_tracking_config',
    'resolve_tracking_uri',
]


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = False
    experiment_prefix: str = 'seisai'
    exp_name: str = 'baseline'
    tracking_uri: str = 'file:./mlruns'
    vis_best_only: bool = True
    vis_max_files: int = 50

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            msg = 'tracking.enabled must be bool'
            raise TypeError(msg)
        if not isinstance(self.experiment_prefix, str) or not self.experiment_prefix:
            msg = 'tracking.experiment_prefix must be non-empty str'
            raise TypeError(msg)
        if not isinstance(self.exp_name, str) or not self.exp_name:
            msg = 'tracking.exp_name must be non-empty str'
            raise TypeError(msg)
        if not isinstance(self.tracking_uri, str) or not self.tracking_uri:
            msg = 'tracking.tracking_uri must be non-empty str'
            raise TypeError(msg)
        if not isinstance(self.vis_best_only, bool):
            msg = 'tracking.vis_best_only must be bool'
            raise TypeError(msg)
        if not isinstance(self.vis_max_files, int):
            msg = 'tracking.vis_max_files must be int'
            raise TypeError(msg)
        if self.vis_max_files < 0:
            msg = 'tracking.vis_max_files must be >= 0'
            raise ValueError(msg)


def _optional_bool(d: dict, key: str, default: bool) -> bool:
    if key not in d:
        return bool(default)
    value = d[key]
    if not isinstance(value, bool):
        msg = f'tracking.{key} must be bool'
        raise TypeError(msg)
    return bool(value)


def _optional_int(d: dict, key: str, default: int) -> int:
    if key not in d:
        return int(default)
    value = d[key]
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f'tracking.{key} must be int'
        raise TypeError(msg)
    return int(value)


def _optional_str(d: dict, key: str, default: str) -> str:
    if key not in d:
        return str(default)
    value = d[key]
    if not isinstance(value, str):
        msg = f'tracking.{key} must be str'
        raise TypeError(msg)
    return str(value)


def resolve_tracking_uri(tracking_uri: str, base_dir: str | Path) -> str:
    if not isinstance(tracking_uri, str) or not tracking_uri:
        msg = 'tracking_uri must be non-empty str'
        raise TypeError(msg)

    base = Path(base_dir).expanduser().resolve()

    if not tracking_uri.startswith('file:'):
        return tracking_uri

    raw = tracking_uri[len('file:') :]
    if not raw:
        msg = 'tracking_uri file path must be non-empty'
        raise ValueError(msg)

    if raw.startswith('//'):
        raw = '/' + raw.lstrip('/')

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    return f'file:{path}'


def load_tracking_config(cfg: dict, base_dir: str | Path) -> TrackingConfig:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    tracking_raw = cfg.get('tracking')
    if tracking_raw is None:
        return TrackingConfig()
    if not isinstance(tracking_raw, dict):
        msg = 'cfg.tracking must be dict'
        raise TypeError(msg)

    enabled = _optional_bool(tracking_raw, 'enabled', False)
    experiment_prefix = _optional_str(tracking_raw, 'experiment_prefix', 'seisai')
    exp_name = _optional_str(tracking_raw, 'exp_name', 'baseline')
    tracking_uri = _optional_str(tracking_raw, 'tracking_uri', 'file:./mlruns')
    vis_best_only = _optional_bool(tracking_raw, 'vis_best_only', True)
    vis_max_files = _optional_int(tracking_raw, 'vis_max_files', 50)

    tracking_uri = resolve_tracking_uri(tracking_uri, base_dir)

    return TrackingConfig(
        enabled=enabled,
        experiment_prefix=experiment_prefix,
        exp_name=exp_name,
        tracking_uri=tracking_uri,
        vis_best_only=vis_best_only,
        vis_max_files=vis_max_files,
    )
