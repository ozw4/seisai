from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

from .sanitize import sanitize_key
from .tracker import BaseTracker

__all__ = ['MLflowTracker']


def _require_mlflow() -> Any:
    spec = importlib.util.find_spec('mlflow')
    if spec is None:
        msg = 'mlflow is required for MLflowTracker but is not installed'
        raise ImportError(msg)
    return importlib.import_module('mlflow')


def _sanitize_mapping(mapping: dict[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        msg = f'{label} must be dict'
        raise TypeError(msg)

    sanitized: dict[str, Any] = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key:
            msg = f'{label} key must be non-empty str'
            raise TypeError(msg)
        safe_key = sanitize_key(key)
        if safe_key in sanitized:
            msg = f'duplicate {label} key after sanitization: {safe_key}'
            raise ValueError(msg)
        sanitized[safe_key] = value
    return sanitized


def _coerce_params(params: dict[str, Any]) -> dict[str, str]:
    return {k: str(v) for k, v in params.items()}


def _coerce_tags(tags: dict[str, Any]) -> dict[str, str]:
    return {k: str(v) for k, v in tags.items()}


def _coerce_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    coerced: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            msg = f'metric {key} must be numeric'
            raise TypeError(msg)
        coerced[key] = float(value)
    return coerced


def _validate_artifacts(artifacts: dict[str, Any]) -> dict[str, Path]:
    if not isinstance(artifacts, dict):
        msg = 'artifacts must be dict'
        raise TypeError(msg)
    out: dict[str, Path] = {}
    for key, value in artifacts.items():
        if not isinstance(key, str) or not key:
            msg = 'artifact key must be non-empty str'
            raise TypeError(msg)
        if isinstance(value, Path):
            path = value
        elif isinstance(value, str):
            path = Path(value)
        else:
            msg = f'artifact {key} must be Path or str'
            raise TypeError(msg)
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            msg = f'expected file: {path}'
            raise ValueError(msg)
        out[key] = path
    return out


def _list_vis_files(vis_epoch_dir: Path, *, vis_max_files: int) -> list[Path]:
    if not vis_epoch_dir.exists():
        raise FileNotFoundError(vis_epoch_dir)
    if not vis_epoch_dir.is_dir():
        msg = f'expected directory: {vis_epoch_dir}'
        raise ValueError(msg)

    files = [p for p in vis_epoch_dir.rglob('*') if p.is_file()]
    files.sort(key=lambda p: str(p))
    return files[:vis_max_files]


class MLflowTracker(BaseTracker):
    def __init__(self) -> None:
        self._mlflow = _require_mlflow()

    def start_run(
        self,
        *,
        tracking_uri: str,
        experiment: str,
        run_name: str,
        tags: dict,
        params: dict,
        artifacts: dict[str, Path],
    ) -> None:
        if not isinstance(tracking_uri, str) or not tracking_uri:
            msg = 'tracking_uri must be non-empty str'
            raise TypeError(msg)
        if not isinstance(experiment, str) or not experiment:
            msg = 'experiment must be non-empty str'
            raise TypeError(msg)
        if not isinstance(run_name, str) or not run_name:
            msg = 'run_name must be non-empty str'
            raise TypeError(msg)

        safe_tags = _sanitize_mapping(tags, label='tags')
        safe_params = _sanitize_mapping(params, label='params')
        safe_artifacts = _validate_artifacts(artifacts)

        self._mlflow.set_tracking_uri(tracking_uri)
        self._mlflow.set_experiment(experiment)
        self._mlflow.start_run(run_name=run_name)

        if safe_tags:
            self._mlflow.set_tags(_coerce_tags(safe_tags))
        if safe_params:
            self._mlflow.log_params(_coerce_params(safe_params))

        for key, path in safe_artifacts.items():
            self._mlflow.log_artifact(str(path), artifact_path=str(key))

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        if not isinstance(step, int) or isinstance(step, bool):
            msg = 'step must be int'
            raise TypeError(msg)
        safe_metrics = _sanitize_mapping(metrics, label='metrics')
        if not safe_metrics:
            return None
        coerced = _coerce_metrics(safe_metrics)
        for key, value in coerced.items():
            self._mlflow.log_metric(key, value, step=step)
        return None

    def log_best(
        self,
        *,
        ckpt_path: Path,
        vis_epoch_dir: Path | None,
        vis_max_files: int,
    ) -> None:
        if not isinstance(ckpt_path, Path):
            msg = 'ckpt_path must be Path'
            raise TypeError(msg)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        if not ckpt_path.is_file():
            msg = f'expected file: {ckpt_path}'
            raise ValueError(msg)

        if not isinstance(vis_max_files, int) or isinstance(vis_max_files, bool):
            msg = 'vis_max_files must be int'
            raise TypeError(msg)

        self._mlflow.log_artifact(str(ckpt_path), artifact_path='ckpt')

        if vis_epoch_dir is None:
            return None
        if vis_max_files <= 0:
            return None

        if not isinstance(vis_epoch_dir, Path):
            msg = 'vis_epoch_dir must be Path'
            raise TypeError(msg)

        files = _list_vis_files(vis_epoch_dir, vis_max_files=vis_max_files)
        if not files:
            return None

        for path in files:
            rel = path.relative_to(vis_epoch_dir)
            artifact_path = Path('vis') / vis_epoch_dir.name / rel.parent
            self._mlflow.log_artifact(str(path), artifact_path=str(artifact_path))
        return None

    def log_artifacts(self, artifacts: dict[str, Path]) -> None:
        safe_artifacts = _validate_artifacts(artifacts)
        if not safe_artifacts:
            return None
        for key, path in safe_artifacts.items():
            self._mlflow.log_artifact(str(path), artifact_path=str(key))
        return None

    def end_run(self, *, status: str) -> None:
        if not isinstance(status, str) or not status:
            msg = 'status must be non-empty str'
            raise TypeError(msg)
        self._mlflow.end_run(status=status)
