"""Thin entrypoint for fbpick physics-lite robustification."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

__all__ = ['main', 'run_pipeline']


def _load_runtime() -> SimpleNamespace:
    from seisai_engine.pipelines.common import load_cfg_with_base_dir, resolve_cfg_paths
    from seisai_engine.pipelines.fbpick.physics.config import load_physics_lite_config
    from seisai_engine.pipelines.fbpick.physics.progress import build_progress_reporter
    from seisai_engine.pipelines.fbpick.physics.run import run_physics_lite

    return SimpleNamespace(
        build_progress_reporter=build_progress_reporter,
        load_physics_lite_config=load_physics_lite_config,
        load_cfg_with_base_dir=load_cfg_with_base_dir,
        resolve_cfg_paths=resolve_cfg_paths,
        run_physics_lite=run_physics_lite,
    )


def _require_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg.get('paths')
    if not isinstance(paths, dict):
        msg = 'paths must be dict'
        raise TypeError(msg)
    return paths


def _apply_progress_overrides(
    cfg: dict[str, Any],
    *,
    progress: bool | None,
    progress_level: str | None,
    progress_interval_sec: float | None,
) -> dict[str, Any]:
    out = deepcopy(cfg)
    runtime_cfg = out.setdefault('physical_runtime', {})
    if not isinstance(runtime_cfg, dict):
        msg = 'physical_runtime must be dict'
        raise TypeError(msg)
    progress_cfg = runtime_cfg.setdefault('progress', {})
    if not isinstance(progress_cfg, dict):
        msg = 'physical_runtime.progress must be dict'
        raise TypeError(msg)
    if progress is not None:
        progress_cfg['enabled'] = bool(progress)
    if progress_level is not None:
        progress_cfg['level'] = str(progress_level)
    if progress_interval_sec is not None:
        progress_cfg['interval_sec'] = float(progress_interval_sec)
    return out


def run_pipeline(
    config_path: str | Path,
    *,
    progress: bool | None = None,
    progress_level: str | None = None,
    progress_interval_sec: float | None = None,
) -> Path:
    runtime = _load_runtime()
    cfg, base_dir = runtime.load_cfg_with_base_dir(Path(config_path))
    cfg = _apply_progress_overrides(
        cfg,
        progress=progress,
        progress_level=progress_level,
        progress_interval_sec=progress_interval_sec,
    )
    paths = _require_paths(cfg)
    if 'coarse_npz_path' not in paths:
        msg = 'config missing key: paths.coarse_npz_path'
        raise KeyError(msg)
    path_keys = ['paths.coarse_npz_path']
    if paths.get('out_path') is not None:
        path_keys.append('paths.out_path')
    runtime.resolve_cfg_paths(cfg, base_dir, keys=path_keys)

    coarse_npz_path = paths['coarse_npz_path']
    if not isinstance(coarse_npz_path, str):
        msg = 'config.paths.coarse_npz_path must be str'
        raise TypeError(msg)
    out_path = paths.get('out_path')
    if out_path is not None and not isinstance(out_path, str):
        msg = 'config.paths.out_path must be str or null'
        raise TypeError(msg)

    typed_cfg = runtime.load_physics_lite_config(cfg)
    reporter = runtime.build_progress_reporter(typed_cfg.physical_runtime.progress)
    run_kwargs: dict[str, object] = {}
    if bool(typed_cfg.physical_runtime.progress.enabled):
        run_kwargs['progress'] = reporter
        run_kwargs['progress_context'] = {
            'batch_index': 1,
            'batch_total': 1,
            'config': Path(config_path),
            'started_at': datetime.now(timezone.utc).isoformat(),
        }

    result = runtime.run_physics_lite(
        coarse_npz_path,
        cfg=cfg,
        out_path=out_path,
        **run_kwargs,
    )
    print(str(result))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument('--progress', action='store_true', dest='progress')
    progress_group.add_argument('--no-progress', action='store_false', dest='progress')
    parser.set_defaults(progress=None)
    parser.add_argument('--progress-level', choices=['none', 'batch', 'sgy', 'stage', 'fit'])
    parser.add_argument('--progress-interval-sec', type=float)
    args = parser.parse_args(argv)
    run_pipeline(
        args.config,
        progress=args.progress,
        progress_level=args.progress_level,
        progress_interval_sec=args.progress_interval_sec,
    )


if __name__ == '__main__':
    main()
