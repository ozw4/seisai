"""Thin entrypoint for batched fbpick physics-lite robustification."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

__all__ = ['main', 'run_pipeline']


def _load_runtime() -> SimpleNamespace:
    from seisai_utils.listfiles import expand_cfg_listfiles

    from seisai_engine.pipelines.common import load_cfg_with_base_dir, resolve_cfg_paths
    from seisai_engine.pipelines.fbpick.physics.config import load_physics_lite_config
    from seisai_engine.pipelines.fbpick.physics.progress import build_progress_reporter
    from seisai_engine.pipelines.fbpick.physics.run import run_physics_lite

    return SimpleNamespace(
        build_progress_reporter=build_progress_reporter,
        expand_cfg_listfiles=expand_cfg_listfiles,
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


def _prepare_cfg(
    cfg: dict[str, Any],
    *,
    base_dir: Path,
    runtime: SimpleNamespace,
) -> dict[str, Any]:
    runtime.expand_cfg_listfiles(cfg, keys=['paths.segy_files'])
    runtime.resolve_cfg_paths(
        cfg,
        base_dir,
        keys=['paths.segy_files', 'paths.coarse_npz_dir', 'paths.out_dir'],
    )
    return cfg


def _build_prefixed_name(*, segy_path: str | Path, suffix: str) -> str:
    segy = Path(segy_path)
    parent_name = segy.parent.name
    if not parent_name:
        msg = 'fbpick physics batch output prefix parent dir name is empty'
        raise ValueError(msg)
    return parent_name + '__' + segy.stem + suffix


def _build_coarse_npz_path(*, segy_path: str | Path, coarse_npz_dir: str | Path) -> Path:
    return Path(coarse_npz_dir) / _build_prefixed_name(
        segy_path=segy_path,
        suffix='.coarse.npz',
    )


def _build_out_path(*, segy_path: str | Path, out_dir: str | Path) -> Path:
    return Path(out_dir) / _build_prefixed_name(
        segy_path=segy_path,
        suffix='.robust.npz',
    )


def _iter_single_physics_cfgs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    paths = _require_paths(cfg)
    segy_files = paths.get('segy_files')
    if not isinstance(segy_files, list):
        msg = 'paths.segy_files must be list[str]'
        raise TypeError(msg)
    coarse_npz_dir = paths.get('coarse_npz_dir')
    if not isinstance(coarse_npz_dir, str):
        msg = 'paths.coarse_npz_dir must be str'
        raise TypeError(msg)
    out_dir = paths.get('out_dir')
    if not isinstance(out_dir, str):
        msg = 'paths.out_dir must be str'
        raise TypeError(msg)

    out_paths = [_build_out_path(segy_path=segy_path, out_dir=out_dir) for segy_path in segy_files]
    if len(set(out_paths)) != len(out_paths):
        msg = 'physics batch output path collision in paths.segy_files'
        raise ValueError(msg)

    single_cfgs: list[dict[str, Any]] = []
    for segy_path in segy_files:
        coarse_npz_path = _build_coarse_npz_path(
            segy_path=segy_path,
            coarse_npz_dir=coarse_npz_dir,
        )
        if not coarse_npz_path.is_file():
            raise FileNotFoundError(coarse_npz_path)
        single_cfgs.append(
            {
                **deepcopy(cfg),
                'paths': {
                    **deepcopy(paths),
                    'segy_files': [segy_path],
                    'coarse_npz_path': str(coarse_npz_path),
                    'out_path': str(_build_out_path(segy_path=segy_path, out_dir=out_dir)),
                },
            }
        )
    return single_cfgs


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
    prepared = _prepare_cfg(cfg, base_dir=base_dir, runtime=runtime)
    typed_cfg = runtime.load_physics_lite_config(prepared)
    reporter = runtime.build_progress_reporter(typed_cfg.physical_runtime.progress)
    progress_enabled = bool(typed_cfg.physical_runtime.progress.enabled)
    single_cfgs = _iter_single_physics_cfgs(prepared)
    paths = _require_paths(prepared)
    started_at = datetime.now(timezone.utc).isoformat()
    batch_start = perf_counter()
    if progress_enabled:
        reporter.emit(
            'physics-batch.start',
            config=Path(config_path),
            files=len(single_cfgs),
            coarse_npz_dir=paths.get('coarse_npz_dir'),
            out_dir=paths.get('out_dir'),
            started_at=started_at,
        )

    out_path: Path | None = None
    for index, single_cfg in enumerate(single_cfgs, start=1):
        paths = _require_paths(single_cfg)
        coarse_npz_path = paths['coarse_npz_path']
        run_kwargs: dict[str, object] = {}
        if progress_enabled:
            run_kwargs['progress'] = reporter
            run_kwargs['progress_context'] = {
                'batch_index': index,
                'batch_total': len(single_cfgs),
                'config': Path(config_path),
                'segy': paths['segy_files'][0],
            }
        out_path = runtime.run_physics_lite(
            coarse_npz_path,
            cfg=single_cfg,
            out_path=paths['out_path'],
            **run_kwargs,
        )
        print(str(out_path))

    if out_path is None:
        msg = 'paths.segy_files must contain at least one entry'
        raise ValueError(msg)
    if progress_enabled:
        reporter.emit(
            'physics-batch.done',
            files=len(single_cfgs),
            elapsed=perf_counter() - batch_start,
        )
    return out_path


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
