"""Thin entrypoint for batched fbpick physics-lite robustification."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

__all__ = ['main', 'run_pipeline']


def _load_runtime() -> SimpleNamespace:
    from seisai_utils.listfiles import expand_cfg_listfiles

    from seisai_engine.pipelines.common import load_cfg_with_base_dir, resolve_cfg_paths
    from seisai_engine.pipelines.fbpick.physics.run import run_physics_lite

    return SimpleNamespace(
        expand_cfg_listfiles=expand_cfg_listfiles,
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


def run_pipeline(config_path: str | Path) -> Path:
    runtime = _load_runtime()
    cfg, base_dir = runtime.load_cfg_with_base_dir(Path(config_path))
    prepared = _prepare_cfg(cfg, base_dir=base_dir, runtime=runtime)

    out_path: Path | None = None
    for single_cfg in _iter_single_physics_cfgs(prepared):
        paths = _require_paths(single_cfg)
        coarse_npz_path = paths['coarse_npz_path']
        out_path = runtime.run_physics_lite(
            coarse_npz_path,
            cfg=single_cfg,
            out_path=paths['out_path'],
        )
        print(str(out_path))

    if out_path is None:
        msg = 'paths.segy_files must contain at least one entry'
        raise ValueError(msg)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    run_pipeline(args.config)


if __name__ == '__main__':
    main()
