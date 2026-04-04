"""Thin entrypoint for fbpick physics-lite robustification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from seisai_engine.pipelines.common import load_cfg_with_base_dir, resolve_cfg_paths
from seisai_engine.pipelines.fbpick.physics import run_physics_lite

__all__ = ['main']


def _require_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg.get('paths')
    if not isinstance(paths, dict):
        msg = 'paths must be dict'
        raise TypeError(msg)
    return paths


def run_pipeline(config_path: str | Path) -> Path:
    cfg, base_dir = load_cfg_with_base_dir(Path(config_path))
    paths = _require_paths(cfg)
    if 'coarse_npz_path' not in paths:
        msg = 'config missing key: paths.coarse_npz_path'
        raise KeyError(msg)
    path_keys = ['paths.coarse_npz_path']
    if paths.get('out_path') is not None:
        path_keys.append('paths.out_path')
    resolve_cfg_paths(cfg, base_dir, keys=path_keys)

    coarse_npz_path = paths['coarse_npz_path']
    if not isinstance(coarse_npz_path, str):
        msg = 'config.paths.coarse_npz_path must be str'
        raise TypeError(msg)
    out_path = paths.get('out_path')
    if out_path is not None and not isinstance(out_path, str):
        msg = 'config.paths.out_path must be str or null'
        raise TypeError(msg)

    result = run_physics_lite(
        coarse_npz_path,
        cfg=cfg,
        out_path=out_path,
    )
    print(str(result))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    run_pipeline(args.config)


if __name__ == '__main__':
    main()
