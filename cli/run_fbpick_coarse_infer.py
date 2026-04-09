"""Thin entrypoint for fbpick coarse inference."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

__all__ = ['main', 'run_pipeline']


def _load_runtime() -> SimpleNamespace:
    from seisai_utils.listfiles import expand_cfg_listfiles

    from seisai_engine.infer.segy2segy_cli_common import select_state_dict
    from seisai_engine.pipelines.common import (
        load_cfg_with_base_dir,
        load_checkpoint,
        resolve_cfg_paths,
        resolve_device,
    )
    from seisai_engine.pipelines.fbpick.coarse.build_model import build_model
    from seisai_engine.pipelines.fbpick.coarse.config import load_coarse_infer_config
    from seisai_engine.pipelines.fbpick.coarse.infer import run_coarse_infer

    return SimpleNamespace(
        expand_cfg_listfiles=expand_cfg_listfiles,
        select_state_dict=select_state_dict,
        load_cfg_with_base_dir=load_cfg_with_base_dir,
        load_checkpoint=load_checkpoint,
        resolve_cfg_paths=resolve_cfg_paths,
        resolve_device=resolve_device,
        build_model=build_model,
        load_coarse_infer_config=load_coarse_infer_config,
        run_coarse_infer=run_coarse_infer,
    )


def _require_infer_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    infer_cfg = cfg.get('infer')
    if not isinstance(infer_cfg, dict):
        msg = 'infer must be dict'
        raise TypeError(msg)
    return infer_cfg


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
        keys=['paths.segy_files', 'paths.out_dir', 'infer.ckpt_path'],
    )
    return cfg


def _resolve_cli_device(cfg: dict[str, Any], *, runtime: SimpleNamespace):
    infer_cfg = _require_infer_cfg(cfg)
    device_raw = infer_cfg.get('device', 'auto')
    if device_raw is not None and not isinstance(device_raw, str):
        msg = 'infer.device must be str or null'
        raise TypeError(msg)
    return runtime.resolve_device(device_raw)


def _resolve_ckpt_path(cfg: dict[str, Any]) -> Path:
    infer_cfg = _require_infer_cfg(cfg)
    ckpt_path = infer_cfg.get('ckpt_path')
    if not isinstance(ckpt_path, str) or not ckpt_path.strip():
        msg = 'infer.ckpt_path must be non-empty str'
        raise ValueError(msg)
    path = Path(ckpt_path)
    if not path.is_file():
        msg = f'checkpoint not found: {path}'
        raise FileNotFoundError(msg)
    return path


def _validate_checkpoint_for_infer(ckpt: dict[str, Any], *, model_sig: dict[str, Any]) -> None:
    pipeline = ckpt.get('pipeline')
    if pipeline != 'fbpick':
        msg = f'coarse infer checkpoint pipeline must be "fbpick", got {pipeline!r}'
        raise ValueError(msg)

    ckpt_model_sig = ckpt.get('model_sig')
    if not isinstance(ckpt_model_sig, dict):
        msg = 'checkpoint model_sig must be dict'
        raise TypeError(msg)
    for key, value in ckpt_model_sig.items():
        if key in model_sig and model_sig[key] != value:
            msg = f'checkpoint model_sig mismatch for {key}: {value!r} != {model_sig[key]!r}'
            raise ValueError(msg)

    output_ids = ckpt.get('output_ids')
    if output_ids is not None:
        if not isinstance(output_ids, (list, tuple)):
            msg = 'checkpoint output_ids must be list[str] or tuple[str, ...]'
            raise TypeError(msg)
        if list(output_ids) != ['P']:
            msg = f'coarse infer checkpoint output_ids must be ["P"], got {output_ids!r}'
            raise ValueError(msg)

    softmax_axis = ckpt.get('softmax_axis')
    if softmax_axis is not None and softmax_axis != 'time':
        msg = f'coarse infer checkpoint softmax_axis must be "time", got {softmax_axis!r}'
        raise ValueError(msg)


def _build_out_path(*, segy_path: str | Path, out_dir: str | Path) -> Path:
    segy = Path(segy_path)
    parent_name = segy.parent.name
    if not parent_name:
        msg = 'coarse infer output prefix parent dir name is empty'
        raise ValueError(msg)
    return Path(out_dir) / (parent_name + '__' + segy.stem + '.coarse.npz')


def _iter_single_segy_cfgs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    paths = cfg.get('paths')
    if not isinstance(paths, dict):
        msg = 'paths must be dict'
        raise TypeError(msg)
    segy_files = paths.get('segy_files')
    if not isinstance(segy_files, list):
        msg = 'paths.segy_files must be list[str]'
        raise TypeError(msg)
    out_dir = paths.get('out_dir')
    if not isinstance(out_dir, str):
        msg = 'paths.out_dir must be str'
        raise TypeError(msg)

    out_paths = [_build_out_path(segy_path=segy_path, out_dir=out_dir) for segy_path in segy_files]
    if len(set(out_paths)) != len(out_paths):
        msg = 'coarse infer output path collision in paths.segy_files'
        raise ValueError(msg)

    return [
        {
            **deepcopy(cfg),
            'paths': {
                **deepcopy(paths),
                'segy_files': [segy_path],
            },
        }
        for segy_path in segy_files
    ]


def run_pipeline(config_path: str | Path) -> Path:
    runtime = _load_runtime()
    cfg, base_dir = runtime.load_cfg_with_base_dir(Path(config_path))
    prepared = _prepare_cfg(cfg, base_dir=base_dir, runtime=runtime)
    typed = runtime.load_coarse_infer_config(prepared)
    ckpt_path = _resolve_ckpt_path(prepared)
    device = _resolve_cli_device(prepared, runtime=runtime)

    ckpt = runtime.load_checkpoint(ckpt_path)
    _validate_checkpoint_for_infer(ckpt, model_sig=typed.model_sig)

    model = runtime.build_model(dict(typed.model_sig))
    state_dict, _ = runtime.select_state_dict(ckpt)
    model.load_state_dict(state_dict)
    model.to(device)

    out_path: Path | None = None
    for single_cfg in _iter_single_segy_cfgs(prepared):
        paths = single_cfg['paths']
        segy_path = paths['segy_files'][0]
        target_out_path = _build_out_path(segy_path=segy_path, out_dir=paths['out_dir'])
        out_path = runtime.run_coarse_infer(
            model=model,
            cfg=single_cfg,
            device=device,
        )
        if out_path != target_out_path:
            out_path = out_path.replace(target_out_path)
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
