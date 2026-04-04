"""Thin entrypoint for fbpick coarse inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from seisai_utils.listfiles import expand_cfg_listfiles

from seisai_engine.infer.segy2segy_cli_common import select_state_dict
from seisai_engine.pipelines.common import (
    load_cfg_with_base_dir,
    load_checkpoint,
    resolve_cfg_paths,
    resolve_device,
)
from seisai_engine.pipelines.fbpick.coarse import (
    build_model,
    load_coarse_infer_config,
    run_coarse_infer,
)

__all__ = ['main']



def _require_infer_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    infer_cfg = cfg.get('infer')
    if not isinstance(infer_cfg, dict):
        msg = 'infer must be dict'
        raise TypeError(msg)
    return infer_cfg



def _prepare_cfg(cfg: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    expand_cfg_listfiles(cfg, keys=['paths.segy_files'])
    resolve_cfg_paths(
        cfg,
        base_dir,
        keys=['paths.segy_files', 'paths.out_dir', 'infer.ckpt_path'],
    )
    return cfg



def _resolve_cli_device(cfg: dict[str, Any]):
    infer_cfg = _require_infer_cfg(cfg)
    device_raw = infer_cfg.get('device', 'auto')
    if device_raw is not None and not isinstance(device_raw, str):
        msg = 'infer.device must be str or null'
        raise TypeError(msg)
    return resolve_device(device_raw)



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



def run_pipeline(config_path: str | Path) -> Path:
    cfg, base_dir = load_cfg_with_base_dir(Path(config_path))
    prepared = _prepare_cfg(cfg, base_dir=base_dir)
    typed = load_coarse_infer_config(prepared)
    ckpt_path = _resolve_ckpt_path(prepared)
    device = _resolve_cli_device(prepared)

    ckpt = load_checkpoint(ckpt_path)
    _validate_checkpoint_for_infer(ckpt, model_sig=typed.model_sig)

    model = build_model(dict(typed.model_sig))
    state_dict, _ = select_state_dict(ckpt)
    model.load_state_dict(state_dict)
    model.to(device)

    out_path = run_coarse_infer(
        model=model,
        cfg=prepared,
        device=device,
    )
    print(str(out_path))
    return out_path



def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    run_pipeline(args.config)


if __name__ == '__main__':
    main()
