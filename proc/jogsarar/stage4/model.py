from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from seisai_engine.pipelines.common.checkpoint_io import load_checkpoint
from seisai_engine.pipelines.psn.build_model import build_model as build_psn_model
from seisai_engine.pipelines.psn.config import load_psn_train_config
from seisai_models.models.encdec2d import EncDec2D
from seisai_utils import config_yaml
from stage4.cfg import DEFAULT_STAGE4_CFG, Stage4Cfg


def _resolve_config_loader():
    if hasattr(config_yaml, 'load_yaml_config'):
        fn = config_yaml.load_yaml_config
        if callable(fn):
            return fn
    if hasattr(config_yaml, 'load_yaml'):
        fn = config_yaml.load_yaml
        if callable(fn):
            return fn
    msg = 'seisai_utils.config_yaml must expose load_yaml_config() or load_yaml()'
    raise AttributeError(msg)


def _resolve_explicit_ckpt_path(path: Path) -> Path:
    ckpt_path = path.expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = (Path.cwd() / ckpt_path).resolve()
    if not ckpt_path.is_file():
        msg = f'checkpoint not found: {ckpt_path}'
        raise FileNotFoundError(msg)
    return ckpt_path


def _resolve_ckpt_path(
    loaded_cfg: dict | None,
    cfg_yaml_path: Path | None,
    *,
    cfg: Stage4Cfg = DEFAULT_STAGE4_CFG,
) -> Path:
    if cfg_yaml_path is None:
        if cfg.ckpt_path is None:
            msg = 'ckpt_path must be set when cfg_yaml is None'
            raise ValueError(msg)
        return _resolve_explicit_ckpt_path(cfg.ckpt_path)

    if cfg.ckpt_path is not None:
        return _resolve_explicit_ckpt_path(cfg.ckpt_path)

    if loaded_cfg is None:
        msg = 'loaded_cfg must be provided when cfg_yaml is set'
        raise ValueError(msg)

    paths = loaded_cfg.get('paths')
    if not isinstance(paths, dict):
        msg = 'config.paths must be dict'
        raise TypeError(msg)
    out_dir_val = paths.get('out_dir')
    if not isinstance(out_dir_val, str) or not out_dir_val.strip():
        msg = 'config.paths.out_dir must be non-empty str'
        raise ValueError(msg)

    out_dir = Path(out_dir_val).expanduser()
    if not out_dir.is_absolute():
        out_dir = cfg_yaml_path.parent / out_dir
    ckpt_path = (out_dir / 'ckpt' / 'best.pt').resolve()
    if not ckpt_path.is_file():
        msg = (
            f'checkpoint not found: {ckpt_path} '
            '(set cfg.ckpt_path explicitly if needed)'
        )
        raise FileNotFoundError(msg)
    return ckpt_path


def _resolve_device(*, cfg: Stage4Cfg = DEFAULT_STAGE4_CFG) -> torch.device:
    dev = torch.device(str(cfg.device))
    if dev.type == 'cuda' and not torch.cuda.is_available():
        msg = 'cfg.device is cuda but CUDA is not available'
        raise RuntimeError(msg)
    return dev


def load_psn_model_and_eps(
    *, cfg: Stage4Cfg = DEFAULT_STAGE4_CFG
) -> tuple[torch.nn.Module, float, Path]:
    if cfg.cfg_yaml is not None:
        cfg_yaml_path = cfg.cfg_yaml.expanduser().resolve()
        if not cfg_yaml_path.is_file():
            msg = f'cfg_yaml not found: {cfg_yaml_path}'
            raise FileNotFoundError(msg)

        load_yaml_fn = _resolve_config_loader()
        loaded_cfg = load_yaml_fn(cfg_yaml_path)
        if not isinstance(loaded_cfg, dict):
            msg = f'loaded config must be dict, got {type(loaded_cfg).__name__}'
            raise TypeError(msg)

        typed = load_psn_train_config(loaded_cfg)
        ckpt_path = _resolve_ckpt_path(loaded_cfg, cfg_yaml_path, cfg=cfg)
        ckpt = load_checkpoint(ckpt_path)
        if ckpt['pipeline'] != 'psn':
            msg = f'checkpoint pipeline must be "psn", got {ckpt["pipeline"]!r}'
            raise ValueError(msg)

        model_sig = ckpt['model_sig']
        expected_sig = asdict(typed.model)
        if model_sig != expected_sig:
            msg = 'checkpoint model_sig does not match cfg_yaml model definition'
            raise ValueError(msg)

        transform_cfg = loaded_cfg.get('transform', {})
        if not isinstance(transform_cfg, dict):
            msg = 'config.transform must be dict'
            raise TypeError(msg)
        standardize_eps = float(transform_cfg.get('standardize_eps', 1.0e-8))
        if standardize_eps <= 0.0:
            msg = f'transform.standardize_eps must be > 0, got {standardize_eps}'
            raise ValueError(msg)
        model = build_psn_model(typed.model)
    else:
        ckpt_path = _resolve_ckpt_path(None, None, cfg=cfg)
        ckpt = load_checkpoint(ckpt_path)
        if ckpt['pipeline'] != 'psn':
            msg = f'checkpoint pipeline must be "psn", got {ckpt["pipeline"]!r}'
            raise ValueError(msg)
        model_sig = ckpt['model_sig']
        if not isinstance(model_sig, dict):
            msg = f'checkpoint model_sig must be dict, got {type(model_sig).__name__}'
            raise TypeError(msg)
        model_kwargs = dict(model_sig)
        if 'pretrained' in model_kwargs:
            model_kwargs['pretrained'] = False
        if 'backbone_pretrained' in model_kwargs:
            if 'pretrained' not in model_kwargs:
                model_kwargs['pretrained'] = False
            model_kwargs.pop('backbone_pretrained')
        standardize_eps = float(cfg.standardize_eps)
        if standardize_eps <= 0.0:
            msg = f'cfg.standardize_eps must be > 0, got {standardize_eps}'
            raise ValueError(msg)
        model = EncDec2D(**model_kwargs)

    device = _resolve_device(cfg=cfg)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device)
    model.eval()
    return model, float(standardize_eps), ckpt_path


__all__ = [
    '_resolve_ckpt_path',
    '_resolve_config_loader',
    '_resolve_device',
    '_resolve_explicit_ckpt_path',
    'load_psn_model_and_eps',
]
