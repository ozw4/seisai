from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import torch
from _model import NetAE as EncDec2D
from stage1.cfg import DEFAULT_STAGE1_CFG, Stage1Cfg, WEIGHTS_PATH


def _strip_prefix(sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith('model.'):
            out[k.removeprefix('model.')] = v
        elif k.startswith('module.'):
            out[k.removeprefix('module.')] = v
        else:
            out[k] = v
    return out


def build_model(*, weights_path: Path = WEIGHTS_PATH) -> torch.nn.Module:
    cfg = replace(DEFAULT_STAGE1_CFG, weights_path=Path(weights_path))
    return build_model_from_cfg(cfg)


def build_model_from_cfg(cfg: Stage1Cfg) -> torch.nn.Module:
    device_name = str(cfg.device)
    backbone = str(cfg.backbone)
    use_tta = bool(cfg.use_tta)
    weights_path = Path(cfg.weights_path)

    if device_name.startswith('cuda') and not torch.cuda.is_available():
        msg = 'CUDA requested but not available'
        raise RuntimeError(msg)

    ckpt_path = Path(weights_path).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = (Path.cwd() / ckpt_path).resolve()
    if not ckpt_path.is_file():
        msg = f'weights not found: {ckpt_path}'
        raise FileNotFoundError(msg)

    model = EncDec2D(
        backbone=backbone,
        in_chans=1,
        out_chans=1,
        pretrained=True,
        stage_strides=[(2, 4), (2, 2), (2, 4), (2, 2)],
        pre_stages=2,
        pre_stage_strides=((1, 1), (1, 2)),
    )
    model.out_chans = 1
    model.use_tta = use_tta

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model_ema']

    if (
        not isinstance(sd, Mapping)
        or not sd
        or not all(isinstance(v, torch.Tensor) for v in sd.values())
    ):
        msg = "ckpt['model_ema'] is not a state_dict"
        raise ValueError(msg)

    sd = _strip_prefix(sd)
    print(f'[CKPT] load from: {ckpt_path} (model_ema direct state_dict)')

    model.load_state_dict(sd, strict=True)
    model.to(device=torch.device(device_name))
    model.eval()
    return model


__all__ = [
    '_strip_prefix',
    'build_model',
    'build_model_from_cfg',
]
