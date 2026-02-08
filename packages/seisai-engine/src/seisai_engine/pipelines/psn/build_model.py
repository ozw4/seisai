from __future__ import annotations

from seisai_models.models.encdec2d import EncDec2D
from seisai_utils.config import (
    optional_bool,
    optional_int,
    optional_str,
    require_dict,
)

__all__ = ['build_model']


def build_model(cfg: dict) -> EncDec2D:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    model_cfg = require_dict(cfg, 'model')
    backbone = optional_str(model_cfg, 'backbone', 'resnet18')
    pretrained = optional_bool(model_cfg, 'pretrained', default=False)
    in_chans = optional_int(model_cfg, 'in_chans', 1)
    out_chans = optional_int(model_cfg, 'out_chans', 3)

    if int(in_chans) != 1:
        msg = 'model.in_chans must be 1 (waveform only)'
        raise ValueError(msg)
    if int(out_chans) != 3:
        msg = 'model.out_chans must be 3 (P/S/Noise)'
        raise ValueError(msg)

    model = EncDec2D(
        backbone=str(backbone),
        in_chans=int(in_chans),
        out_chans=int(out_chans),
        pretrained=bool(pretrained),
    )
    model.use_tta = False
    return model
