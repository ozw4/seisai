from __future__ import annotations

from dataclasses import asdict

from seisai_models.models.encdec2d import EncDec2D

from seisai_engine.pipelines.common.encdec2d_model import build_encdec2d_model

from .config import CoarseModelCfg

__all__ = ['build_model']


def build_model(cfg: CoarseModelCfg) -> EncDec2D:
    if not isinstance(cfg, CoarseModelCfg):
        msg = 'cfg must be CoarseModelCfg'
        raise TypeError(msg)
    if int(cfg.in_chans) != 3:
        msg = 'config.model.in_chans must be 3 (amplitude / abs offset / absolute time)'
        raise ValueError(msg)
    if int(cfg.out_chans) != 1:
        msg = 'config.model.out_chans must be 1 (coarse first-break probability)'
        raise ValueError(msg)
    return build_encdec2d_model(asdict(cfg))
