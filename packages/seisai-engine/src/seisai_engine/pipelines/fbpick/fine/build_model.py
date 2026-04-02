from __future__ import annotations

from dataclasses import asdict

from seisai_models.models.encdec2d import EncDec2D

from seisai_engine.pipelines.common.encdec2d_model import build_encdec2d_model

from .config import FineModelCfg

__all__ = ['build_model']


def build_model(cfg: FineModelCfg) -> EncDec2D:
    if not isinstance(cfg, FineModelCfg):
        msg = 'cfg must be FineModelCfg'
        raise TypeError(msg)
    if int(cfg.in_chans) != 1:
        msg = 'config.model.in_chans must be 1 (fine v1 amplitude only)'
        raise ValueError(msg)
    if int(cfg.out_chans) != 1:
        msg = 'config.model.out_chans must be 1 (fine local probability map)'
        raise ValueError(msg)
    return build_encdec2d_model(asdict(cfg))
