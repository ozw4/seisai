from __future__ import annotations

from typing import TYPE_CHECKING

from seisai_models.models.encdec2d import EncDec2D

if TYPE_CHECKING:
    from .config import PairModelCfg

__all__ = ['build_model']


def build_model(cfg: PairModelCfg) -> EncDec2D:
    model = EncDec2D(
        backbone=cfg.backbone,
        in_chans=int(cfg.in_chans),
        out_chans=int(cfg.out_chans),
        pretrained=bool(cfg.pretrained),
    )
    model.use_tta = False
    return model
