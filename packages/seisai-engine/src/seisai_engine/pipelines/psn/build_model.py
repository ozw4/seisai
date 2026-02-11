from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from seisai_models.models.encdec2d import EncDec2D

if TYPE_CHECKING:
    from .config import PsnModelCfg

__all__ = ['build_model']


def build_model(cfg: PsnModelCfg) -> EncDec2D:
    model = EncDec2D(**asdict(cfg))
    model.use_tta = False
    return model
