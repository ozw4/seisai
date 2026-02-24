from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from seisai_models.models.encdec2d import EncDec2D

from seisai_engine.pipelines.common.encdec2d_model import build_encdec2d_model

if TYPE_CHECKING:
    from .config import PsnModelCfg

__all__ = ['build_model']


def build_model(cfg: PsnModelCfg) -> EncDec2D:
    return build_encdec2d_model(asdict(cfg))
