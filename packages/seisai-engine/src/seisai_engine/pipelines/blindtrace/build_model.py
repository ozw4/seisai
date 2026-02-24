from __future__ import annotations

from seisai_models.models.encdec2d import EncDec2D

from seisai_engine.pipelines.common.encdec2d_model import build_encdec2d_model

__all__ = ['build_model']


def build_model(encdec2d_kwargs: dict) -> EncDec2D:
    if not isinstance(encdec2d_kwargs, dict):
        raise TypeError('encdec2d_kwargs must be dict')
    return build_encdec2d_model(encdec2d_kwargs)
