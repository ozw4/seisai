from __future__ import annotations

from seisai_models.models.encdec2d import EncDec2D

__all__ = ['build_model']


def build_model(encdec2d_kwargs: dict) -> EncDec2D:
    if not isinstance(encdec2d_kwargs, dict):
        raise TypeError('encdec2d_kwargs must be dict')
    model = EncDec2D(**encdec2d_kwargs)
    model.use_tta = False
    return model
