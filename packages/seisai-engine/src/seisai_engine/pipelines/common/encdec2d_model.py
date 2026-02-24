from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from seisai_models.models.encdec2d import EncDec2D

__all__ = ['build_encdec2d_model']


def build_encdec2d_model(encdec2d_kwargs: Mapping[str, Any]) -> EncDec2D:
    if not isinstance(encdec2d_kwargs, Mapping):
        msg = 'encdec2d_kwargs must be a mapping'
        raise TypeError(msg)
    model = EncDec2D(**dict(encdec2d_kwargs))
    model.use_tta = False
    return model
