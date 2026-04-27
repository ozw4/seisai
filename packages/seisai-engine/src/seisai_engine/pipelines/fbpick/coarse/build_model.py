from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from seisai_engine.pipelines.common import build_encdec2d_model

__all__ = ['build_model']


def build_model(model_sig: Mapping[str, Any]):
    return build_encdec2d_model(model_sig)
