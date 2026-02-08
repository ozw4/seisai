from __future__ import annotations

from seisai_models.models.encdec2d import EncDec2D

__all__ = ['build_model']


def build_model(
    *, backbone: str, in_chans: int, out_chans: int, pretrained: bool
) -> EncDec2D:
    model = EncDec2D(
        backbone=str(backbone),
        in_chans=int(in_chans),
        out_chans=int(out_chans),
        pretrained=bool(pretrained),
    )
    model.use_tta = False
    return model
