from __future__ import annotations

from dataclasses import dataclass

from seisai_utils.config import require_dict, require_float

__all__ = ['FBPickNormRefs', 'load_norm_refs_cfg']


@dataclass(frozen=True)
class FBPickNormRefs:
    time_ref_sec: float
    offset_ref_m: float


def load_norm_refs_cfg(cfg: dict) -> FBPickNormRefs:
    if not isinstance(cfg, dict):
        msg = 'cfg must be dict'
        raise TypeError(msg)

    norm_refs = require_dict(cfg, 'norm_refs')
    time_ref_sec = float(require_float(norm_refs, 'time_ref_sec'))
    offset_ref_m = float(require_float(norm_refs, 'offset_ref_m'))
    if time_ref_sec <= 0.0:
        msg = 'norm_refs.time_ref_sec must be > 0'
        raise ValueError(msg)
    if offset_ref_m <= 0.0:
        msg = 'norm_refs.offset_ref_m must be > 0'
        raise ValueError(msg)
    return FBPickNormRefs(
        time_ref_sec=float(time_ref_sec),
        offset_ref_m=float(offset_ref_m),
    )
