from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    'apply_shift_linear': ('.residual_statics', 'apply_shift_linear'),
    'detect_event_pick_cluster': ('.detectors', 'detect_event_pick_cluster'),
    'detect_event_stalta_majority': ('.detectors', 'detect_event_stalta_majority'),
    'estimate_shift_ncc': ('.residual_statics', 'estimate_shift_ncc'),
    'make_local_reference': ('.residual_statics', 'make_local_reference'),
    'refine_firstbreak_residual_statics': (
        '.residual_statics',
        'refine_firstbreak_residual_statics',
    ),
    'smooth_shifts': ('.residual_statics', 'smooth_shifts'),
    'stalta_1d': ('.stalta', 'stalta_1d'),
}

__all__ = [
    'apply_shift_linear',
    'detect_event_pick_cluster',
    'detect_event_stalta_majority',
    'estimate_shift_ncc',
    'make_local_reference',
    'refine_firstbreak_residual_statics',
    'smooth_shifts',
    'stalta_1d',
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
