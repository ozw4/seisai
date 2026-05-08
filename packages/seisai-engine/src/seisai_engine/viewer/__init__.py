from __future__ import annotations

from importlib import import_module

__all__ = [
    'clear_model_cache',
    'infer_denoise_hw',
    'infer_prob_hw',
    'render_fbpick_overview',
    'save_fbpick_fine_qc_gather_png',
    'save_fbpick_overview_png',
]

_EXPORTS = {
    'clear_model_cache': ('.model_cache', 'clear_model_cache'),
    'infer_denoise_hw': ('.denoise', 'infer_denoise_hw'),
    'infer_prob_hw': ('.fbpick', 'infer_prob_hw'),
    'render_fbpick_overview': ('.fbpick', 'render_fbpick_overview'),
    'save_fbpick_fine_qc_gather_png': (
        '.fbpick',
        'save_fbpick_fine_qc_gather_png',
    ),
    'save_fbpick_overview_png': ('.fbpick', 'save_fbpick_overview_png'),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        msg = f'module {__name__!r} has no attribute {name!r}'
        raise AttributeError(msg)

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals()) + __all__))
