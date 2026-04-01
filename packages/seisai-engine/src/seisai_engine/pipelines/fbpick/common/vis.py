from __future__ import annotations

from pathlib import Path

__all__ = [
    'save_coarse_debug_figure',
    'save_fine_debug_figure',
    'save_global_qc_debug_figure',
]


def _raise_vis_stub(*, stage: str, out_path: str | Path) -> None:
    msg = (
        f'fbpick {stage} debug visualization is not implemented in Phase 1. '
        f'Requested output path: {Path(out_path)}'
    )
    raise NotImplementedError(msg)


def save_coarse_debug_figure(*, out_path: str | Path, **_: object) -> None:
    _raise_vis_stub(stage='coarse', out_path=out_path)


def save_fine_debug_figure(*, out_path: str | Path, **_: object) -> None:
    _raise_vis_stub(stage='fine', out_path=out_path)


def save_global_qc_debug_figure(*, out_path: str | Path, **_: object) -> None:
    _raise_vis_stub(stage='global_qc', out_path=out_path)
