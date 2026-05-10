from __future__ import annotations

from pathlib import Path

__all__ = ['build_fbpick_tag', 'build_final_npz_name']


def build_fbpick_tag(segy_path: str | Path) -> str:
    segy = Path(segy_path)
    parent_name = segy.parent.name
    if parent_name:
        return parent_name + '__' + segy.stem
    return segy.stem


def build_final_npz_name(segy_path: str | Path) -> str:
    return build_fbpick_tag(segy_path) + '.fbpick_final.npz'
