from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def iter_tag(iter_id: int) -> str:
    if isinstance(iter_id, bool) or not isinstance(iter_id, int):
        msg = f'iter_id must be int, got {type(iter_id).__name__}'
        raise TypeError(msg)
    if int(iter_id) < 0:
        msg = f'iter_id must be >= 0, got {iter_id}'
        raise ValueError(msg)
    return f'iter{int(iter_id):02d}'


@dataclass(frozen=True)
class IterLayout:
    iter_id: int
    iter_root: Path
    stage1_out: Path
    stage2_out: Path
    stage3_out: Path
    stage4_out: Path


def resolve_iter_layout(out_root: Path, *, iter_id: int) -> IterLayout:
    out_dir = Path(out_root).expanduser().resolve()
    if out_dir.exists() and (not out_dir.is_dir()):
        msg = f'out_root must be a directory path: {out_dir}'
        raise NotADirectoryError(msg)

    tag = iter_tag(iter_id)
    iter_root = out_dir / tag
    return IterLayout(
        iter_id=int(iter_id),
        iter_root=iter_root,
        stage1_out=iter_root / 'stage1',
        stage2_out=iter_root / 'stage2_win512',
        stage3_out=iter_root / 'stage3',
        stage4_out=iter_root / 'stage4_pred',
    )


__all__ = [
    'IterLayout',
    'iter_tag',
    'resolve_iter_layout',
]
