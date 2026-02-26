from __future__ import annotations

from pathlib import Path


def guess_stage4_model_id(stage4_pred_root: Path) -> str:
    root = Path(stage4_pred_root).expanduser().resolve()
    if not root.exists() or (not root.is_dir()):
        return ''
    return ''


__all__ = ['guess_stage4_model_id']
