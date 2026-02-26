from __future__ import annotations

from pathlib import Path

import pytest

from common.iter_layout import iter_tag, resolve_iter_layout


def test_iter_tag_formats_and_validates() -> None:
    assert iter_tag(0) == 'iter00'
    assert iter_tag(7) == 'iter07'
    assert iter_tag(123) == 'iter123'

    with pytest.raises(ValueError):
        iter_tag(-1)


def test_resolve_iter_layout_builds_stage_paths(tmp_path: Path) -> None:
    out_root = tmp_path / 'out_root'
    layout = resolve_iter_layout(out_root, iter_id=3)

    assert layout.iter_id == 3
    assert layout.iter_root == out_root.resolve() / 'iter03'
    assert layout.stage1_out == layout.iter_root / 'stage1'
    assert layout.stage2_out == layout.iter_root / 'stage2_win512'
    assert layout.stage3_out == layout.iter_root / 'stage3'
    assert layout.stage4_out == layout.iter_root / 'stage4_pred'


def test_resolve_iter_layout_rejects_file_path(tmp_path: Path) -> None:
    out_file = tmp_path / 'out.txt'
    out_file.write_text('x', encoding='utf-8')

    with pytest.raises(NotADirectoryError):
        resolve_iter_layout(out_file, iter_id=0)
