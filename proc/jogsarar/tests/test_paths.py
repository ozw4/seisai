from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip('segyio')

import stage2_make_psn512_windows as st2
import stage4_psn512_infer_to_raw as st4
from common import paths as p


def test_stage2_wrapper_paths_follow_existing_rules() -> None:
    cfg = replace(
        st2.DEFAULT_STAGE2_CFG,
        in_segy_root=Path('/data/raw'),
        in_infer_root=Path('/data/infer'),
        out_segy_root=Path('/data/win'),
    )
    segy = Path('/data/raw/lineA/shot01.sgy')

    infer_npz = st2.infer_npz_path_for_segy(segy, cfg=cfg)
    assert infer_npz == Path('/data/infer/lineA/shot01.prob.npz')

    win512 = st2.out_segy_path_for_in(segy, cfg=cfg)
    assert win512 == Path('/data/win/lineA/shot01.win512.sgy')

    sidecar = st2.out_sidecar_npz_path_for_out(win512, cfg=cfg)
    assert sidecar == Path('/data/win/lineA/shot01.win512.sidecar.npz')

    csr = st2.out_pick_csr_npz_path_for_out(win512, cfg=cfg)
    assert csr == Path('/data/win/lineA/shot01.win512.phase_pick.csr.npz')


def test_sidecar_candidate_paths_count_depends_on_win512_suffix() -> None:
    win_named = Path('/data/win/lineA/shot01.win512.sgy')
    cands_win_named = p.sidecar_candidate_paths(win_named)
    assert cands_win_named == [Path('/data/win/lineA/shot01.win512.sidecar.npz')]

    plain_named = Path('/data/win/lineA/shot01.sgy')
    cands_plain = p.sidecar_candidate_paths(plain_named)
    assert cands_plain == [
        Path('/data/win/lineA/shot01.sidecar.npz'),
        Path('/data/win/lineA/shot01.win512.sidecar.npz'),
    ]


def test_stem_without_win512_and_lookup_key_match_stage4_behavior() -> None:
    assert p.stem_without_win512('a.win512') == 'a'
    assert p.stem_without_win512('a') == 'a'
    assert st4._stem_without_win512('a.win512') == 'a'
    assert st4._stem_without_win512('a') == 'a'

    win_root = Path('/data/win')
    win_path = Path('/data/win/sub/shot01.win512.sgy')
    key = p.win512_lookup_key(win_path, win_root=win_root)
    assert key == ('sub', 'shot01')


def test_stage4_pred_paths_preserve_relative_directories() -> None:
    raw_root = Path('/data/raw')
    out_root = Path('/data/pred')
    raw_path = Path('/data/raw/lineA/day1/shot01.sgy')

    out_dir = p.stage4_pred_out_dir(
        raw_path, in_raw_root=raw_root, out_pred_root=out_root
    )
    assert out_dir == Path('/data/pred/lineA/day1')

    out_npz = p.stage4_pred_npz_path(
        raw_path, in_raw_root=raw_root, out_pred_root=out_root
    )
    assert out_npz == Path('/data/pred/lineA/day1/shot01.psn_pred.npz')

    out_crd = p.stage4_pred_crd_path(
        raw_path, in_raw_root=raw_root, out_pred_root=out_root
    )
    assert out_crd == Path('/data/pred/lineA/day1/shot01.fb.crd')
