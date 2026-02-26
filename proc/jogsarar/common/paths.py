from __future__ import annotations

from pathlib import Path


def stage1_prob_npz_path(
    segy_path: Path, *, in_segy_root: Path, out_infer_root: Path
) -> Path:
    rel = segy_path.relative_to(in_segy_root)
    return out_infer_root / rel.parent / f'{segy_path.stem}.prob.npz'


def stage1_out_dir(
    segy_path: Path, *, in_segy_root: Path, out_infer_root: Path
) -> Path:
    rel = segy_path.relative_to(in_segy_root)
    return out_infer_root / rel.parent


def stage2_win512_segy_path(
    segy_path: Path, *, in_segy_root: Path, out_segy_root: Path
) -> Path:
    rel = segy_path.relative_to(in_segy_root)
    out_rel = rel.with_suffix('')
    return out_segy_root / out_rel.parent / f'{out_rel.name}.win512.sgy'


def stage2_sidecar_npz_path(win512_segy_path: Path) -> Path:
    return win512_segy_path.with_suffix('.sidecar.npz')


def stage2_phase_pick_csr_npz_path(win512_segy_path: Path) -> Path:
    return win512_segy_path.with_suffix('.phase_pick.csr.npz')


def stem_without_win512(stem: str) -> str:
    tag = '.win512'
    if stem.endswith(tag):
        return stem[: -len(tag)]
    return stem


def win512_lookup_key(win_path: Path, *, win_root: Path) -> tuple[str, str]:
    rel = win_path.relative_to(win_root)
    return rel.parent.as_posix(), stem_without_win512(rel.stem)


def sidecar_candidate_paths(win_path: Path) -> list[Path]:
    cands: list[Path] = [win_path.with_suffix('.sidecar.npz')]
    if not win_path.stem.endswith('.win512'):
        cands.append(win_path.with_suffix('.win512.sidecar.npz'))
    return cands


def resolve_sidecar_path(win_path: Path) -> Path | None:
    for p in sidecar_candidate_paths(win_path):
        if p.is_file():
            return p
    return None


def stage4_pred_out_dir(
    raw_path: Path, *, in_raw_root: Path, out_pred_root: Path
) -> Path:
    rel = raw_path.relative_to(in_raw_root)
    return out_pred_root / rel.parent


def stage4_pred_npz_path(
    raw_path: Path, *, in_raw_root: Path, out_pred_root: Path
) -> Path:
    out_dir = stage4_pred_out_dir(
        raw_path, in_raw_root=in_raw_root, out_pred_root=out_pred_root
    )
    return out_dir / f'{raw_path.stem}.psn_pred.npz'


def stage4_pred_crd_path(
    raw_path: Path, *, in_raw_root: Path, out_pred_root: Path
) -> Path:
    out_dir = stage4_pred_out_dir(
        raw_path, in_raw_root=in_raw_root, out_pred_root=out_pred_root
    )
    return out_dir / f'{raw_path.stem}.fb.crd'


__all__ = [
    'resolve_sidecar_path',
    'sidecar_candidate_paths',
    'stage1_out_dir',
    'stage1_prob_npz_path',
    'stage2_phase_pick_csr_npz_path',
    'stage2_sidecar_npz_path',
    'stage2_win512_segy_path',
    'stage4_pred_crd_path',
    'stage4_pred_npz_path',
    'stage4_pred_out_dir',
    'stem_without_win512',
    'win512_lookup_key',
]
