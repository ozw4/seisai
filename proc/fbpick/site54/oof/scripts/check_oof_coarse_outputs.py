#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

FOLDS = [f"fold{i:02d}" for i in range(6)]
DEFAULT_CV_ROOT = Path('/workspace/proc/fbpick/site54/oof')
DEFAULT_RUN_ID = 'baseline_physical_center'


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return -1
    return sum(
        1
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.strip().startswith('#')
    )


def main() -> None:
    ap = argparse.ArgumentParser(description='Check OOF coarse .coarse.npz outputs for each fold.')
    ap.add_argument('--cv-root', type=Path, default=DEFAULT_CV_ROOT)
    ap.add_argument('--run-id', default=DEFAULT_RUN_ID)
    ap.add_argument('--run-root', type=Path, default=None)
    ap.add_argument(
        '--fold-list-root',
        type=Path,
        default=None,
        help='Defaults to <cv-root>/fold_lists.',
    )
    args = ap.parse_args()

    run_root = args.run_root or (args.cv_root / 'runs' / args.run_id)
    fold_list_root = args.fold_list_root or (args.cv_root / 'fold_lists')
    ok = True
    for fold in FOLDS:
        heldout_list = fold_list_root / 'folds' / fold / 'heldout_sgy.txt'
        out_dir = run_root / fold / '02_coarse_infer'
        expected = _count_lines(heldout_list)
        files = sorted(out_dir.glob('*.coarse.npz')) if out_dir.is_dir() else []
        print(f'{fold}: heldout_sgy={expected}, coarse_npz={len(files)}, out_dir={out_dir}')
        if expected < 0 or len(files) != expected:
            ok = False
    if not ok:
        raise SystemExit('some folds have incomplete coarse outputs')


if __name__ == '__main__':
    main()
