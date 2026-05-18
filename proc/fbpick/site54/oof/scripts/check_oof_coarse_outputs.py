#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


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
    ap.add_argument('--oof-root', type=Path, default=Path('/workspace/proc/fbpick/site54/oof'))
    ap.add_argument(
        '--fold-list-root',
        type=Path,
        default=None,
        help='Defaults to <oof-root>/site54_oof_6fold_lists.',
    )
    args = ap.parse_args()

    fold_list_root = args.fold_list_root or (args.oof_root / 'site54_oof_6fold_lists')
    ok = True
    for i in range(6):
        fold = f'fold{i:02d}'
        heldout_list = fold_list_root / 'folds' / fold / 'heldout_sgy.txt'
        out_dir = args.oof_root / 'coarse_oof' / fold
        expected = _count_lines(heldout_list)
        files = sorted(out_dir.glob('*.coarse.npz')) if out_dir.is_dir() else []
        print(f'{fold}: heldout_sgy={expected}, coarse_npz={len(files)}, out_dir={out_dir}')
        if expected > 0 and len(files) == 0:
            ok = False
    if not ok:
        raise SystemExit('some folds have no coarse outputs')


if __name__ == '__main__':
    main()
