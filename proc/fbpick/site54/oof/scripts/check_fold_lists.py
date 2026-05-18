#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path


def read_list(path: Path) -> list[Path]:
    items: list[Path] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        items.append(Path(s))
    return items


def assert_disjoint(fold: str, suffix: str, train: list[Path], valid: list[Path], held: list[Path]) -> None:
    checks = {
        'train/inner_valid': set(train).intersection(valid),
        'train/heldout': set(train).intersection(held),
        'inner_valid/heldout': set(valid).intersection(held),
    }
    overlaps = {name: values for name, values in checks.items() if values}
    if overlaps:
        details = ', '.join(f'{name}={len(values)}' for name, values in overlaps.items())
        raise SystemExit(f'{fold}: {suffix} split overlap: {details}')


def main() -> None:
    ap = argparse.ArgumentParser(description='Check OOF fold list files and optionally referenced data files.')
    ap.add_argument('--fold-list-root', type=Path, default=Path('/workspace/proc/fbpick/site54/oof/fold_lists'))
    ap.add_argument('--check-exists', action='store_true', help='Also check that every SGY/FB path exists on this machine.')
    args = ap.parse_args()
    root = args.fold_list_root
    if not root.is_dir():
        raise SystemExit(f'fold list root not found: {root}')

    total_heldout = 0
    heldout_sgy_seen: set[Path] = set()
    for i in range(6):
        fold = f'fold{i:02d}'
        d = root / 'folds' / fold
        train_sgy = read_list(d / 'train_sgy.txt')
        train_fb = read_list(d / 'train_fb.txt')
        valid_sgy = read_list(d / 'inner_valid_sgy.txt')
        valid_fb = read_list(d / 'inner_valid_fb.txt')
        held_sgy = read_list(d / 'heldout_sgy.txt')
        held_fb = read_list(d / 'heldout_fb.txt')
        if len(train_sgy) != len(train_fb):
            raise SystemExit(f'{fold}: train length mismatch: SGY={len(train_sgy)} FB={len(train_fb)}')
        if len(valid_sgy) != len(valid_fb):
            raise SystemExit(f'{fold}: inner_valid length mismatch: SGY={len(valid_sgy)} FB={len(valid_fb)}')
        if len(held_sgy) != len(held_fb):
            raise SystemExit(f'{fold}: heldout length mismatch: SGY={len(held_sgy)} FB={len(held_fb)}')
        assert_disjoint(fold, 'SGY', train_sgy, valid_sgy, held_sgy)
        assert_disjoint(fold, 'FB', train_fb, valid_fb, held_fb)
        duplicate_heldout = heldout_sgy_seen.intersection(held_sgy)
        if duplicate_heldout:
            raise SystemExit(f'duplicate heldout SGY in {fold}: {sorted(map(str, duplicate_heldout))}')
        heldout_sgy_seen.update(held_sgy)
        total_heldout += len(held_sgy)
        print(f'{fold}: train={len(train_sgy):2d}, inner_valid={len(valid_sgy):2d}, heldout={len(held_sgy):2d}')
        if args.check_exists:
            for p in train_sgy + train_fb + valid_sgy + valid_fb + held_sgy + held_fb:
                if not p.exists():
                    raise SystemExit(f'missing referenced file in {fold}: {p}')
    if total_heldout != 54:
        raise SystemExit(f'expected 54 total heldout regions, got {total_heldout}')
    print(f'OK: total heldout regions = {total_heldout}; duplicate heldout SGY = 0')

if __name__ == '__main__':
    main()
