#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import csv

FOLDS = [f"fold{i:02d}" for i in range(6)]


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip() and not line.strip().startswith('#')]


def prefixed_name(segy_path: str, suffix: str) -> str:
    p = Path(segy_path)
    return p.parent.name + '__' + p.stem + suffix


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect OOF coarse/robust lists in original SGY/FB order.')
    parser.add_argument('--oof-root', default='/workspace/proc/fbpick/site54/oof')
    parser.add_argument('--out-dir', default='/workspace/proc/fbpick/site54/oof/lists')
    args = parser.parse_args()

    oof_root = Path(args.oof_root)
    lists_root = oof_root / 'fold_lists' / 'lists'
    all_sgy = read_list(lists_root / 'all_sgy.txt')
    all_fb = read_list(lists_root / 'all_fb.txt')
    if len(all_sgy) != len(all_fb):
        raise ValueError(f'all_sgy/all_fb length mismatch: {len(all_sgy)} vs {len(all_fb)}')

    robust_by_name: dict[str, tuple[str, Path]] = {}
    coarse_by_name: dict[str, tuple[str, Path]] = {}
    for fold in FOLDS:
        robust_dir = oof_root / 'robust_oof' / fold
        coarse_dir = oof_root / 'coarse_oof' / fold
        for p in robust_dir.glob('*.robust.npz'):
            if p.name in robust_by_name:
                raise ValueError(f'duplicate robust output name: {p.name}')
            robust_by_name[p.name] = (fold, p.resolve())
        for p in coarse_dir.glob('*.coarse.npz'):
            if p.name in coarse_by_name:
                raise ValueError(f'duplicate coarse output name: {p.name}')
            coarse_by_name[p.name] = (fold, p.resolve())

    out_sgy: list[str] = []
    out_fb: list[str] = []
    out_robust: list[str] = []
    out_coarse: list[str] = []
    rows: list[dict[str, str]] = []
    for idx, (sgy, fb) in enumerate(zip(all_sgy, all_fb), start=1):
        robust_name = prefixed_name(sgy, '.robust.npz')
        coarse_name = prefixed_name(sgy, '.coarse.npz')
        if robust_name not in robust_by_name:
            raise FileNotFoundError(f'missing robust output for {sgy}: expected name {robust_name}')
        if coarse_name not in coarse_by_name:
            raise FileNotFoundError(f'missing coarse output for {sgy}: expected name {coarse_name}')
        r_fold, r_path = robust_by_name[robust_name]
        c_fold, c_path = coarse_by_name[coarse_name]
        if r_fold != c_fold:
            raise ValueError(f'coarse/robust fold mismatch for {sgy}: coarse={c_fold}, robust={r_fold}')
        out_sgy.append(sgy)
        out_fb.append(fb)
        out_robust.append(str(r_path))
        out_coarse.append(str(c_path))
        rows.append({
            'index': str(idx),
            'fold': r_fold,
            'sgy': sgy,
            'fb': fb,
            'coarse_npz': str(c_path),
            'robust_npz': str(r_path),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'oof_train_sgy_all.txt').write_text('\n'.join(out_sgy) + '\n', encoding='utf-8')
    (out_dir / 'oof_train_fb_all.txt').write_text('\n'.join(out_fb) + '\n', encoding='utf-8')
    (out_dir / 'oof_train_robust_all.txt').write_text('\n'.join(out_robust) + '\n', encoding='utf-8')
    (out_dir / 'oof_train_coarse_all.txt').write_text('\n'.join(out_coarse) + '\n', encoding='utf-8')
    with (out_dir / 'oof_train_mapping.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index','fold','sgy','fb','coarse_npz','robust_npz'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'OK: wrote {len(rows)} OOF entries to {out_dir}')


if __name__ == '__main__':
    main()
