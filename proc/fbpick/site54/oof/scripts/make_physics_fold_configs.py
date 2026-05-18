#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

FOLDS = [f"fold{i:02d}" for i in range(6)]


def main() -> None:
    parser = argparse.ArgumentParser(description='Render site54 OOF physics batch configs.')
    parser.add_argument('--fold-list-root', default='/workspace/proc/fbpick/site54/oof/fold_lists')
    parser.add_argument('--oof-root', default='/workspace/proc/fbpick/site54/oof')
    parser.add_argument('--config-dir', default='/workspace/proc/fbpick/site54/oof/configs')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    fold_list_root = Path(args.fold_list_root).resolve()
    oof_root = Path(args.oof_root).resolve()
    config_dir = Path(args.config_dir).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)

    for fold in FOLDS:
        heldout_sgy = fold_list_root / 'folds' / fold / 'heldout_sgy.txt'
        if not heldout_sgy.is_file():
            raise FileNotFoundError(heldout_sgy)
        coarse_dir = oof_root / 'coarse_oof' / fold
        out_dir = oof_root / 'robust_oof' / fold
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            'paths': {
                'segy_files': str(heldout_sgy),
                'coarse_npz_dir': str(coarse_dir),
                'out_dir': str(out_dir),
            },
            'feasible_band': {},
            'trend': {},
            'residual_statics': {},
            'keep_reject': {},
            'robust_center': {},
        }
        out = config_dir / f'config_run_fbpick_physics_{fold}_heldout.yaml'
        if out.exists() and not args.overwrite:
            print(f'[skip] exists: {out}')
            continue
        out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        print(f'[write] {out}')


if __name__ == '__main__':
    main()
