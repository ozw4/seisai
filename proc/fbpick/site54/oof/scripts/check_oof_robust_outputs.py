#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

FOLDS = [f"fold{i:02d}" for i in range(6)]


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip() and not line.strip().startswith('#')]


def check_schema(path: Path) -> None:
    with np.load(path, allow_pickle=True) as z:
        required = [
            'robust_pick_i', 'robust_pick_t_sec', 'robust_conf', 'dt_sec',
            'n_samples_orig', 'n_traces', 'trace_indices', 'offsets_m',
        ]
        missing = [k for k in required if k not in z]
        if missing:
            raise ValueError(f'{path}: missing keys {missing}')
        n_traces = int(np.asarray(z['n_traces']))
        n_samples = int(np.asarray(z['n_samples_orig']))
        dt_sec = float(np.asarray(z['dt_sec']))
        pick = np.asarray(z['robust_pick_i'])
        tsec = np.asarray(z['robust_pick_t_sec'])
        conf = np.asarray(z['robust_conf'])
        if pick.shape != (n_traces,):
            raise ValueError(f'{path}: robust_pick_i shape {pick.shape} != {(n_traces,)}')
        if tsec.shape != (n_traces,):
            raise ValueError(f'{path}: robust_pick_t_sec shape {tsec.shape} != {(n_traces,)}')
        if conf.shape != (n_traces,):
            raise ValueError(f'{path}: robust_conf shape {conf.shape} != {(n_traces,)}')
        if not np.all((0 <= pick) & (pick < n_samples)):
            raise ValueError(f'{path}: robust_pick_i outside sample range')
        if not np.all(np.isfinite(tsec)):
            raise ValueError(f'{path}: robust_pick_t_sec contains non-finite')
        if not np.all(np.isfinite(conf)):
            raise ValueError(f'{path}: robust_conf contains non-finite')
        np.testing.assert_allclose(
            tsec,
            pick.astype(np.float32) * np.float32(dt_sec),
            rtol=0,
            atol=1.0e-6,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description='Check site54 OOF robust outputs.')
    parser.add_argument('--oof-root', default='/workspace/proc/fbpick/site54/oof')
    parser.add_argument('--check-schema', action='store_true')
    args = parser.parse_args()
    oof_root = Path(args.oof_root)
    fold_root = oof_root / 'fold_lists' / 'folds'
    total_expected = 0
    total_found = 0
    for fold in FOLDS:
        heldout = read_list(fold_root / fold / 'heldout_sgy.txt')
        out_dir = oof_root / 'robust_oof' / fold
        files = sorted(out_dir.glob('*.robust.npz'))
        total_expected += len(heldout)
        total_found += len(files)
        print(f'{fold}: heldout_sgy={len(heldout)}, robust_npz={len(files)}, out_dir={out_dir}')
        if len(files) != len(heldout):
            raise SystemExit(f'{fold}: expected {len(heldout)} robust npz, found {len(files)}')
        if args.check_schema:
            for p in files:
                check_schema(p)
    if total_expected != 54:
        raise SystemExit(f'expected total heldout 54, got {total_expected}')
    if total_found != total_expected:
        raise SystemExit(f'expected total robust {total_expected}, got {total_found}')
    print(f'OK: robust outputs complete ({total_found}/{total_expected})')


if __name__ == '__main__':
    main()
