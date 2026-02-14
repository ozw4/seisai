#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_psn_npz_from_picks(picks_1d: np.ndarray) -> dict[str, np.ndarray]:
    if picks_1d.ndim != 1:
        raise ValueError(f'Expected 1D array, got shape={picks_1d.shape}')

    n_traces = int(picks_1d.shape[0])

    if picks_1d.dtype.kind in ('i', 'u'):
        valid = picks_1d >= 0
        p_data = picks_1d[valid].astype(np.int64, copy=False)
    elif picks_1d.dtype.kind == 'f':
        finite = np.isfinite(picks_1d)
        nonneg = picks_1d >= 0
        is_int = picks_1d == np.floor(picks_1d)
        valid = finite & nonneg & is_int
        p_data = picks_1d[valid].astype(np.int64, copy=False)
    else:
        raise ValueError(f'Unsupported dtype: {picks_1d.dtype}')

    # CSR-style indptr: length n_traces+1, increments by 1 only where valid pick exists
    inc = valid.astype(np.int64, copy=False)
    p_indptr = np.empty(n_traces + 1, dtype=np.int64)
    p_indptr[0] = 0
    np.cumsum(inc, out=p_indptr[1:])

    # S is absent: empty CSR
    s_indptr = np.zeros(n_traces + 1, dtype=np.int64)
    s_data = np.empty((0,), dtype=np.int64)

    return {
        'p_indptr': p_indptr,
        'p_data': p_data,
        's_indptr': s_indptr,
        's_data': s_data,
    }


def process_field_dir(field_dir: Path, force: bool, dry_run: bool) -> tuple[bool, str]:
    npy_files = sorted(field_dir.glob('*.npy'))
    if len(npy_files) == 0:
        return False, f'SKIP  {field_dir}: no .npy'
    if len(npy_files) != 1:
        names = ', '.join(p.name for p in npy_files)
        return (
            False,
            f'SKIP  {field_dir}: expected 1 .npy, found {len(npy_files)} [{names}]',
        )

    npy_path = npy_files[0]
    npz_path = npy_path.with_suffix('.npz')

    if npz_path.exists() and not force:
        return (
            False,
            f'SKIP  {field_dir}: {npz_path.name} already exists (use --force to overwrite)',
        )

    arr = np.load(npy_path)
    data = build_psn_npz_from_picks(arr)

    n_traces = int(arr.shape[0]) if arr.ndim == 1 else -1
    n_picks = int(data['p_data'].shape[0])

    if dry_run:
        return (
            True,
            f'DRY   {field_dir}: {npy_path.name} -> {npz_path.name} (traces={n_traces}, p_picks={n_picks})',
        )

    np.savez_compressed(npz_path, **data)
    return (
        True,
        f'DONE  {field_dir}: {npy_path.name} -> {npz_path.name} (traces={n_traces}, p_picks={n_picks})',
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Create PSN phase-pick .npz from per-field 1D .npy (P picks only).'
    )
    ap.add_argument(
        '--root',
        type=Path,
        default=Path('/home/dcuser/data/ActiveSeisField'),
        help='Root directory containing field subdirectories.',
    )
    ap.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing .npz if present.',
    )
    ap.add_argument(
        '--dry-run',
        action='store_true',
        help='Print actions without writing files.',
    )
    args = ap.parse_args()

    root: Path = args.root
    if not root.exists():
        raise FileNotFoundError(str(root))
    if not root.is_dir():
        raise NotADirectoryError(str(root))

    field_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not field_dirs:
        raise RuntimeError(f'No field directories found under {root}')

    ok = 0
    skipped = 0
    for d in field_dirs:
        success, msg = process_field_dir(d, force=args.force, dry_run=args.dry_run)
        print(msg)
        if success:
            ok += 1
        else:
            skipped += 1

    print(f'\nSummary: processed={ok}, skipped={skipped}, root={root}')


if __name__ == '__main__':
    main()
