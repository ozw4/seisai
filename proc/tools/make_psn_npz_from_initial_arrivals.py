#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEISAI_UTILS_SRC = _REPO_ROOT / 'packages' / 'seisai-utils' / 'src'
if _SEISAI_UTILS_SRC.is_dir() and str(_SEISAI_UTILS_SRC) not in sys.path:
	sys.path.insert(0, str(_SEISAI_UTILS_SRC))


def _load_module_attr(module_path: Path, attr: str):
	spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f'failed to create spec for {module_path}')
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return getattr(module, attr)


validate_phase_pick_csr = _load_module_attr(
	_REPO_ROOT
	/ 'packages'
	/ 'seisai-dataset'
	/ 'src'
	/ 'seisai_dataset'
	/ 'phase_pick_io.py',
	'validate_phase_pick_csr',
)
from seisai_utils.listfiles import load_path_listfile


def build_psn_npz_from_picks(picks_1d: np.ndarray) -> dict[str, np.ndarray]:
	picks = np.asarray(picks_1d)
	if picks.ndim != 1:
		raise ValueError(f'expected 1D array, got shape={picks.shape}')

	n_traces = int(picks.shape[0])
	if np.issubdtype(picks.dtype, np.integer):
		valid = picks > 0
		p_data = picks[valid].astype(np.int64, copy=False)
	elif np.issubdtype(picks.dtype, np.floating):
		finite = np.isfinite(picks)
		positive = picks > 0
		is_int = picks == np.floor(picks)
		valid = finite & positive & is_int
		p_data = picks[valid].astype(np.int64, copy=False)
	else:
		raise ValueError(f'unsupported dtype: {picks.dtype}')

	p_indptr = np.empty(n_traces + 1, dtype=np.int64)
	p_indptr[0] = 0
	np.cumsum(valid.astype(np.int64, copy=False), out=p_indptr[1:])

	s_indptr = np.zeros(n_traces + 1, dtype=np.int64)
	s_data = np.empty(0, dtype=np.int64)

	validate_phase_pick_csr(
		p_indptr=p_indptr,
		p_data=p_data,
		s_indptr=s_indptr,
		s_data=s_data,
	)
	return {
		'p_indptr': p_indptr,
		'p_data': p_data,
		's_indptr': s_indptr,
		's_data': s_data,
	}


def convert_one(
	*,
	npy_path: Path,
	out_path: Path | None = None,
	force: bool = False,
) -> Path:
	npy_path = Path(npy_path)
	if not npy_path.exists():
		raise FileNotFoundError(npy_path)
	if not npy_path.is_file():
		raise ValueError(f'expected file: {npy_path}')
	if npy_path.suffix.lower() != '.npy':
		raise ValueError(f'expected .npy file: {npy_path}')

	if out_path is None:
		out_path = npy_path.with_suffix('.npz')
	out_path = Path(out_path)
	if out_path.exists() and not force:
		raise FileExistsError(f'already exists: {out_path} (use --force to overwrite)')
	out_path.parent.mkdir(parents=True, exist_ok=True)

	arr = np.load(npy_path, allow_pickle=False)
	data = build_psn_npz_from_picks(arr)
	np.savez_compressed(out_path, **data)

	n_traces = int(arr.shape[0])
	n_picks = int(data['p_data'].shape[0])
	print(
		f'[DONE] {npy_path.name} -> {out_path} '
		f'(traces={n_traces}, p_picks={n_picks}, invalid={n_traces - n_picks})'
	)
	return out_path


def convert_many(
	*,
	npy_list: Path,
	out_dir: Path | None = None,
	force: bool = False,
) -> list[Path]:
	npy_paths = [Path(p) for p in load_path_listfile(npy_list)]
	outputs: list[Path] = []
	for npy_path in npy_paths:
		out_path = (
			(Path(out_dir) / f'{npy_path.stem}.npz') if out_dir is not None else None
		)
		outputs.append(convert_one(npy_path=npy_path, out_path=out_path, force=force))
	return outputs


def build_parser() -> argparse.ArgumentParser:
	ap = argparse.ArgumentParser(
		description='Convert simple 1D initial-arrivals .npy to phase-pick CSR .npz (P only).'
	)
	sub = ap.add_subparsers(dest='cmd', required=True)

	one = sub.add_parser('one', help='Convert one .npy to one CSR .npz.')
	one.add_argument(
		'--npy', type=Path, required=True, help='Input 1D pick array .npy file.'
	)
	one.add_argument(
		'--out',
		type=Path,
		default=None,
		help='Output .npz path. Default: <input_stem>.npz',
	)
	one.add_argument(
		'--force', action='store_true', help='Overwrite existing output .npz.'
	)

	many = sub.add_parser('many', help='Convert all .npy files listed in a listfile.')
	many.add_argument(
		'--npy-list',
		type=Path,
		required=True,
		help='Listfile of input .npy files, one path per line.',
	)
	many.add_argument(
		'--out-dir',
		type=Path,
		default=None,
		help='Optional directory to collect outputs. Default: next to each input file.',
	)
	many.add_argument(
		'--force', action='store_true', help='Overwrite existing output .npz files.'
	)
	return ap


def main() -> None:
	args = build_parser().parse_args()

	if args.cmd == 'one':
		convert_one(
			npy_path=args.npy,
			out_path=args.out,
			force=bool(args.force),
		)
		return

	if args.cmd == 'many':
		out_dir = Path(args.out_dir) if args.out_dir is not None else None
		outputs = convert_many(
			npy_list=args.npy_list,
			out_dir=out_dir,
			force=bool(args.force),
		)
		print(f'\nSummary: converted={len(outputs)}')
		return

	raise ValueError(f'unknown command: {args.cmd}')


if __name__ == '__main__':
	main()
