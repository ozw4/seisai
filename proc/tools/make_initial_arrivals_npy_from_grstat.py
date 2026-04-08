#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import segyio

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEISAI_PICK_SRC = _REPO_ROOT / 'packages' / 'seisai-pick' / 'src'
_SEISAI_UTILS_SRC = _REPO_ROOT / 'packages' / 'seisai-utils' / 'src'
for _p in (_SEISAI_PICK_SRC, _SEISAI_UTILS_SRC):
	if _p.is_dir() and str(_p) not in sys.path:
		sys.path.insert(0, str(_p))

from seisai_pick.pickio.io_grstat import load_fb_irasformat
from seisai_utils.listfiles import load_path_listfile


def _read_dt_ms_from_segy(segy_path: Path) -> float:
	segy_path = Path(segy_path)
	if not segy_path.exists():
		raise FileNotFoundError(segy_path)
	if not segy_path.is_file():
		raise ValueError(f'expected file: {segy_path}')

	with segyio.open(str(segy_path), 'r', ignore_geometry=True) as src:
		dt_us = int(src.bin[segyio.BinField.Interval])
	if dt_us <= 0:
		raise ValueError(f'invalid SEG-Y sample interval: {dt_us} us in {segy_path}')
	return float(dt_us) / 1000.0


def _resolve_dt(*, dt: float | None, segy_path: Path | None) -> float:
	if (dt is None) == (segy_path is None):
		raise ValueError('specify exactly one of dt or segy_path')
	if dt is not None:
		if dt <= 0:
			raise ValueError(f'dt must be > 0, got {dt}')
		return float(dt)
	return _read_dt_ms_from_segy(Path(segy_path))


def _common_parent(paths: list[Path]) -> Path:
	if not paths:
		raise ValueError('empty path list')
	parents = [str(Path(p).resolve().parent) for p in paths]
	return Path(os.path.commonpath(parents))


def _validate_paired_directory_layout(
	*,
	grstat_paths: list[Path],
	sgy_paths: list[Path],
) -> tuple[Path, Path]:
	if len(grstat_paths) != len(sgy_paths):
		raise ValueError(
			f'list length mismatch: len(grstat_list)={len(grstat_paths)} != '
			f'len(sgy_list)={len(sgy_paths)}'
		)

	grstat_root = _common_parent(grstat_paths)
	sgy_root = _common_parent(sgy_paths)

	for i, (grstat_path, sgy_path) in enumerate(zip(grstat_paths, sgy_paths), start=1):
		grstat_parent = Path(grstat_path).resolve().parent
		sgy_parent = Path(sgy_path).resolve().parent
		grstat_rel_parent = grstat_parent.relative_to(grstat_root)
		sgy_rel_parent = sgy_parent.relative_to(sgy_root)
		if grstat_rel_parent != sgy_rel_parent:
			raise ValueError(
				'directory layout mismatch between --grstat-list and --sgy-list: '
				f'line={i}, '
				f'grstat_parent={grstat_parent}, sgy_parent={sgy_parent}, '
				f'grstat_rel_parent={grstat_rel_parent}, sgy_rel_parent={sgy_rel_parent}, '
				f'grstat_root={grstat_root}, sgy_root={sgy_root}'
			)

	print(
		'[CHECK] paired directory layout OK: '
		f'grstat_root={grstat_root}, sgy_root={sgy_root}, pairs={len(grstat_paths)}'
	)
	return grstat_root, sgy_root


def convert_one(
	*,
	grstat_path: Path,
	dt: float | None = None,
	segy_path: Path | None = None,
	out_path: Path | None = None,
	strict: bool = True,
	verbose: bool = True,
) -> Path:
	grstat_path = Path(grstat_path)
	if not grstat_path.exists():
		raise FileNotFoundError(grstat_path)
	if not grstat_path.is_file():
		raise ValueError(f'expected file: {grstat_path}')

	resolved_dt = _resolve_dt(dt=dt, segy_path=segy_path)

	if out_path is None:
		out_path = grstat_path.with_suffix('.npy')
	out_path = Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	picks_1d = load_fb_irasformat(
		str(grstat_path),
		resolved_dt,
		strict=bool(strict),
		verbose=bool(verbose),
	)
	np.save(out_path, np.asarray(picks_1d, dtype=np.int32))

	n_traces = int(picks_1d.size)
	n_valid = int(np.count_nonzero(picks_1d > 0))
	dt_src = f'{resolved_dt:g} ms'
	if segy_path is not None:
		dt_src += f' from {Path(segy_path).name}'
	print(
		f'[DONE] {grstat_path.name} -> {out_path} '
		f'(dt={dt_src}, traces={n_traces}, valid={n_valid}, invalid={n_traces - n_valid})'
	)
	return out_path


def convert_many(
	*,
	grstat_list: Path,
	dt: float | None = None,
	sgy_list: Path | None = None,
	out_dir: Path | None = None,
	strict: bool = True,
	verbose: bool = True,
) -> list[Path]:
	grstat_paths = [Path(p) for p in load_path_listfile(grstat_list)]
	sgy_paths: list[Path] | None = None
	if sgy_list is not None:
		sgy_paths = [Path(p) for p in load_path_listfile(sgy_list)]
		_validate_paired_directory_layout(
			grstat_paths=grstat_paths, sgy_paths=sgy_paths
		)

	outputs: list[Path] = []
	for i, grstat_path in enumerate(grstat_paths):
		out_path = (
			(Path(out_dir) / f'{grstat_path.stem}.npy') if out_dir is not None else None
		)
		segy_path = None if sgy_paths is None else sgy_paths[i]
		outputs.append(
			convert_one(
				grstat_path=grstat_path,
				dt=dt,
				segy_path=segy_path,
				out_path=out_path,
				strict=strict,
				verbose=verbose,
			)
		)
	return outputs


def build_parser() -> argparse.ArgumentParser:
	ap = argparse.ArgumentParser(
		description='Convert grstat first-break text dump to simple 1D pick array .npy.'
	)
	sub = ap.add_subparsers(dest='cmd', required=True)

	one = sub.add_parser('one', help='Convert one grstat text file to one .npy.')
	one.add_argument(
		'--grstat', type=Path, required=True, help='Input grstat text file.'
	)
	one_dt = one.add_mutually_exclusive_group(required=True)
	one_dt.add_argument(
		'--dt',
		type=float,
		default=None,
		help='Sampling interval in ms used in the grstat file. Example: 1.0 or 0.5.',
	)
	one_dt.add_argument(
		'--sgy',
		type=Path,
		default=None,
		help='SEG-Y file to read the sampling interval from. BinField.Interval is treated as microseconds and converted to ms.',
	)
	one.add_argument(
		'--out',
		type=Path,
		default=None,
		help='Output .npy path. Default: <input_stem>.npy',
	)
	one.add_argument(
		'--no-strict',
		action='store_true',
		help='Disable contiguous-block / channel-count validation.',
	)
	one.add_argument(
		'--quiet',
		action='store_true',
		help='Suppress parser progress prints from load_fb_irasformat.',
	)

	many = sub.add_parser('many', help='Convert all grstat files in a listfile.')
	many.add_argument(
		'--grstat-list',
		type=Path,
		required=True,
		help='Listfile of grstat text files, one path per line.',
	)
	many_dt = many.add_mutually_exclusive_group(required=True)
	many_dt.add_argument(
		'--dt',
		type=float,
		default=None,
		help='Sampling interval in ms used in the grstat files. Example: 1.0 or 0.5.',
	)
	many_dt.add_argument(
		'--sgy-list',
		type=Path,
		default=None,
		help='Listfile of SEG-Y files, paired with --grstat-list line by line. Each SEG-Y BinField.Interval is used as dt.',
	)
	many.add_argument(
		'--out-dir',
		type=Path,
		default=None,
		help='Optional directory to collect outputs. Default: next to each input file.',
	)
	many.add_argument(
		'--no-strict',
		action='store_true',
		help='Disable contiguous-block / channel-count validation.',
	)
	many.add_argument(
		'--quiet',
		action='store_true',
		help='Suppress parser progress prints from load_fb_irasformat.',
	)
	return ap


def main() -> None:
	args = build_parser().parse_args()
	strict = not bool(args.no_strict)
	verbose = not bool(args.quiet)

	if args.cmd == 'one':
		convert_one(
			grstat_path=args.grstat,
			dt=args.dt,
			segy_path=args.sgy,
			out_path=args.out,
			strict=strict,
			verbose=verbose,
		)
		return

	if args.cmd == 'many':
		out_dir = Path(args.out_dir) if args.out_dir is not None else None
		outputs = convert_many(
			grstat_list=args.grstat_list,
			dt=args.dt,
			sgy_list=args.sgy_list,
			out_dir=out_dir,
			strict=strict,
			verbose=verbose,
		)
		print(f'\nSummary: converted={len(outputs)}')
		return

	raise ValueError(f'unknown command: {args.cmd}')


if __name__ == '__main__':
	main()
