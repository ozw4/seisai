from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import segyio


def _find_repo_root(start: Path) -> Path:
	start = start.resolve()
	for base in (start, *start.parents):
		packages_dir = base / 'packages'
		if not packages_dir.is_dir():
			continue
		utils_src = packages_dir / 'seisai-utils' / 'src'
		dataset_src = packages_dir / 'seisai-dataset' / 'src'
		if utils_src.is_dir() and dataset_src.is_dir():
			return base
	raise FileNotFoundError(
		f'could not find repo root from {start}; expected packages/seisai-utils/src and '
		'packages/seisai-dataset/src'
	)


def _bootstrap_repo_imports() -> None:
	repo_root = _find_repo_root(Path(__file__).resolve())
	src_roots = (
		repo_root / 'packages' / 'seisai-utils' / 'src',
		repo_root / 'packages' / 'seisai-dataset' / 'src',
	)
	for src_root in reversed(src_roots):
		sys.path.insert(0, str(src_root))


_bootstrap_repo_imports()

from seisai_dataset.phase_pick_io import csr_first_positive, load_phase_pick_csr_npz
from seisai_utils.listfiles import load_path_listfile

_DENSE_NPZ_KEYS = ('fb', 'fb_idx', 'arr_0')
_PHASE_PICK_NPZ_KEYS = ('p_indptr', 'p_data', 's_indptr', 's_data')


class FbLoadMode:
	DENSE = 'dense'
	PHASE_PICK_CSR = 'phase_pick_csr'


def _deduce_hfull_and_axis(
	fb: np.ndarray | None,
	nshot: int,
	chnos: np.ndarray,
) -> tuple[int, int, bool]:
	"""Infer full channel count and channel base.

	Returns
	-------
	hfull, ch_base, used_header_fallback

	"""
	ch_min = int(np.nanmin(chnos)) if chnos.size else 1
	ch_base = 1 if ch_min == 1 else (0 if ch_min == 0 else 1)

	hfull: int | None = None
	if isinstance(fb, np.ndarray) and fb.ndim == 2:
		if fb.shape[0] == nshot:
			hfull = int(fb.shape[1])
		elif fb.shape[1] == nshot:
			hfull = int(fb.shape[0])

	if hfull is not None:
		return hfull, ch_base, False

	ch_max = int(np.nanmax(chnos)) if chnos.size else ch_base
	hfull = max(1, ch_max - ch_base + 1)
	return hfull, ch_base, True


def _build_trace_index(ffids: np.ndarray, chnos: np.ndarray) -> dict[int, int]:
	ffids64 = ffids.astype(np.int64, copy=False)
	chnos64 = chnos.astype(np.int64, copy=False)
	key = (ffids64 << 32) | (chnos64 & 0xFFFFFFFF)
	idx = np.arange(ffids64.size, dtype=np.int64)
	return dict(zip(key.tolist(), idx.tolist(), strict=True))


def _read_npz_first_available(
	z: np.lib.npyio.NpzFile, keys: tuple[str, ...]
) -> np.ndarray | None:
	for key in keys:
		if key in z.files:
			return np.asarray(z[key])
	return None


def _load_fb_array(fb_path: Path) -> tuple[np.ndarray, str]:
	"""Load FB labels from .npy or .npz.

	Supported .npz formats
	----------------------
	1. Dense arrays under one of: fb, fb_idx, arr_0
	2. PhasePickCSR used in this repo: p_indptr, p_data, s_indptr, s_data
	   In this case the per-trace first positive P pick is extracted.
	"""
	suffix = fb_path.suffix.lower()
	if suffix == '.npy':
		fb = np.load(fb_path, allow_pickle=False)
		if not isinstance(fb, np.ndarray):
			raise TypeError(f'expected ndarray from {fb_path}, got {type(fb).__name__}')
		return fb, FbLoadMode.DENSE

	if suffix != '.npz':
		raise ValueError(f'unsupported fb file extension: {fb_path}')

	with np.load(fb_path, allow_pickle=False) as z:
		fb = _read_npz_first_available(z, _DENSE_NPZ_KEYS)
		if fb is not None:
			return fb, FbLoadMode.DENSE
		if all(key in z.files for key in _PHASE_PICK_NPZ_KEYS):
			picks = load_phase_pick_csr_npz(fb_path)
			return csr_first_positive(
				indptr=picks.p_indptr, data=picks.p_data
			), FbLoadMode.PHASE_PICK_CSR
		keys = ', '.join(sorted(z.files))
		raise ValueError(f'unsupported fb npz keys in {fb_path}: [{keys}]')


def _fb_trace_count_hint(fb_path: Path) -> int:
	"""Return a trace-count hint used for skip judgement."""
	suffix = fb_path.suffix.lower()
	if suffix == '.npy':
		fb = np.load(fb_path, allow_pickle=False)
		if not isinstance(fb, np.ndarray):
			raise TypeError(f'expected ndarray from {fb_path}, got {type(fb).__name__}')
		return int(np.size(fb))

	if suffix != '.npz':
		raise ValueError(f'unsupported fb file extension: {fb_path}')

	with np.load(fb_path, allow_pickle=False) as z:
		fb = _read_npz_first_available(z, _DENSE_NPZ_KEYS)
		if fb is not None:
			return int(np.size(fb))
		if all(key in z.files for key in _PHASE_PICK_NPZ_KEYS):
			picks = load_phase_pick_csr_npz(fb_path)
			return int(picks.n_traces)
		keys = ', '.join(sorted(z.files))
		raise ValueError(f'unsupported fb npz keys in {fb_path}: [{keys}]')


def _lininterp_nan(a: np.ndarray) -> np.ndarray:
	x = np.arange(a.size, dtype=np.float64)
	valid = ~np.isnan(a)
	n_valid = int(valid.sum())
	if n_valid >= 2:
		a[~valid] = np.interp(x[~valid], x[valid], a[valid])
		return a
	if n_valid == 1:
		a[:] = a[valid][0]
	return a


def rectify_one(
	segy_path: str | Path,
	fb_path: str | Path | None = None,
	out_path: str | Path | None = None,
) -> Path:
	segy_path = Path(segy_path).expanduser().resolve()
	if not segy_path.is_file():
		raise FileNotFoundError(segy_path)

	fb: np.ndarray | None = None
	fb_mode: str | None = None
	if fb_path is not None:
		fb_path = Path(fb_path).expanduser().resolve()
		if not fb_path.is_file():
			raise FileNotFoundError(fb_path)
		fb, fb_mode = _load_fb_array(fb_path)

	if out_path is None:
		out_path = segy_path.with_name(f'{segy_path.stem}_rectified.sgy')
	else:
		out_path = Path(out_path).expanduser().resolve()

	with segyio.open(segy_path, 'r', ignore_geometry=True) as src:
		n_traces = int(src.tracecount)
		ns = int(src.bin[segyio.BinField.Samples])
		dt = int(src.bin[segyio.BinField.Interval])
		fmt = int(src.bin[segyio.BinField.Format])

		ffid_f = segyio.TraceField.FieldRecord
		chno_f = segyio.TraceField.TraceNumber
		tic_f = segyio.TraceField.TraceIdentificationCode

		ffids = np.asarray(list(src.attributes(ffid_f)), dtype=np.int64)
		chnos = np.asarray(list(src.attributes(chno_f)), dtype=np.int64)
		u_ff = np.unique(ffids)

		hfull, ch_base, used_header_fallback = _deduce_hfull_and_axis(
			fb, u_ff.size, chnos
		)
		if fb_mode == FbLoadMode.PHASE_PICK_CSR and used_header_fallback:
			print(
				'[WARN] FB .npz is PhasePickCSR; Hfull was inferred from SEG-Y TraceNumber range '
				f'({ch_base}..{ch_base + hfull - 1}).'
			)
		ch_list = np.arange(ch_base, ch_base + hfull, dtype=np.int64)

		trace_index = _build_trace_index(ffids, chnos)

		gx_field = getattr(segyio.TraceField, 'GroupX', None)
		gy_field = getattr(segyio.TraceField, 'GroupY', None)
		interp_xy: dict[int, tuple[np.ndarray | None, np.ndarray | None]] = {}

		for ff in u_ff.tolist():
			gx_arr = (
				np.full(hfull, np.nan, dtype=np.float64)
				if gx_field is not None
				else None
			)
			gy_arr = (
				np.full(hfull, np.nan, dtype=np.float64)
				if gy_field is not None
				else None
			)

			for ch in ch_list.tolist():
				key = (int(ff) << 32) | (int(ch) & 0xFFFFFFFF)
				if key not in trace_index:
					continue
				header = src.header[trace_index[key]]
				pos = int(ch - ch_base)
				if gx_arr is not None:
					gx_arr[pos] = float(header.get(gx_field, np.nan))
				if gy_arr is not None:
					gy_arr[pos] = float(header.get(gy_field, np.nan))

			if gx_arr is not None:
				gx_arr = _lininterp_nan(gx_arr)
			if gy_arr is not None:
				gy_arr = _lininterp_nan(gy_arr)
			interp_xy[int(ff)] = (gx_arr, gy_arr)

		spec = segyio.spec()
		if hasattr(segyio, 'TraceSortingFormat') and hasattr(
			segyio.TraceSortingFormat, 'Unknown'
		):
			spec.sorting = segyio.TraceSortingFormat.Unknown
		spec.format = 5 if fmt not in (1, 2, 3, 5) else fmt
		spec.samples = range(ns)
		spec.ilines = u_ff.astype(np.int32)
		spec.xlines = ch_list.astype(np.int32)

		out_path.parent.mkdir(parents=True, exist_ok=True)
		with segyio.create(str(out_path), spec) as dst:
			if len(src.text) > 0:
				for i, block in enumerate(src.text):
					dst.text[i] = block

			dst.bin = src.bin
			dst.bin = src.bin
			dst.bin[segyio.BinField.Samples] = ns
			dst.bin[segyio.BinField.Interval] = dt
			dst.bin[segyio.BinField.Format] = spec.format

			out_idx = 0
			for ff in u_ff.tolist():
				gx_arr, gy_arr = interp_xy[int(ff)]
				template_header: dict[int, int] | None = None

				for ch in ch_list.tolist():
					key = (int(ff) << 32) | (int(ch) & 0xFFFFFFFF)
					if key in trace_index:
						template_header = dict(src.header[trace_index[key]])
						break

				if template_header is None:
					raise RuntimeError(
						f'no template trace found for ffid={ff} in {segy_path}'
					)

				for ch in ch_list.tolist():
					key = (int(ff) << 32) | (int(ch) & 0xFFFFFFFF)
					if key in trace_index:
						src_idx = trace_index[key]
						trace = src.trace[src_idx]
						header = dict(src.header[src_idx])
					else:
						trace = np.zeros(ns, dtype=np.float32)
						header = dict(template_header)
						header[tic_f] = 2

					header[ffid_f] = int(ff)
					header[chno_f] = int(ch)
					header[segyio.TraceField.TRACE_SAMPLE_COUNT] = ns
					header[segyio.TraceField.TRACE_SAMPLE_INTERVAL] = dt

					pos = int(ch - ch_base)
					if (
						gx_field is not None
						and gx_arr is not None
						and not np.isnan(gx_arr[pos])
					):
						header[gx_field] = int(round(gx_arr[pos]))
					if (
						gy_field is not None
						and gy_arr is not None
						and not np.isnan(gy_arr[pos])
					):
						header[gy_field] = int(round(gy_arr[pos]))

					dst.trace[out_idx] = trace
					dst.header[out_idx] = header
					out_idx += 1

	print(
		f'[OK] {segy_path.name}: {n_traces} -> {u_ff.size * hfull} traces '
		f'(shots={u_ff.size}, Hfull={hfull})'
	)
	return out_path


def rectify_many(
	segy_listfile: str | Path,
	fb_listfile: str | Path,
) -> list[Path]:
	segy_files = load_path_listfile(segy_listfile)
	fb_files = load_path_listfile(fb_listfile)

	if len(segy_files) != len(fb_files):
		raise ValueError(
			f'list length mismatch: len(segy_files)={len(segy_files)} '
			f'!= len(fb_files)={len(fb_files)}'
		)

	outputs: list[Path] = []
	for segy_file, fb_file in zip(segy_files, fb_files, strict=True):
		segy_path = Path(segy_file).expanduser().resolve()
		fb_path = Path(fb_file).expanduser().resolve()
		if not segy_path.is_file():
			raise FileNotFoundError(segy_path)
		if not fb_path.is_file():
			raise FileNotFoundError(fb_path)

		with segyio.open(segy_path, 'r', ignore_geometry=True) as src:
			n_traces = int(src.tracecount)

		fb_trace_count = _fb_trace_count_hint(fb_path)
		if fb_trace_count == n_traces:
			print(
				f'[SKIP] {segy_path.name}: FB trace count ({fb_trace_count}) == '
				f'SEGY traces ({n_traces}) -> already aligned'
			)
			continue

		out_path = rectify_one(
			segy_path=segy_path,
			fb_path=fb_path,
			out_path=segy_path.with_name(f'{segy_path.stem}_rectified.sgy'),
		)
		outputs.append(out_path)
		print(f' -> {out_path}')

	return outputs


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser(
		description='Insert dead traces into shot gathers so each shot has a full channel grid.',
	)
	subparsers = parser.add_subparsers(dest='command', required=True)

	one_parser = subparsers.add_parser('one', help='Rectify one SEG-Y file.')
	one_parser.add_argument('--segy', required=True, type=Path)
	one_parser.add_argument('--fb', type=Path, default=None)
	one_parser.add_argument('--out', type=Path, default=None)

	many_parser = subparsers.add_parser(
		'many',
		help='Rectify a paired SEG-Y/FB list using current repo-style listfiles.',
	)
	many_parser.add_argument('--segy-list', required=True, type=Path)
	many_parser.add_argument('--fb-list', required=True, type=Path)

	args = parser.parse_args(argv)

	if args.command == 'one':
		rectify_one(segy_path=args.segy, fb_path=args.fb, out_path=args.out)
		return

	if args.command == 'many':
		rectify_many(segy_listfile=args.segy_list, fb_listfile=args.fb_list)
		return

	raise RuntimeError(f'unsupported command: {args.command}')


if __name__ == '__main__':
	main()
