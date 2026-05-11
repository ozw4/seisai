#!/usr/bin/env python3
"""Export fbpick robust.npz physical_center_i to grstat after local phase snap."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np

PickMode = Literal['peak', 'trough', 'rising', 'trailing']


def _load_runtime():
	from seisai_dataset.file_info import open_segy_with_endian
	from seisai_engine.pipelines.fbpick.common import load_robust_npz
	from seisai_pick.pickio.io_grstat import numpy2fbcrd
	from seisai_pick.snap_picks_to_phase import snap_picks_to_phase

	return {
		'open_segy_with_endian': open_segy_with_endian,
		'load_robust_npz': load_robust_npz,
		'numpy2fbcrd': numpy2fbcrd,
		'snap_picks_to_phase': snap_picks_to_phase,
	}


def _choose_offset(candidates: np.ndarray) -> int:
	if candidates.size == 0:
		return 0
	abs_vals = np.abs(candidates)
	best_abs = abs_vals.min()
	best = candidates[abs_vals == best_abs]
	neg = best[best < 0]
	if neg.size:
		return int(neg.max())
	return int(best.min())


def _snap_zero_crossing_windowed(
	picks: np.ndarray,
	seis: np.ndarray,
	*,
	mode: PickMode,
	ltcor: int,
) -> np.ndarray:
	"""Bound rising/trailing zero-crossing snap to +/- ltcor samples."""
	if mode not in ('rising', 'trailing'):
		raise ValueError('mode must be rising or trailing')
	if ltcor < 0:
		raise ValueError('ltcor must be >= 0')
	arr = np.asarray(seis)
	picks_arr = np.asarray(picks)
	if arr.ndim != 2:
		raise ValueError(f'seis must be 2D, got {arr.shape}')
	if picks_arr.ndim != 1 or picks_arr.shape[0] != arr.shape[0]:
		raise ValueError('picks must be 1D with length n_traces')

	n_traces, n_samples = arr.shape
	out = picks_arr.astype(np.int32, copy=True)
	for tr in range(n_traces):
		p0 = int(picks_arr[tr])
		if p0 == 0:
			continue
		if not (0 <= p0 < n_samples):
			raise ValueError(
				f'pick out of bounds: trace={tr}, pick={p0}, n_samples={n_samples}'
			)
		amp = float(arr[tr, p0])
		if amp == 0.0:
			continue

		left = max(0, p0 - ltcor)
		right = min(n_samples - 1, p0 + ltcor)
		center = p0 - left
		win = arr[tr, left : right + 1]

		if mode == 'trailing':
			if amp < 0.0:
				cand = np.flatnonzero(win[: center + 1] >= 0).astype(np.int32) - center
			else:
				cand = np.flatnonzero(win[center:] <= 0).astype(np.int32)
		elif amp > 0.0:  # rising, backward
			cand = np.flatnonzero(win[: center + 1] <= 0).astype(np.int32) - center
		else:  # rising, forward
			cand = np.flatnonzero(win[center:] >= 0).astype(np.int32)

		out[tr] = np.int32(p0 + _choose_offset(cand))
	return out


def _read_trace_subset(f, trace_indices: np.ndarray) -> np.ndarray:
	return np.stack([np.asarray(f.trace.raw[int(i)]) for i in trace_indices], axis=0)


def _sorted_unique_int(values: np.ndarray) -> list[int]:
	return sorted(int(v) for v in np.unique(np.asarray(values, dtype=np.int64)))


def _build_fb_matrix(
	*,
	picks_i: np.ndarray,
	ffid_values: np.ndarray,
	chno_values: np.ndarray,
	duplicate_policy: str,
) -> tuple[np.ndarray, list[int], dict[str, int]]:
	ffid = np.asarray(ffid_values, dtype=np.int64)
	chno = np.asarray(chno_values, dtype=np.int64)
	picks = np.asarray(picks_i, dtype=np.int32)
	if picks.ndim != 1 or ffid.shape != picks.shape or chno.shape != picks.shape:
		raise ValueError(
			'picks_i, ffid_values, chno_values must be 1D arrays of the same length'
		)
	if np.any(chno < 1):
		raise ValueError('chno_values must be >= 1 for grstat export')

	ffids_sorted = _sorted_unique_int(ffid)
	ffid_to_row = {v: i for i, v in enumerate(ffids_sorted)}
	max_chno = int(np.max(chno)) if chno.size else 0
	fb_mat = np.zeros((len(ffids_sorted), max_chno), dtype=np.int32)
	seen: dict[tuple[int, int], int] = {}
	duplicate_count = 0
	duplicate_overwritten = 0

	for i, (ff, cn, pick) in enumerate(zip(ffid, chno, picks, strict=True)):
		key = (int(ff), int(cn))
		if key in seen:
			duplicate_count += 1
			if duplicate_policy == 'error':
				raise ValueError(
					f'duplicate ffid/chno pair {key}: robust rows {seen[key]} and {i}'
				)
			if duplicate_policy == 'first':
				continue
			duplicate_overwritten += 1
		seen[key] = i
		fb_mat[ffid_to_row[int(ff)], int(cn) - 1] = int(pick)

	return (
		fb_mat,
		ffids_sorted,
		{
			'n_gathers': len(ffids_sorted),
			'max_chno': max_chno,
			'duplicate_ffid_chno_count': duplicate_count,
			'duplicate_overwritten_count': duplicate_overwritten,
		},
	)


def _status_counts(arr: np.ndarray | None) -> str:
	if arr is None:
		return ''
	c = Counter(np.asarray(arr).reshape(-1).astype(np.int64).tolist())
	return '; '.join(f'{k}={v}' for k, v in sorted(c.items()))


def run(args: argparse.Namespace) -> None:
	rt = _load_runtime()
	segy_path = Path(args.segy).expanduser().resolve()
	robust_path = Path(args.robust_npz).expanduser().resolve()
	out_crd = Path(args.out_crd).expanduser().resolve()
	out_npz = Path(args.out_npz).expanduser().resolve() if args.out_npz else None
	out_crd.parent.mkdir(parents=True, exist_ok=True)
	if out_npz is not None:
		out_npz.parent.mkdir(parents=True, exist_ok=True)

	robust = rt['load_robust_npz'](robust_path)
	if args.pick_key not in robust:
		raise KeyError(f'robust npz does not contain pick key: {args.pick_key}')

	picks_in = np.asarray(robust[args.pick_key], dtype=np.int32)
	ffid = np.asarray(robust['ffid_values'], dtype=np.int32)
	chno = np.asarray(robust['chno_values'], dtype=np.int32)
	trace_indices = np.asarray(
		robust.get('trace_indices', np.arange(picks_in.size)), dtype=np.int64
	)
	dt_sec = float(np.asarray(robust['dt_sec']).item())
	n_traces = int(np.asarray(robust['n_traces']).item())
	n_samples = int(np.asarray(robust['n_samples_orig']).item())

	if picks_in.shape != (n_traces,):
		raise ValueError(
			f'{args.pick_key} shape mismatch: {picks_in.shape} != {(n_traces,)}'
		)
	if trace_indices.shape != (n_traces,):
		raise ValueError(
			f'trace_indices shape mismatch: {trace_indices.shape} != {(n_traces,)}'
		)
	if np.any(picks_in < 0) or np.any(picks_in >= n_samples):
		raise ValueError(f'{args.pick_key} contains out-of-bounds picks')

	picks_out = picks_in.astype(np.int32, copy=True)
	with rt['open_segy_with_endian'](
		str(segy_path), 'r', ignore_geometry=True, segy_endian=args.endian
	) as f:
		f.mmap()
		if args.strict_trace_count and int(f.tracecount) <= int(np.max(trace_indices)):
			raise ValueError(
				f'SEG-Y tracecount={int(f.tracecount)} is not enough for max trace_indices={int(np.max(trace_indices))}'
			)
		if args.strict_sample_count and len(np.asarray(f.trace.raw[0])) != n_samples:
			raise ValueError(
				f'SEG-Y n_samples={len(np.asarray(f.trace.raw[0]))} != robust n_samples_orig={n_samples}'
			)

		for ff in _sorted_unique_int(ffid):
			idx = np.flatnonzero(ffid == ff)
			order = np.lexsort((idx, chno[idx].astype(np.int64)))
			idx_sorted = idx[order]
			seis_g = _read_trace_subset(f, trace_indices[idx_sorted])
			p_g = picks_in[idx_sorted]
			if (
				args.phase_mode in ('rising', 'trailing')
				and not args.unbounded_zero_crossing
			):
				snapped_g = _snap_zero_crossing_windowed(
					p_g, seis_g, mode=args.phase_mode, ltcor=int(args.max_shift_samples)
				)
			else:
				snapped_g = rt['snap_picks_to_phase'](
					p_g, seis_g, mode=args.phase_mode, ltcor=int(args.max_shift_samples)
				)
			picks_out[idx_sorted] = snapped_g.astype(np.int32, copy=False)

	delta = picks_out.astype(np.int64) - picks_in.astype(np.int64)
	if np.any(np.abs(delta) > int(args.max_shift_samples)) and not (
		args.phase_mode in ('rising', 'trailing') and args.unbounded_zero_crossing
	):
		raise RuntimeError('snap delta exceeded max_shift_samples')

	fb_mat, ffids_sorted, mat_stats = _build_fb_matrix(
		picks_i=picks_out,
		ffid_values=ffid,
		chno_values=chno,
		duplicate_policy=args.duplicate_policy,
	)
	dt_multiplier = (
		float(args.dt_multiplier) if args.dt_multiplier is not None else dt_sec * 1000.0
	)
	written = rt['numpy2fbcrd'](
		dt=dt_multiplier,
		fbnum=fb_mat,
		gather_range=ffids_sorted,
		output_name=str(out_crd),
		original=None,
		mode='gather',
		header_comment=args.header_comment,
	)

	changed = delta != 0
	summary = {
		'segy': str(segy_path),
		'robust_npz': str(robust_path),
		'out_crd': str(out_crd),
		'out_npz': str(out_npz) if out_npz is not None else '',
		'pick_key': args.pick_key,
		'phase_mode': args.phase_mode,
		'max_shift_samples': int(args.max_shift_samples),
		'dt_multiplier': dt_multiplier,
		'n_traces': n_traces,
		'changed_count': int(np.count_nonzero(changed)),
		'changed_rate': float(np.mean(changed)) if changed.size else float('nan'),
		'delta_abs_p50': float(np.percentile(np.abs(delta[changed]), 50))
		if np.any(changed)
		else 0.0,
		'delta_abs_p90': float(np.percentile(np.abs(delta[changed]), 90))
		if np.any(changed)
		else 0.0,
		'delta_abs_max': int(np.max(np.abs(delta))) if delta.size else 0,
		'physical_model_status_counts': _status_counts(
			robust.get('physical_model_status')
		),
		**mat_stats,
	}

	if out_npz is not None:
		np.savez_compressed(
			out_npz,
			dt_sec=np.asarray(dt_sec, dtype=np.float32),
			dt_multiplier=np.asarray(dt_multiplier, dtype=np.float64),
			n_samples_orig=np.asarray(n_samples, dtype=np.int32),
			n_traces=np.asarray(n_traces, dtype=np.int32),
			ffid_values=ffid.astype(np.int32, copy=False),
			chno_values=chno.astype(np.int32, copy=False),
			trace_indices=trace_indices.astype(np.int64, copy=False),
			pick_key=np.asarray(args.pick_key),
			pick_input_i=picks_in.astype(np.int32, copy=False),
			pick_snapped_i=picks_out.astype(np.int32, copy=False),
			pick_snap_delta_i=delta.astype(np.int32, copy=False),
			pick_snap_changed_mask=changed.astype(np.bool_),
			phase_mode=np.asarray(args.phase_mode),
			max_shift_samples=np.asarray(int(args.max_shift_samples), dtype=np.int32),
			fb_mat_samples=fb_mat.astype(np.int32, copy=False),
			fb_mat_grstat_values=written.astype(np.int32, copy=False),
			gather_range_ffids=np.asarray(ffids_sorted, dtype=np.int32),
			summary_json=np.asarray(
				json.dumps(summary, ensure_ascii=False, sort_keys=True)
			),
		)
	print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--segy', required=True)
	parser.add_argument('--robust-npz', required=True)
	parser.add_argument('--out-crd', required=True)
	parser.add_argument('--out-npz', default=None)
	parser.add_argument('--pick-key', default='physical_center_i')
	parser.add_argument(
		'--phase-mode', choices=['peak', 'trough', 'rising', 'trailing'], default='peak'
	)
	parser.add_argument('--max-shift-samples', type=int, default=2)
	parser.add_argument('--unbounded-zero-crossing', action='store_true')
	parser.add_argument(
		'--dt-multiplier', type=float, default=None, help='default: dt_sec*1000.0'
	)
	parser.add_argument('--endian', choices=['big', 'little'], default='big')
	parser.add_argument('--header-comment', default='physical_center_i snap to phase')
	parser.add_argument(
		'--duplicate-policy', choices=['error', 'first', 'last'], default='error'
	)
	parser.add_argument(
		'--no-strict-trace-count', dest='strict_trace_count', action='store_false'
	)
	parser.add_argument(
		'--no-strict-sample-count', dest='strict_sample_count', action='store_false'
	)
	parser.set_defaults(strict_trace_count=True, strict_sample_count=True)
	args = parser.parse_args(argv)
	if args.max_shift_samples < 0:
		parser.error('--max-shift-samples must be >= 0')
	run(args)


if __name__ == '__main__':
	main()
