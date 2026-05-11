"""Export fbpick robust picks to grstat and optionally evaluate them."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np

PickMode = Literal['peak', 'trough', 'rising', 'trailing']
DuplicatePolicy = Literal['error', 'first', 'last']


class GrstatMatrix(NamedTuple):
    """Parsed grstat first-break matrix in sample-index domain."""

    record_numbers: np.ndarray
    samples: np.ndarray
    raw_values: np.ndarray


def _load_runtime():
    from seisai_dataset.file_info import open_segy_with_endian
    from seisai_engine.pipelines.fbpick.common import load_robust_npz
    from seisai_pick.pickio.io_grstat import numpy2fbcrd
    from seisai_pick.snap_picks_to_phase import snap_picks_to_phase

    return {
        'load_robust_npz': load_robust_npz,
        'numpy2fbcrd': numpy2fbcrd,
        'open_segy_with_endian': open_segy_with_endian,
        'snap_picks_to_phase': snap_picks_to_phase,
    }


def _parse_int_or_none(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    if s[0] == '-':
        return int(s) if s[1:].isdigit() else None
    return int(s) if s.isdigit() else None


def _iter_fixed_width(payload: str, width: int) -> list[str]:
    return [payload[i : i + width] for i in range(0, len(payload), width)]


def load_grstat_matrix(
    path: str | Path,
    *,
    dt_multiplier: float,
    strict_blocks: bool = True,
) -> GrstatMatrix:
    """Load grstat CRD text into a 2D sample-index matrix.

    This parser preserves ``rec.no.`` so prediction rows can be aligned by FFID
    during evaluation.
    """
    if dt_multiplier <= 0:
        msg = 'dt_multiplier must be > 0'
        raise ValueError(msg)

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)

    records: list[tuple[int, dict[int, int]]] = []
    current_rec: int | None = None
    current_values: dict[int, int] = {}
    expected_next_start = 1

    def _flush_record() -> None:
        nonlocal current_rec, current_values
        if current_rec is not None:
            records.append((current_rec, dict(current_values)))
        current_rec = None
        current_values = {}

    with p.open('r', encoding='utf-8', errors='replace') as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.rstrip('\n')
            if line.startswith('* rec.no.='):
                _flush_record()
                rec = _parse_int_or_none(line[len('* rec.no.=') :])
                if rec is None:
                    msg = f'invalid rec.no. at line {line_no}: {line!r}'
                    raise ValueError(msg)
                current_rec = rec
                expected_next_start = 1
                continue

            if not line.startswith('fb'):
                continue

            if current_rec is None:
                msg = f'fb line appears before rec.no. at line {line_no}'
                raise ValueError(msg)
            if len(line) < 20:
                msg = f'fb line too short at line {line_no}: {line!r}'
                raise ValueError(msg)

            start = _parse_int_or_none(line[2:15])
            end = _parse_int_or_none(line[15:20])
            if start is None or end is None or end < start:
                msg = f'invalid fb start/end at line {line_no}: {line!r}'
                raise ValueError(msg)
            if strict_blocks and start != expected_next_start:
                msg = (
                    f'non-contiguous fb block at line {line_no}: '
                    f'start={start}, expected={expected_next_start}'
                )
                raise ValueError(msg)
            expected_next_start = end + 1

            chunks = _iter_fixed_width(line[20:], 5)
            expected_count = end - start + 1
            if len(chunks) != expected_count:
                msg = (
                    f'fb value count mismatch at line {line_no}: '
                    f'{len(chunks)} != {expected_count}'
                )
                raise ValueError(msg)

            for offset, chunk in enumerate(chunks):
                value = _parse_int_or_none(chunk)
                if value is None:
                    msg = f'invalid fb value at line {line_no}: {chunk!r}'
                    raise ValueError(msg)
                current_values[start + offset] = value

    _flush_record()

    if not records:
        msg = f'no records found in grstat file: {p}'
        raise ValueError(msg)

    max_chno = max((max(vals) if vals else 0) for _, vals in records)
    record_numbers = np.asarray([rec for rec, _ in records], dtype=np.int32)
    raw_values = np.zeros((len(records), max_chno), dtype=np.int32)

    for row, (_, values) in enumerate(records):
        for chno, value in values.items():
            if chno < 1:
                msg = f'invalid channel number {chno} in {p}'
                raise ValueError(msg)
            raw_values[row, chno - 1] = int(value)

    samples = np.floor(raw_values.astype(np.float64) / float(dt_multiplier)).astype(
        np.int32
    )
    samples[raw_values <= 0] = 0
    samples[raw_values == -9999] = 0
    samples[samples < 0] = 0

    return GrstatMatrix(
        record_numbers=record_numbers,
        samples=samples,
        raw_values=raw_values,
    )


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
        msg = 'mode must be rising or trailing'
        raise ValueError(msg)
    if ltcor < 0:
        msg = 'ltcor must be >= 0'
        raise ValueError(msg)

    arr = np.asarray(seis)
    picks_arr = np.asarray(picks)
    if arr.ndim != 2:
        msg = f'seis must be 2D, got {arr.shape}'
        raise ValueError(msg)
    if picks_arr.ndim != 1 or picks_arr.shape[0] != arr.shape[0]:
        msg = 'picks must be 1D with length n_traces'
        raise ValueError(msg)

    n_traces, n_samples = arr.shape
    out = picks_arr.astype(np.int32, copy=True)
    for tr in range(n_traces):
        p0 = int(picks_arr[tr])
        if p0 == 0:
            continue
        if not (0 <= p0 < n_samples):
            msg = f'pick out of bounds: trace={tr}, pick={p0}, n_samples={n_samples}'
            raise ValueError(msg)

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
        elif amp > 0.0:
            cand = np.flatnonzero(win[: center + 1] <= 0).astype(np.int32) - center
        else:
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
    duplicate_policy: DuplicatePolicy,
) -> tuple[np.ndarray, list[int], dict[str, int]]:
    ffid = np.asarray(ffid_values, dtype=np.int64)
    chno = np.asarray(chno_values, dtype=np.int64)
    picks = np.asarray(picks_i, dtype=np.int32)
    if picks.ndim != 1 or ffid.shape != picks.shape or chno.shape != picks.shape:
        msg = 'picks_i, ffid_values, chno_values must be 1D arrays of the same length'
        raise ValueError(msg)
    if np.any(chno < 1):
        msg = 'chno_values must be >= 1 for grstat export'
        raise ValueError(msg)

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
                msg = f'duplicate ffid/chno pair {key}: robust rows {seen[key]} and {i}'
                raise ValueError(msg)
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


def _pct(x: np.ndarray, q: float) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan')
    return float(np.percentile(arr, q))


def _align_reference_samples(
    *,
    reference: GrstatMatrix,
    prediction_ffids: list[int],
    prediction_max_chno: int,
    strict_gather_numbers: bool,
    strict_shape: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    pred_ffids = [int(v) for v in prediction_ffids]
    ref_records = [int(v) for v in reference.record_numbers.tolist()]
    ref_by_rec = {int(rec): i for i, rec in enumerate(ref_records)}

    if all(ff in ref_by_rec for ff in pred_ffids):
        row_indices = np.asarray([ref_by_rec[ff] for ff in pred_ffids], dtype=np.int64)
        aligned = reference.samples[row_indices]
        raw_aligned = reference.raw_values[row_indices]
        align_mode = 'record_number'
    elif strict_gather_numbers:
        missing = [ff for ff in pred_ffids if ff not in ref_by_rec]
        msg = (
            'reference grstat rec.no. values do not cover prediction ffids; '
            f'missing first values={missing[:10]}'
        )
        raise ValueError(msg)
    elif len(ref_records) == len(pred_ffids):
        aligned = reference.samples
        raw_aligned = reference.raw_values
        align_mode = 'row_order'
    else:
        msg = (
            'cannot align reference grstat by row order: '
            f'{len(ref_records)} reference rows != {len(pred_ffids)} prediction rows'
        )
        raise ValueError(msg)

    if aligned.shape[1] != prediction_max_chno:
        if strict_shape:
            msg = (
                'reference grstat channel count does not match prediction: '
                f'{aligned.shape[1]} != {prediction_max_chno}'
            )
            raise ValueError(msg)
        if aligned.shape[1] < prediction_max_chno:
            pad_width = prediction_max_chno - aligned.shape[1]
            aligned = np.pad(aligned, ((0, 0), (0, pad_width)), constant_values=0)
            raw_aligned = np.pad(raw_aligned, ((0, 0), (0, pad_width)), constant_values=0)
        else:
            aligned = aligned[:, :prediction_max_chno]
            raw_aligned = raw_aligned[:, :prediction_max_chno]

    return (
        aligned.astype(np.int32, copy=False),
        raw_aligned.astype(np.int32, copy=False),
        align_mode,
    )


def evaluate_grstat_matrix(
    *,
    prediction_samples: np.ndarray,
    prediction_ffids: list[int],
    reference: GrstatMatrix,
    dt_multiplier: float,
    strict_gather_numbers: bool = True,
    strict_shape: bool = True,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate a predicted grstat sample matrix against reference grstat picks."""
    pred = np.asarray(prediction_samples, dtype=np.int32)
    if pred.ndim != 2:
        msg = f'prediction_samples must be 2D, got {pred.shape}'
        raise ValueError(msg)
    if len(prediction_ffids) != pred.shape[0]:
        msg = 'prediction_ffids length must match prediction rows'
        raise ValueError(msg)

    ref, ref_raw, align_mode = _align_reference_samples(
        reference=reference,
        prediction_ffids=prediction_ffids,
        prediction_max_chno=pred.shape[1],
        strict_gather_numbers=strict_gather_numbers,
        strict_shape=strict_shape,
    )

    ref_valid = ref > 0
    pred_valid = pred > 0
    eval_mask = ref_valid & pred_valid
    missing_pred = ref_valid & ~pred_valid

    err = pred.astype(np.int64) - ref.astype(np.int64)
    err_eval = err[eval_mask]
    abs_eval = np.abs(err_eval)
    err_ms = err_eval.astype(np.float64) * float(dt_multiplier)
    abs_ms = abs_eval.astype(np.float64) * float(dt_multiplier)

    summary: dict[str, object] = {
        'alignment_mode': align_mode,
        'dt_multiplier': float(dt_multiplier),
        'n_prediction_gathers': int(pred.shape[0]),
        'n_prediction_channels': int(pred.shape[1]),
        'n_reference_valid': int(np.count_nonzero(ref_valid)),
        'n_prediction_valid': int(np.count_nonzero(pred_valid)),
        'n_eval': int(np.count_nonzero(eval_mask)),
        'n_missing_prediction_at_reference': int(np.count_nonzero(missing_pred)),
        'prediction_valid_at_reference_rate': float(np.mean(pred_valid[ref_valid]))
        if np.any(ref_valid)
        else float('nan'),
    }

    if err_eval.size:
        summary.update(
            {
                'bias_samples_mean': float(np.mean(err_eval)),
                'bias_samples_p50': _pct(err_eval, 50),
                'mae_samples_mean': float(np.mean(abs_eval)),
                'mae_samples_p50': _pct(abs_eval, 50),
                'mae_samples_p90': _pct(abs_eval, 90),
                'mae_samples_p95': _pct(abs_eval, 95),
                'mae_samples_p99': _pct(abs_eval, 99),
                'mae_samples_max': int(np.max(abs_eval)),
                'bias_ms_mean': float(np.mean(err_ms)),
                'bias_ms_p50': _pct(err_ms, 50),
                'mae_ms_mean': float(np.mean(abs_ms)),
                'mae_ms_p50': _pct(abs_ms, 50),
                'mae_ms_p90': _pct(abs_ms, 90),
                'mae_ms_p95': _pct(abs_ms, 95),
                'mae_ms_p99': _pct(abs_ms, 99),
                'mae_ms_max': float(np.max(abs_ms)),
            }
        )
        for tol in (1, 2, 4, 8, 16, 32, 64, 127):
            summary[f'R{tol}'] = float(np.mean(abs_eval <= tol))
    else:
        summary.update(
            {
                'bias_samples_mean': float('nan'),
                'bias_samples_p50': float('nan'),
                'mae_samples_mean': float('nan'),
                'mae_samples_p50': float('nan'),
                'mae_samples_p90': float('nan'),
                'mae_samples_p95': float('nan'),
                'mae_samples_p99': float('nan'),
                'mae_samples_max': float('nan'),
                'bias_ms_mean': float('nan'),
                'bias_ms_p50': float('nan'),
                'mae_ms_mean': float('nan'),
                'mae_ms_p50': float('nan'),
                'mae_ms_p90': float('nan'),
                'mae_ms_p95': float('nan'),
                'mae_ms_p99': float('nan'),
                'mae_ms_max': float('nan'),
            }
        )
        for tol in (1, 2, 4, 8, 16, 32, 64, 127):
            summary[f'R{tol}'] = float('nan')

    rows: list[dict[str, object]] = []
    ffids = np.asarray(prediction_ffids, dtype=np.int32)
    for row_i, ffid in enumerate(ffids):
        valid_cols = np.flatnonzero(ref_valid[row_i])
        for col_i in valid_cols:
            pred_i = int(pred[row_i, col_i])
            ref_i = int(ref[row_i, col_i])
            error_i: int | str = pred_i - ref_i if pred_i > 0 else ''
            rows.append(
                {
                    'ffid': int(ffid),
                    'chno': int(col_i + 1),
                    'reference_sample': ref_i,
                    'reference_grstat_value': int(ref_raw[row_i, col_i]),
                    'prediction_sample': pred_i,
                    'error_samples': error_i,
                    'abs_error_samples': abs(error_i) if error_i != '' else '',
                    'error_ms': float(error_i) * float(dt_multiplier)
                    if error_i != ''
                    else '',
                    'abs_error_ms': abs(float(error_i) * float(dt_multiplier))
                    if error_i != ''
                    else '',
                }
            )

    return summary, rows


def _write_summary_csv(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'ffid',
        'chno',
        'reference_sample',
        'reference_grstat_value',
        'prediction_sample',
        'error_samples',
        'abs_error_samples',
        'error_ms',
        'abs_error_ms',
    ]
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_robust_pick_to_grstat(
    *,
    segy_path: str | Path,
    robust_npz_path: str | Path,
    out_crd_path: str | Path,
    out_npz_path: str | Path | None = None,
    pick_key: str = 'physical_center_i',
    phase_mode: PickMode = 'peak',
    max_shift_samples: int = 2,
    endian: Literal['big', 'little'] = 'big',
    header_comment: str = 'physical_center_i snap to phase',
    duplicate_policy: DuplicatePolicy = 'error',
    dt_multiplier: float | None = None,
    unbounded_zero_crossing: bool = False,
    strict_trace_count: bool = True,
    strict_sample_count: bool = True,
    reference_grstat_path: str | Path | None = None,
    eval_summary_json_path: str | Path | None = None,
    eval_summary_csv_path: str | Path | None = None,
    eval_per_trace_csv_path: str | Path | None = None,
    eval_strict_gather_numbers: bool = True,
    eval_strict_shape: bool = True,
) -> dict[str, object]:
    """Export ``pick_key`` from robust.npz to grstat after phase snapping."""
    if max_shift_samples < 0:
        msg = 'max_shift_samples must be >= 0'
        raise ValueError(msg)
    if phase_mode not in ('peak', 'trough', 'rising', 'trailing'):
        msg = "phase_mode must be one of: 'peak', 'trough', 'rising', 'trailing'"
        raise ValueError(msg)
    if endian not in ('big', 'little'):
        msg = "endian must be 'big' or 'little'"
        raise ValueError(msg)
    if duplicate_policy not in ('error', 'first', 'last'):
        msg = "duplicate_policy must be one of: 'error', 'first', 'last'"
        raise ValueError(msg)

    rt = _load_runtime()
    segy = Path(segy_path).expanduser().resolve()
    robust_path = Path(robust_npz_path).expanduser().resolve()
    out_crd = Path(out_crd_path).expanduser().resolve()
    out_npz = Path(out_npz_path).expanduser().resolve() if out_npz_path else None
    out_crd.parent.mkdir(parents=True, exist_ok=True)
    if out_npz is not None:
        out_npz.parent.mkdir(parents=True, exist_ok=True)

    robust = rt['load_robust_npz'](robust_path)
    if pick_key not in robust:
        msg = f'robust npz does not contain pick key: {pick_key}'
        raise KeyError(msg)

    picks_in = np.asarray(robust[pick_key], dtype=np.int32)
    ffid = np.asarray(robust['ffid_values'], dtype=np.int32)
    chno = np.asarray(robust['chno_values'], dtype=np.int32)
    trace_indices = np.asarray(
        robust.get('trace_indices', np.arange(picks_in.size)), dtype=np.int64
    )
    dt_sec = float(np.asarray(robust['dt_sec']).item())
    n_traces = int(np.asarray(robust['n_traces']).item())
    n_samples = int(np.asarray(robust['n_samples_orig']).item())

    if picks_in.shape != (n_traces,):
        msg = f'{pick_key} shape mismatch: {picks_in.shape} != {(n_traces,)}'
        raise ValueError(msg)
    if trace_indices.shape != (n_traces,):
        msg = f'trace_indices shape mismatch: {trace_indices.shape} != {(n_traces,)}'
        raise ValueError(msg)
    if np.any(picks_in < 0) or np.any(picks_in >= n_samples):
        msg = f'{pick_key} contains out-of-bounds picks'
        raise ValueError(msg)

    picks_out = picks_in.astype(np.int32, copy=True)
    with rt['open_segy_with_endian'](
        str(segy), 'r', ignore_geometry=True, segy_endian=endian
    ) as f:
        f.mmap()
        if strict_trace_count and int(f.tracecount) <= int(np.max(trace_indices)):
            msg = (
                f'SEG-Y tracecount={int(f.tracecount)} is not enough for '
                f'max trace_indices={int(np.max(trace_indices))}'
            )
            raise ValueError(msg)
        if strict_sample_count and len(np.asarray(f.trace.raw[0])) != n_samples:
            msg = (
                f'SEG-Y n_samples={len(np.asarray(f.trace.raw[0]))} '
                f'!= robust n_samples_orig={n_samples}'
            )
            raise ValueError(msg)

        for ff in _sorted_unique_int(ffid):
            idx = np.flatnonzero(ffid == ff)
            order = np.lexsort((idx, chno[idx].astype(np.int64)))
            idx_sorted = idx[order]
            seis_g = _read_trace_subset(f, trace_indices[idx_sorted])
            p_g = picks_in[idx_sorted]
            if phase_mode in ('rising', 'trailing') and not unbounded_zero_crossing:
                snapped_g = _snap_zero_crossing_windowed(
                    p_g,
                    seis_g,
                    mode=phase_mode,
                    ltcor=int(max_shift_samples),
                )
            else:
                snapped_g = rt['snap_picks_to_phase'](
                    p_g,
                    seis_g,
                    mode=phase_mode,
                    ltcor=int(max_shift_samples),
                )
            picks_out[idx_sorted] = snapped_g.astype(np.int32, copy=False)

    delta = picks_out.astype(np.int64) - picks_in.astype(np.int64)
    if np.any(np.abs(delta) > int(max_shift_samples)) and not (
        phase_mode in ('rising', 'trailing') and unbounded_zero_crossing
    ):
        msg = 'snap delta exceeded max_shift_samples'
        raise RuntimeError(msg)

    fb_mat, ffids_sorted, mat_stats = _build_fb_matrix(
        picks_i=picks_out,
        ffid_values=ffid,
        chno_values=chno,
        duplicate_policy=duplicate_policy,
    )
    dt_mult = float(dt_multiplier) if dt_multiplier is not None else dt_sec * 1000.0
    written = rt['numpy2fbcrd'](
        dt=dt_mult,
        fbnum=fb_mat,
        gather_range=ffids_sorted,
        output_name=str(out_crd),
        original=None,
        mode='gather',
        header_comment=header_comment,
    )

    changed = delta != 0
    summary: dict[str, object] = {
        'segy': str(segy),
        'robust_npz': str(robust_path),
        'out_crd': str(out_crd),
        'out_npz': str(out_npz) if out_npz is not None else '',
        'pick_key': pick_key,
        'phase_mode': phase_mode,
        'max_shift_samples': int(max_shift_samples),
        'dt_multiplier': dt_mult,
        'n_traces': n_traces,
        'changed_count': int(np.count_nonzero(changed)),
        'changed_rate': float(np.mean(changed)) if changed.size else float('nan'),
        'delta_abs_p50': _pct(np.abs(delta[changed]), 50) if np.any(changed) else 0.0,
        'delta_abs_p90': _pct(np.abs(delta[changed]), 90) if np.any(changed) else 0.0,
        'delta_abs_max': int(np.max(np.abs(delta))) if delta.size else 0,
        'physical_model_status_counts': _status_counts(
            robust.get('physical_model_status')
        ),
        **mat_stats,
    }

    eval_summary: dict[str, object] | None = None
    if reference_grstat_path:
        ref_path = Path(reference_grstat_path).expanduser().resolve()
        reference = load_grstat_matrix(ref_path, dt_multiplier=dt_mult)
        eval_summary, eval_rows = evaluate_grstat_matrix(
            prediction_samples=fb_mat,
            prediction_ffids=ffids_sorted,
            reference=reference,
            dt_multiplier=dt_mult,
            strict_gather_numbers=eval_strict_gather_numbers,
            strict_shape=eval_strict_shape,
        )
        eval_summary = {
            'reference_grstat': str(ref_path),
            'prediction_crd': str(out_crd),
            **eval_summary,
        }
        summary['evaluation'] = eval_summary

        if eval_summary_json_path:
            path = Path(eval_summary_json_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(eval_summary, ensure_ascii=False, indent=2, sort_keys=True),
                encoding='utf-8',
            )
        if eval_summary_csv_path:
            _write_summary_csv(
                Path(eval_summary_csv_path).expanduser().resolve(), eval_summary
            )
        if eval_per_trace_csv_path:
            _write_rows_csv(
                Path(eval_per_trace_csv_path).expanduser().resolve(), eval_rows
            )

    if out_npz is not None:
        arrays: dict[str, object] = {
            'dt_sec': np.asarray(dt_sec, dtype=np.float32),
            'dt_multiplier': np.asarray(dt_mult, dtype=np.float64),
            'n_samples_orig': np.asarray(n_samples, dtype=np.int32),
            'n_traces': np.asarray(n_traces, dtype=np.int32),
            'ffid_values': ffid.astype(np.int32, copy=False),
            'chno_values': chno.astype(np.int32, copy=False),
            'trace_indices': trace_indices.astype(np.int64, copy=False),
            'pick_key': np.asarray(pick_key),
            'pick_input_i': picks_in.astype(np.int32, copy=False),
            'pick_snapped_i': picks_out.astype(np.int32, copy=False),
            'pick_snap_delta_i': delta.astype(np.int32, copy=False),
            'pick_snap_changed_mask': changed.astype(np.bool_),
            'phase_mode': np.asarray(phase_mode),
            'max_shift_samples': np.asarray(int(max_shift_samples), dtype=np.int32),
            'fb_mat_samples': fb_mat.astype(np.int32, copy=False),
            'fb_mat_grstat_values': written.astype(np.int32, copy=False),
            'gather_range_ffids': np.asarray(ffids_sorted, dtype=np.int32),
            'summary_json': np.asarray(
                json.dumps(summary, ensure_ascii=False, sort_keys=True)
            ),
        }
        if eval_summary is not None:
            arrays['evaluation_summary_json'] = np.asarray(
                json.dumps(eval_summary, ensure_ascii=False, sort_keys=True)
            )
        np.savez_compressed(out_npz, **arrays)

    return summary
