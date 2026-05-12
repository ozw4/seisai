"""Compare fbpick physics robust outputs and runtime diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from seisai_engine.pipelines.fbpick.physics.physical_center import (
    PHYSICAL_MODEL_FAILURE_LABELS,
    PHYSICAL_MODEL_STATUS_LABELS,
)
from seisai_engine.pipelines.fbpick.physics.runtime_diagnostics import (
    PHYSICS_RUNTIME_DIAGNOSTIC_KEYS,
    PHYSICS_RUNTIME_STRING_DIAGNOSTIC_KEYS,
    derive_physics_runtime_summary_path,
    runtime_summary_from_npz_fields,
)

__all__ = [
    'compare_paths',
    'main',
    'write_compare_csv',
]

SAMPLE_DIFF_KEYS = (
    'physical_center_i',
    'fine_center_i',
)
TIME_DIFF_KEYS = (
    'physical_center_t_sec',
    'fine_center_t_sec',
)
EXPORT_SAMPLE_DIFF_KEYS = (
    'pick_snapped_i',
    'pick_snap_delta_i',
)
RUNTIME_TOTAL_KEYS = (
    'physics_total_sec',
    'physical_center_total_sec',
    'ransac_fit_total_sec',
)
RUNTIME_COUNT_KEYS = (
    'n_fit_calls',
    'n_anchor_fit_calls',
    'observation_sampling_enabled',
    'max_obs_per_fit',
    'n_offset_bins',
    'n_source_groups',
    'n_non_anchor_groups',
    'n_reused_predictions',
    'n_t0_shifted_groups',
    'n_t0_shifted_predictions',
    'n_adaptive_refit_calls',
    'n_adaptive_refit_success',
    'n_adaptive_refit_failed',
    'n_fallback_full_fit_no_compatible_anchor',
    'n_anchor_groups',
    'anchor_stride_source_groups',
)
RUNTIME_RATE_KEYS = (
    'cache_hit_rate',
    'fit_call_reduction_rate_vs_full',
    'adaptive_refit_rate',
)
RUNTIME_ANCHOR_VALUE_KEYS = (
    'anchor_selection_mode',
    'observation_sampling_method',
    'anchor_source_distance_p50_m',
    'anchor_source_distance_p90_m',
    'anchor_source_distance_max_m',
    'obs_count_before_p50',
    'obs_count_before_p90',
    'obs_count_before_p99',
    'obs_count_after_p50',
    'obs_count_after_p90',
    'obs_count_after_p99',
    'obs_downsample_rate_p50',
    'obs_downsample_rate_p90',
    't0_shift_ms_p50',
    't0_shift_ms_p90',
    't0_shift_ms_p99',
    'reuse_resid_p90_ms_p50',
    'reuse_resid_p90_ms_p90',
)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    npz_path = Path(path).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_scalar(value.item())
        return [_json_scalar(item) for item in value.tolist()]
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_scalar(item) for item in value]
    return value


def _valid_numeric_mask(arr: np.ndarray, *, negative_is_missing: bool) -> np.ndarray:
    values = np.asarray(arr)
    if np.issubdtype(values.dtype, np.floating):
        valid = np.isfinite(values)
    else:
        valid = np.ones(values.shape, dtype=np.bool_)
    if negative_is_missing and np.issubdtype(values.dtype, np.number):
        valid &= values >= 0
    return valid


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values.astype(np.float64, copy=False), float(q)))


def _rate(mask: np.ndarray) -> float:
    if mask.size == 0:
        return float('nan')
    return float(np.mean(mask.astype(np.float64)))


def _missing_diff_stats(
    *,
    key: str,
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, Any]:
    return {
        'available': False,
        'key': key,
        'missing_baseline_field': key not in baseline,
        'missing_candidate_field': key not in candidate,
    }


def _diff_stats(
    *,
    key: str,
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    unit: str,
    include_sample_rates: bool,
    negative_is_missing: bool = True,
) -> dict[str, Any]:
    if key not in baseline or key not in candidate:
        return _missing_diff_stats(key=key, baseline=baseline, candidate=candidate)

    base = np.asarray(baseline[key]).reshape(-1)
    cand = np.asarray(candidate[key]).reshape(-1)
    if base.shape != cand.shape:
        msg = f'{key} shape mismatch: baseline {base.shape}, candidate {cand.shape}'
        raise ValueError(msg)

    valid_base = _valid_numeric_mask(base, negative_is_missing=negative_is_missing)
    valid_cand = _valid_numeric_mask(cand, negative_is_missing=negative_is_missing)
    valid = valid_base & valid_cand
    diff = cand[valid].astype(np.float64, copy=False) - base[valid].astype(
        np.float64,
        copy=False,
    )
    abs_diff = np.abs(diff)

    stats: dict[str, Any] = {
        'available': True,
        'key': key,
        'n': int(base.size),
        'n_valid_both': int(np.count_nonzero(valid)),
        'n_missing_baseline': int(base.size - np.count_nonzero(valid_base)),
        'n_missing_candidate': int(cand.size - np.count_nonzero(valid_cand)),
    }
    if diff.size == 0:
        stats.update(
            {
                f'bias_mean_{unit}': None,
                f'abs_diff_mean_{unit}': None,
                f'abs_diff_p50_{unit}': None,
                f'abs_diff_p90_{unit}': None,
                f'abs_diff_p95_{unit}': None,
                f'abs_diff_p99_{unit}': None,
                f'abs_diff_max_{unit}': None,
            }
        )
        if include_sample_rates:
            for threshold in (1, 2, 4, 8, 16):
                stats[f'within_{threshold}_sample_rate'] = None
        return stats

    stats.update(
        {
            f'bias_mean_{unit}': float(np.mean(diff)),
            f'abs_diff_mean_{unit}': float(np.mean(abs_diff)),
            f'abs_diff_p50_{unit}': _percentile(abs_diff, 50.0),
            f'abs_diff_p90_{unit}': _percentile(abs_diff, 90.0),
            f'abs_diff_p95_{unit}': _percentile(abs_diff, 95.0),
            f'abs_diff_p99_{unit}': _percentile(abs_diff, 99.0),
            f'abs_diff_max_{unit}': float(np.max(abs_diff)),
        }
    )
    if include_sample_rates:
        for threshold in (1, 2, 4, 8, 16):
            stats[f'within_{threshold}_sample_rate'] = _rate(abs_diff <= threshold)
    return stats


def _status_counts(
    *,
    key: str,
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    labels: dict[int, str],
) -> dict[str, Any]:
    if key not in baseline or key not in candidate:
        return {
            'available': False,
            'key': key,
            'missing_baseline_field': key not in baseline,
            'missing_candidate_field': key not in candidate,
        }

    base = np.asarray(baseline[key]).reshape(-1)
    cand = np.asarray(candidate[key]).reshape(-1)
    if base.shape != cand.shape:
        msg = f'{key} shape mismatch: baseline {base.shape}, candidate {cand.shape}'
        raise ValueError(msg)

    base_counts = Counter(int(value) for value in base.tolist())
    cand_counts = Counter(int(value) for value in cand.tolist())
    value_order = sorted(set(labels) | set(base_counts) | set(cand_counts))

    def _format_counts(counts: Counter[int]) -> dict[str, int]:
        out: dict[str, int] = {}
        for value in value_order:
            label = labels.get(value, f'value_{value}')
            out[label] = int(counts.get(value, 0))
        return out

    baseline_counts = _format_counts(base_counts)
    candidate_counts = _format_counts(cand_counts)
    return {
        'available': True,
        'key': key,
        'n': int(base.size),
        'counts_match': baseline_counts == candidate_counts,
        'arrays_match': bool(np.array_equal(base, cand)),
        'baseline': baseline_counts,
        'candidate': candidate_counts,
    }


def _load_runtime_json(path: Path) -> dict[str, float | int | str] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        msg = f'runtime summary must be a JSON object: {path}'
        raise TypeError(msg)

    out: dict[str, float | int | str] = {}
    for key in PHYSICS_RUNTIME_DIAGNOSTIC_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if key in PHYSICS_RUNTIME_STRING_DIAGNOSTIC_KEYS:
            if not isinstance(value, str):
                msg = f'runtime summary key {key} must be string: {path}'
                raise TypeError(msg)
            out[key] = value
            continue
        if not isinstance(value, int | float):
            msg = f'runtime summary key {key} must be numeric: {path}'
            raise TypeError(msg)
        if key.startswith('n_') or key in {
            'observation_sampling_enabled',
            'max_obs_per_fit',
        }:
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


def _load_runtime_summary(
    *,
    robust_path: Path,
    robust_payload: dict[str, np.ndarray],
    runtime_json_path: str | Path | None,
) -> tuple[dict[str, float | int | str] | None, Path | None, str | None]:
    path = (
        Path(runtime_json_path).expanduser().resolve()
        if runtime_json_path is not None
        else derive_physics_runtime_summary_path(robust_path)
    )
    summary = _load_runtime_json(path)
    if summary is not None:
        return summary, path, 'json'

    embedded = runtime_summary_from_npz_fields(robust_payload)
    if embedded is not None:
        return embedded, None, 'robust_npz'
    return None, path, None


def _runtime_speedup(
    baseline: dict[str, float | int | str],
    candidate: dict[str, float | int | str],
    key: str,
) -> float | None:
    base = baseline.get(key)
    cand = candidate.get(key)
    if base is None or cand is None:
        return None
    cand_float = float(cand)
    if cand_float <= 0.0:
        return None
    return float(base) / cand_float


def _fit_call_reduction_rate(
    baseline: dict[str, float | int | str],
    candidate: dict[str, float | int | str],
) -> float | None:
    base = baseline.get('n_fit_calls')
    cand = candidate.get('n_fit_calls')
    if base is None or cand is None:
        return None
    base_float = float(base)
    if base_float <= 0.0:
        return None
    return (base_float - float(cand)) / base_float


def _runtime_compare(
    *,
    baseline_robust_path: Path,
    candidate_robust_path: Path,
    baseline_robust: dict[str, np.ndarray],
    candidate_robust: dict[str, np.ndarray],
    baseline_runtime_json: str | Path | None,
    candidate_runtime_json: str | Path | None,
) -> dict[str, Any]:
    baseline_summary, baseline_path, baseline_source = _load_runtime_summary(
        robust_path=baseline_robust_path,
        robust_payload=baseline_robust,
        runtime_json_path=baseline_runtime_json,
    )
    candidate_summary, candidate_path, candidate_source = _load_runtime_summary(
        robust_path=candidate_robust_path,
        robust_payload=candidate_robust,
        runtime_json_path=candidate_runtime_json,
    )

    out: dict[str, Any] = {
        'available': baseline_summary is not None and candidate_summary is not None,
        'baseline_available': baseline_summary is not None,
        'candidate_available': candidate_summary is not None,
        'baseline_runtime_json': (
            str(baseline_path) if baseline_path is not None else None
        ),
        'candidate_runtime_json': (
            str(candidate_path) if candidate_path is not None else None
        ),
        'baseline_source': baseline_source,
        'candidate_source': candidate_source,
    }

    for key in RUNTIME_TOTAL_KEYS:
        out[f'{key}_baseline'] = (
            baseline_summary.get(key) if baseline_summary is not None else None
        )
        out[f'{key}_candidate'] = (
            candidate_summary.get(key) if candidate_summary is not None else None
        )
        out[f'speedup_{key.removesuffix("_sec")}'] = _runtime_speedup(
            baseline_summary or {},
            candidate_summary or {},
            key,
        )
    for key in RUNTIME_COUNT_KEYS:
        out[f'{key}_baseline'] = (
            baseline_summary.get(key) if baseline_summary is not None else None
        )
        out[f'{key}_candidate'] = (
            candidate_summary.get(key) if candidate_summary is not None else None
        )
    for key in RUNTIME_RATE_KEYS:
        out[f'{key}_baseline'] = (
            baseline_summary.get(key) if baseline_summary is not None else None
        )
        out[f'{key}_candidate'] = (
            candidate_summary.get(key) if candidate_summary is not None else None
        )
    for key in RUNTIME_ANCHOR_VALUE_KEYS:
        out[f'{key}_baseline'] = (
            baseline_summary.get(key) if baseline_summary is not None else None
        )
        out[f'{key}_candidate'] = (
            candidate_summary.get(key) if candidate_summary is not None else None
        )
    out['fit_call_reduction_rate'] = _fit_call_reduction_rate(
        baseline_summary or {},
        candidate_summary or {},
    )
    return out


def compare_paths(
    *,
    baseline_robust: str | Path,
    candidate_robust: str | Path,
    baseline_export: str | Path | None = None,
    candidate_export: str | Path | None = None,
    baseline_runtime_json: str | Path | None = None,
    candidate_runtime_json: str | Path | None = None,
) -> dict[str, Any]:
    baseline_robust_path = Path(baseline_robust).expanduser().resolve()
    candidate_robust_path = Path(candidate_robust).expanduser().resolve()
    baseline = _load_npz(baseline_robust_path)
    candidate = _load_npz(candidate_robust_path)

    center_diffs: dict[str, Any] = {}
    for key in SAMPLE_DIFF_KEYS:
        center_diffs[f'{key}_diff'] = _diff_stats(
            key=key,
            baseline=baseline,
            candidate=candidate,
            unit='samples',
            include_sample_rates=True,
        )
    for key in TIME_DIFF_KEYS:
        center_diffs[f'{key}_diff'] = _diff_stats(
            key=key,
            baseline=baseline,
            candidate=candidate,
            unit='sec',
            include_sample_rates=False,
        )

    status_counts = {
        'physical_model_status': _status_counts(
            key='physical_model_status',
            baseline=baseline,
            candidate=candidate,
            labels=PHYSICAL_MODEL_STATUS_LABELS,
        ),
        'physical_model_failure_reason': _status_counts(
            key='physical_model_failure_reason',
            baseline=baseline,
            candidate=candidate,
            labels=PHYSICAL_MODEL_FAILURE_LABELS,
        ),
    }

    if (baseline_export is None) != (candidate_export is None):
        msg = '--baseline-export and --candidate-export must be provided together'
        raise ValueError(msg)

    export_diffs: dict[str, Any] = {}
    baseline_export_path: Path | None = None
    candidate_export_path: Path | None = None
    if baseline_export is not None and candidate_export is not None:
        baseline_export_path = Path(baseline_export).expanduser().resolve()
        candidate_export_path = Path(candidate_export).expanduser().resolve()
        baseline_export_payload = _load_npz(baseline_export_path)
        candidate_export_payload = _load_npz(candidate_export_path)
        for key in EXPORT_SAMPLE_DIFF_KEYS:
            export_diffs[f'{key}_diff'] = _diff_stats(
                key=key,
                baseline=baseline_export_payload,
                candidate=candidate_export_payload,
                unit='samples',
                include_sample_rates=True,
                negative_is_missing=key != 'pick_snap_delta_i',
            )

    runtime = _runtime_compare(
        baseline_robust_path=baseline_robust_path,
        candidate_robust_path=candidate_robust_path,
        baseline_robust=baseline,
        candidate_robust=candidate,
        baseline_runtime_json=baseline_runtime_json,
        candidate_runtime_json=candidate_runtime_json,
    )

    return _json_scalar(
        {
            'baseline_robust': str(baseline_robust_path),
            'candidate_robust': str(candidate_robust_path),
            'baseline_export': (
                str(baseline_export_path) if baseline_export_path is not None else None
            ),
            'candidate_export': (
                str(candidate_export_path)
                if candidate_export_path is not None
                else None
            ),
            'center_diffs': center_diffs,
            'status_counts': status_counts,
            'runtime': runtime,
            'export_diffs': export_diffs,
        }
    )


def _flatten_rows(result: dict[str, Any]) -> list[tuple[str, str, Any]]:
    rows: list[tuple[str, str, Any]] = []
    for group_name, group in result['center_diffs'].items():
        for key, value in group.items():
            rows.append((group_name, key, value))
    for field_name, group in result['status_counts'].items():
        rows.append(('status_counts', f'{field_name}_available', group['available']))
        if not group['available']:
            rows.append(
                (
                    'status_counts',
                    f'{field_name}_missing_baseline_field',
                    group['missing_baseline_field'],
                )
            )
            rows.append(
                (
                    'status_counts',
                    f'{field_name}_missing_candidate_field',
                    group['missing_candidate_field'],
                )
            )
            continue
        rows.append(
            ('status_counts', f'{field_name}_counts_match', group['counts_match'])
        )
        rows.append(
            ('status_counts', f'{field_name}_arrays_match', group['arrays_match'])
        )
        for side in ('baseline', 'candidate'):
            for label, count in group[side].items():
                rows.append(('status_counts', f'{side}_{label}', count))
    for key, value in result['runtime'].items():
        rows.append(('runtime', key, value))
    for group_name, group in result['export_diffs'].items():
        for key, value in group.items():
            rows.append((group_name, key, value))
    return rows


def write_compare_csv(path: str | Path, result: dict[str, Any]) -> Path:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(('metric_group', 'key', 'value'))
        for group, key, value in _flatten_rows(result):
            writer.writerow((group, key, '' if value is None else value))
    return out_path


def _write_json(path: str | Path, result: dict[str, Any]) -> Path:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    return out_path


def _format_optional_float(value: Any, *, suffix: str = '') -> str:
    if value is None:
        return 'n/a'
    return f'{float(value):.6g}{suffix}'


def _print_summary(result: dict[str, Any]) -> None:
    print('compare_fbpick_physics_runtime')
    print(f'baseline: {result["baseline_robust"]}')
    print(f'candidate: {result["candidate_robust"]}')
    for group_name in ('physical_center_i_diff', 'fine_center_i_diff'):
        group = result['center_diffs'][group_name]
        if not group['available']:
            print(f'{group_name}: missing optional field')
            continue
        p90 = _format_optional_float(group['abs_diff_p90_samples'])
        max_value = _format_optional_float(group['abs_diff_max_samples'])
        within4 = _format_optional_float(group['within_4_sample_rate'])
        print(
            f'{group_name}: n_valid={group["n_valid_both"]} '
            f'p90={p90} samples max={max_value} samples within4={within4}'
        )

    status = result['status_counts']['physical_model_status']
    failure = result['status_counts']['physical_model_failure_reason']
    print(
        'status_counts: '
        f'physical_model_status_match={status.get("counts_match")} '
        f'failure_reason_match={failure.get("counts_match")}'
    )

    runtime = result['runtime']
    if runtime['available']:
        speedup = _format_optional_float(runtime['speedup_physics_total'])
        baseline_sec = _format_optional_float(
            runtime['physics_total_sec_baseline'],
            suffix='s',
        )
        candidate_sec = _format_optional_float(
            runtime['physics_total_sec_candidate'],
            suffix='s',
        )
        print(
            'runtime: '
            f'physics_total baseline={baseline_sec} '
            f'candidate={candidate_sec} speedup={speedup}'
        )
    else:
        print(
            'runtime: unavailable '
            f'baseline_available={runtime["baseline_available"]} '
            f'candidate_available={runtime["candidate_available"]}'
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--baseline',
        '--baseline-robust',
        dest='baseline_robust',
        required=True,
    )
    parser.add_argument(
        '--candidate',
        '--candidate-robust',
        dest='candidate_robust',
        required=True,
    )
    parser.add_argument('--baseline-export')
    parser.add_argument('--candidate-export')
    parser.add_argument('--baseline-runtime-json')
    parser.add_argument('--candidate-runtime-json')
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--out-csv')
    args = parser.parse_args(argv)

    result = compare_paths(
        baseline_robust=args.baseline_robust,
        candidate_robust=args.candidate_robust,
        baseline_export=args.baseline_export,
        candidate_export=args.candidate_export,
        baseline_runtime_json=args.baseline_runtime_json,
        candidate_runtime_json=args.candidate_runtime_json,
    )
    json_path = _write_json(args.out_json, result)
    csv_path = write_compare_csv(args.out_csv, result) if args.out_csv else None
    _print_summary(result)
    print(f'wrote_json: {json_path}')
    if csv_path is not None:
        print(f'wrote_csv: {csv_path}')


if __name__ == '__main__':
    main()
