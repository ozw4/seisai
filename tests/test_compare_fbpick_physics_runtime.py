from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cli.compare_fbpick_physics_runtime import compare_paths, main


def _write_robust_npz(
    path: Path,
    *,
    physical_center_i: np.ndarray,
    fine_center_i: np.ndarray | None = None,
    physical_model_status: np.ndarray | None = None,
    physical_model_failure_reason: np.ndarray | None = None,
) -> Path:
    arrays: dict[str, np.ndarray] = {
        'physical_center_i': physical_center_i.astype(np.int32, copy=False),
    }
    if fine_center_i is not None:
        arrays['fine_center_i'] = fine_center_i.astype(np.int32, copy=False)
    if physical_model_status is not None:
        arrays['physical_model_status'] = physical_model_status.astype(
            np.uint8,
            copy=False,
        )
    if physical_model_failure_reason is not None:
        arrays['physical_model_failure_reason'] = physical_model_failure_reason.astype(
            np.uint8,
            copy=False,
        )
    np.savez_compressed(path, **arrays)
    return path


def test_identical_synthetic_robust_npz_returns_zero_diff(tmp_path: Path) -> None:
    baseline = _write_robust_npz(
        tmp_path / 'baseline.robust.npz',
        physical_center_i=np.asarray([10, 20, 30]),
        fine_center_i=np.asarray([11, 21, 31]),
        physical_model_status=np.asarray([0, 4, 8]),
        physical_model_failure_reason=np.asarray([0, 4, 1]),
    )
    candidate = _write_robust_npz(
        tmp_path / 'candidate.robust.npz',
        physical_center_i=np.asarray([10, 20, 30]),
        fine_center_i=np.asarray([11, 21, 31]),
        physical_model_status=np.asarray([0, 4, 8]),
        physical_model_failure_reason=np.asarray([0, 4, 1]),
    )

    result = compare_paths(baseline_robust=baseline, candidate_robust=candidate)

    physical = result['center_diffs']['physical_center_i_diff']
    fine = result['center_diffs']['fine_center_i_diff']
    assert physical['abs_diff_max_samples'] == 0.0
    assert physical['within_1_sample_rate'] == 1.0
    assert fine['abs_diff_p90_samples'] == 0.0
    assert result['status_counts']['physical_model_status']['counts_match'] is True


def test_known_sample_shifts_return_expected_percentiles(tmp_path: Path) -> None:
    baseline_values = np.asarray([100, 100, 100, 100, 100])
    candidate_values = baseline_values + np.asarray([1, 1, 1, 10, 10])
    baseline = _write_robust_npz(
        tmp_path / 'baseline.robust.npz',
        physical_center_i=baseline_values,
    )
    candidate = _write_robust_npz(
        tmp_path / 'candidate.robust.npz',
        physical_center_i=candidate_values,
    )

    result = compare_paths(baseline_robust=baseline, candidate_robust=candidate)

    stats = result['center_diffs']['physical_center_i_diff']
    assert stats['bias_mean_samples'] == 4.6
    assert stats['abs_diff_p50_samples'] == 1.0
    assert stats['abs_diff_p90_samples'] == 10.0
    assert stats['abs_diff_max_samples'] == 10.0
    assert stats['within_4_sample_rate'] == 0.6


def test_missing_optional_runtime_json_does_not_crash(tmp_path: Path) -> None:
    baseline = _write_robust_npz(
        tmp_path / 'baseline.robust.npz',
        physical_center_i=np.asarray([1, 2, 3]),
    )
    candidate = _write_robust_npz(
        tmp_path / 'candidate.robust.npz',
        physical_center_i=np.asarray([1, 2, 3]),
    )

    result = compare_paths(baseline_robust=baseline, candidate_robust=candidate)

    assert result['runtime']['available'] is False
    assert result['runtime']['baseline_available'] is False
    assert result['runtime']['candidate_available'] is False


def test_status_count_comparison_works(tmp_path: Path) -> None:
    baseline = _write_robust_npz(
        tmp_path / 'baseline.robust.npz',
        physical_center_i=np.asarray([1, 2, 3, 4]),
        physical_model_status=np.asarray([0, 0, 4, 8]),
        physical_model_failure_reason=np.asarray([0, 0, 4, 1]),
    )
    candidate = _write_robust_npz(
        tmp_path / 'candidate.robust.npz',
        physical_center_i=np.asarray([1, 2, 3, 4]),
        physical_model_status=np.asarray([0, 4, 4, 8]),
        physical_model_failure_reason=np.asarray([0, 4, 4, 1]),
    )

    result = compare_paths(baseline_robust=baseline, candidate_robust=candidate)

    status = result['status_counts']['physical_model_status']
    assert status['counts_match'] is False
    assert status['baseline']['two_piece_ok'] == 2
    assert status['candidate']['two_piece_ok'] == 1
    assert status['candidate']['fallback_robust'] == 2


def test_cli_writes_json_csv_and_compares_export_npz(tmp_path: Path) -> None:
    baseline = _write_robust_npz(
        tmp_path / 'baseline.robust.npz',
        physical_center_i=np.asarray([10, 20]),
        fine_center_i=np.asarray([11, 21]),
    )
    candidate = _write_robust_npz(
        tmp_path / 'candidate.robust.npz',
        physical_center_i=np.asarray([10, 22]),
        fine_center_i=np.asarray([11, 23]),
    )
    baseline_export = tmp_path / 'baseline.snap.npz'
    candidate_export = tmp_path / 'candidate.snap.npz'
    np.savez_compressed(
        baseline_export,
        pick_snapped_i=np.asarray([10, 20], dtype=np.int32),
        pick_snap_delta_i=np.asarray([-10, 1], dtype=np.int32),
    )
    np.savez_compressed(
        candidate_export,
        pick_snapped_i=np.asarray([10, 25], dtype=np.int32),
        pick_snap_delta_i=np.asarray([-7, 3], dtype=np.int32),
    )
    out_json = tmp_path / 'compare.json'
    out_csv = tmp_path / 'compare.csv'

    main(
        [
            '--baseline',
            str(baseline),
            '--candidate',
            str(candidate),
            '--baseline-export',
            str(baseline_export),
            '--candidate-export',
            str(candidate_export),
            '--out-json',
            str(out_json),
            '--out-csv',
            str(out_csv),
        ]
    )

    payload = json.loads(out_json.read_text(encoding='utf-8'))
    assert payload['center_diffs']['physical_center_i_diff']['abs_diff_max_samples'] == 2.0
    assert payload['export_diffs']['pick_snapped_i_diff']['abs_diff_max_samples'] == 5.0
    assert payload['export_diffs']['pick_snap_delta_i_diff']['n_valid_both'] == 2
    assert payload['export_diffs']['pick_snap_delta_i_diff']['abs_diff_max_samples'] == 3.0
    assert out_csv.read_text(encoding='utf-8').splitlines()[0] == 'metric_group,key,value'
