from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from cli.compare_fbpick_physics_runtime import compare_paths, write_compare_csv
from seisai_engine.pipelines.fbpick.common import save_robust_npz


def _minimal_robust_payload() -> dict[str, np.ndarray]:
    return {
        'dt_sec': np.asarray(0.004, dtype=np.float32),
        'n_samples_orig': np.asarray(512, dtype=np.int32),
        'n_traces': np.asarray(2, dtype=np.int32),
        'ffid_values': np.array([1, 1], dtype=np.int32),
        'chno_values': np.array([1, 2], dtype=np.int32),
        'offsets_m': np.array([100.0, 200.0], dtype=np.float32),
        'trace_indices': np.array([0, 1], dtype=np.int64),
        'robust_pick_i': np.array([100, 101], dtype=np.int32),
        'robust_pick_t_sec': np.array([0.4, 0.404], dtype=np.float32),
        'robust_conf': np.array([0.9, 0.8], dtype=np.float32),
        'robust_source': np.array([0, 0], dtype=np.uint8),
        'used_theoretical_mask': np.array([False, False], dtype=np.bool_),
        'reason_mask': np.array([0, 0], dtype=np.uint8),
        'conf_prob1': np.array([0.9, 0.8], dtype=np.float32),
        'conf_trend1': np.array([0.9, 0.8], dtype=np.float32),
        'conf_rs1': np.array([1.0, 1.0], dtype=np.float32),
        'lineage': np.asarray('{"stage":"test"}'),
    }


def test_compare_paths_reports_detailed_runtime_rows(tmp_path: Path) -> None:
    baseline = save_robust_npz(
        tmp_path / 'baseline.robust.npz',
        **_minimal_robust_payload(),
    )
    candidate = save_robust_npz(
        tmp_path / 'candidate.robust.npz',
        **_minimal_robust_payload(),
    )
    baseline_json = tmp_path / 'baseline.physics_runtime_summary.json'
    candidate_json = tmp_path / 'candidate.physics_runtime_summary.json'
    baseline_json.write_text(
        json.dumps(
            {
                'physics_total_sec': 100.0,
                'physical_center_total_sec': 90.0,
                'ransac_fit_total_sec': 20.0,
                'non_ransac_total_sec': 70.0,
                'n_fit_calls': 10,
                'n_fit_contexts': 10,
                'n_reuse_contexts': 0,
                'n_adaptive_refit_calls': 0,
                'n_fallback_full_fit_no_compatible_anchor': 0,
            }
        ),
        encoding='utf-8',
    )
    candidate_json.write_text(
        json.dumps(
            {
                'physics_total_sec': 80.0,
                'physical_center_total_sec': 75.0,
                'ransac_fit_total_sec': 10.0,
                'non_ransac_total_sec': 65.0,
                'n_fit_calls': 5,
                'n_fit_contexts': 7,
                'n_reuse_contexts': 2,
                'n_adaptive_refit_calls': 1,
                'n_fallback_full_fit_no_compatible_anchor': 1,
            }
        ),
        encoding='utf-8',
    )

    result = compare_paths(
        baseline_robust=baseline,
        candidate_robust=candidate,
        baseline_runtime_json=baseline_json,
        candidate_runtime_json=candidate_json,
    )
    runtime = result['runtime']
    assert runtime['speedup_physics_total'] == pytest.approx(1.25)
    assert runtime['speedup_physical_center_total'] == pytest.approx(1.2)
    assert runtime['speedup_ransac_fit_total'] == pytest.approx(2.0)
    assert runtime['speedup_non_ransac_total'] == pytest.approx(70.0 / 65.0)
    assert runtime['n_fit_calls_baseline'] == 10
    assert runtime['n_fit_calls_candidate'] == 5
    assert runtime['n_reuse_contexts_candidate'] == 2

    csv_path = write_compare_csv(tmp_path / 'compare.csv', result)
    csv_text = csv_path.read_text(encoding='utf-8')
    assert 'speedup_non_ransac_total' in csv_text
    assert 'n_fit_contexts_candidate' in csv_text
